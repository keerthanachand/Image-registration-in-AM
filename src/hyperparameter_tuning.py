#!/usr/bin/env python3
import os, json, numpy as np, tensorflow as tf, optuna, h5py
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import voxelmorph as vxm
from voxelmorph import losses
from train_utils import split_by_index, initialize_generator_parameters, vxm_data_generator
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.integration import TFKerasPruningCallback

# ========= CONFIG =========
MODEL_JSON     = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
MODEL_WEIGHTS  = "/home/kchand/results/cross_validation/vxm_weights_fold/weights_fold0.h5"
TRAIN_HDF5     = "/home/kchand/input_data/all_samples_simple_structures.h5"
SAVE_ROOT      = "/home/kchand/results/finetune_two_stage_optuna"
PATCH_SIZE     = (128, 128, 128)
STUDY_NAME     = "vxm_two_stage_val"
STORAGE_URL    = "sqlite:////home/kchand/results/voxelmorph_two_stage.db"
TRIALS_PER_WORKER = 32
SEED_BASE      = 1000

# Explicit split by index
NUM_SAMPLES = 16
TEST_IDX = [12, 13, 14, 15]
VAL_IDX  = [6, 11]

#create data 
import time, os

# only array task 0 makes the split
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
train_h5 = "/home/kchand/input_data/train_data_temp.h5"
val_h5   = "/home/kchand/input_data/val_data_temp.h5"
test_h5  = "/home/kchand/input_data/test_data_temp.h5"

if task_id == "0":
    train_h5, val_h5, test_h5 = split_by_index(TRAIN_HDF5, TEST_IDX, VAL_IDX, NUM_SAMPLES)
else:
    # wait until files exist (created by task 0)
    for _ in range(120):  # up to ~2 minutes
        if all(os.path.exists(p) for p in [train_h5, val_h5, test_h5]):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Split files not found after waiting.")


# ========= TF setup =========
def set_tf_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    print("Visible GPUs:", gpus)

# ========= Helpers =========
def load_pretrained(json_path, weights_path):
    with open(json_path, "r") as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={
            "SpatialTransformer": vxm.layers.SpatialTransformer,
            "VxmDense": vxm.networks.VxmDense,
        },
    )
    model.load_weights(weights_path)
    return model

def freeze_first_k_encoder_levels_convs(model, k_levels=2):
    frozen = 0
    for layer in model.layers:
        for lvl in range(k_levels):
            if layer.name.startswith(f"vxm_dense_unet_enc_conv_{lvl}_") and isinstance(layer, tf.keras.layers.Conv3D):
                layer.trainable = False
                frozen += 1
    print(f"[freeze] Froze {frozen} Conv3D layers (first {k_levels} encoder levels).")

def compile_vm(model, lr, lambda_smooth):
    ncc  = losses.NCC().loss
    grad = losses.Grad("l2").loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[ncc, grad],
        loss_weights=[1.0, lambda_smooth],
    )

# ========= Objective (optimize val loss) =========
def objective(trial, cfg):
    # Seed
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed = cfg["seed_base"] + array_id * 1000 + trial.number
    np.random.seed(seed); tf.random.set_seed(seed)

    # --- Search space (fixed & corrected) ---
    batch_size      = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    steps_per_epoch = trial.suggest_categorical("steps_per_epoch", [20, 50, 80])

    freeze_levels   = trial.suggest_int("freeze_levels_stage1", 2, 3)  # only deeper layers train in stage 1

    # discrete values -> categorical
    lambda_smooth   = trial.suggest_categorical("lambda_smooth", [0.1, 0.2, 0.25, 0.3])

    # widened ranges; low < high
    lr_stage1       = trial.suggest_categorical("lr_stage1", [1e-5, 1e-4, 1e-3])
    lr_stage2       = trial.suggest_categorical("lr_stage2", [1e-6, 1e-5, 5e-4])

    # discrete epoch choices -> categorical
    epochs_stage1   = trial.suggest_categorical("epochs_stage1", [50, 70, 100])
    epochs_stage2   = trial.suggest_categorical("epochs_stage2", [10, 25, 50])
    #epochs_stage1   = trial.suggest_categorical("epochs_stage1", [10])
    #epochs_stage2   = trial.suggest_categorical("epochs_stage2", [10])

    # scheduler per stage (don’t overwrite; and use them below)
    lr_factor_s1    = trial.suggest_float("lr_factor_s1", 0.3, 0.8)
    lr_factor_s2    = trial.suggest_float("lr_factor_s2", 0.3, 0.8)
    lr_patience_s1  = trial.suggest_int("lr_patience_stage1", 5, 10)
    lr_patience_s2  = trial.suggest_int("lr_patience_stage2", 2, 5)
    lr_min          = trial.suggest_float("lr_min", 1e-8, 1e-7, log=True)

    # Trial dir & params
    tdir = os.path.join(cfg["save_root"], f"trial_{trial.number:04d}")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "params.json"), "w") as f:
        json.dump({**trial.params, "seed": seed}, f, indent=2)

    # Generators (separate params for val if your util supports no-aug/no-shuffle)
    train_params = initialize_generator_parameters(hdf5_file=cfg["train_hdf5"], patch_size=cfg["patch_size"])
    val_params   = initialize_generator_parameters(hdf5_file=cfg["val_hdf5"],   patch_size=cfg["patch_size"])
    train_gen = vxm_data_generator(cfg["train_hdf5"], cfg["patch_size"], batch_size, train_params)
    val_gen   = vxm_data_generator(cfg["val_hdf5"],   cfg["patch_size"], batch_size, val_params)

    # Reasonable default if your generator is infinite
    import math
    VAL_SAMPLES = 2048  # try 4096 if val loss looks noisy
    val_steps   = max(1, math.ceil(VAL_SAMPLES / batch_size))


    # ===== Stage 1 =====
    model = load_pretrained(cfg["model_json"], cfg["model_weights"])
    freeze_first_k_encoder_levels_convs(model, freeze_levels)
    compile_vm(model, lr_stage1, lambda_smooth)

    cb1 = [
    CSVLogger(os.path.join(tdir, "hist_stage1.csv")),
    ModelCheckpoint(os.path.join(tdir, "weights_stage1.h5"),
                    save_best_only=True, save_weights_only=True,
                    monitor="val_loss", mode="min"),
    ReduceLROnPlateau(monitor="val_loss", factor=lr_factor_s1,
                      patience=lr_patience_s1, min_lr=lr_min, verbose=0),
    TFKerasPruningCallback(trial, monitor="val_loss"),   # 👈
    ]
    try:
        model.fit(train_gen,
                  validation_data=val_gen,
                  epochs=epochs_stage1,
                  steps_per_epoch=steps_per_epoch,
                  validation_steps=val_steps,
                  verbose=0,
                  callbacks=cb1)
    except optuna.TrialPruned:
        tf.keras.backend.clear_session()
        raise
    except tf.errors.ResourceExhaustedError:
        tf.keras.backend.clear_session()
        return float("inf")

    # ===== Stage 2 =====
    for l in model.layers: l.trainable = True
    compile_vm(model, lr_stage2, lambda_smooth)

    cb2 = [
    CSVLogger(os.path.join(tdir, "hist_stage2.csv")),
    ModelCheckpoint(os.path.join(tdir, "weights_stage2.h5"),
                    save_best_only=True, save_weights_only=True,
                    monitor="val_loss", mode="min"),
    ReduceLROnPlateau(monitor="val_loss", factor=lr_factor_s2,
                      patience=lr_patience_s2, min_lr=lr_min, verbose=0),
    TFKerasPruningCallback(trial, monitor="val_loss"),   # 👈
    ]
    try:
        hist2 = model.fit(train_gen,
                      validation_data=val_gen,
                      epochs=epochs_stage2,
                      steps_per_epoch=steps_per_epoch,
                      validation_steps=val_steps,
                      verbose=0,
                      callbacks=cb2)
    except optuna.TrialPruned:
        tf.keras.backend.clear_session()
        raise
    except tf.errors.ResourceExhaustedError:
        tf.keras.backend.clear_session()
        return float("inf")


    # ✅ Optimize on validation loss
    best_val2 = float(min(hist2.history["val_loss"]))
    tf.keras.backend.clear_session()
    return best_val2



# Wrapper (no lambda)
def objective_with_args(trial):
    return objective(trial, CONFIG)

# ========= Main =========
if __name__ == "__main__":
    set_tf_memory_growth()
    os.makedirs(SAVE_ROOT, exist_ok=True)

    CONFIG = {
        "model_json": MODEL_JSON,
        "model_weights": MODEL_WEIGHTS,
        "train_hdf5": train_h5,
        "val_hdf5": val_h5,
        "save_root": SAVE_ROOT,
        "patch_size": PATCH_SIZE,
        "seed_base": SEED_BASE,
    }

    study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    direction="minimize",
    load_if_exists=True,
    sampler=TPESampler(
        seed=42,
        multivariate=True,
        group=True,
        constant_liar=True,   # avoids duplicate suggestions across workers
    ),
    pruner=HyperbandPruner(
        min_resource=20,      # wait ≥10 epochs before judging
        reduction_factor=3,   # keep ~1/3 “best” each round
    ),
)


    study.optimize(objective_with_args, n_trials=TRIALS_PER_WORKER)

    print("Best value:", study.best_value)
    print("Best params:", json.dumps(study.best_params, indent=2))
    pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    comp   = sum(t.state.name == "COMPLETE" for t in study.trials)
    print(f"Trials — PRUNED: {pruned}, COMPLETE: {comp}")











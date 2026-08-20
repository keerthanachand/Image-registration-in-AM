#!/usr/bin/env python3
import os, shutil
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # set before importing h5py
import json, numpy as np, tensorflow as tf, optuna, h5py, time
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import voxelmorph as vxm
from voxelmorph import losses
from src.training.train_utils import split_by_index, initialize_generator_parameters, vxm_data_generator, build_and_train_vxm_model

from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping

# ========= CONFIG =========
MODEL_JSON     = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
#MODEL_WEIGHTS  = "/home/kchand/results/cross_validation/vxm_weights_fold/weights_fold0.h5"
TRAIN_HDF5     = "/home/kchand/input_data/all_data_for_cross_validation_TPMS.h5"
SAVE_ROOT      = "/home/kchand/results/hyperparameter_TPMS"
PATCH_SIZE     = (128, 128, 128)

# IMPORTANT: use ONE shared study name across all workers
STUDY_NAME     = "vxm_TPMS_hyperparameter"

# IMPORTANT: point this to a SHARED RDB (e.g., Postgres/MySQL) that all PCs can reach.
# You can override via env var STORAGE_URL without editing the file.
STORAGE_URL    = os.environ.get(
    "STORAGE_URL",
    "sqlite:////home/kchand/results/finetune_two_stage_optuna/hyperparameter_tuning_TPMS.db?timeout=300"
)

TRIALS_PER_WORKER = 15
SEED_BASE      = 2000

# Explicit split by index
NUM_SAMPLES = 7
TEST_IDX = [6]
VAL_IDX  = [4]

# ====== Dataset split coordination (original logic) ======
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
train_h5 = "/home/kchand/input_data/train_data_temp.h5"
val_h5   = "/home/kchand/input_data/val_data_temp.h5"
test_h5  = "/home/kchand/input_data/test_data_temp.h5"


if task_id == "0":
    train_h5, val_h5, test_h5 = split_by_index(TRAIN_HDF5, TEST_IDX, VAL_IDX, NUM_SAMPLES)
else:
    for _ in range(120):
        if all(os.path.exists(p) for p in [train_h5, val_h5, test_h5]):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Split files not found after waiting.")

# ====== Minimal change: localize read-only HDF5s for non-zero tasks ======
def _localize(src_path: str, tag: str) -> str:
    dst = f"/tmp/{os.path.basename(src_path)}.{tag}"
    if not os.path.exists(dst):
        try:
            os.link(src_path, dst)  # hardlink if same filesystem
        except Exception:
            shutil.copyfile(src_path, dst)  # fallback to full copy
    return dst

if task_id != "0":
    train_h5 = _localize(train_h5, f"arr{task_id}.train")
    val_h5   = _localize(val_h5,   f"arr{task_id}.val")
    test_h5  = _localize(test_h5,  f"arr{task_id}.test")

print("GPUs seen by TF:", tf.config.list_physical_devices("GPU"))

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

# ========= Objective =========
def objective(trial, cfg):
    tf.keras.backend.clear_session()
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed = cfg["seed_base"] + array_id * 1000 + trial.number
    np.random.seed(seed); tf.random.set_seed(seed); tf.keras.utils.set_random_seed(seed)

    # --- Search space (as-is) ---
    batch_size      = trial.suggest_categorical("batch_size", [4, 8, 16])
    steps_per_epoch = trial.suggest_int("steps_per_epoch", 20, 80, step=10)
    lambda_smooth   = trial.suggest_float("lambda_smooth", 0.05, 0.3)
    lr     = trial.suggest_float("lr_stage1", 1e-6, 1e-3, log=True)
    epochs   = trial.suggest_int("epochs_stage1", 30, 200, step=5)
    #epochs   = trial.suggest_int("epochs_stage1", 1, 3, step=1)
    lr_factor    = trial.suggest_float("lr_factor_s1", 0.3, 0.8)
    lr_patience  = trial.suggest_int("lr_patience_stage1", 2, 10)
    lr_min          = trial.suggest_float("lr_min", 1e-8, 1e-7, log=True)
    print("SAMPLED:", trial.number, json.dumps(trial.params, sort_keys=True))

    # Trial dir & params
    tdir = os.path.join(cfg["save_root"], f"trial_{trial.number:04d}")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "params.json"), "w") as f:
        json.dump({**trial.params, "seed": seed}, f, indent=2)

    t0 = time.time()
    train_params = initialize_generator_parameters(hdf5_file=cfg["train_hdf5"], patch_size=cfg["patch_size"])
    val_params   = initialize_generator_parameters(hdf5_file=cfg["val_hdf5"],   patch_size=cfg["patch_size"])
    train_gen = vxm_data_generator(cfg["train_hdf5"], cfg["patch_size"], batch_size, train_params)
    val_gen   = vxm_data_generator(cfg["val_hdf5"],   cfg["patch_size"], batch_size, val_params)
    print(f"Generators initialized in {time.time()-t0:.2f}s")
    val_steps = 10

    # ===== Stage 1 =====
    #print("Loading pretrained model...")
    #model = load_pretrained(cfg["model_json"], cfg["model_weights"])
    #print(f"Freezing {freeze_levels} encoder levels.")
    #freeze_first_k_encoder_levels_convs(model, freeze_levels)
    
    # Load model architecture
    json_path = cfg["model_json"]
    with open(json_path, "r") as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={
            "SpatialTransformer": vxm.layers.SpatialTransformer,
            "VxmDense": vxm.networks.VxmDense,
        },
    )
    print(f"Compiling Model (lr={lr}, λ={lambda_smooth})...")
    compile_vm(model, lr, lambda_smooth)
    print("✅ Compilation complete.")

    cb = [
        CSVLogger(os.path.join(tdir, "hist.csv")),
        ModelCheckpoint(os.path.join(tdir, "weights.h5"),
                        save_best_only=True, save_weights_only=True,
                        monitor="val_loss", mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=lr_factor,
                          patience=lr_patience, min_lr=lr_min, verbose=2),
        TFKerasPruningCallback(trial, monitor="val_loss"),
    ]
    print(f"🚀 Training started for {epochs} epochs...")
    try:
        hist = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            workers=0,
            use_multiprocessing=False,
            verbose=2,
            callbacks=cb
        )
    except optuna.TrialPruned:
        tf.keras.backend.clear_session()
        raise
    except tf.errors.ResourceExhaustedError:
        tf.keras.backend.clear_session()
        return float("inf")

    print(f"✅ Best val={min(hist.history['val_loss']):.4f}")

    best_val = float(min(hist.history["val_loss"]))
    tf.keras.backend.clear_session()
    print(f"🎯 Trial {trial.number} complete — best val loss = {best_val:.6f}")
    return best_val

def objective_with_args(trial):
    return objective(trial, CONFIG)

# ========= Main =========
if __name__ == "__main__":
    set_tf_memory_growth()
    os.makedirs(SAVE_ROOT, exist_ok=True)

    CONFIG = {
        "model_json": MODEL_JSON,
        "train_hdf5": train_h5,
        "val_hdf5": val_h5,
        "save_root": SAVE_ROOT,
        "patch_size": PATCH_SIZE,
        "seed_base": SEED_BASE,
    }

    # ---- Use RDBStorage when possible (shared across PCs). SQLite must be on truly shared storage. ----
    try:
        from optuna.storages import RDBStorage, RetryFailedTrialCallback
        storage = RDBStorage(
            url=STORAGE_URL,
            engine_kwargs={"pool_pre_ping": True},
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=RetryFailedTrialCallback(),
        )
    except Exception:
        # Fallback to passing the URL string; still works for sqlite.
        storage = STORAGE_URL

    # Sampler: remove fixed seed to avoid identical startup suggestions across workers
    sampler = TPESampler(
        seed=None,                 # diversity across workers
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=10,
    )

    pruner = HyperbandPruner(min_resource=10, reduction_factor=3)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(objective_with_args, n_trials=TRIALS_PER_WORKER)

    print("Best value:", study.best_value)
    print("Best params:", json.dumps(study.best_params, indent=2))
    pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    comp   = sum(t.state.name == "COMPLETE" for t in study.trials)
    print(f"Trials — PRUNED: {pruned}, COMPLETE: {comp}")

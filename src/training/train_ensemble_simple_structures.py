#!/usr/bin/env python3
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"  # optional, harmless if missing


import json
import time
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(False)

import optuna
import voxelmorph as vxm
from voxelmorph import losses

from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)

from train_utils import (
    initialize_generator_parameters,
    vxm_data_generator, vxm_data_generator_fast
)

print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", tf.config.list_physical_devices("GPU"))

# ========= CONFIG =========
MODEL_JSON     = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
MODEL_WEIGHTS  = "/home/kchand/results/cross_validation/vxm_weights_fold/weights_fold0.h5"  # (kept; used now)
SAVE_ROOT      = "/home/kchand/results/finetune_two_stage_optuna"
ENSEMBLE_ROOT  = os.path.join(SAVE_ROOT, "ensemble_best_from_optuna")
PATCH_SIZE     = (128, 128, 128)

# Pre-split files (already exist)
TRAIN_H5_PATH = os.environ.get(
    "TRAIN_H5",
    "/home/kchand/input_data/data_split_Simple_structures/train_data_temp.h5",
)

VAL_H5_PATH = os.environ.get(
    "VAL_H5",
    "/home/kchand/input_data/data_split_Simple_structures/val_data_temp.h5",
)

TEST_H5_PATH   = "/home/kchand/input_data/data_split_Simple_structures/test_data_temp.h5"

STUDY_NAME     = "vxm_two_stage_hyperparameter"
STORAGE_URL    = os.environ.get(
    "STORAGE_URL",
    "sqlite:////home/kchand/results/finetune_two_stage_optuna/voxelmorph_two_stage.db?timeout=300",
)

SEED_BASE      = 5000

# Input pipeline knobs (threading only; safe for HDF5)
FIT_WORKERS    = int(os.environ.get("FIT_WORKERS", "1"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "1"))
VAL_STEPS      = int(os.environ.get("VAL_STEPS", "10"))


def set_tf_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print("Visible GPUs:", gpus)


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_pretrained(json_path: str, weights_path: str):
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
    """
    Matches your Optuna tuning script naming:
    vxm_dense_unet_enc_conv_{lvl}_...
    """
    frozen = 0
    for layer in model.layers:
        for lvl in range(k_levels):
            if layer.name.startswith(f"vxm_dense_unet_enc_conv_{lvl}_") and isinstance(layer, tf.keras.layers.Conv3D):
                layer.trainable = False
                frozen += 1
    print(f"[freeze] Froze {frozen} Conv3D layers (first {k_levels} encoder levels).")


def compile_vm(model, lr: float, lambda_smooth: float):
    ncc  = losses.NCC().loss
    grad = losses.Grad("l2").loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[ncc, grad],
        loss_weights=[1.0, lambda_smooth],
    )


def train_single_ensemble_model(
    ens_id: int,
    best_params: dict,
    train_h5: str,
    val_h5: str,
    patch_size,
):
    """
    Modified to match your TWO-STAGE Optuna tuning:
      - load pretrained weights
      - stage1: freeze N encoder levels, train with lr_stage1/epochs_stage1
      - reload best stage1 weights
      - stage2: unfreeze all, train with lr_stage2/epochs_stage2
    """
    tf.keras.backend.clear_session()

    seed = SEED_BASE + ens_id
    set_global_seed(seed)
    print(f"\n===== [ENSEMBLE] Training member {ens_id:02d} with seed {seed} =====")

    # Required keys from your Optuna tuning script
    required = [
        "batch_size",
        "steps_per_epoch",
        "freeze_levels_stage1",
        "lambda_smooth",
        "lr_stage1",
        "lr_stage2",
        "epochs_stage1",
        "epochs_stage2",
        "lr_factor_s1",
        "lr_factor_s2",
        "lr_patience_stage1",
        "lr_patience_stage2",
        "lr_min",
    ]
    missing = [k for k in required if k not in best_params]
    if missing:
        raise KeyError(f"Optuna best_params missing keys: {missing}")

    # Cast values
    batch_size      = int(best_params["batch_size"])
    steps_per_epoch = int(best_params["steps_per_epoch"])
    freeze_levels   = int(best_params["freeze_levels_stage1"])
    lambda_smooth   = float(best_params["lambda_smooth"])

    lr_stage1       = float(best_params["lr_stage1"])
    lr_stage2       = float(best_params["lr_stage2"])
    epochs_stage1   = int(best_params["epochs_stage1"])
    epochs_stage2   = int(best_params["epochs_stage2"])

    lr_factor_s1    = float(best_params["lr_factor_s1"])
    lr_factor_s2    = float(best_params["lr_factor_s2"])
    lr_patience_s1  = int(best_params["lr_patience_stage1"])
    lr_patience_s2  = int(best_params["lr_patience_stage2"])
    lr_min          = float(best_params["lr_min"])
    os.makedirs(ENSEMBLE_ROOT, exist_ok=True)
    tdir = os.path.join(ENSEMBLE_ROOT, f"ens_{ens_id:02d}")
    os.makedirs(tdir, exist_ok=True)

    with open(os.path.join(tdir, "used_params.json"), "w") as f:
        json.dump({**best_params, "seed": seed}, f, indent=2)

    # --- Generators ---
    t0 = time.time()
    train_params = initialize_generator_parameters(hdf5_file=train_h5, patch_size=patch_size)
    val_params   = initialize_generator_parameters(hdf5_file=val_h5,   patch_size=patch_size)

    train_gen = vxm_data_generator_fast(train_h5, patch_size, batch_size, train_params,cache_size=4)
    val_gen   = vxm_data_generator_fast(val_h5, patch_size, batch_size, val_params, cache_size=2)
    print(f"[ENSEMBLE {ens_id:02d}] Generators initialized in {time.time()-t0:.2f}s")

    # =========================
    # Stage 1
    # =========================
    print(f"[ENSEMBLE {ens_id:02d}] Loading pretrained model from JSON+weights ...")
    model = load_pretrained(MODEL_JSON, MODEL_WEIGHTS)

    print(f"[ENSEMBLE {ens_id:02d}] Freezing {freeze_levels} encoder levels for Stage 1 ...")
    freeze_first_k_encoder_levels_convs(model, freeze_levels)

    print(f"[ENSEMBLE {ens_id:02d}] Compiling Stage 1 (lr={lr_stage1:.2e}, λ={lambda_smooth}) ...")
    compile_vm(model, lr_stage1, lambda_smooth)

    ckpt_s1 = os.path.join(tdir, "weights_stage1.h5")
    log_s1  = os.path.join(tdir, "hist_stage1.csv")

    cb1 = [
        CSVLogger(log_s1),
        ModelCheckpoint(
            ckpt_s1,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor_s1,
            patience=lr_patience_s1,
            min_lr=lr_min,
            verbose=2,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"[ENSEMBLE {ens_id:02d}] 🚀 Stage 1 training for {epochs_stage1} epochs ...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=VAL_STEPS,
        workers=FIT_WORKERS,
        use_multiprocessing=False,  # HDF5 safety
        max_queue_size=MAX_QUEUE_SIZE,
        verbose=2,
        callbacks=cb1,
    )

    # Reload best stage1 weights (matches your tuning script)
    if os.path.exists(ckpt_s1):
        model.load_weights(ckpt_s1)
    else:
        print(f"[ENSEMBLE {ens_id:02d}] ⚠️ Stage 1 best checkpoint missing; using current weights.")

    # =========================
    # Stage 2
    # =========================
    print(f"[ENSEMBLE {ens_id:02d}] Unfreezing all layers for Stage 2 ...")
    for l in model.layers:
        l.trainable = True

    print(f"[ENSEMBLE {ens_id:02d}] Compiling Stage 2 (lr={lr_stage2:.2e}, λ={lambda_smooth}) ...")
    compile_vm(model, lr_stage2, lambda_smooth)

    ckpt_s2 = os.path.join(tdir, "weights_stage2.h5")
    log_s2  = os.path.join(tdir, "hist_stage2.csv")

    cb2 = [
        CSVLogger(log_s2),
        ModelCheckpoint(
            ckpt_s2,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor_s2,
            patience=lr_patience_s2,
            min_lr=lr_min,
            verbose=2,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    train_gen = vxm_data_generator_fast(train_h5, patch_size, batch_size, train_params,cache_size=4)
    val_gen   = vxm_data_generator_fast(val_h5, patch_size, batch_size, val_params, cache_size=2)
    print(f"[ENSEMBLE {ens_id:02d}] Generators initialized in {time.time()-t0:.2f}s")
    
    print(f"[ENSEMBLE {ens_id:02d}] 🚀 Stage 2 training for {epochs_stage2} epochs ...")
    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=VAL_STEPS,
        workers=FIT_WORKERS,
        use_multiprocessing=False,
        max_queue_size=MAX_QUEUE_SIZE,
        verbose=2,
        callbacks=cb2,
    )

    best_val2 = float(min(hist2.history["val_loss"]))
    print(f"[ENSEMBLE {ens_id:02d}] ✅ Finished. Best Stage-2 val_loss={best_val2:.6f}")
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    set_tf_memory_growth()
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Decide ensemble member ID from SLURM
    ens_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    print(f"Running ensemble member id = {ens_id}")

    # Load split files directly
    train_h5 = TRAIN_H5_PATH
    val_h5   = VAL_H5_PATH
    test_h5  = TEST_H5_PATH

    # Sanity check that files exist
    for p in [train_h5, val_h5, test_h5]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Split file not found: {p}")

    print("[split] Train H5:", train_h5)
    print("[split] Val   H5:", val_h5)
    print("[split] Test  H5:", test_h5)

    # Load best hyperparameters from Optuna
    print(f"Connecting to Optuna study '{STUDY_NAME}' at:")
    print(f"  {STORAGE_URL}")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

    print("\n===== BEST RESULT FROM STUDY =====")
    print("Best value:", study.best_value)
    print("Best params:")
    print(json.dumps(study.best_params, indent=2))

    best_params = dict(study.best_params)

    # Train exactly ONE ensemble member (this task's ens_id) using TWO-STAGE best params
    train_single_ensemble_model(
        ens_id=ens_id,
        best_params=best_params,
        train_h5=train_h5,
        val_h5=val_h5,
        patch_size=PATCH_SIZE,
    )

    # Clean exit (kept from your original)
    os._exit(0)




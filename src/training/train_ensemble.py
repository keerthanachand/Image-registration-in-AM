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
    vxm_data_generator,
)



# ========= CONFIG =========
MODEL_JSON     = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
SAVE_ROOT      = "/home/kchand/results/hyperparameter_TPMS"
ENSEMBLE_ROOT  = os.path.join(SAVE_ROOT, "ensemble")
PATCH_SIZE     = (128, 128, 128)

# Pre-split files (already exist)
TRAIN_H5_PATH  = "/home/kchand/input_data/train_data_temp.h5"
VAL_H5_PATH    = "/home/kchand/input_data/val_data_temp.h5"
TEST_H5_PATH   = "/home/kchand/input_data/test_data_temp.h5"  # not used in training, but kept for sanity check

STUDY_NAME     = "vxm_TPMS_hyperparameter"
STORAGE_URL    = "sqlite:////home/kchand/results/finetune_two_stage_optuna/hyperparameter_tuning_TPMS.db?timeout=300"

SEED_BASE      = 5000


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


def build_model_from_json(json_path: str):
    with open(json_path, "r") as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={
            "SpatialTransformer": vxm.layers.SpatialTransformer,
            "VxmDense": vxm.networks.VxmDense,
        },
    )
    return model


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
    model_json_path: str,
    train_h5: str,
    val_h5: str,
    patch_size,
):
    tf.keras.backend.clear_session()

    seed = SEED_BASE + ens_id
    set_global_seed(seed)
    print(f"\n===== [ENSEMBLE] Training member {ens_id:02d} with seed {seed} =====")

    # Validate keys early for a nicer error than KeyError later
    required = [
        "batch_size",
        "steps_per_epoch",
        "lambda_smooth",
        "lr_stage1",
        "epochs_stage1",
        "lr_factor_s1",
        "lr_patience_stage1",
        "lr_min",
    ]
    missing = [k for k in required if k not in best_params]
    if missing:
        raise KeyError(f"Optuna best_params missing keys: {missing}")

    batch_size      = int(best_params["batch_size"])
    steps_per_epoch = int(best_params["steps_per_epoch"])
    lambda_smooth   = float(best_params["lambda_smooth"])
    lr              = float(best_params["lr_stage1"])
    epochs          = int(best_params["epochs_stage1"])
    lr_factor       = float(best_params["lr_factor_s1"])
    lr_patience     = int(best_params["lr_patience_stage1"])
    lr_min          = float(best_params["lr_min"])

    os.makedirs(ENSEMBLE_ROOT, exist_ok=True)
    tdir = os.path.join(ENSEMBLE_ROOT, f"ens_{ens_id:02d}")
    os.makedirs(tdir, exist_ok=True)

    with open(os.path.join(tdir, "used_params.json"), "w") as f:
        json.dump({**best_params, "seed": seed}, f, indent=2)

    t0 = time.time()
    train_params = initialize_generator_parameters(
        hdf5_file=train_h5, patch_size=patch_size
    )
    val_params = initialize_generator_parameters(
        hdf5_file=val_h5, patch_size=patch_size
    )
    train_gen = vxm_data_generator(
        train_h5, patch_size, batch_size, train_params
    )
    val_gen = vxm_data_generator(
        val_h5, patch_size, batch_size, val_params
    )
    print(f"[ENSEMBLE {ens_id:02d}] Generators initialized in {time.time()-t0:.2f}s")

    val_steps = 10  # keep as-is from your original script

    model = build_model_from_json(model_json_path)
    print(f"[ENSEMBLE {ens_id:02d}] Compiling model (lr={lr}, λ={lambda_smooth}) ...")
    compile_vm(model, lr, lambda_smooth)
    print(f"[ENSEMBLE {ens_id:02d}] Compilation complete.")

    ckpt_path = os.path.join(tdir, "weights.h5")
    log_path  = os.path.join(tdir, "hist.csv")

    callbacks = [
        CSVLogger(log_path),
        ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=lr_patience,
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

    print(f"[ENSEMBLE {ens_id:02d}] 🚀 Training for {epochs} epochs ...")
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        workers=4,
        use_multiprocessing=False,
        verbose=2,
        callbacks=callbacks,
    )

    best_val = float(min(hist.history["val_loss"]))
    print(f"[ENSEMBLE {ens_id:02d}] ✅ Finished. Best val_loss={best_val:.6f}")
    tf.keras.backend.clear_session()


if __name__ == "__main__":

    print("TF version:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU available:", tf.config.list_physical_devices("GPU"))

    set_tf_memory_growth()
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Decide ensemble member ID from SLURM
    ens_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    print(f"Running ensemble member id = {ens_id}")

    # Load split files directly (no splitting here)
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

    best_params = study.best_params

    # Train exactly ONE ensemble member (this task's ens_id)
    train_single_ensemble_model(
        ens_id=ens_id,
        best_params=best_params,
        model_json_path=MODEL_JSON,
        train_h5=train_h5,
        val_h5=val_h5,
        patch_size=PATCH_SIZE,
    )

    
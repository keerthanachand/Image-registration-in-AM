# === Two-stage VoxelMorph fine-tuning with Optuna (uses your train_utils) ===
import os, json, time, numpy as np, optuna, tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
import voxelmorph as vxm
from voxelmorph import losses

# ---- your utilities ----
from train_utils import (
    initialize_generator_parameters,
    vxm_data_generator
)

# ========= PATHS (edit to your env) =========
MODEL_JSON     = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
MODEL_WEIGHTS  = "/home/kchand/results/cross_validation/vxm_weights_fold/weights_fold0.h5"
TRAIN_HDF5     = "/home/kchand/input_data/all_samples_simple_structures.h5"
SAVE_ROOT      = "/home/kchand/results/finetune_two_stage_optuna"
DB_URI         = "sqlite:///voxelmorph_two_stage.db"
PATCH_SIZE     = (128, 128, 128)
os.makedirs(SAVE_ROOT, exist_ok=True)

# --- Optuna configuration ---
STORAGE="sqlite:////home/kchand/results/voxelmorph_two_stage.db"   # shared DB file
SEED=$((1000 + SLURM_ARRAY_TASK_ID))

#Each worker will run 15 Optuna trials before stopping.
TRIALS_PER_WORKER=15


# ========= helpers =========


def pick_gpu_from_slurm():
    """
    Map SLURM task rank to a single GPU index.
    Works for array jobs or mpi-like launchers.
    """
    # Prefer SLURM_LOCALID (0..gpus-1 on the node)
    local_id = os.environ.get("SLURM_LOCALID")
    if local_id is None:
        # fallback to array index
        local_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    gpu_idx = int(local_id) % max(1, len(tf.config.list_physical_devices("GPU")))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    return gpu_idx


def load_pretrained(json_path, weights_path):
    with open(json_path, "r") as f:
        arch_json = f.read()
    m = tf.keras.models.model_from_json(
        arch_json,
        custom_objects={
            "SpatialTransformer": vxm.layers.SpatialTransformer,
            "VxmDense": vxm.networks.VxmDense,
        },
    )
    m.load_weights(weights_path)
    return m

def freeze_first_k_encoder_levels_convs(model, k_levels=2):
    """Freeze ONLY Conv3D layers in encoder levels [0..k_levels-1]."""
    frozen = 0
    for layer in model.layers:
        for lvl in range(k_levels):
            if layer.name.startswith(f"vxm_dense_unet_enc_conv_{lvl}_"):
                if isinstance(layer, tf.keras.layers.Conv3D):
                    layer.trainable = False
                    frozen += 1
    print(f"[freeze] froze {frozen} Conv3D layers (first {k_levels} encoder levels).")

def compile_vm(model, lr, lambda_smooth):
    ncc  = losses.NCC().loss
    grad = losses.Grad("l2").loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[ncc, grad],
        loss_weights=[1.0, lambda_smooth],
    )

def make_train_gen(h5, patch_size, batch_size):
    gen_params = initialize_generator_parameters(hdf5_file=h5, patch_size=patch_size)
    train_gen = vxm_data_generator(
        hdf5_file=h5, patch_size=patch_size, batch_size=batch_size, generator_params=gen_params
    )
    return train_gen

# ... (your imports and code unchanged above)

def objective(trial: optuna.Trial):
    # --- seed for reproducibility ---
    seed = 1337 + trial.number
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # --- sample hyperparams (your code unchanged) ---
    # ... your existing sampling code ...

    # --- per-trial dir & params.json ---
    tdir = os.path.join(SAVE_ROOT, f"trial_{trial.number:04d}")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "params.json"), "w") as f:
        json.dump(trial.params, f, indent=2)

    # --- GPU memory growth (your code unchanged) ---
    # ... your GPU code ...

    # --- data (unchanged) ---
    train_gen = make_train_gen(TRAIN_HDF5, PATCH_SIZE, batch_size)

    # ========== STAGE 1 ==========
    model = load_pretrained(MODEL_JSON, MODEL_WEIGHTS)
    freeze_first_k_encoder_levels_convs(model, k_levels=freeze_levels)
    compile_vm(model, lr=lr_stage1, lambda_smooth=lambda_smooth)

    cb1 = [
        CSVLogger(os.path.join(tdir, "hist_stage1.csv")),
        ModelCheckpoint(os.path.join(tdir, "weights_stage1.h5"),
                        save_best_only=True, save_weights_only=True,
                        monitor="loss", mode="min"),
        ReduceLROnPlateau(monitor="loss", factor=rlrop_factor,
                          patience=rlrop_patience, min_lr=rlrop_min_lr, verbose=0),
        # Optional:
        # EarlyStopping(monitor="loss", patience=max(10, rlrop_patience), restore_best_weights=True, verbose=0),
    ]

    try:
        hist1 = model.fit(
            train_gen,
            epochs=epochs_stage1,
            steps_per_epoch=steps_per_epoch,
            verbose=0,
            callbacks=cb1,
            # keep workers=1 to avoid HDF5 issues
        )
    except tf.errors.ResourceExhaustedError:
        import tensorflow.keras.backend as K
        K.clear_session()
        return float("inf")

    best1 = float(np.min(hist1.history["loss"]))

    # ========== STAGE 2 ==========
    for l in model.layers: l.trainable = True
    compile_vm(model, lr=lr_stage2, lambda_smooth=lambda_smooth)

    cb2 = [
        CSVLogger(os.path.join(tdir, "hist_stage2.csv")),
        ModelCheckpoint(os.path.join(tdir, "weights_stage2.h5"),
                        save_best_only=True, save_weights_only=True,
                        monitor="loss", mode="min"),
        ReduceLROnPlateau(monitor="loss", factor=rlrop_factor,
                          patience=max(3, rlrop_patience // 2),
                          min_lr=rlrop_min_lr, verbose=0),
        # Optional:
        # EarlyStopping(monitor="loss", patience=max(8, rlrop_patience//2), restore_best_weights=True, verbose=0),
    ]

    hist2 = model.fit(
        train_gen,
        epochs=epochs_stage2,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        callbacks=cb2,
    )
    best2 = float(np.min(hist2.history["loss"]))

    score = best2 + 0.05 * max(0.0, best2 - best1)

    # --- VERY IMPORTANT: free graph memory ---
    import tensorflow.keras.backend as K
    K.clear_session()

    return score

if __name__ == "__main__":
    # Make sure each SLURM task sticks to one GPU
    _ = pick_gpu_from_slurm()

    study = optuna.create_study(
        study_name="vxm_two_stage",
        storage=DB_URI,  # switch to PostgreSQL if you run multiple workers
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=25, gc_after_trial=True)

    print("Best value:", study.best_value)
    print("Best params:", json.dumps(study.best_params, indent=2))

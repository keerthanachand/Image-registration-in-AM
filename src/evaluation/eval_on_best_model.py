#!/usr/bin/env python3
"""
Single-model inference (no ensemble) + Dice + diff-map "BDM" stats

This script supports TWO ways to choose a model:

(A) Direct load from explicit paths (architecture JSON + weights .h5)
    - set USE_OPTUNA_BEST = False
    - set MODEL_ARCH_JSON and MODEL_WEIGHTS_H5

(B) Optional: pick "best" model from an Optuna study (hyperparameter tuning)
    - set USE_OPTUNA_BEST = True
    - set STUDY_NAME + STORAGE_URL
    - optionally set TRIAL_WEIGHTS_PATH_TEMPLATE / WEIGHTS_SEARCH_ROOT

Notes:
- The architecture JSON MUST match the weights.
- Test data is read from an HDF5 with datasets static_0..N, moving_0..N

"""

import os
import gc
import json
import time
import glob
import h5py
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# ENV / TF SETTINGS
# =============================================================================
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(False)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


# =============================================================================
# YOUR UTILS
# =============================================================================
from src.evaluation.evaluation_utils import (
    save_image_as_vtk,
    save_displacement_vector_as_vtk,
    dice_coefficient,
    save_as_tiff_uint8,
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages,
    gaussian_weight,
)


# =============================================================================
# CONFIG
# =============================================================================

# -------------------------
# Model selection mode
# -------------------------
USE_OPTUNA_BEST = False  # <-- set True if you want to fetch "best trial" from Optuna

# -------------------------
# (A) DIRECT LOAD MODE
# -------------------------
MODEL_ARCH_JSON = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
MODEL_WEIGHTS_H5 = "/home/kchand/results/cross_validation/vxm_weights_fold/weights_fold0.h5"   # <-- set this when USE_OPTUNA_BEST=False

# -------------------------
# (B) OPTUNA BEST MODE (optional)
# -------------------------
STUDY_NAME = "vxm_TPMS_hyperparameter"
STORAGE_URL = "sqlite:////home/kchand/results/finetune_two_stage_optuna/hyperparameter_tuning_TPMS.db?timeout=300"

# If you know how to map trial->weights path, set a template (recommended):
# Example template:
# TRIAL_WEIGHTS_PATH_TEMPLATE = "/home/kchand/results/hyperparameter_TPMS/trial_{trial_number}/weights_stage2.h5"
TRIAL_WEIGHTS_PATH_TEMPLATE = None

# Otherwise, fall back to "newest .h5 under root"
WEIGHTS_SEARCH_ROOT = "/home/kchand/results/hyperparameter_TPMS"

# If you want Optuna mode but still override weights explicitly, set this (takes priority):
OPTUNA_OVERRIDE_WEIGHTS_H5 = None
OPTUNA_OVERRIDE_ARCH_JSON = None

# -------------------------
# Data / inference / output
# -------------------------
TEST_H5_PATH = "/home/kchand/input_data/data_split_simple_structures_BAM_padded/test_data.h5"
OUT_ROOT = "/home/kchand/results/BAM_simple_struct_eval_on_TPMS_model_no_hypertuning"

PATCH_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)

SAVE_TIFF = True
SAVE_VTK = False


# =============================================================================
# PLOTTING (unchanged; uses disp magnitude as "uncertainty panel")
# =============================================================================
def plot_overlay_fixed_moving_moved_mag_unc(
    fixed_image,
    moving_image,
    moved_image,
    disp_field,
    uncertainty,
    *,
    plane="XY",
    slice_index=None,
    mag_vmax=40.0,
    unc_vmax=None,
    save_path=None,
):
    Dx, Dy, Dz = fixed_image.shape
    plane = plane.upper()

    if plane == "YZ":
        si = Dx // 2 if slice_index is None else slice_index
        fixed_sl = fixed_image[si, :, :]
        moving_sl = moving_image[si, :, :]
        moved_sl = moved_image[si, :, :]
        unc_sl = uncertainty[si, :, :]
        disp_sl = disp_field[si, :, :, :]
    elif plane == "XZ":
        si = Dz // 2 if slice_index is None else slice_index
        fixed_sl = np.rot90(fixed_image[:, :, si], k=-1)
        moving_sl = np.rot90(moving_image[:, :, si], k=-1)
        moved_sl = np.rot90(moved_image[:, :, si], k=-1)
        unc_sl = np.rot90(uncertainty[:, :, si], k=-1)
        disp_sl = np.rot90(disp_field[:, :, si, :], k=-1)
    else:  # XY
        si = Dy // 2 if slice_index is None else slice_index
        fixed_sl = fixed_image[:, si, :]
        moving_sl = moving_image[:, si, :]
        moved_sl = moved_image[:, si, :]
        unc_sl = uncertainty[:, si, :]
        disp_sl = disp_field[:, si, :, :]

    disp_mag = np.linalg.norm(disp_sl, axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].imshow(moving_sl, cmap="gray")
    axes[0, 0].imshow(fixed_sl, cmap="Greens", alpha=0.5)
    axes[0, 0].set_title("Fixed vs Moving")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(moved_sl, cmap="gray")
    axes[0, 1].imshow(fixed_sl, cmap="Greens", alpha=0.5)
    axes[0, 1].set_title("Fixed vs Moved")
    axes[0, 1].axis("off")

    im1 = axes[1, 0].imshow(disp_mag, cmap="viridis", vmin=0, vmax=mag_vmax)
    axes[1, 0].set_title("Disp Magnitude")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(unc_sl, cmap="viridis", vmin=0, vmax=unc_vmax)
    axes[1, 1].set_title("Disp Magnitude (BDM panel placeholder)")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_plane_fixed_moving_moved_mag_quiver_unc(
    fixed_image,
    moving_image,
    moved_image,
    disp_field,
    uncertainty,
    *,
    plane="XY",
    slice_index=None,
    disp_vmax=40,
    save_path=None,
):
    Dx, Dy, Dz = fixed_image.shape
    plane = plane.upper()

    if plane == "YZ":
        si = Dx // 2 if slice_index is None else slice_index
        fixed_sl = fixed_image[si]
        disp_sl = disp_field[si]
        unc_sl = uncertainty[si]
    elif plane == "XZ":
        si = Dz // 2 if slice_index is None else slice_index
        fixed_sl = np.rot90(fixed_image[:, :, si], k=-1)
        disp_sl = np.rot90(disp_field[:, :, si, :], k=-1)
        unc_sl = np.rot90(uncertainty[:, :, si], k=-1)
    else:
        si = Dy // 2 if slice_index is None else slice_index
        fixed_sl = fixed_image[:, si, :]
        disp_sl = disp_field[:, si, :, :]
        unc_sl = uncertainty[:, si, :]

    disp_mag = np.linalg.norm(disp_sl, axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(fixed_sl, cmap="gray")
    axes[0].set_title("Fixed")
    axes[0].axis("off")

    im1 = axes[1].imshow(disp_mag, cmap="viridis", vmin=0, vmax=disp_vmax)
    axes[1].set_title("Disp Magnitude")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(unc_sl, cmap="viridis")
    axes[2].set_title("Disp Magnitude")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MODEL LOADING HELPERS
# =============================================================================
def build_model_from_json(json_path: str):
    if not json_path or not os.path.exists(json_path):
        raise FileNotFoundError(f"Model architecture JSON not found: {json_path}")
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


def find_weights_auto(root: str) -> str:
    if not root:
        raise ValueError("WEIGHTS_SEARCH_ROOT is None/empty; cannot auto-find weights.")
    patterns = [
        os.path.join(root, "**", "weights_stage2.h5"),
        os.path.join(root, "**", "weights.h5"),
        os.path.join(root, "**", "*.weights.h5"),
        os.path.join(root, "**", "*.h5"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        raise FileNotFoundError(f"No .h5 weights found under: {root}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def resolve_model_paths():
    """
    Returns:
      (arch_json_path, weights_h5_path, extra_info_dict)
    """
    info = {}

    if not USE_OPTUNA_BEST:
        # DIRECT MODE
        arch = MODEL_ARCH_JSON
        weights = MODEL_WEIGHTS_H5
        if not weights or not os.path.exists(weights):
            raise FileNotFoundError(
                f"MODEL_WEIGHTS_H5 not set or not found: {weights}\n"
                f"Set MODEL_WEIGHTS_H5 or switch USE_OPTUNA_BEST=True."
            )
        info["mode"] = "direct"
        return arch, weights, info

    # OPTUNA MODE (optional)
    info["mode"] = "optuna"

    # Optional overrides
    arch = OPTUNA_OVERRIDE_ARCH_JSON or MODEL_ARCH_JSON
    if OPTUNA_OVERRIDE_WEIGHTS_H5:
        weights = OPTUNA_OVERRIDE_WEIGHTS_H5
        if not os.path.exists(weights):
            raise FileNotFoundError(f"OPTUNA_OVERRIDE_WEIGHTS_H5 not found: {weights}")
        info["weights_source"] = "optuna_override_weights"
        return arch, weights, info

    # Import optuna only if needed (so script runs without optuna installed in direct mode)
    try:
        import optuna
    except Exception as e:
        raise RuntimeError(
            "USE_OPTUNA_BEST=True but optuna is not available in this environment.\n"
            "Either install optuna or set USE_OPTUNA_BEST=False."
        ) from e

    print(f"\nConnecting to Optuna study '{STUDY_NAME}' at:\n  {STORAGE_URL}")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

    info["best_value"] = float(study.best_value)
    info["best_trial_number"] = int(study.best_trial.number)
    info["best_params"] = dict(study.best_params)

    # Resolve weights path
    if TRIAL_WEIGHTS_PATH_TEMPLATE:
        weights = TRIAL_WEIGHTS_PATH_TEMPLATE.format(trial_number=study.best_trial.number)
        info["weights_source"] = "trial_template"
        if not os.path.exists(weights):
            raise FileNotFoundError(
                "TRIAL_WEIGHTS_PATH_TEMPLATE produced a path that doesn't exist:\n"
                f"  {weights}\n"
                "Fix TRIAL_WEIGHTS_PATH_TEMPLATE or use WEIGHTS_SEARCH_ROOT / OPTUNA_OVERRIDE_WEIGHTS_H5."
            )
    else:
        weights = find_weights_auto(WEIGHTS_SEARCH_ROOT)
        info["weights_source"] = "newest_under_root"
        info["weights_search_root"] = WEIGHTS_SEARCH_ROOT

    return arch, weights, info


# =============================================================================
# DATA HELPERS
# =============================================================================
def get_num_pairs(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as hf:
        statics = sorted([k for k in hf.keys() if k.startswith("static_")])
        movings = sorted([k for k in hf.keys() if k.startswith("moving_")])
    n = min(len(statics), len(movings))
    if n == 0:
        raise RuntimeError(f"No static_/moving_ datasets found in {h5_path}")
    return n


def _to_scalar_volume(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"Expected scalar volume (D,H,W), got {x.shape}")
    return x.astype(np.float32, copy=False)


def _to_disp_volume(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u)
    if u.ndim == 5 and u.shape[0] == 1:
        u = u[0]
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"Expected disp volume (D,H,W,3), got {u.shape}")
    return u.astype(np.float32, copy=False)


# =============================================================================
# PATCH-WISE INFERENCE (single pair)
# =============================================================================
def test_data_generator_single(
    vxm_model,
    hdf5_file,
    pair_idx: int = 0,
    patch_size=(128, 128, 128),
    stride=(64, 64, 64),
    verbose: bool = True,
):
    def _next_multiple(n, base):
        r = n % base
        return n if r == 0 else n + (base - r)

    start_time = time.time()

    with h5py.File(hdf5_file, "r") as hf:
        static_key = f"static_{pair_idx}"
        moving_key = f"moving_{pair_idx}"
        if static_key not in hf or moving_key not in hf:
            raise KeyError(f"Missing {static_key} or {moving_key} in {hdf5_file}")

        sample_name = hf[static_key].attrs.get("sample_name", f"sample_{pair_idx}")
        if verbose:
            print(f"Predicting for sample: {sample_name} (idx={pair_idx})")

        fixed_image = hf[static_key][...].astype(np.float32, copy=False)
        moving_image = hf[moving_key][...].astype(np.float32, copy=False)

        vol_shape = fixed_image.shape
        moving_shape = moving_image.shape
        fixed_shape = fixed_image.shape

        target_shape = tuple(
            max(_next_multiple(moving_shape[i], patch_size[i]),
                _next_multiple(fixed_shape[i], patch_size[i]))
            for i in range(3)
        )

        pad_moving = [(0, target_shape[i] - moving_shape[i]) for i in range(3)]
        pad_fixed = [(0, target_shape[i] - fixed_shape[i]) for i in range(3)]

        padded_moving = np.pad(moving_image, pad_moving, mode="constant", constant_values=0)
        padded_fixed = np.pad(fixed_image, pad_fixed, mode="constant", constant_values=0)
        padded_vol_shape = padded_fixed.shape

    if verbose:
        print("Padded data with zeros")

    patches_per_dim = [
        (padded_vol_shape[i] - patch_size[i]) // stride[i] + 1
        for i in range(3)
    ]

    reconstructed_moved = np.zeros(padded_vol_shape, dtype=np.float32)
    reconstructed_displacement = np.zeros((*padded_vol_shape, 3), dtype=np.float32)
    weight_volume = np.zeros(padded_vol_shape, dtype=np.float16)

    gaussian_weights = gaussian_weight(patch_size).astype(np.float32)

    if verbose:
        print("Initialized arrays")

    for z in range(patches_per_dim[0]):
        for y in range(patches_per_dim[1]):
            for x in range(patches_per_dim[2]):
                start_z = z * stride[0]
                start_y = y * stride[1]
                start_x = x * stride[2]

                moving_patch = padded_moving[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2],
                ]
                fixed_patch = padded_fixed[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2],
                ]

                inputs = [moving_patch[None, ..., None], fixed_patch[None, ..., None]]
                processed_patch, displacement_patch = vxm_model.predict(inputs, verbose=0)

                processed_patch = processed_patch.squeeze().astype(np.float32, copy=False)
                displacement_patch = displacement_patch.squeeze().astype(np.float32, copy=False)

                reconstructed_moved[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2],
                ] += processed_patch * gaussian_weights

                reconstructed_displacement[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2],
                    :
                ] += displacement_patch * gaussian_weights[..., None]

                weight_volume[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2],
                ] += gaussian_weights.astype(np.float16)

    w = np.maximum(weight_volume.astype(np.float32), 1.0)
    reconstructed_moved /= w
    reconstructed_displacement /= w[..., None]

    if verbose:
        print("Crop padded area :::>")

    reconstructed_moved = reconstructed_moved[:vol_shape[0], :vol_shape[1], :vol_shape[2]]
    reconstructed_displacement = reconstructed_displacement[:vol_shape[0], :vol_shape[1], :vol_shape[2], :]

    elapsed_time = (time.time() - start_time) / 60
    if verbose:
        print(f"Time taken to test one sample: {elapsed_time:.2f} minutes")

    yield reconstructed_moved, reconstructed_displacement, fixed_image, moving_image, str(sample_name)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(TEST_H5_PATH):
        raise FileNotFoundError(f"Test H5 not found: {TEST_H5_PATH}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Resolve model paths (direct or optuna)
    arch_path, weights_path, info = resolve_model_paths()

    # Log info
    with open(os.path.join(OUT_ROOT, "model_selection_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\n===== MODEL SELECTION =====")
    print("Mode:", info.get("mode"))
    if info.get("mode") == "optuna":
        print("Best value:", info.get("best_value"))
        print("Best trial:", info.get("best_trial_number"))
        print("Weights source:", info.get("weights_source"))
    print("Architecture JSON:", arch_path)
    print("Weights H5:", weights_path)

    with open(os.path.join(OUT_ROOT, "used_weights_path.txt"), "w") as f:
        f.write(weights_path + "\n")
    with open(os.path.join(OUT_ROOT, "used_arch_path.txt"), "w") as f:
        f.write(arch_path + "\n")

    # Load model
    tf.keras.backend.clear_session()
    model = build_model_from_json(arch_path)
    model.load_weights(weights_path)
    print("✅ Model loaded.")

    n_pairs = get_num_pairs(TEST_H5_PATH)
    print(f"\nFound {n_pairs} test pair(s) in: {TEST_H5_PATH}")

    rows = []
    dice_before_list, dice_after_list = [], []

    t_all0 = time.time()

    for pair_idx in range(n_pairs):
        print(f"\n========== Evaluating TEST PAIR {pair_idx} ==========")

        with h5py.File(TEST_H5_PATH, "r") as hf:
            fixed_full = _to_scalar_volume(hf[f"static_{pair_idx}"][...])
            moving_full = _to_scalar_volume(hf[f"moving_{pair_idx}"][...])
            sample_name = hf[f"static_{pair_idx}"].attrs.get("sample_name", f"test_{pair_idx:02d}")

        out_dir = os.path.join(OUT_ROOT, f"{sample_name}_idx_{pair_idx:02d}")
        os.makedirs(out_dir, exist_ok=True)

        # Dice BEFORE
        dice_before = float(dice_coefficient(fixed_full, moving_full))
        print(f"Sample name: {sample_name}")
        print(f"Dice BEFORE reg: {dice_before:.4f}")

        # BDM BEFORE (diff-map stats)
        binary_fixed = binarize_volume(fixed_full)
        binary_moving = binarize_volume(moving_full)

        print("\n--- BDM BEFORE (Fixed vs Moving) ---")
        diff_map_before = compute_diff_map(binary_fixed, binary_moving)
        bdm_before_stats = report_combined_difference_percentages(diff_map_before, binary_fixed, binary_moving)

        # Predict moved
        t0 = time.time()
        moved, disp, fixed_img, moving_img, _ = next(
            test_data_generator_single(
                model,
                TEST_H5_PATH,
                pair_idx=pair_idx,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                verbose=True,
            )
        )

        moved = _to_scalar_volume(moved)
        disp = _to_disp_volume(disp)
        fixed_img = _to_scalar_volume(fixed_img)
        moving_img = _to_scalar_volume(moving_img)

        # Dice AFTER
        dice_after = float(dice_coefficient(fixed_img, moved))
        print(f"\nDice AFTER  reg: {dice_after:.4f}")

        # BDM AFTER (diff-map stats)
        binary_moved = binarize_volume(moved)
        print("\n--- BDM AFTER (Fixed vs Moved) ---")
        diff_map_after = compute_diff_map(binary_fixed, binary_moved)
        bdm_after_stats = report_combined_difference_percentages(diff_map_after, binary_fixed, binary_moved)

        # Plots (use disp magnitude as placeholder "uncertainty")
        disp_mag = np.linalg.norm(disp, axis=-1).astype(np.float32, copy=False)

        plot_plane_fixed_moving_moved_mag_quiver_unc(
            fixed_img,
            moving_img,
            moved,
            disp,
            disp_mag,
            plane="XY",
            save_path=os.path.join(out_dir, "quiver_plot.png"),
        )
        plot_overlay_fixed_moving_moved_mag_unc(
            fixed_img,
            moving_img,
            moved,
            disp,
            disp_mag,
            plane="XY",
            save_path=os.path.join(out_dir, "overlay_plot.png"),
        )

        # Saves
        if SAVE_TIFF:
            save_as_tiff_uint8(moved, os.path.join(out_dir, "moved.tiff"))
            save_as_tiff_uint8(fixed_img, os.path.join(out_dir, "fixed.tiff"))
            save_as_tiff_uint8(moving_img, os.path.join(out_dir, "moving.tiff"))
            save_as_tiff_uint8(disp_mag, os.path.join(out_dir, "disp_mag.tiff"))

        if SAVE_VTK:
            save_image_as_vtk(moved, os.path.join(out_dir, "moved.vtk"))
            save_displacement_vector_as_vtk(disp, os.path.join(out_dir, "disp.vtk"))

        elapsed = (time.time() - t0) / 60.0
        print(f"\n✅ Saved outputs to: {out_dir}")
        print(f"⏱️  Pair time: {elapsed:.2f} minutes")

        # Collect
        dice_before_list.append(dice_before)
        dice_after_list.append(dice_after)

        rows.append(
            {
                "pair_idx": pair_idx,
                "sample_name": str(sample_name),
                "dice_before": dice_before,
                "dice_after": dice_after,

                # BDM BEFORE
                "bdm_before_total_fg": bdm_before_stats["Total Foreground Voxels"],
                "bdm_before_percent_-1": bdm_before_stats["Percent -1"],
                "bdm_before_percent_0": bdm_before_stats["Percent  0"],
                "bdm_before_percent_+1": bdm_before_stats["Percent +1"],

                # BDM AFTER
                "bdm_after_total_fg": bdm_after_stats["Total Foreground Voxels"],
                "bdm_after_percent_-1": bdm_after_stats["Percent -1"],
                "bdm_after_percent_0": bdm_after_stats["Percent  0"],
                "bdm_after_percent_+1": bdm_after_stats["Percent +1"],

                "pair_minutes": elapsed,
            }
        )

        # free per-pair arrays
        del moved, disp, fixed_img, moving_img, fixed_full, moving_full, disp_mag
        del binary_moving, binary_moved, diff_map_before, diff_map_after
        gc.collect()

    # Print all scores
    dice_before_arr = np.asarray(dice_before_list, dtype=np.float32)
    dice_after_arr = np.asarray(dice_after_list, dtype=np.float32)

    print("\n==============================")
    print("ALL TEST SAMPLES — DICE")
    print("==============================")
    for i in range(n_pairs):
        print(f"[{i:02d}] Dice: {dice_before_arr[i]:.4f} -> {dice_after_arr[i]:.4f}")

    print("\n===== OVERALL DICE SUMMARY =====")
    print(f"Dice BEFORE: mean={dice_before_arr.mean():.4f} std={dice_before_arr.std(ddof=1) if n_pairs>1 else 0.0:.4f}")
    print(f"Dice AFTER : mean={dice_after_arr.mean():.4f} std={dice_after_arr.std(ddof=1) if n_pairs>1 else 0.0:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_ROOT, "test_metrics_per_sample.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Wrote CSV: {csv_path}")

    summary = {
        "n_samples": int(n_pairs),
        "dice_before_mean": float(dice_before_arr.mean()),
        "dice_after_mean": float(dice_after_arr.mean()),
        "dice_before_std": float(dice_before_arr.std(ddof=1) if n_pairs > 1 else 0.0),
        "dice_after_std": float(dice_after_arr.std(ddof=1) if n_pairs > 1 else 0.0),
        "weights_path": weights_path,
        "arch_path": arch_path,
        "model_selection": info,
        "total_minutes": float((time.time() - t_all0) / 60.0),
    }
    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ Wrote summary.json")

    del model
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()

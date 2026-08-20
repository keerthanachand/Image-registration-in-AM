#!/usr/bin/env python3
"""
Ensemble inference + uncertainty (streaming, OOM-safe)

Key changes vs your original:
- No moved_list/disp_list, no np.stack -> uses Welford streaming mean/std (constant memory)
- Loads ONE ensemble model at a time -> predicts -> frees model -> clear_session()
- Keeps your patch-wise reconstruction + gaussian blending logic
- Saves plots, TIFFs, NPY std maps, and dice summary CSV

Author: rewritten for Keerthana (2026-02-17)
"""

import os
import gc
import time
import h5py
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

# -------------------------
# ENV / TF SETTINGS
# -------------------------
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
tf.get_logger().setLevel("ERROR")

# Optional: avoid TF grabbing all VRAM at once
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# -------------------------
# YOUR UTILS
# -------------------------
from evaluation_utils import (
    save_image_as_vtk,
    save_displacement_vector_as_vtk,
    dice_coefficient,
    save_as_tiff_uint8,
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages,
    gaussian_weight,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# CONFIG
# -------------------------
ARCH_PATH = "/home/kchand/results/cross_validation/vxm_model_architecture.json"

ENSEMBLE_ROOT = "/home/kchand/results/finetune_two_stage_optuna/ensemble_best_from_optuna"
N_ENSEMBLE = 10

TEST_H5_PATH = "/home/kchand/input_data/BAM_inconel_samples_simple_structures_padded64.h5"

PATCH_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)

OUT_ROOT = "/home/kchand/results/BAM_Inconel_simple_str_ensemble"


# -------------------------
# PLOTTING
# -------------------------
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
    axes[1, 1].set_title("Uncertainty")
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
    axes[2].set_title("Uncertainty")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# TYPE HELPERS
# -------------------------
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


# -------------------------
# WELFORD (STREAMING MEAN/STD)
# -------------------------
def welford_init(shape, dtype=np.float32):
    mean = np.zeros(shape, dtype=dtype)
    m2 = np.zeros(shape, dtype=dtype)
    n = 0
    return mean, m2, n


def welford_update(mean, m2, n, x):
    n1 = n + 1
    delta = x - mean
    mean = mean + delta / n1
    delta2 = x - mean
    m2 = m2 + delta * delta2
    return mean, m2, n1


def welford_finalize(mean, m2, n, ddof=1):
    if n <= ddof:
        var = np.zeros_like(mean)
    else:
        var = m2 / (n - ddof)
    std = np.sqrt(var, dtype=np.float32)
    return mean, std


# -------------------------
# MODEL LOADING
# -------------------------
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


def load_one_member(mi: int):
    weights_path = os.path.join(ENSEMBLE_ROOT, f"ens_{mi:02d}", "weights_stage2.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing ensemble weights: {weights_path}")

    m = build_model_from_json(ARCH_PATH)
    m.load_weights(weights_path)
    print(f"Loaded ensemble member {mi:02d}: {weights_path}")
    return m


def save_volume_float32_npy(volume: np.ndarray, path: str):
    volume = np.asarray(volume, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, volume)


# -------------------------
# H5 PAIRS
# -------------------------
def get_num_pairs(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as hf:
        statics = sorted([k for k in hf.keys() if k.startswith("static_")])
        movings = sorted([k for k in hf.keys() if k.startswith("moving_")])
    n = min(len(statics), len(movings))
    if n == 0:
        raise RuntimeError(f"No static_/moving_ datasets found in {h5_path}")
    return n


# -------------------------
# PATCH-WISE INFERENCE (SINGLE PAIR)
# -------------------------
def test_data_generator_single(
    vxm_model,
    hdf5_file,
    pair_idx: int = 0,
    patch_size=(128, 128, 128),
    stride=(64, 64, 64),
    verbose: bool = True,
):
    """
    Patch-wise prediction for a single (static_{pair_idx}, moving_{pair_idx}).
    Reconstructs moved image + displacement using gaussian blending and crops back to static shape.
    """

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

    # keep weights lighter
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

                # model expects (B,D,H,W,1)
                inputs = [
                    moving_patch[None, ..., None],
                    fixed_patch[None, ..., None],
                ]

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

    # normalize (cast weights once)
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

    yield reconstructed_moved, reconstructed_displacement, fixed_image, moving_image


# -------------------------
# MAIN
# -------------------------
def main():
    if not os.path.exists(TEST_H5_PATH):
        raise FileNotFoundError(f"Test H5 not found: {TEST_H5_PATH}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # reproducibility (inference should be deterministic anyway)
    tf.random.set_seed(0)
    np.random.seed(0)

    n_pairs = get_num_pairs(TEST_H5_PATH)
    print(f"Found {n_pairs} test pair(s) in: {TEST_H5_PATH}")

    for pair_idx in range(n_pairs):
        print(f"\n========== Evaluating TEST PAIR {pair_idx} ==========")

        # Read baseline fixed/moving directly for Dice-before and names
        with h5py.File(TEST_H5_PATH, "r") as hf:
            fixed_full = _to_scalar_volume(hf[f"static_{pair_idx}"][...])
            moving_full = _to_scalar_volume(hf[f"moving_{pair_idx}"][...])
            sample_name = hf[f"static_{pair_idx}"].attrs.get("sample_name", f"test_{pair_idx:02d}")

        out_dir = os.path.join(OUT_ROOT, f"{sample_name}_idx_{pair_idx:02d}")
        os.makedirs(out_dir, exist_ok=True)

        dice_init = dice_coefficient(fixed_full, moving_full)
        print(f"Sample name: {sample_name}")
        print(f"Dice BEFORE reg: {dice_init:.4f}")

        # Streaming aggregators (initialized once we see shapes)
        moved_mean = moved_m2 = None
        disp_mean = disp_m2 = None
        mag_mean = mag_m2 = None
        n_moved = n_disp = n_mag = 0

        dice_after_list = []

        fixed_image = None
        moving_image = None

        for mi in range(N_ENSEMBLE):
            print(f"--- Running ensemble member {mi:02d} ---")

            model = load_one_member(mi)

            gen = test_data_generator_single(
                model,
                TEST_H5_PATH,
                pair_idx=pair_idx,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                verbose=True,
            )
            reconstructed_moved, reconstructed_disp, fixed_img, moving_img = next(gen)

            moved = _to_scalar_volume(reconstructed_moved)
            disp = _to_disp_volume(reconstructed_disp)

            fixed_image = _to_scalar_volume(fixed_img)
            moving_image = _to_scalar_volume(moving_img)

            # init aggregators when shapes known
            if moved_mean is None:
                moved_mean, moved_m2, n_moved = welford_init(moved.shape)
                disp_mean, disp_m2, n_disp = welford_init(disp.shape)          # (D,H,W,3)
                mag_mean, mag_m2, n_mag = welford_init(moved.shape)            # (D,H,W)

            moved_mean, moved_m2, n_moved = welford_update(moved_mean, moved_m2, n_moved, moved)
            disp_mean, disp_m2, n_disp = welford_update(disp_mean, disp_m2, n_disp, disp)

            disp_mag = np.linalg.norm(disp, axis=-1).astype(np.float32, copy=False)
            mag_mean, mag_m2, n_mag = welford_update(mag_mean, mag_m2, n_mag, disp_mag)

            d = dice_coefficient(fixed_image, moved)
            dice_after_list.append(float(d))
            print(f"Dice AFTER reg (member {mi:02d}): {d:.4f}")

            # free per-member memory
            del reconstructed_moved, reconstructed_disp, moved, disp, disp_mag
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        dice_after_arr = np.asarray(dice_after_list, dtype=np.float32)
        print("\n===== ENSEMBLE DICE SUMMARY =====")
        print(f"Dice BEFORE reg: {dice_init:.4f}")
        print(f"Dice AFTER reg: mean={dice_after_arr.mean():.4f} std={dice_after_arr.std(ddof=1):.4f}")

        # finalize ensemble stats
        moved_mean, moved_std = welford_finalize(moved_mean, moved_m2, n_moved, ddof=1)
        disp_mean, disp_std_xyz = welford_finalize(disp_mean, disp_m2, n_disp, ddof=1)

        # A) std of displacement magnitude across ensemble
        _, disp_std_mag_A = welford_finalize(mag_mean, mag_m2, n_mag, ddof=1)

        # B) magnitude of component-wise std
        disp_std_mag_B = np.linalg.norm(disp_std_xyz, axis=-1)

        dice_after_mean = dice_coefficient(fixed_image, moved_mean)
        print(f"Dice AFTER reg (ensemble mean moved): {dice_after_mean:.4f}")

        # Optional: binarize + diff stats
        binary_fixed = binarize_volume(fixed_image)
        binary_moving = binarize_volume(moving_image)
        binary_moved_mean = binarize_volume(moved_mean)

        diff_map_before = compute_diff_map(binary_fixed, binary_moving)
        _ = report_combined_difference_percentages(diff_map_before, binary_fixed, binary_moving)

        diff_map_after = compute_diff_map(binary_fixed, binary_moved_mean)
        _ = report_combined_difference_percentages(diff_map_after, binary_fixed, binary_moved_mean)

        # Plots
        plot_plane_fixed_moving_moved_mag_quiver_unc(
            fixed_image,
            moving_image,
            moved_mean,
            disp_mean,
            disp_std_mag_B,
            plane="XY",
            save_path=os.path.join(out_dir, "quiver_plot.png"),
        )

        plot_overlay_fixed_moving_moved_mag_unc(
            fixed_image,
            moving_image,
            moved_mean,
            disp_mean,
            disp_std_mag_B,
            plane="XY",
            save_path=os.path.join(out_dir, "overlay_plot.png"),
        )

 
       
        
        save_image_as_vtk(fixed_image, os.path.join(out_dir, "fixed_image.vtk"))
        save_image_as_vtk(moving_image, os.path.join(out_dir, "moving_image.vtk"))
        save_image_as_vtk(moved_mean, os.path.join(out_dir, "moved_mean.vtk"))
        save_image_as_vtk(disp_std_mag_B, os.path.join(out_dir, "disp_std_mag_B.vtk"))
        save_displacement_vector_as_vtk(disp_mean, os.path.join(out_dir, "disp_mean.vtk"))

        # Dice summary CSV
        import pandas as pd
        df = pd.DataFrame({"member": list(range(N_ENSEMBLE)), "dice_after": dice_after_arr})
        df.loc[len(df)] = ["mean", float(dice_after_arr.mean())]
        df.loc[len(df)] = ["std", float(dice_after_arr.std(ddof=1))]
        df.loc[len(df)] = ["dice_before", float(dice_init)]
        df.loc[len(df)] = ["dice_after_mean_moved", float(dice_after_mean)]
        df.to_csv(os.path.join(out_dir, "dice_ensemble_summary.csv"), index=False)

        print(f"✅ Saved outputs to: {out_dir}")

        # free per-pair big arrays
        del moved_mean, moved_std, disp_mean, disp_std_xyz, disp_std_mag_A, disp_std_mag_B
        del fixed_image, moving_image, fixed_full, moving_full
        gc.collect()


if __name__ == "__main__":
    main()





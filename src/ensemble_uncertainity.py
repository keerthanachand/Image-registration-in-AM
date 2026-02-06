#!/usr/bin/env python3
import os
import h5py
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
tf.get_logger().setLevel("ERROR")

from evaluation_utils import (
    save_image_as_vtk,
    save_displacement_vector_as_vtk,
    dice_coefficient,
    save_as_tiff_uint8,
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages,
    gaussian_weight
)

# -------------------------
# CONFIG
# -------------------------
ARCH_PATH = "/home/kchand/results/cross_validation/vxm_model_architecture.json"

ENSEMBLE_ROOT = "/home/kchand/results/hyperparameter_TPMS/ensemble"
N_ENSEMBLE = 10

TEST_H5_PATH = "/home/kchand/input_data/test_data_temp.h5"

PATCH_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)

OUT_ROOT = "/home/kchand/results/TPMS_ensemble_eval"
os.makedirs(OUT_ROOT, exist_ok=True)


# -------------------------
# Helpers: shape normalization
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
    Same logic as your original test_data_generator, but:
      - processes ONLY static_{pair_idx}/moving_{pair_idx}
      - yields exactly once
    Keeps:
      - your padding logic
      - your weight normalization (np.maximum(..., 1))
      - your cropping behavior (cropping to vol_shape = static shape)
    """

    hf = h5py.File(hdf5_file, "r")

    def _next_multiple(n, base):
        r = n % base
        return n if r == 0 else n + (base - r)

    start_time = time.time()

    static_key = f"static_{pair_idx}"
    moving_key = f"moving_{pair_idx}"
    if static_key not in hf or moving_key not in hf:
        hf.close()
        raise KeyError(f"Missing {static_key} or {moving_key} in {hdf5_file}")

    sample_name = hf[static_key].attrs.get("sample_name", f"sample_{pair_idx}")
    if verbose:
        print(f"Predicting for sample: {sample_name} (idx={pair_idx})")

    vol_shape = hf[static_key].shape
    moving_image = hf[moving_key][...]
    fixed_image  = hf[static_key][...]

    moving_shape = moving_image.shape
    fixed_shape  = fixed_image.shape

    target_shape = tuple(
        max(_next_multiple(moving_shape[i], patch_size[i]),
            _next_multiple(fixed_shape[i],  patch_size[i]))
        for i in range(3)
    )

    pad_moving = [(0, target_shape[i] - moving_shape[i]) for i in range(3)]
    pad_fixed  = [(0, target_shape[i] - fixed_shape[i])  for i in range(3)]

    padded_moving = np.pad(moving_image, pad_moving, mode="constant", constant_values=0)
    padded_fixed  = np.pad(fixed_image,  pad_fixed,  mode="constant", constant_values=0)
    padded_vol_shape = padded_fixed.shape

    if verbose:
        print("Padded data with zeros")

    patches_per_dim = [
        (padded_vol_shape[i] - patch_size[i]) // stride[i] + 1
        for i in range(len(padded_vol_shape))
    ]

    reconstructed_moved = np.zeros(padded_vol_shape, dtype=np.float32)
    reconstructed_displacement = np.zeros((*padded_vol_shape, 3), dtype=np.float32)
    weight_volume = np.zeros(padded_vol_shape, dtype=np.float32)

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

                patch_input = np.expand_dims(moving_patch, axis=-1)
                fixed_patch = np.expand_dims(fixed_patch, axis=-1)

                inputs = [
                    np.expand_dims(patch_input, axis=0),
                    np.expand_dims(fixed_patch, axis=0),
                ]

                processed_patch, displacement_patch = vxm_model.predict(inputs, verbose=0)
                processed_patch = processed_patch.squeeze().astype(np.float32)
                displacement_patch = displacement_patch.squeeze().astype(np.float32)

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
                ] += displacement_patch * gaussian_weights[..., np.newaxis]

                weight_volume[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2],
                ] += gaussian_weights

    # Your original normalization
    reconstructed_moved /= np.maximum(weight_volume, 1)
    reconstructed_displacement /= np.maximum(weight_volume[..., np.newaxis], 1)

    if verbose:
        print("Crop padded area :::>")

    reconstructed_moved = reconstructed_moved[:vol_shape[0], :vol_shape[1], :vol_shape[2]]
    reconstructed_displacement = reconstructed_displacement[:vol_shape[0], :vol_shape[1], :vol_shape[2], :]

    hf.close()

    elapsed_time = (time.time() - start_time) / 60
    if verbose:
        print(f"Time taken to test one sample: {elapsed_time:.2f} minutes")

    yield reconstructed_moved, reconstructed_displacement, fixed_image, moving_image


def _to_scalar_volume(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"Expected scalar volume -> (D,H,W), got {x.shape}")
    return x.astype(np.float32, copy=False)


def _to_disp_volume(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u)
    if u.ndim == 5 and u.shape[0] == 1:
        u = u[0]
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"Expected disp volume -> (D,H,W,3), got {u.shape}")
    return u.astype(np.float32, copy=False)


# -------------------------
# Model loader
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


def load_ensemble_models(n_members: int):
    models = []
    for i in range(n_members):
        weights_path = os.path.join(ENSEMBLE_ROOT, f"ens_{i:02d}", "weights.h5")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing ensemble weights: {weights_path}")

        m = build_model_from_json(ARCH_PATH)
        m.load_weights(weights_path)
        models.append(m)
        print(f"Loaded ensemble member {i:02d}: {weights_path}")
    return models


# -------------------------
# Ensemble aggregation
# -------------------------
def aggregate_ensemble(moved_list, disp_list, ddof: int = 1):
    moved_stack = np.stack(moved_list, axis=0)   # (N,D,H,W)
    moved_mean = moved_stack.mean(axis=0)
    moved_std  = moved_stack.std(axis=0, ddof=ddof)

    disp_stack = np.stack(disp_list, axis=0)     # (N,D,H,W,3)
    disp_mean = disp_stack.mean(axis=0)
    disp_std_xyz = disp_stack.std(axis=0, ddof=ddof)

    # A) std of displacement magnitude across ensemble
    disp_mag = np.linalg.norm(disp_stack, axis=-1)          # (N,D,H,W)
    disp_std_mag_A = disp_mag.std(axis=0, ddof=ddof)        # (D,H,W)

    # B) magnitude of component-wise std
    disp_std_mag_B = np.linalg.norm(disp_std_xyz, axis=-1)  # (D,H,W)

    return moved_mean, moved_std, disp_mean, disp_std_xyz, disp_std_mag_A, disp_std_mag_B


# -------------------------
# Determine number of test pairs in the H5
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
# MAIN
# -------------------------
if __name__ == "__main__":

    if not os.path.exists(TEST_H5_PATH):
        raise FileNotFoundError(f"Test H5 not found: {TEST_H5_PATH}")

    # reproducibility (inference should be deterministic anyway)
    tf.random.set_seed(0)
    np.random.seed(0)

    n_pairs = get_num_pairs(TEST_H5_PATH)
    print(f"Found {n_pairs} test pair(s) in: {TEST_H5_PATH}")

    models = load_ensemble_models(N_ENSEMBLE)

    for pair_idx in range(n_pairs):
        print(f"\n========== Evaluating TEST PAIR {pair_idx} ==========")

        # Read baseline fixed/moving directly for Dice-before
        with h5py.File(TEST_H5_PATH, "r") as hf:
            fixed_full  = _to_scalar_volume(hf[f"static_{pair_idx}"][...])
            moving_full = _to_scalar_volume(hf[f"moving_{pair_idx}"][...])
            sample_name = hf[f"static_{pair_idx}"].attrs.get("sample_name", f"test_{pair_idx:02d}")

        OUT_DIR = os.path.join(OUT_ROOT, f"{sample_name}_idx_{pair_idx:02d}")
        os.makedirs(OUT_DIR, exist_ok=True)

        Dice_init = dice_coefficient(fixed_full, moving_full)
        print(f"Sample name: {sample_name}")
        print(f"Dice BEFORE reg: {Dice_init:.4f}")

        moved_list = []
        disp_list = []
        dice_after_list = []

        fixed_image = None
        moving_image = None

        for mi, model in enumerate(models):
            print(f"--- Running ensemble member {mi:02d} ---")

            # IMPORTANT: your generator must know which pair to use.
            # If your test_data_generator currently always returns *_0,
            # update it to accept pair_idx and read static_{pair_idx}/moving_{pair_idx}.
            gen = test_data_generator_single(
                model,
                TEST_H5_PATH,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                pair_idx=pair_idx,  # <-- add this to your generator signature
            )

            reconstructed_moved, reconstructed_disp, fixed_img, moving_img = next(gen)

            moved = _to_scalar_volume(reconstructed_moved)
            disp  = _to_disp_volume(reconstructed_disp)

            fixed_image  = _to_scalar_volume(fixed_img)
            moving_image = _to_scalar_volume(moving_img)

            moved_list.append(moved)
            disp_list.append(disp)

            d = dice_coefficient(fixed_image, moved)
            dice_after_list.append(d)
            print(f"Dice AFTER reg (member {mi:02d}): {d:.4f}")

        dice_after_arr = np.asarray(dice_after_list, dtype=np.float32)
        print("\n===== ENSEMBLE DICE SUMMARY =====")
        print(f"Dice BEFORE reg: {Dice_init:.4f}")
        print(f"Dice AFTER reg: mean={dice_after_arr.mean():.4f} std={dice_after_arr.std(ddof=1):.4f}")

        moved_mean, moved_std, disp_mean, disp_std_xyz, disp_std_mag_A, disp_std_mag_B = aggregate_ensemble(
            moved_list, disp_list, ddof=1
        )

        Dice_after_mean = dice_coefficient(fixed_image, moved_mean)
        print(f"Dice AFTER reg (ensemble mean moved): {Dice_after_mean:.4f}")

        # Optional: binarize + diff stats
        binary_fixed      = binarize_volume(fixed_image)
        binary_moving     = binarize_volume(moving_image)
        binary_moved_mean = binarize_volume(moved_mean)

        diff_map_before = compute_diff_map(binary_fixed, binary_moving)
        _ = report_combined_difference_percentages(diff_map_before, binary_fixed, binary_moving)

        diff_map_after = compute_diff_map(binary_fixed, binary_moved_mean)
        _ = report_combined_difference_percentages(diff_map_after, binary_fixed, binary_moved_mean)

        # Save mean predictions
        #save_image_as_vtk(moved_mean, os.path.join(OUT_DIR, "moved_mean.vtk"))
        #save_image_as_vtk(fixed_image, os.path.join(OUT_DIR, "fixed.vtk"))
        #save_image_as_vtk(moving_image, os.path.join(OUT_DIR, "moving.vtk"))
        #save_displacement_vector_as_vtk(disp_mean, os.path.join(OUT_DIR, "disp_mean.vtk"))

        # Save uncertainty maps
        #save_image_as_vtk(moved_std, os.path.join(OUT_DIR, "moved_std_uncertainty.vtk"))
        #save_image_as_vtk(disp_std_mag_A, os.path.join(OUT_DIR, "disp_std_mag_A_std_of_magnitude.vtk"))
        #save_image_as_vtk(disp_std_mag_B, os.path.join(OUT_DIR, "disp_std_mag_B_mag_of_stdxyz.vtk"))

        # Per-component std
        #save_image_as_vtk(disp_std_xyz[..., 0], os.path.join(OUT_DIR, "disp_std_dx.vtk"))
        #save_image_as_vtk(disp_std_xyz[..., 1], os.path.join(OUT_DIR, "disp_std_dy.vtk"))
        #save_image_as_vtk(disp_std_xyz[..., 2], os.path.join(OUT_DIR, "disp_std_dz.vtk"))

        # TIFFs
        save_as_tiff_uint8(moved_mean, os.path.join(OUT_DIR, "moved_mean.tiff"))
        save_as_tiff_uint8(fixed_image, os.path.join(OUT_DIR, "fixed.tiff"))
        save_as_tiff_uint8(moving_image, os.path.join(OUT_DIR, "moving.tiff"))
        save_as_tiff_uint8(moved_std, os.path.join(OUT_DIR, "moved_std_uncertainty.tiff"))
        #save_as_tiff_uint8(disp_std_mag_A, os.path.join(OUT_DIR, "disp_std_mag_A.tiff"))
        save_as_tiff_uint8(disp_std_mag_B, os.path.join(OUT_DIR, "disp_std_mag_B.tiff"))

        # Dice summary CSV
        import pandas as pd
        df = pd.DataFrame({"member": list(range(N_ENSEMBLE)), "dice_after": dice_after_arr})
        df.loc[len(df)] = ["mean", float(dice_after_arr.mean())]
        df.loc[len(df)] = ["std",  float(dice_after_arr.std(ddof=1))]
        df.to_csv(os.path.join(OUT_DIR, "dice_ensemble_summary.csv"), index=False)

        print(f"✅ Saved outputs to: {OUT_DIR}")

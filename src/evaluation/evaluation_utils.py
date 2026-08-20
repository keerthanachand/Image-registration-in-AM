import os
import h5py
import numpy as np
import tensorflow as tf
import tifffile as tiff
import voxelmorph as vxm
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import time
from skimage import filters
from pathlib import Path

def plot_two_histories(hist1_path, hist2_path, label1="Model A", label2="Model B",
                       show_val=False, out_path=None):
    """
    Plot total loss from two CSV history files on one chart.
    - y-axis fixed to [-1, 0]
    - optionally includes val_loss
    """
    def _load(csv):
        df = pd.read_csv(csv)
        epoch = df['epoch'] if 'epoch' in df.columns else pd.Series(range(len(df)))
        loss = pd.to_numeric(df.get('loss'), errors='coerce').values
        vloss = pd.to_numeric(df.get('val_loss'), errors='coerce').values if 'val_loss' in df.columns else None
        return epoch.values, loss, vloss

    e1, l1, vl1 = _load(hist1_path)
    e2, l2, vl2 = _load(hist2_path)

    # clip to [-1, 0]
    l1 = np.clip(l1, -1.0, 0.0)
    l2 = np.clip(l2, -1.0, 0.0)
    if vl1 is not None: vl1 = np.clip(vl1, -1.0, 0.0)
    if vl2 is not None: vl2 = np.clip(vl2, -1.0, 0.0)

    plt.figure(figsize=(8,5))
    plt.plot(e1, l1, label=f"{label1} — loss", linewidth=2)
    plt.plot(e2, l2, label=f"{label2} — loss", linewidth=2, linestyle="--")
    if show_val:
        if vl1 is not None and not np.all(np.isnan(vl1)):
            plt.plot(e1, vl1, label=f"{label1} — val_loss", alpha=0.7)
        if vl2 is not None and not np.all(np.isnan(vl2)):
            plt.plot(e2, vl2, label=f"{label2} — val_loss", alpha=0.7)

    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch"); plt.ylabel("Total loss")
    plt.ylim(-1.0, 0.0); plt.grid(True, linestyle=":", linewidth=0.8); plt.legend()
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    else:
        plt.show()

def binarize_volume(data):
    """
    Binarise using otsu thresholding

    Parameters:
        data (ndarray): 2D or 3D input volume or image.

    Returns:
        ndarray: Binarized array with values 0 or 1.
    """
    flattened_data = data.flatten()
    threshold = filters.threshold_otsu(flattened_data)
    thresholded_data = (data >= threshold).astype(np.uint8)
    return thresholded_data


def compute_diff_map(binary_ct, binary_cad):
    """
    Compute difference map: moved CT minus CAD.

    Parameters:
        binary_ct (ndarray): Binarized moved CT volume.
        binary_cad (ndarray): Binarized CAD volume.

    Returns:
        diff_map (ndarray): Difference map with values in {-1, 0, 1}.
    """
    return binary_ct.astype(np.int8) - binary_cad.astype(np.int8)



def report_combined_difference_percentages(diff_map, binary_cad, binary_ct):
    """
    Computes and prints percentage of -1, 0, and 1 in the diff_map,
    restricted to the union of foreground regions in CAD and CT.

    Parameters:
        diff_map (ndarray): Difference map with values in {-1, 0, 1}.
        binary_cad (ndarray): Binarized CAD volume (0 or 1).
        binary_ct (ndarray): Binarized CT or moved CT volume (0 or 1).

    Returns:
        dict: Dictionary with voxel counts and percentages.
    """
    # Union of foreground regions
    combined_foreground = (binary_cad == 1) | (binary_ct == 1)
    total_voxels = np.sum(combined_foreground)

    # Count occurrences
    count_minus1 = np.sum((diff_map == -1) & combined_foreground)
    count_zero   = np.sum((diff_map == 0)  & combined_foreground)
    count_plus1  = np.sum((diff_map == 1)  & combined_foreground)

    # Percentages
    percent_minus1 = (count_minus1 / total_voxels) * 100 if total_voxels else 0
    percent_zero   = (count_zero   / total_voxels) * 100 if total_voxels else 0
    percent_plus1  = (count_plus1  / total_voxels) * 100 if total_voxels else 0

    # Log to console
    print("=== Difference Map Percentage Analysis ===")
    print(f"Total foreground voxels (union): {total_voxels}")
    print(f"-1 (CAD=1, CT=0):   {count_minus1} voxels  →  {percent_minus1:.2f}%")
    print(f" 0 (Match):         {count_zero} voxels    →  {percent_zero:.2f}%")
    print(f" 1 (CAD=0, CT=1):   {count_plus1} voxels   →  {percent_plus1:.2f}%")

    return {
        "Total Foreground Voxels": int(total_voxels),
        "Count -1 (CAD=1, CT=0)": int(count_minus1),
        "Count  0 (Match)": int(count_zero),
        "Count +1 (CAD=0, CT=1)": int(count_plus1),
        "Percent -1": percent_minus1,
        "Percent  0": percent_zero,
        "Percent +1": percent_plus1
    }


def save_as_tiff_uint8(image, filename):
    """
    Save a 3D numpy array as a TIFF file after normalizing and converting to uint8.
    
    Args:
        image (numpy array): The 3D float64 image to be saved.
        filename (str): The filename where the image will be saved.
    """
    # Normalize image to [0, 255]
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        raise ValueError("Image has no dynamic range (min == max).")

    image_normalized = (image - min_val) / (max_val - min_val)
    image_uint8 = (image_normalized * 255).astype(np.uint8)

    # Save as TIFF
    tiff.imwrite(filename, image_uint8)
    print(f"Saved normalized uint8 TIFF to {filename}")

def save_image_as_vtk(moved_image, filename):
    # Create a PyVista grid for the moved image
    moved_image_shape = moved_image.shape
    x = np.arange(moved_image_shape[0])
    y = np.arange(moved_image_shape[1])
    z = np.arange(moved_image_shape[2])
    grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))

    # Add the moved image data to the grid
    grid["image"] = moved_image.flatten(order="F")  # Flatten in Fortran order

    # Save the moved image grid to a VTK file
    grid.save(filename)

# Function to save the displacement vector field as a VTK file
def save_displacement_vector_as_vtk(displacement_vector, filename):
    # Create a PyVista grid for the displacement vector
    vector_shape = displacement_vector.shape[:-1]
    x = np.arange(vector_shape[0])
    y = np.arange(vector_shape[1])
    z = np.arange(vector_shape[2])
    grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))

    # Add the displacement vector data to the grid
    vectors = np.zeros((np.prod(vector_shape), 3))
    for i in range(3):
        vectors[:, i] = displacement_vector[..., i].flatten(order="F")
    grid["displacement"] = vectors

    # Save the displacement vector grid to a VTK file
    grid.save(filename)


def dice_coefficient(volume_A, volume_B):

    # Get the middle third region of the volumes


    #binarize the volumes
    volume_A = binarize_volume(volume_A)
    volume_B = binarize_volume(volume_B)

    #make sure diemsion is same
    min_dim0 = min(volume_A.shape[0], volume_B.shape[0])
    min_dim1 = min(volume_A.shape[1], volume_B.shape[1])
    min_dim2 = min(volume_A.shape[2], volume_B.shape[2])

    volume_A = volume_A[:min_dim0, :min_dim1, :min_dim2]
    volume_B = volume_B[:min_dim0, :min_dim1, :min_dim2]

    #calculate the dice score
    volume_A = np.array(volume_A, dtype=np.float64)
    volume_B = np.array(volume_B, dtype=np.float64)
    intersection = np.sum(np.logical_and(volume_A, volume_B))
    total_voxels_A = np.sum(volume_A)
    total_voxels_B = np.sum(volume_B)
    dice = (2.0 * intersection) / (total_voxels_A + total_voxels_B)
    return dice

def plot_images(fixed_image, moving_image, reconstructed_image, slice_idx=None):
    """
    Function to plot the fixed image, moving image, and reconstructed image (moved image)
    after testing on a single slice.
    
    Args:
        fixed_image (numpy array): The fixed image volume.
        moving_image (numpy array): The moving image volume.
        reconstructed_image (numpy array): The reconstructed image volume (moved image).
        slice_idx (int, optional): The index of the slice to visualize. If None, the middle slice is used.
    """
    # If no slice index is provided, use the middle slice
    if slice_idx is None:
        slice_idx = fixed_image.shape[0] // 2  # Middle slice of the 3D volume

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot fixed image
    axes[0].imshow(fixed_image[slice_idx, :, :], cmap='gray')
    axes[0].set_title("Fixed Image")
    axes[0].axis('off')

    # Plot moving image
    axes[1].imshow(moving_image[slice_idx, :, :], cmap='gray')
    axes[1].set_title("Moving Image")
    axes[1].axis('off')

    # Plot reconstructed (moved) image
    axes[2].imshow(reconstructed_image[slice_idx, :, :], cmap='gray')
    axes[2].set_title("Reconstructed Image (Moved)")
    axes[2].axis('off')

    plt.show()



def plot_3x3_images(fixed_image, moving_image, reconstructed_image):
    """
    Function to plot a 3x3 grid for fixed, moving, and moved images.
    The central slices from the x, y, and z axes are visualized.
    
    Args:
        fixed_image (numpy array): The fixed image volume.
        moving_image (numpy array): The moving image volume.
        reconstructed_image (numpy array): The reconstructed image volume (moved image).
    """
    # Calculate central indices for each axis
    center_x = fixed_image.shape[0] // 2
    center_y = fixed_image.shape[1] // 2
    center_z = fixed_image.shape[2] // 2

    # Prepare slices from x, y, and z axes
    fixed_slices = [fixed_image[center_x, :, :], fixed_image[:, center_y, :], fixed_image[:, :, center_z]]
    moving_slices = [moving_image[center_x, :, :], moving_image[:, center_y, :], moving_image[:, :, center_z]]
    reconstructed_slices = [reconstructed_image[center_x, :, :], reconstructed_image[:, center_y, :], reconstructed_image[:, :, center_z]]

    # Titles for subplots
    row_titles = ["Central Slice (X-axis)", "Central Slice (Y-axis)", "Central Slice (Z-axis)"]
    column_titles = ["Fixed Image", "Moving Image", "Reconstructed Image"]

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):  # For each axis
        # Plot fixed, moving, and reconstructed images for each axis
        axes[i, 0].imshow(fixed_slices[i], cmap='gray')
        axes[i, 0].set_title(f"{row_titles[i]} - {column_titles[0]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(moving_slices[i], cmap='gray')
        axes[i, 1].set_title(f"{row_titles[i]} - {column_titles[1]}")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(reconstructed_slices[i], cmap='gray')
        axes[i, 2].set_title(f"{row_titles[i]} - {column_titles[2]}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def gaussian_weight(shape, sigma=0.5):
    """Generate a Gaussian weighting function to reduce block artifacts."""
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, shape[0]), 
        np.linspace(-1, 1, shape[1]), 
        np.linspace(-1, 1, shape[2]), indexing='ij')
    weight = np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    return weight / np.max(weight)  # Normalize to max 1

def test_data_generator(vxm_model, hdf5_file, patch_size=(128, 128, 128), stride=(64, 64, 64)):
    """
    Generator that extracts consecutive patches from a 3D volume, processes them, 
    and then stitches them back together for one sample.

    Args:
        hdf5_file: Path to the HDF5 file containing the volume data.
        patch_size: Size of the 3D patches to extract.
        stride: The step size between consecutive patches.
    
    Yields:
        A tuple of stitched volumes (moved image, displacement field) reconstructed from the processed patches.
    """
    hf = h5py.File(hdf5_file, 'r')
    num_samples = len(hf.keys()) // 2  # Assuming paired 'static' and 'moving' datasets
    
    def _next_multiple(n, base):
        r = n % base
        return n if r == 0 else n + (base - r) 


    # Calculate the necessary padding for each dimension
    #pad = [(0, (patch_size[i] - (vol_shape[i] % patch_size[i])) % patch_size[i]) for i in range(len(vol_shape))]
    for idx in range(num_samples):
        start_time = time.time()
        sample_name = hf[f'static_{idx}'].attrs.get('sample_name', f'sample_{idx}')
        print(f'Predicting for sample: {sample_name}')

        vol_shape = hf[f'static_{idx}'].shape
        moving_image = hf[f'moving_{idx}'][...]
        fixed_image = hf[f'static_{idx}'][...]


        moving_shape = moving_image.shape
        fixed_shape = fixed_image.shape

        # per-axis target = max of the two, each rounded up to next multiple of patch_size
        target_shape = tuple(
            max(_next_multiple(moving_shape[i], patch_size[i]),
                _next_multiple(fixed_shape[i],  patch_size[i]))
            for i in range(3)
        )
        # right-side zero pad both to the same target shape (pads only if needed)
        pad_moving = [(0, target_shape[i] - moving_shape[i]) for i in range(3)]
        pad_fixed  = [(0, target_shape[i] - fixed_shape[i])  for i in range(3)]

        padded_moving = np.pad(moving_image, pad_moving, mode='constant', constant_values=0)
        padded_fixed  = np.pad(fixed_image,  pad_fixed,  mode='constant', constant_values=0)
        padded_vol_shape = padded_fixed.shape
        print('Padded data with zeros')

        # Calculate the number of patches in each dimension
        patches_per_dim = [(padded_vol_shape[i] - patch_size[i]) // stride[i] + 1 for i in range(len(padded_vol_shape))]

        # Initialize empty arrays for the stitched moved image and displacement field
        reconstructed_moved = np.zeros(padded_vol_shape)
        reconstructed_displacement = np.zeros((*padded_vol_shape, 3))
        weight_volume = np.zeros(padded_vol_shape)  # To handle overlapping regions

        # Precompute Gaussian weights for blending
        gaussian_weights = gaussian_weight(patch_size)

        print('Initialized arrays')
        for z in range(patches_per_dim[0]):
            for y in range(patches_per_dim[1]):
                for x in range(patches_per_dim[2]):
                    start_z = z * stride[0]
                    start_y = y * stride[1]
                    start_x = x * stride[2]
                    print('Extracting Patch :::>')
                    # Extract patch
                    moving_patch = padded_moving[start_z:start_z + patch_size[0],
                                                 start_y:start_y + patch_size[1],
                                                 start_x:start_x + patch_size[2]]

                    # Normalize the patch
                    #moving_patch = (moving_patch - np.min(moving_patch)) / (np.max(moving_patch) - np.min(moving_patch))

                    # Prepare input for the model
                    patch_input = np.expand_dims(moving_patch, axis=-1)  # Add channel dimension
                    fixed_patch = padded_fixed[start_z:start_z + patch_size[0],
                                               start_y:start_y + patch_size[1],
                                               start_x:start_x + patch_size[2]]
                    fixed_patch = np.expand_dims(fixed_patch, axis=-1)
                    inputs = [np.expand_dims(patch_input, axis=0), np.expand_dims(fixed_patch, axis=0)]
                    
                    # Model prediction
                    print('Model prediction :::>')
                    processed_patch, displacement_patch = vxm_model.predict(inputs)
                    processed_patch = processed_patch.squeeze()
                    displacement_patch = displacement_patch.squeeze()

                    # Stitch with Gaussian blending
                    reconstructed_moved[start_z:start_z + patch_size[0],
                                        start_y:start_y + patch_size[1],
                                        start_x:start_x + patch_size[2]] += processed_patch * gaussian_weights

                    reconstructed_displacement[start_z:start_z + patch_size[0],
                                               start_y:start_y + patch_size[1],
                                               start_x:start_x + patch_size[2], :] += displacement_patch * gaussian_weights[..., np.newaxis]

                    weight_volume[start_z:start_z + patch_size[0],
                                  start_y:start_y + patch_size[1],
                                  start_x:start_x + patch_size[2]] += gaussian_weights

                    print('Stitching prediction :::>')
                    

        # Normalize to handle overlapping regions
        reconstructed_moved /= np.maximum(weight_volume, 1)  # Avoid division by zero
        reconstructed_displacement /= np.maximum(weight_volume[..., np.newaxis], 1)  # Normalize the vector field
        print('Crop padded area :::>')
        # Crop the padded area out to restore the original volume shape
        reconstructed_moved = reconstructed_moved[:vol_shape[0], :vol_shape[1], :vol_shape[2]]
        reconstructed_displacement = reconstructed_displacement[:vol_shape[0], :vol_shape[1], :vol_shape[2], :]
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f"Time taken to test one sample: {elapsed_time:.2f} minutes")
        yield reconstructed_moved, reconstructed_displacement, fixed_image, moving_image  # Yield the reconstructed volumes and fixed image


def add_scalebar(ax, length_pixels=100, label="1.5 mm", height=8, pad=20):
    """
    Adds a clean horizontal scalebar with the label above it.
    Parameters:
    - ax: matplotlib axis
    - length_pixels: length of the scalebar (in pixels)
    - label: string to display above the bar
    - height: thickness of the black bar
    - pad: padding from the bottom of the image (in pixels)
    """
    # Adjust for axis direction (handles flipped Y axes)
    ylim = ax.get_ylim()
    y_direction = -1 if ylim[0] > ylim[1] else 1
    y_start = ylim[0] + y_direction * pad
    x_start = 20  # fixed x offset from left
    # Draw the black scalebar
    ax.add_patch(
        plt.Rectangle((x_start, y_start), length_pixels, height,
                      color='black', zorder=10)
    )
    # Draw the label clearly ABOVE the bar (not overlapping)
    ax.text(
        x_start + length_pixels / 2,  # center of the bar
        y_start + height + 40 * y_direction,  # position above the bar
        label,
        color='black',
        fontsize=16,
        ha='center',
        va='bottom' if y_direction == 1 else 'top',
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.7',
            alpha=0.5
        )
    )

def generate_plot_overlays(ct_image, cad_image, moved_ct):
    Dx, Dy, Dz = cad_image.shape
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2
    fig, axes = plt.subplots(2, 3, figsize=(25, 10))
    titles = ["XCT vs CAD", "XCT vs CAD", "XCT vs CAD",
              "MOVED XCT vs CAD", "MOVED XCT vs CAD", "MOVED XCT vs CAD"]
    slices = [
        # First column (no rotation)
        (cad_image[x_mid, :, :], ct_image[x_mid, :, :]),

        # Middle column → 90° clockwise
        (np.rot90(cad_image[:, :, z_mid], k=-1),
         np.rot90(ct_image[:, :, z_mid], k=-1)),

        # Last column → 180°
        (np.rot90(cad_image[:, y_mid, :], k=2),
         np.rot90(ct_image[:, y_mid, :], k=2)),

        # Second row
        (cad_image[x_mid, :, :], moved_ct[x_mid, :, :]),

        # Middle column → 90° clockwise
        (np.rot90(cad_image[:, :, z_mid], k=-1),
         np.rot90(moved_ct[:, :, z_mid], k=-1)),

        # Last column → 180°
        (np.rot90(cad_image[:, y_mid, :], k=2),
         np.rot90(moved_ct[:, y_mid, :], k=2)),
    ]

    for ax, (cad, ct), title in zip(axes.flat, slices, titles):
        plot_overlay(ax, cad, ct, title)
        add_scalebar(ax, length_pixels=100, label="1.5 mm")  # 100 pixels = 1 mm
    plt.tight_layout()
    plt.show()

def plot_overlay(ax, cad, ct, title, cmap_cad="Greens", cmap_ct="gray", alpha=0.5):
    ax.imshow(cad, cmap=cmap_cad, alpha=1.0)
    ax.imshow(ct, cmap=cmap_ct, alpha=alpha)
    ax.set_title(title, fontsize=22)
    ax.axis("off")
    ax.set_aspect('equal')













    
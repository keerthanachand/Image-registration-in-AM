import numpy as np
import os
import h5py
import nibabel as nib
import tifffile as tiff
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import filters, morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, gaussian_filter, uniform_filter, binary_dilation
import random
from skimage.filters import threshold_local

def adaptive_threshold_ct(ct_data, block_size=41, offset=5, min_size=500, slice_index=None, visualize=False):
    """
    Apply adaptive thresholding to a 3D CT volume (slice-by-slice).
    Keeps only the largest connected component in each slice,
    and applies the binary mask to return thresholded CT data.
    Parameters:
        ct_data (ndarray): 3D grayscale CT image.
        block_size (int): Size of the neighborhood for thresholding.
        offset (int or float): Subtracted from local mean to determine threshold.
        min_size (int): Minimum object size to retain (removes small specks).
        slice_index (int): If provided and visualize=True, visualizes that slice.
        visualize (bool): Whether to show comparison for a given slice.
    Returns:
        masked_ct_data (ndarray): CT volume with only the thresholded foreground retained.
    """
    binary_mask = np.zeros_like(ct_data, dtype=bool)
    for i in range(ct_data.shape[0]):  # slice by slice
        slice_img = ct_data[i]
        # Adaptive thresholding
        local_thresh = threshold_local(slice_img, block_size=block_size, offset=offset)
        binarized = slice_img > local_thresh
        # Remove small noise
        binarized = remove_small_objects(binarized, min_size=min_size)
        # Keep only largest connected component
        labeled = label(binarized)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest_region = max(regions, key=lambda r: r.area)
            largest_mask = labeled == largest_region.label
        else:
            largest_mask = np.zeros_like(slice_img, dtype=bool)
        binary_mask[i] = largest_mask
    # Apply binary mask to original CT image
    masked_ct_data = ct_data * binary_mask
    # Optional visualization
    if visualize and slice_index is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ct_data[slice_index], cmap="gray")
        axes[0].set_title("Original CT Slice")
        axes[1].imshow(masked_ct_data[slice_index], cmap="gray")
        axes[1].set_title("Thresholded CT Slice")
        plt.tight_layout()
        plt.show()
    return masked_ct_data


def read_data(file_path):
    """Detects file type and reads the corresponding 3D volume."""
    if file_path.endswith(('.tiff', '.tif')):
        return tiff.imread(file_path)
    elif file_path.endswith(('.hdr', '.img')):
        return nib.load(file_path).get_fdata()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")



def plot_slices(static, moving, sample_name):
    """Plots corresponding slices from CAD and CT data in x, y, and z directions."""
    mid_x, mid_y, mid_z = static.shape[0] // 2, static.shape[1] // 2, static.shape[2] // 2
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    # X-direction
    axes[0, 0].imshow(static[mid_x, :, :], cmap='gray')
    axes[0, 0].set_title(f'{sample_name} - CAD (X-Direction)')
    axes[0, 1].imshow(moving[mid_x, :, :], cmap='gray')
    axes[0, 1].set_title(f'{sample_name} - CT (X-Direction)')
    # Y-direction
    axes[1, 0].imshow(static[:, mid_y, :], cmap='gray')
    axes[1, 0].set_title(f'{sample_name} - CAD (Y-Direction)')
    axes[1, 1].imshow(moving[:, mid_y, :], cmap='gray')
    axes[1, 1].set_title(f'{sample_name} - CT (Y-Direction)')
    # Z-direction
    axes[2, 0].imshow(static[:, :, mid_z], cmap='gray')
    axes[2, 0].set_title(f'{sample_name} - CAD (Z-Direction)')
    axes[2, 1].imshow(moving[:, :, mid_z], cmap='gray')
    axes[2, 1].set_title(f'{sample_name} - CT (Z-Direction)')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def global_otsu_thresholding(data, use_roi=False):
    """
    Apply Otsu's thresholding to a 3D CT volume.

    Parameters:
        data (ndarray): 3D CT volume.
        use_roi (bool): If True, applies Otsu's thresholding using the middle 1/3rd of the volume.
                        If False, applies Otsu's thresholding to the entire volume.

    Returns:
        thresholded_data (ndarray): Binary mask after Otsu's thresholding.
        threshold (float): Computed Otsu threshold.
    """
    if use_roi:
        # Get dimensions
        z_dim, y_dim, x_dim = data.shape

        # Define ROI: Middle 1/3rd along all axes
        z_start, z_end = z_dim // 3, 2 * z_dim // 3
        y_start, y_end = y_dim // 3, 2 * y_dim // 3
        x_start, x_end = x_dim // 3, 2 * x_dim // 3

        # Extract the ROI
        roi = data[z_start:z_end, y_start:y_end, x_start:x_end]
    else:
        # Use the entire volume
        roi = data

    # Flatten the ROI and compute Otsu threshold
    flattened_data = roi.flatten()
    if flattened_data.size == 0:
        raise ValueError("ROI is empty, cannot apply thresholding.")

    threshold = filters.threshold_otsu(flattened_data)

    # Apply thresholding to the entire volume
    thresholded_data = (data >= threshold).astype(np.uint8)

    return thresholded_data, threshold


def clean_powder(CT_data, erosion_radius=2, opening_radius=3, use_roi=True):
    """
    Process the CT data using morphological operations to remove powder particles.

    Parameters:
        CT_data (ndarray): 3D CT scan data.
        erosion_radius (int): Radius for erosion.
        opening_radius (int): Radius for opening.
        use_roi (bool): If True, use ROI for thresholding; otherwise, apply globally.
    """
    # Apply Otsu's thresholding with ROI selection
    binary_image, threshold = global_otsu_thresholding(CT_data, use_roi=use_roi)

    labeled_image, num_labels = label(binary_image, return_num=True)
    print(f"Number of connected components: {num_labels}")

    regions = regionprops(labeled_image)
    largest_region = max(regions, key=lambda region: region.area)

    largest_component_mask = np.zeros_like(binary_image)
    largest_component_mask[labeled_image == largest_region.label] = 1

    eroded_mask = morphology.erosion(largest_component_mask, morphology.ball(erosion_radius))
    opened_mask = morphology.opening(eroded_mask, morphology.ball(opening_radius))

    return opened_mask.astype(np.float64) * CT_data




def fill_keyhole_pores_grey(CT_data, variation_factor=0.05, filter_size=10, dilation_iters=1):
    """
    pore filling for CT images with adaptive smoothing and multi-pass filtering.
    Parameters:
        CT_data (ndarray): 3D volumetric CT scan (grayscale).
        variation_factor (float): Controls variation in filled regions.
        filter_size (int): Neighborhood size for local mean filtering.
        dilation_iters (int): Number of iterations for dilation to expand filling.
        sigma_low (float): First-pass Gaussian smoothing for local corrections.
        sigma_high (float): Final-pass Gaussian smoothing for uniform blending.
        diffusion_weight (float): Strength of anisotropic diffusion filtering.
    Returns:
        filled_CT_data (ndarray): CT scan with keyhole porosities perfectly blended.
    """
    # Step 1: Apply Otsu thresholding to segment foreground
    #threshold_value = threshold_otsu(CT_data)
    #binary_mask = CT_data > threshold_value
    binary_mask = np.zeros_like(CT_data, dtype=bool)
    for z in range(CT_data.shape[0]):
        slice_img = CT_data[z]
        local_thresh = threshold_local(slice_img, block_size=101, offset=3)
        binary_mask[z] = slice_img > local_thresh
    # Step 2: Fill small holes in the binary mask
    fully_filled_mask = binary_fill_holes(binary_mask)
    # Step 3: Identify pore locations (inside filled mask but not in original mask)
    hole_locations = np.logical_and(fully_filled_mask, ~binary_mask)
    # Step 4: Expand filling region slightly more
    expanded_hole_mask = binary_dilation(hole_locations, iterations=dilation_iters)
    # Step 5: Compute global foreground mean for fallback
    global_foreground_mean = np.mean(CT_data[binary_mask])
    # Step 6: Compute local mean using uniform filtering
    CT_times_mask = CT_data * binary_mask
    nonzero_mask = (CT_times_mask != 0)
    CT_masked = CT_data * nonzero_mask
    local_sum = uniform_filter(CT_masked.astype(float), size=filter_size)
    local_count = uniform_filter(nonzero_mask.astype(float), size=filter_size)
    epsilon = 1e-6
    local_means = local_sum / (local_count + epsilon)
    #local_means = uniform_filter(CT_data, size=filter_size)
    # Step 7: Fill with local mean + adaptive variation
    variation = local_means * variation_factor * np.random.uniform(-1, 1, size=CT_data.shape)
    CT_data_filled = np.where(expanded_hole_mask, local_means + variation, CT_data)
    # Step 8: Apply First-Pass Gaussian Smoothing
    #smoothed_filled_data = gaussian_filter(CT_data_filled, sigma=sigma_low)
    # Step 9: Apply Anisotropic Diffusion (Perona-Malik)
    #smoothed_filled_data = denoise_tv_chambolle(smoothed_filled_data, weight=diffusion_weight)
    # Step 10: Apply Final-Pass Gaussian Smoothing
    #smoothed_filled_data = gaussian_filter(smoothed_filled_data, sigma=sigma_high)
    # Step 11: Blend filled region into original data
    # final_CT = np.where(expanded_hole_mask, smoothed_filled_data, CT_data)
    return CT_data_filled



def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def save_to_hdf5(static, moving, sample_names, hdf5_filename):
    with h5py.File(hdf5_filename, 'w') as hf:
        for i, (static_vol, moving_vol) in enumerate(zip(static, moving)):
            static_dset = hf.create_dataset(f'static_{i}', data=static_vol)
            moving_dset = hf.create_dataset(f'moving_{i}', data=moving_vol)
            static_dset.attrs['sample_name'] = sample_names[i]
            moving_dset.attrs['sample_name'] = sample_names[i]

def find_data_path(data_dir, sample, data_type):
    """Finds the correct file path for CAD or CT data based on available extensions."""
    for ext in ['.tiff', '.tif', '.hdr']:
        file_path = os.path.join(data_dir, sample, f'{data_type}{ext}')
        if os.path.exists(file_path):
            return file_path
    return None

def apply_otsu_mask(XCT_data):
    """
    Applies Otsu's thresholding to XCT data and keeps only the foreground, setting the background to zero.
    Parameters:
        XCT_data (ndarray): 3D XCT scan data.
    Returns:
        XCT_foreground (ndarray): XCT data with only the foreground retained, background set to zero.
        binary_mask (ndarray): Binary mask used for foreground extraction.
    """
    # Step 1: Compute Otsu threshold
    threshold_value = threshold_otsu(XCT_data)
    # Step 2: Create binary mask (1 for foreground, 0 for background)
    binary_mask = XCT_data >= threshold_value
    # Step 3: Apply mask to retain only foreground, set background to zero
    XCT_foreground = XCT_data * binary_mask
    return XCT_foreground, binary_mask

def pad_3d_array(array, pad_width=(1,1,1)):
    """
    Adds zero padding to a 3D NumPy array along all three dimensions.
    Parameters:
    array (numpy.ndarray): The input 3D array.
    pad_width (tuple): A tuple of three integers specifying the padding for each dimension.
    Returns:
    numpy.ndarray: The zero-padded 3D array.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array.")
    pad_x, pad_y, pad_z = pad_width
    return np.pad(array, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), mode='constant', constant_values=0)


def process_and_save_data(data_dir, test_samples, save_dir, use_roi=None, apply_clean_powder=None, apply_fill_keyholes=None):
    """
    Processes the CAD and CT data, applies optional preprocessing steps, and saves the output.

    Parameters:
        data_dir (str): Directory containing the dataset.
        test_samples (list): List of sample names to process.
        save_dir (str): Directory where processed HDF5 files will be saved.
        use_roi (bool, optional): If None, prompts user to choose whether to apply ROI-based thresholding.
        apply_clean_powder (bool, optional): If None, prompts user to choose whether to apply powder cleaning.
        apply_fill_keyholes (bool, optional): If None, prompts user to choose whether to apply keyhole filling.
    """
    start_time = time.time()
    test_static = []
    test_moving = []

    # Ask user for options if not provided
    if use_roi is None:
        use_roi = input("Would you like to apply ROI-based thresholding? (yes/no): ").strip().lower() == 'yes'
    if apply_clean_powder is None:
        apply_clean_powder = input("Would you like to apply powder cleaning? (yes/no): ").strip().lower() == 'yes'
    if apply_fill_keyholes is None:
        apply_fill_keyholes = input("Would you like to apply keyhole pores filling? (yes/no): ").strip().lower() == 'yes'

    print("\nProcessing Options:")
    print(f" - ROI-based thresholding: {'Enabled' if use_roi else 'Disabled'}")
    print(f" - Powder Cleaning: {'Enabled' if apply_clean_powder else 'Disabled'}")
    print(f" - Keyhole Pores Filling: {'Enabled' if apply_fill_keyholes else 'Disabled'}\n")

    print("Testing data processing started...")

    for sample in test_samples:
        static_path = find_data_path(data_dir, sample, 'CAD_data')
        moving_path = find_data_path(data_dir, sample, 'CT_data')

        if not static_path or not moving_path:
            print(f"Skipping {sample}: File not found.")
            continue

        static = read_data(static_path)
        moving = read_data(moving_path)

        print(f"{sample} -> Loaded static: {static.shape}, moving: {moving.shape}")

        # Apply ROI-based thresholding only if enabled
        if apply_clean_powder:
            moving = clean_powder(moving, use_roi=use_roi)

        # Apply keyhole pores filling if enabled
        if apply_fill_keyholes:
            moving = fill_keyhole_pores_grey(moving, variation_factor=0.05, filter_size=10, dilation_iters=5, sigma=4)

        #segmenting forground
        moving, binary_mask = apply_otsu_mask(moving)
        # Normalize both volumes
        static = normalize(static)
        moving = normalize(moving)

        #zero padding
        static = pad_3d_array(static, (12, 12, 12))
        moving = pad_3d_array(moving, (12, 12, 12))

        test_static.append(static)
        test_moving.append(moving)

    os.makedirs(save_dir, exist_ok=True)
    save_to_hdf5(test_static, test_moving, test_samples, os.path.join(save_dir, 'test_data.h5'))

    elapsed_time = (time.time() - start_time) / 60
    print(f"Processing completed in {elapsed_time:.2f} minutes")



data_dir = r'F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\original data'
test_samples = ['cuboid_01_01']
save_dir = r'F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\Other_test_data_processedfor_DL'
process_and_save_data(data_dir, test_samples, save_dir)


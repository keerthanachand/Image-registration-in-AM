import os
import numpy as np
import h5py
import skimage.transform
import tifffile as tiff
import time
from scipy.ndimage import binary_fill_holes, gaussian_filter, uniform_filter, binary_dilation
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from skimage import filters
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import tifffile as tiff
from scipy import stats
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.measure import label, regionprops
import random
from scipy.ndimage import binary_fill_holes
from skimage.restoration import denoise_tv_chambolle


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

def global_otsu_thresholding(data, roi=None):
    # Flatten the 3D CT data to a 1D array
    if roi is not None:
        # Apply the ROI to the data
        flattened_data = roi.flatten()
    else:
        # Flatten the entire data if ROI is not specified
        flattened_data = data.flatten()
    # Apply Otsu's thresholding to the flattened data
    threshold = filters.threshold_otsu(flattened_data)
    # Threshold the entire CT data
    thresholded_data = (data >= threshold).astype(np.uint8) * 1
    return thresholded_data, threshold


def clean_powder(CT_data, erosion_radius=2, opening_radius=3):
    """
    Process the CT data using a morphological pipeline: connected component analysis,
    erosion, opening, and mask multiplication, and keep only the largest connected component.

    Parameters:
    - CT_data (ndarray): 3D CT scan data.
    - erosion_radius (int): Radius for the erosion structuring element (default = 1).
    - opening_radius (int): Radius for the opening structuring element (default = 2).

    Returns:
    - processed_ct (ndarray): CT data multiplied by the mask after morphological operations.
    - largest_component_mask (ndarray): The final binary mask with only the largest connected component.
    """
    # Step 1: Connected Component Analysis (CCA) to label connected components in the binary image
    binary_image, threshold = global_otsu_thresholding(CT_data, roi=CT_data[400:600, 200:400, 300:500])
    labeled_image, num_labels = label(binary_image, return_num=True)
    print(f"Number of connected components: {num_labels}")

    # Step 2: Find the largest connected component
    regions = regionprops(labeled_image)
    largest_region = max(regions, key=lambda region: region.area)  # Get the largest component based on area
    print(f"Largest connected component size: {largest_region.area}")

    # Step 3: Create a mask for only the largest connected component
    largest_component_mask = np.zeros_like(binary_image)
    largest_component_mask[labeled_image == largest_region.label] = 1

    # Step 4: Erosion on the largest connected component mask
    selem_erosion = morphology.ball(erosion_radius)  # 3D spherical structuring element for erosion
    eroded_mask = morphology.erosion(largest_component_mask, selem_erosion)

    # Step 5: Opening (erosion followed by dilation) to clean the mask further
    selem_opening = morphology.ball(opening_radius)
    opened_mask = morphology.opening(eroded_mask, selem_opening)

    # Step 6: Multiply the opened mask with the original CT data
    processed_ct = opened_mask.astype(np.float64) * CT_data  # Apply the mask to the CT data
    return processed_ct


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


def load_tiff_volume(file_path):
    return tiff.imread(file_path)

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def resize_volume(volume, target_shape):
    return skimage.transform.resize(volume, target_shape, anti_aliasing=True)


def save_to_hdf5(static, moving, sample_names, hdf5_filename):
    with h5py.File(hdf5_filename, 'w') as hf:
        for i, (static_vol, moving_vol) in enumerate(zip(static, moving)):
            static_dset = hf.create_dataset(f'static_{i}', data=static_vol)
            moving_dset = hf.create_dataset(f'moving_{i}', data=moving_vol)
            static_dset.attrs['sample_name'] = sample_names[i]
            moving_dset.attrs['sample_name'] = sample_names[i]


def process_and_save_data(data_dir, train_samples, validation_samples, test_samples, target_shape, save_dir):
    start_time = time.time()
    train_static = []
    train_moving = []
    validation_static = []
    validation_moving = []
    test_static = []
    test_moving = []

    print("Training data processing started ")
    # Process training data
    for sample in train_samples:
        static_path = os.path.join(data_dir, sample, 'CAD_data.tiff')
        moving_path = os.path.join(data_dir, sample, 'CT_data.tiff')

        static = load_tiff_volume(static_path)
        moving = load_tiff_volume(moving_path)

        #preprocessing the CT data
        #clean powder in moving
        moving = clean_powder(moving)
        #fill holes in moving
        moving = fill_keyhole_pores_grey(CT_data, variation_factor=0.05, filter_size=10, dilation_iters=5,
                                         sigma_low=2, sigma_high=5, diffusion_weight=0.15)
        #normalize static and moving

        static = normalize(static)
        moving = normalize(moving)
        static_resized = resize_volume(static, target_shape)
        moving_resized = resize_volume(moving, target_shape)

        train_static.append(static_resized)
        train_moving.append(moving_resized)

    print("Validation data processing started ")
    # Process training data
    for sample in validation_samples:
        static_path = os.path.join(data_dir, sample, 'CAD_data.tiff')
        moving_path = os.path.join(data_dir, sample, 'CT_data.tiff')

        static = load_tiff_volume(static_path)
        moving = load_tiff_volume(moving_path)

        # preprocessing the CT data
        # clean powder in moving plus thresholding away background
        moving = clean_powder(moving)
        # fill holes in moving
        moving = fill_keyhole_pores_grey(moving)
        # normalize static and moving

        static = normalize(static)
        moving = normalize(moving)
        static_resized = resize_volume(static, target_shape)
        moving_resized = resize_volume(moving, target_shape)

        validation_static.append(static_resized)
        validation_moving.append(moving_resized)

    # Process testing data
    print("Testing data processing started ")
    for sample in test_samples:
        static_path = os.path.join(data_dir, sample, 'CAD_data.tiff')
        moving_path = os.path.join(data_dir, sample, 'CT_data.tiff')

        static = load_tiff_volume(static_path)
        moving = load_tiff_volume(moving_path)

        # preprocessing the CT data
        # clean powder in moving
        moving = clean_powder(moving)

        # fill holes in moving
        moving = fill_keyhole_pores_grey(moving)

        # normalize static and moving
        static = normalize(static)
        moving = normalize(moving)

        static_resized = resize_volume(static, target_shape)
        moving_resized = resize_volume(moving, target_shape)

        test_static.append(static_resized)
        test_moving.append(moving_resized)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save to HDF5
    print("Saving HDF5 ")
    save_to_hdf5(train_static, train_moving, train_samples, os.path.join(save_dir, 'train_data.h5'))
    save_to_hdf5(validation_static, validation_moving, validation_samples, os.path.join(save_dir, 'validation_data.h5'))
    save_to_hdf5(test_static, test_moving, test_samples, os.path.join(save_dir, 'test_data.h5'))
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time taken to complete process: {elapsed_time:.2f} minutes")

# Directory where your data is stored
data_dir = r'F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\processed_data'

# Define which samples are for training and which are for testing
train_samples = ['sample1', 'sample3', 'sample4', 'sample6']
validation_samples = ['sample2', 'sample5']
test_samples = ['sample7']

# Define target shape for resizing (update to desired dimensions)
#target_shape = (208, 144, 208)  # Example target shape, adjust as needed
target_shape = (816, 592, 816)
# Define directory to save the HDF5 files
save_dir = r'F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\data_for_DL_model_cleaned_train_val_test'

# Process the data and save to HDF5
process_and_save_data(data_dir, train_samples, validation_samples, test_samples, target_shape, save_dir)

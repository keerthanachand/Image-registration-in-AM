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

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.get_logger().setLevel('ERROR')


def save_as_tiff(image, filename):
    """
    Function to save a 3D numpy array as a TIFF file.
    
    Args:
        image (numpy array): The 3D image to be saved.
        filename (str): The filename where the image will be saved.
    """
    tiff.imwrite(filename, image)

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

def get_middle_region(volume):
    """
    Extract the middle third region of the 3D volume to avoid outer areas with air.
    
    Args:
        volume (numpy array): The 3D volume.
        
    Returns:
        numpy array: The extracted middle region of the 3D volume.
    """
    # Define the start and end points for the middle third in each dimension
    start_x, end_x = volume.shape[0] // 3, 2 * volume.shape[0] // 3
    start_y, end_y = volume.shape[1] // 3, 2 * volume.shape[1] // 3
    start_z, end_z = volume.shape[2] // 3, 2 * volume.shape[2] // 3
    
    # Extract the middle region
    middle_region = volume[start_x:end_x, start_y:end_y, start_z:end_z]    
    return middle_region

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

    return thresholded_data

def dice_coefficient(volume_A, volume_B):

    # Get the middle third region of the volumes
    roi_A = get_middle_region(volume_A)
    roi_B = get_middle_region(volume_B)

    #binarize the volumes
    volume_A = global_otsu_thresholding(volume_A, roi=roi_A)
    volume_B = global_otsu_thresholding(volume_B, roi=roi_B)

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

def test_data_generator(hdf5_file, patch_size=(128, 128, 128), stride=(64, 64, 64)):
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
        pad_fixed = [(0, (patch_size[i] - (fixed_shape[i] % patch_size[i])) % patch_size[i]) for i in range(len(fixed_shape))]
        pad_moving = [(0, (patch_size[i] - (moving_shape[i] % patch_size[i])) % patch_size[i]) for i in range(len(moving_shape))]
        # Apply padding to the images
        padded_moving = np.pad(moving_image, pad_moving, mode='constant', constant_values=0)
        padded_fixed = np.pad(fixed_image, pad_fixed, mode='constant', constant_values=0)
        padded_vol_shape = padded_fixed.shape
        print('Padded data with zeros')
        # Calculate the number of patches in each dimension
        patches_per_dim = [(padded_vol_shape[i] - patch_size[i]) // stride[i] + 1 for i in range(len(padded_vol_shape))]

        # Initialize empty arrays for the stitched moved image and displacement field
        reconstructed_moved = np.zeros(padded_vol_shape)
        reconstructed_displacement = np.zeros((*padded_vol_shape, 3))
        weight_volume = np.zeros(padded_vol_shape)  # To handle overlapping regions
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

                    print('Stcitching prediction :::>')
                    # Stitch the processed patch and displacement vector back into the reconstructed volumes
                    reconstructed_moved[start_z:start_z + patch_size[0],
                                        start_y:start_y + patch_size[1],
                                        start_x:start_x + patch_size[2]] += processed_patch

                    reconstructed_displacement[start_z:start_z + patch_size[0],
                                               start_y:start_y + patch_size[1],
                                               start_x:start_x + patch_size[2], :] += displacement_patch
                    
                    weight_volume[start_z:start_z + patch_size[0],
                                  start_y:start_y + patch_size[1],
                                  start_x:start_x + patch_size[2]] += 1

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



# Load trained VoxelMorph model
model_path = "/home/kchand/results/vxm_model_weights.h5"
architecture_path = "/home/kchand/results/vxm_model_architecture.json"


# Load model architecture
with open(architecture_path, "r") as json_file:
    model_json = json_file.read()
vxm_model = tf.keras.models.model_from_json(model_json, 
                custom_objects={'SpatialTransformer': vxm.layers.SpatialTransformer, 
                'VxmDense': vxm.networks.VxmDense})

# Load model weights
vxm_model.load_weights(model_path)

#
# test data load
test_hdf5 = r'/home/kchand/input_data/Canister_test_datav2.h5'


# Initialize the test generator
test_generator = test_data_generator(test_hdf5, patch_size=(128, 128, 128), stride=(64, 64, 64))
# Get the output for just one sample
reconstructed_moved, reconstructed_displacement, fixed_image, moving_image = next(test_generator)
# Plot the images
plot_images(fixed_image, moving_image, reconstructed_moved)
plot_3x3_images(fixed_image, moving_image, reconstructed_moved)
# calculate Dice score on the data before and aftr non linear reg excluding the base plate
Dice_init = dice_coefficient(fixed_image[:,:530,:], moving_image[:,:530,:])
print(f'Dice score before non-linear registration on test data is: {Dice_init:.4f}')
Dice_after_reg = dice_coefficient(fixed_image[:,:530,:], reconstructed_moved[:,:530,:])
print(f'Dice score after non-linear registration on test data is: {Dice_after_reg:.4f}')

#save data 
save_image_as_vtk(reconstructed_moved, r'/home/kchand/results/Canister/moved_image.vtk')
save_image_as_vtk(fixed_image, r'/home/kchand/results/Canister/fixed_image.vtk')
save_image_as_vtk(moving_image, r'/home/kchand/results/Canister/moving_image.vtk')
save_displacement_vector_as_vtk(reconstructed_displacement, r'/home/kchand/results/Canisterv2/disp_field.vtk')
save_as_tiff(reconstructed_moved, r'/home/kchand/results/Canisterv2/reconstructed_moved.tiff')
save_as_tiff(fixed_image, r'/home/kchand/results/Canisterv2/fixed_image.tiff')
save_as_tiff(moving_image, r'/home/kchand/results/Canisterv2/moving_image.tiff')


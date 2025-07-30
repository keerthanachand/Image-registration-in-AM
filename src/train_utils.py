import os

import time
import h5py
import numpy as np
import tensorflow as tf
from skimage import filters
import voxelmorph as vxm
import neurite as ne
import skimage
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import tifffile as tiff
from tensorflow.keras.callbacks import ReduceLROnPlateau
from voxelmorph import networks, losses




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

    #calculate the dice score
    volume_A = np.array(volume_A, dtype=np.float64)
    volume_B = np.array(volume_B, dtype=np.float64)
    intersection = np.sum(np.logical_and(volume_A, volume_B))
    total_voxels_A = np.sum(volume_A)
    total_voxels_B = np.sum(volume_B)
    dice = (2.0 * intersection) / (total_voxels_A + total_voxels_B)
    return dice

def save_as_tiff(image, filename):
    """
    Function to save a 3D numpy array as a TIFF file.
    
    Args:
        image (numpy array): The 3D image to be saved.
        filename (str): The filename where the image will be saved.
    """
    tiff.imwrite(filename, image)


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

# Example usage
# fixed_image, moving_image, reconstructed_image should be numpy arrays of the same shape
# plot_3x3_images(fixed_image, moving_image, reconstructed_image)


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
    #vol_shape = hf['static_0'].shape
    start_time = time.time()
    # Calculate the necessary padding for each dimension
    pad = [(0, (patch_size[i] - (vol_shape[i] % patch_size[i])) % patch_size[i]) for i in range(len(vol_shape))]
    
    for idx in range(num_samples):
        moving_image = hf[f'moving_{idx}'][...]
        fixed_image = hf[f'static_{idx}'][...]

        # Apply padding to the images
        padded_moving = np.pad(moving_image, pad, mode='constant', constant_values=0)
        padded_fixed = np.pad(fixed_image, pad, mode='constant', constant_values=0)
        padded_vol_shape = padded_moving.shape

        # Calculate the number of patches in each dimension
        patches_per_dim = [(padded_vol_shape[i] - patch_size[i]) // stride[i] + 1 for i in range(len(padded_vol_shape))]

        # Initialize empty arrays for the stitched moved image and displacement field
        reconstructed_moved = np.zeros(padded_vol_shape)
        reconstructed_displacement = np.zeros((*padded_vol_shape, 3))
        weight_volume = np.zeros(padded_vol_shape)  # To handle overlapping regions

        for z in range(patches_per_dim[0]):
            for y in range(patches_per_dim[1]):
                for x in range(patches_per_dim[2]):
                    start_z = z * stride[0]
                    start_y = y * stride[1]
                    start_x = x * stride[2]

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
                    processed_patch, displacement_patch = vxm_model.predict(inputs)
                    processed_patch = processed_patch.squeeze()
                    displacement_patch = displacement_patch.squeeze()

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

        # Crop the padded area out to restore the original volume shape
        reconstructed_moved = reconstructed_moved[:vol_shape[0], :vol_shape[1], :vol_shape[2]]
        reconstructed_displacement = reconstructed_displacement[:vol_shape[0], :vol_shape[1], :vol_shape[2], :]
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f"Time taken to test one sample: {elapsed_time:.2f} minutes")
        yield reconstructed_moved, reconstructed_displacement, fixed_image, moving_image  # Yield the reconstructed volumes and fixed image


def save_moved_image_as_vtk(moved_image, filename):
    # Create a PyVista grid for the moved image
    moved_image_shape = moved_image.shape
    x = np.arange(moved_image_shape[0])
    y = np.arange(moved_image_shape[1])
    z = np.arange(moved_image_shape[2])
    grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))

    # Add the moved image data to the grid
    grid["moved_image"] = moved_image.flatten(order="F")  # Flatten in Fortran order

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

def plot_history(history, loss_name='loss'):
    fig, ax = plt.subplots()
    ax.plot(history.history[loss_name], label='Training Loss')
    if 'val_' + loss_name in history.history:
        ax.plot(history.history['val_' + loss_name], label='Validation Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('History of ' + loss_name, fontsize=12)
    ax.legend()
    plt.show()


def initialize_generator_parameters(hdf5_file, patch_size):
    """
    This function initializes and returns the constant values
    needed for the generator, such as num_samples, volume shape,
    and max values for coordinates.
    """
    with h5py.File(hdf5_file, 'r') as hf:
        num_samples = len(hf.keys()) // 2  # Assuming paired 'static' and 'moving' datasets
        vol_shape = hf['static_0'].shape
        ndims = len(vol_shape)
        
        # Calculate max_x, max_y, and max_z based on the shape of the volumes
        max_x = vol_shape[0] - patch_size[0]
        max_y = vol_shape[1] - patch_size[1]
        max_z = vol_shape[2] - patch_size[2]

    return num_samples, vol_shape, ndims, max_x, max_y, max_z


def vxm_data_generator(hdf5_file, patch_size=(128, 128, 128), batch_size=1, generator_params=None):
    """
    Generator that loads data from an HDF5 file and yields data for
    the VoxelMorph model using corresponding random patches.
    
    Inputs:  moving [bs, D, H, W, 1], fixed image [bs, D, H, W, 1]
    Outputs: moved image [bs, D, H, W, 1], zero-gradient [bs, D, H, W, 3]
    """
    # Unpack the generator parameters to avoid recalculating them
    num_samples, vol_shape, ndims, max_x, max_y, max_z = generator_params

    with h5py.File(hdf5_file, 'r') as hf:
        # Zero array for the deformation field (used in outputs)
        zero_phi = np.zeros([batch_size, *patch_size, ndims])
        
        # Normalization function (applied during patch extraction)
        
        
        # Function to extract a random patch from the same location in both images
        def extract_corresponding_patches(moving_image, fixed_image, patch_size):
            # Generate a single set of starting coordinates for both moving and fixed images
            start_x = np.random.randint(0, max_x + 1)
            start_y = np.random.randint(0, max_y + 1)
            start_z = np.random.randint(0, max_z + 1)

            # Extract the patches from the same location in both images
            moving_patch = moving_image[start_x:start_x + patch_size[0], 
                                        start_y:start_y + patch_size[1], 
                                        start_z:start_z + patch_size[2]]

            fixed_patch = fixed_image[start_x:start_x + patch_size[0], 
                                      start_y:start_y + patch_size[1], 
                                      start_z:start_z + patch_size[2]]

            return np.expand_dims(moving_patch, axis=-1), np.expand_dims(fixed_patch, axis=-1)

        while True:
            moving_patches = []
            fixed_patches = []
            
            # Process each sample in the batch
            for _ in range(batch_size):
                # Randomly select a sample (load only once per iteration)
                idx = np.random.randint(0, num_samples)
                moving_image = hf[f'moving_{idx}'][...]  # Load image into memory
                fixed_image = hf[f'static_{idx}'][...]   # Load corresponding fixed image
                
                # Extract corresponding patches from the images
                moving_patch, fixed_patch = extract_corresponding_patches(moving_image, fixed_image, patch_size)

                # Append the patches to the batch list
                moving_patches.append(moving_patch)
                fixed_patches.append(fixed_patch)

            # Convert lists of patches to numpy arrays
            moving_patches = np.array(moving_patches)
            fixed_patches = np.array(fixed_patches)
            
            # Yield inputs (moving, fixed images) and outputs (fixed images, zero deformation field)
            inputs = [moving_patches, fixed_patches]
            outputs = [fixed_patches, zero_phi]
            yield inputs, outputs

def create_vxm_model(in_sample): 
    ndim = 3
    vol_shape = in_sample[0].shape[1:4]
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    # Separate inputs for moving and fixed images
    moving_input = tf.keras.layers.Input(shape=(*vol_shape, 1), name='moving_image')
    fixed_input = tf.keras.layers.Input(shape=(*vol_shape, 1), name='fixed_image')

    # Concatenate the inputs along the channel axis
    concat_input = tf.keras.layers.Concatenate(axis=-1)([moving_input, fixed_input])

    # Build the U-Net model
    unet_output = vxm.networks.Unet(inshape=(*vol_shape, 2), nb_features=nb_features)(concat_input)

    # Batch Normalization and LeakyReLU after each Conv3D in the Unet
    def add_batchnorm_and_activation(x):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    # Apply the transformations (batchnorm + activation) to the U-Net layers
    unet_output_bn = add_batchnorm_and_activation(unet_output)

    # Transform the results into a flow field (displacement field)
    disp_tensor = tf.keras.layers.Conv3D(ndim, kernel_size=3, padding='same', name='disp')(unet_output_bn)

    # Apply Batch Normalization after the displacement Conv3D layer
    disp_tensor = tf.keras.layers.BatchNormalization()(disp_tensor)

    # Spatial transformer to warp the moving image
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

    # Warp the moving image using the predicted displacement field
    moved_image_tensor = spatial_transformer([moving_input, disp_tensor])

    # The model will output both the moved image and the displacement field
    outputs = [moved_image_tensor, disp_tensor]

    # Return the final model that takes moving and fixed images as input
    return tf.keras.models.Model(inputs=[moving_input, fixed_input], outputs=outputs)


def plot_patches(inputs, outputs, patch_size=(128, 128, 128)):
    """
    Function to plot corresponding random image patches from inputs and outputs.
    
    inputs:  List of numpy arrays [moving_patches, fixed_patches]
    outputs: List of numpy arrays [fixed_patches, zero_phi]
    patch_size: Size of the 3D patches
    """
    moving_patches, fixed_patches = inputs
    fixed_patches_out, zero_phi = outputs
    
    batch_size = moving_patches.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    
    for i in range(batch_size):
        ax1, ax2, ax3 = axes[i] if batch_size > 1 else axes

        # Extracting central slice for visualization
        moving_slice = moving_patches[i, ..., 0][patch_size[0] // 2, :, :]
        fixed_slice_in = fixed_patches[i, ..., 0][patch_size[0] // 2, :, :]
        fixed_slice_out = fixed_patches_out[i, ..., 0][patch_size[0] // 2, :, :]
        
        ax1.imshow(moving_slice, cmap='gray')
        ax1.set_title(f'Moving Image Patch {i+1}')
        
        ax2.imshow(fixed_slice_in, cmap='gray')
        ax2.set_title(f'Fixed Image Patch (Input) {i+1}')
        
        ax3.imshow(fixed_slice_out, cmap='gray')
        ax3.set_title(f'Fixed Image Patch (Output) {i+1}')
    
    plt.tight_layout()
    plt.show()


def build_and_train_vxm_model(train_generator, in_sample, val_generator=None, nb_features=None, nb_epochs=150, steps_per_epoch=4, validation_steps=2):
    """
    Builds, compiles, and trains a VoxelMorph model on given data generators.

    Parameters:
        train_generator: Generator for training data.
        in_sample: A sample input to determine volume shape.
        val_generator: Generator for validation data (optional).
        nb_features (list): List of encoder and decoder features.
        nb_epochs (int): Number of training epochs.
        steps_per_epoch (int): Training steps per epoch.
        validation_steps (int): Validation steps per epoch.

    Returns:
        model (tf.keras.Model): Trained VoxelMorph model.
        history (History): Training history object.
    """
    import time
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from voxelmorph import networks, losses

    # Get volume shape and default features if not provided
    vol_shape = in_sample[0].shape[1:4]
    if nb_features is None:
        nb_features = [[32, 32, 32, 32], [32, 32, 32, 32, 32, 16]]

    # Initialize model
    model = networks.VxmDense(vol_shape, nb_features, int_steps=0)

    # Compile model with losses and weights
    loss_functions = [losses.NCC().loss, losses.Grad('l2').loss]
    loss_weights = [1, 0.05]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=loss_functions,
                  loss_weights=loss_weights)

    # Define callbacks
    callbacks = []
    if val_generator is not None:
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    else:
        lr_scheduler = ReduceLROnPlateau(
            monitor='loss',
            factor=0.8,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    callbacks.append(lr_scheduler)

    # Train
    start_time = time.time()
    if val_generator is not None:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=nb_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2,
            callbacks=callbacks
        )
    else:
        history = model.fit(
            train_generator,
            epochs=nb_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=2,
            callbacks=callbacks
        )
    end_time = time.time()
    print(f"✅ Training completed in {(end_time - start_time) / 60:.2f} minutes")

    return model, history

if __name__ == '__main__':

    # Paths to the HDF5 files
    train_hdf5 = r'/home/kchand/input_data/train_data.h5'
    val_hdf5 = r'/home/kchand/input_data/validation_data.h5'
    #test_hdf5 = r'/home/kchand/input_data/test_data.h5'

    # Initialize generator parameters (only need to do this once)
    generator_params = initialize_generator_parameters(hdf5_file=train_hdf5, patch_size=(128, 128, 128))
    # Create the generator with the precomputed parameters
    train_generator = vxm_data_generator(hdf5_file=train_hdf5, patch_size=(128, 128, 128), batch_size=8, generator_params=generator_params)

    # Create the validation generator with the precomputed parameters
    val_generator_params = initialize_generator_parameters(hdf5_file=val_hdf5, patch_size=(128, 128, 128))
    val_generator = vxm_data_generator(hdf5_file=val_hdf5, patch_size=(128, 128, 128), batch_size=4, generator_params=val_generator_params)


    # Get sample data for input shapes
    in_sample, out_sample = next(train_generator)
    plot_patches(in_sample, out_sample, patch_size=(128, 128, 128))

    print(f"in_sample shape: {in_sample[0].shape}, {in_sample[1].shape}")
    print(f"out_sample shape: {out_sample[0].shape}, {out_sample[1].shape}")

    vxm_model, history = build_and_train_vxm_model(
        train_generator=train_generator,
        in_sample=in_sample,
        val_generator=val_generator  # or None
    )

    # # Determine volume shape and features for the VoxelMorph model
    # vol_shape = in_sample[0].shape[1:4]
    # nb_features = [
    # [32, 32, 32, 32],         # encoder features
    # [32, 32, 32, 32, 32, 16]  # decoder features and conv 
    # ]

    # # Initialize the VoxelMorph model with 3D convolutions
    # vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    # #function call for vmxmodel
    # #vxm_model = create_vxm_model(in_sample)
    # #losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    # losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    # lambda_param = 0.05
    # loss_weights = [1, lambda_param]

    # vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses, loss_weights=loss_weights)
    # #vxm_model.summary()
    # # Train the model
    # start_time = time.time()
    # #number of times the model will go through the entire training dataset
    # nb_epochs = 150
    # #number of batches to draw from the generator for each epoch
    # steps_per_epoch = 4
    # validation_steps = 2 
    # lr_scheduler = ReduceLROnPlateau(
    # monitor='val_loss',    # Monitor validation loss to decide when to reduce the learning rate
    # factor=0.8,            # Reduce learning rate by a factor of 0.5 (half it)
    # patience=10,            # Number of epochs with no improvement after which to reduce learning rate
    # min_lr=1e-6,           # Lower bound on the learning rate
    # verbose=1              # Verbosity mode (prints updates when learning rate is reduced)
    # )


    # #train the model
    # hist = vxm_model.fit(
    # train_generator,
    # validation_data=val_generator,
    # epochs=nb_epochs,
    # steps_per_epoch=steps_per_epoch,
    # validation_steps=validation_steps,
    # verbose=2,
    # callbacks=[lr_scheduler]  # Add learning rate scheduler to callbacks
    # )
    # end_time = time.time()
    # elapsed_time = (end_time - start_time) / 60
    # print(f"Time taken to complete training: {elapsed_time:.2f} minutes")

    # #visualize history 
    # plot_history(hist)

    plot_history(history)
    #model weights 
    vxm_model.save_weights(r'/home/kchand/results/vxm_model_weights.h5')

    # Save the model architecture in JSON format
    model_json = vxm_model.to_json()
    with open(r"/home/kchand/results/vxm_model_architecture.json", "w") as json_file:
        json_file.write(model_json)


    # Convert training history to a DataFrame and save as CSV
    history_df = pd.DataFrame(hist.history)
    history_df.to_csv(r'/home/kchand/results/training_history.csv', index=False)


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
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from voxelmorph import layers, networks, losses
import voxelmorph as vxm
import matplotlib.pyplot as plt
import time

from evaluation_utils import (
    test_data_generator,
    save_displacement_vector_as_vtk,
    save_as_tiff_uint8,
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages,
    
)

from train_utils import (
    initialize_generator_parameters, 
    vxm_data_generator,
    save_image_as_vtk
)

from cross_validation import (
    dice_coefficient,
    plot_3x3_images
)

import matplotlib.pyplot as plt
import os

def plot_history(history, loss_name='loss', output_dir=None, show=True):
    """
    Plot and save training and validation loss history.

    Args:
        history: Keras History object from model.fit()
        loss_name: Name of the loss key to plot (default 'loss')
        output_dir: Path to save the PNG file. If None, doesn't save.
        show: Whether to display the plot using plt.show()
    """
    fig, ax = plt.subplots()
    ax.plot(history.history[loss_name], label='Training Loss')
    
    val_loss_key = 'val_' + loss_name
    if val_loss_key in history.history:
        ax.plot(history.history[val_loss_key], label='Validation Loss')

    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Training and Validation Loss History', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Save the figure
    if output_dir:
        filename = os.path.join(output_dir, f'{loss_name}_history.png')
        fig.savefig(filename)
        print(f"📊 Saved loss history plot to: {filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def dice_coefficient(volume_A, volume_B):

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

def plot_3x3_images(fixed_image, moving_image, reconstructed_image, save_path=None):
    """
    Plot and optionally save a 3x3 grid of slices from fixed, moving, and reconstructed images.

    Args:
        fixed_image (numpy array): The fixed image volume.
        moving_image (numpy array): The moving image volume.
        reconstructed_image (numpy array): The reconstructed image volume (moved image).
        save_path (str, optional): If provided, saves the figure to this path.
    """
    

    center_x = fixed_image.shape[0] // 2
    center_y = fixed_image.shape[1] // 2
    center_z = fixed_image.shape[2] // 2

    fixed_slices = [fixed_image[center_x, :, :], fixed_image[:, center_y, :], fixed_image[:, :, center_z]]
    moving_slices = [moving_image[center_x, :, :], moving_image[:, center_y, :], moving_image[:, :, center_z]]
    reconstructed_slices = [reconstructed_image[center_x, :, :], reconstructed_image[:, center_y, :], reconstructed_image[:, :, center_z]]

    row_titles = ["Central Slice (X-axis)", "Central Slice (Y-axis)", "Central Slice (Z-axis)"]
    column_titles = ["Fixed Image", "Moving Image", "Reconstructed Image"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
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

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"🖼️ Saved comparison plot to: {save_path}")
        plt.close(fig)  # Avoid showing if saving
    else:
        plt.show()


def build_and_train_vxm_model(train_generator, in_sample, val_generator=None, nb_features=None, nb_epochs=200, steps_per_epoch=5, validation_steps=2):
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
    loss_functions = [losses.MutualInformation().loss, losses.Grad('l2').loss]
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
    test_hdf5 = r'/home/kchand/input_data/test_data.h5'

    # Initialize generator parameters (only need to do this once)
    generator_params = initialize_generator_parameters(hdf5_file=train_hdf5, patch_size=(128, 128, 128))
    # Create the generator with the precomputed parameters
    train_generator = vxm_data_generator(hdf5_file=train_hdf5, patch_size=(128, 128, 128), batch_size=8, generator_params=generator_params)

    # Create the validation generator with the precomputed parameters
    val_generator_params = initialize_generator_parameters(hdf5_file=val_hdf5, patch_size=(128, 128, 128))
    val_generator = vxm_data_generator(hdf5_file=val_hdf5, patch_size=(128, 128, 128), batch_size=4, generator_params=val_generator_params)


    # Get sample data for input shapes
    in_sample, out_sample = next(train_generator)  

    vxm_model, history = build_and_train_vxm_model(
        train_generator=train_generator,
        in_sample=in_sample,
        val_generator=val_generator, nb_epochs=250  # or None
    )

    output_dir = "/home/kchand/results/MI"
    os.makedirs(output_dir, exist_ok=True)

    #model weights 
    vxm_model.save_weights(os.path.join(output_dir, "vxm_model_weights.h5"))


    model_json = vxm_model.to_json()
    with open(os.path.join(output_dir, "vxm_model_architecture.json"), "w") as json_file:
        json_file.write(model_json)



    history_df = pd.DataFrame(history.history)

    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    plot_history(history, output_dir=output_dir, show = False)


    #test on test sample 
    print("🧠 Running inference on test sample...")
    
    # Initialize the test generator
    test_generator = test_data_generator(vxm_model, test_hdf5, patch_size=(128, 128, 128), stride=(64, 64, 64))
    # Get the output for just one sample
    start_time = time.time()
    reconstructed_moved, reconstructed_displacement, fixed_image, moving_image = next(test_generator)
    inference_time_sec = time.time() - start_time
    print(f"⏱️ Inference completed in {inference_time_sec:.2f} seconds")

    print("🧼 Computing Dice and BDM metrics...")
    fixed_crop = fixed_image[:, :530, :]
    moving_crop = moving_image[:, :530, :]
    moved_crop = reconstructed_moved[:, :530, :]
    # print("🧼 Computing Dice and BDM metrics...")
    # fixed_crop = fixed_image
    # moving_crop = moving_image
    # moved_crop = reconstructed_moved

    binary_fixed = binarize_volume(fixed_crop)
    binary_moving = binarize_volume(moving_crop)
    binary_moved = binarize_volume(moved_crop)

    plot_path = os.path.join(output_dir, "slice_comparison.png")
    plot_3x3_images(binary_fixed, binary_moving, binary_moved, save_path=plot_path)
    

    dice_before = dice_coefficient(binary_fixed, binary_moving)
    dice_after = dice_coefficient(binary_fixed, binary_moved)

    print(f"🎯 Dice BEFORE registration: {dice_before:.4f}")
    print(f"✅ Dice AFTER registration:  {dice_after:.4f}")

    diff_map_before = compute_diff_map(binary_fixed, binary_moving)
    #return difference map dictionary of BDM
    diff_stats_before = report_combined_difference_percentages(diff_map_before, binary_fixed, binary_moving)

    diff_map_after = compute_diff_map(binary_fixed, binary_moved)
    diff_stats_after = report_combined_difference_percentages(diff_map_after, binary_fixed, binary_moved)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_image_as_vtk(reconstructed_moved, os.path.join(output_dir, "moved_image.vtk"))
    save_image_as_vtk(fixed_image, os.path.join(output_dir, "fixed_image.vtk"))
    save_image_as_vtk(moving_image, os.path.join(output_dir, "moving_image.vtk"))
    save_displacement_vector_as_vtk(reconstructed_displacement, os.path.join(output_dir, "disp_field.vtk"))


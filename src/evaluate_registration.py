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


from evaluation_utils import (
    test_data_generator,
    plot_images, 
    plot_3x3_images, 
    save_image_as_vtk
    save_moved_image_as_vtk,
    save_displacement_vector_as_vtk,
    dice_coefficient,
    save_as_tiff, save_as_tiff_uint8
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages
)



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
test_hdf5 = r'/home/kchand/input_data/test_data.h5'


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
save_image_as_vtk(reconstructed_moved, r'/home/kchand/results/TPMS7/v1/moved_image.vtk')
save_image_as_vtk(fixed_image, r'/home/kchand/results//TPMS7/v1/fixed_image.vtk')
save_image_as_vtk(moving_image, r'/home/kchand/results//TPMS7/v1/moving_image.vtk')
save_displacement_vector_as_vtk(reconstructed_displacement, r'/home/kchand/results//TPMS7/v1/disp_field.vtk')
save_as_tiff_uint8(reconstructed_moved, r'/home/kchand/results//TPMS7/v1/reconstructed_moved.tiff')
save_as_tiff_uint8(fixed_image, r'/home/kchand/results//TPMS7/v1/fixed_image.tiff')
save_as_tiff_uint8(moving_image, r'/home/kchand/results//TPMS7/v1/moving_image.tiff')

print('All data is saved!')
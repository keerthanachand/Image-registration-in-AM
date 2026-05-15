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
    save_image_as_vtk,
    plot_overlay,
    generate_plot_overlays,
    add_scalebar,
    save_moved_image_as_vtk,
    save_displacement_vector_as_vtk,
    dice_coefficient,
    save_as_tiff, 
    save_as_tiff_uint8,
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages
)




# Load trained VoxelMorph model
model_path = "/home/kchand/results/cross_validation/vxm_weights_fold/weights_fold0.h5"
architecture_path = "/home/kchand/results/cross_validation/vxm_model_architecture.json"

# Load model architecture
with open(architecture_path, "r") as json_file:
    model_json = json_file.read()
vxm_model = tf.keras.models.model_from_json(model_json, 
                custom_objects={'SpatialTransformer': vxm.layers.SpatialTransformer, 
                'VxmDense': vxm.networks.VxmDense})

# Load model weights
vxm_model.load_weights(model_path)

#
# load data 
#test_hdf5 = r'/home/kchand/input_data/test_data.h5'
num_samples = 16

#toggle for a single fold (optional) e.g., set to 0 or 3 to test a single fold; max = num_samples - 1
only_fold = 15
all_data_path = '/home/kchand/input_data/all_samples_simple_structures.h5'
#load data
folds_to_run = [only_fold] if only_fold is not None else range(num_samples)

for fold_idx in folds_to_run:
    print(f"========== Starting Fold {fold_idx} ==========")
    train_hdf5, test_hdf5 = prepare_loocv_fold(all_data_path, fold_idx, num_samples)


with h5py.File(test_hdf5, 'r') as hf:
        moving = hf['moving_0'][...][np.newaxis, ..., np.newaxis]
        fixed = hf['static_0'][...][np.newaxis, ..., np.newaxis]
        #get the sample name from the attributes
        sample_name = hf['static_0'].attrs.get('sample_name', 'unknown_sample')

vol_shape = moving.shape[1:4]
# Initialize the test generator
test_generator = test_data_generator(vxm_model, test_hdf5, patch_size=(128, 128, 128), stride=(64, 64, 64))
# Get the output for just one sample
reconstructed_moved, reconstructed_displacement, fixed_image, moving_image = next(test_generator)
# Plot the images
plot_images(fixed_image, moving_image, reconstructed_moved)
plot_3x3_images(fixed_image, moving_image, reconstructed_moved)

#binarize and plot 
#binarise fixed and moving
binary_fixed = binarize_volume(fixed_image)
binary_moving = binarize_volume(moving_image)
binary_moved = binarize_volume(reconstructed_moved)
plot_images(binary_fixed, binary_moving, binary_moved)
plot_3x3_images(binary_fixed, binary_moving, binary_moved)

# Plot the images
plot_images(fixed_image, moving_image, reconstructed_moved)
plot_3x3_images(fixed_image, moving_image, reconstructed_moved)

# calculate Dice score on the data before and aftr non linear reg excluding the base plate
Dice_init = dice_coefficient(fixed_image[:,:,:], moving_image[:,:,:])
print(f'Dice score before non-linear registration on test data is: {Dice_init:.4f}')
Dice_after_reg = dice_coefficient(fixed_image[:,:,:], reconstructed_moved[:,:,:])
print(f'Dice score after non-linear registration on test data is: {Dice_after_reg:.4f}')



diff_map_before = compute_diff_map(binary_fixed, binary_moving)
#return difference map dictionary of BDM
diff_stats_before = report_combined_difference_percentages(diff_map_before, binary_fixed, binary_moving)

diff_map_after = compute_diff_map(binary_fixed, binary_moved)
diff_stats_after = report_combined_difference_percentages(diff_map_after, binary_fixed, binary_moved)
#save data 


save_image_as_vtk(reconstructed_moved, r'/home/kchand/results/TPMS7/v1/moved_image.vtk')
save_image_as_vtk(fixed_image, r'/home/kchand/results//TPMS7/v1/fixed_image.vtk')
save_image_as_vtk(moving_image, r'/home/kchand/results//TPMS7/v1/moving_image.vtk')
save_displacement_vector_as_vtk(reconstructed_displacement, r'/home/kchand/results//TPMS7/v1/disp_field.vtk')
save_as_tiff_uint8(reconstructed_moved, r'/home/kchand/results//TPMS7/v1/reconstructed_moved.tiff')
save_as_tiff_uint8(fixed_image, r'/home/kchand/results//TPMS7/v1/fixed_image.tiff')
save_as_tiff_uint8(moving_image, r'/home/kchand/results//TPMS7/v1/moving_image.tiff')

print('All data is saved!')'






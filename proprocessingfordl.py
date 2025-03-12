import os
import numpy as np
import h5py
import skimage.transform
import tifffile as tiff
import time

def load_tiff_volume(file_path):
    return tiff.imread(file_path)


def resize_volume(volume, target_shape):
    return skimage.transform.resize(volume, target_shape, anti_aliasing=True)


def save_to_hdf5(static, moving, sample_names, hdf5_filename):
    with h5py.File(hdf5_filename, 'w') as hf:
        for i, (static_vol, moving_vol) in enumerate(zip(static, moving)):
            static_dset = hf.create_dataset(f'static_{i}', data=static_vol)
            moving_dset = hf.create_dataset(f'moving_{i}', data=moving_vol)
            static_dset.attrs['sample_name'] = sample_names[i]
            moving_dset.attrs['sample_name'] = sample_names[i]


def process_and_save_data(data_dir, train_samples, test_samples, target_shape, save_dir):
    start_time = time.time()
    train_static = []
    train_moving = []
    test_static = []
    test_moving = []

    print("Training data processing started ")
    # Process training data
    for sample in train_samples:
        static_path = os.path.join(data_dir, sample, 'CAD_data.tiff')
        moving_path = os.path.join(data_dir, sample, 'CT_data.tiff')

        static = load_tiff_volume(static_path)
        moving = load_tiff_volume(moving_path)

        static_resized = resize_volume(static, target_shape)
        moving_resized = resize_volume(moving, target_shape)

        train_static.append(static_resized)
        train_moving.append(moving_resized)

    # Process testing data
    print("Testing data processing started ")
    for sample in test_samples:
        static_path = os.path.join(data_dir, sample, 'CAD_data.tiff')
        moving_path = os.path.join(data_dir, sample, 'CT_data.tiff')

        static = load_tiff_volume(static_path)
        moving = load_tiff_volume(moving_path)

        static_resized = resize_volume(static, target_shape)
        moving_resized = resize_volume(moving, target_shape)

        test_static.append(static_resized)
        test_moving.append(moving_resized)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save to HDF5
    print("Saving HDF5 ")
    save_to_hdf5(train_static, train_moving, train_samples, os.path.join(save_dir, 'train_data.h5'))
    save_to_hdf5(test_static, test_moving, test_samples, os.path.join(save_dir, 'test_data.h5'))
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time taken to complete process: {elapsed_time:.2f} minutes")

# Directory where your data is stored
data_dir = r'F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\processed_data'

# Define which samples are for training and which are for testing
train_samples = ['sample1', 'sample2', 'sample4','sample5', 'sample6']
test_samples = ['sample3', 'sample7']

# Define target shape for resizing (update to desired dimensions)
target_shape = (208, 144, 208)  # Example target shape, adjust as needed

# Define directory to save the HDF5 files
save_dir = r'F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\data_for_DL_model'

# Process the data and save to HDF5
process_and_save_data(data_dir, train_samples, test_samples, target_shape, save_dir)
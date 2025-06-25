# imports
import os
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
import numpy as np
import skimage
import matplotlib.pyplot as plt

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    fig, ax = plt.subplots()
    ax.plot(hist.epoch, hist.history[loss_name], '.-')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_title('History of loss', fontsize=(12, 12))

def vxm_data_generator(x_data, y_data, batch_size=1):
    """
    Generator that takes in data of size [N, D, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, D, H, W, 1], fixed image [bs, D, H, W, 1]
    outputs: moved image [bs, D, H, W, 1], zero-gradient [bs, D, H, W, 3]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, D, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = y_data[idx1, ..., np.newaxis]
        fixed_images = x_data[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

# Load the data
data = np.load(r'/home/kchand/deform_reg_project/05012023_preregistrationforDL.npz')

# Access the stored arrays
static = data['CAD']
moving = data['CT']

print(f"Original static shape: {static.shape}")
print(f"Original moving shape: {moving.shape}")

# Check if the data is loaded correctly
assert static.size > 0, "Static array is empty"
assert moving.size > 0, "Moving array is empty"

# If the data is already 4D, don't add a new dimension
if static.ndim == 3:
    static = static[np.newaxis, ...]  # shape: (1, D, H, W)
if moving.ndim == 3:
    moving = moving[np.newaxis, ...]  # shape: (1, D, H, W)

print(f"Reshaped static shape: {static.shape}")
print(f"Reshaped moving shape: {moving.shape}")

# Downsample the data to reduce the volume size
def downsample_volume(volume, factor=2):
    return skimage.transform.resize(volume, 
                                    (volume.shape[0] // factor, 
                                     volume.shape[1] // factor, 
                                     volume.shape[2] // factor),
                                    anti_aliasing=True)

static_downsampled = downsample_volume(static[0], factor=2)
moving_downsampled = downsample_volume(moving[0], factor=2)

static_downsampled = static_downsampled[np.newaxis, ...]  # Add the batch dimension back
moving_downsampled = moving_downsampled[np.newaxis, ...]

print(f"Downsampled static shape: {static_downsampled.shape}")
print(f"Downsampled moving shape: {moving_downsampled.shape}")

# Using the same single sample for both training and testing
x_train = static_downsampled
y_train = moving_downsampled
x_test = static_downsampled
y_test = moving_downsampled

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32') / 255
y_test = y_test.astype('float32') / 255

print(f"Normalized x_train shape: {x_train.shape}")
print(f"Normalized y_train shape: {y_train.shape}")
print(f"Normalized x_test shape: {x_test.shape}")
print(f"Normalized y_test shape: {y_test.shape}")

# pad to a multiple of 32
def pad_to_multiple_of_32(arr):
    """
    Pads a 4D NumPy array with zeros to ensure its dimensions are multiples of 32.
    
    Parameters:
        arr (np.ndarray): The input 4D array.
        
    Returns:
        np.ndarray: The padded 4D array with dimensions that are multiples of 32.
    """
    def pad_dimension(size):
        return (32 - size % 32) % 32
    
    # Get the current shape of the array
    _, d, h, w = arr.shape
    
    # Calculate the required padding for each dimension
    pad_d = pad_dimension(d)
    pad_h = pad_dimension(h)
    pad_w = pad_dimension(w)
    
    # Define the padding for each dimension
    padding = ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w))
    
    # Pad the array
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=0)
    
    return padded_arr

x_train = pad_to_multiple_of_32(x_train)
x_test = pad_to_multiple_of_32(x_test)
y_train = pad_to_multiple_of_32(y_train)
y_test = pad_to_multiple_of_32(y_test)

print(f"Padded x_train shape: {x_train.shape}")
print(f"Padded y_train shape: {y_train.shape}")
print(f"Padded x_test shape: {x_test.shape}")
print(f"Padded y_test shape: {y_test.shape}")

# Check if we have any empty arrays after padding
assert x_train.shape[0] > 0, "x_train is empty after padding"
assert y_train.shape[0] > 0, "y_train is empty after padding"
assert x_test.shape[0] > 0, "x_test is empty after padding"
assert y_test.shape[0] > 0, "y_test is empty after padding"

vol_shape = x_train.shape[1:]
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

# update the VxmDense model to use 3D convolutions
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# generate batch data
train_generator = vxm_data_generator(x_train, y_train, batch_size=1)
in_sample, out_sample = next(train_generator)

print(f"in_sample shape: {in_sample[0].shape}, {in_sample[1].shape}")
print(f"out_sample shape: {out_sample[0].shape}, {out_sample[1].shape}")

hist = vxm_model.fit(train_generator, epochs=50, steps_per_epoch=5, verbose=2)

"""
# test
# create the validation data generator
val_generator = vxm_data_generator(x_test, y_test, batch_size=1)
val_input, _ = next(val_generator)

val_pred = vxm_model.predict(val_input)

# visualize registration
images = [img[0, :, :, 0] for img in val_input + val_pred] 
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);

# visualization
def plot_3d_slices(images, titles=None, cmaps=['gray'], do_colorbars=True):
    nb_vis = len(images)
    fig, axes = plt.subplots(1, nb_vis, figsize=(15, 5))
    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(np.sum(img, axis=0), cmap=cmaps[i % len(cmaps)])
        if titles:
            ax.set_title(titles[i])
        if do_colorbars:
            plt.colorbar(ax.imshow(np.sum(img, axis=0), cmap=cmaps[i % len(cmaps)]), ax=ax)
    plt.show()

images = [img[0, ..., 0] for img in val_input + val_pred]
titles = ['moving', 'fixed', 'moved', 'flow']
plot_3d_slices(images, titles=titles, cmaps=['gray'])

"""
# imports
import os, sys
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
import matplotlib
#matplotlib.use('Agg')
import numpy as np
#import matplotlib.pyplot as plt
import skimage

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    fig, ax = plt.subplots()
    ax.plot([1, 1])
    ax.plot(hist.epoch, hist.history[loss_name], '.-')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_title('History of loss', fontsize=(12, 12))

def vxm_data_generator(x_data, y_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = y_data[idx1, ..., np.newaxis]

        fixed_images = x_data[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

def pad_to_multiple_of_32(arr):
    """
    Pads a 3D NumPy array with zeros to ensure its dimensions are multiples of 32.
    
    Parameters:
        arr (np.ndarray): The input 3D array.
        
    Returns:
        np.ndarray: The padded 3D array with dimensions that are multiples of 32.
    """
    def pad_dimension(size):
        return (32 - size % 32) % 32
    
    # Get the current shape of the array
    d1, d2, d3 = arr.shape
    
    # Calculate the required padding for each dimension
    pad_d1 = pad_dimension(d1)
    pad_d2 = pad_dimension(d2)
    pad_d3 = pad_dimension(d3)
    
    # Define the padding for each dimension
    padding = ((0, pad_d1), (0, pad_d2), (0, pad_d3))
    
    # Pad the array
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=0)
    
    return padded_arr

# Load the data
data = np.load(r'/home/kchand/deform_reg_project/05012023_preregistrationforDL.npz')

# Access the stored arrays
static = data['CAD']
moving = data['CT']

# split data into train and test data
nb_train = int(moving.shape[0]*0.8)
x_train = static[:nb_train,:,:]
y_train = moving[:nb_train,:,:]
x_test = static[nb_train:,:,:]
y_test = moving[nb_train:,:,:]


#visualize
nb_vis = 5

# choose nb_vis sample indexes
idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
example_digits = [f for f in x_train[idx, ...]]

# plot
ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);


#normalize data
x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255
y_train = y_train.astype('float')/255
y_test = y_test.astype('float')/255


#padding
#pad to a multiple of 32
pad_amount = ((0, 13), (45,40), (10,10))
x_train = np.pad(x_train, pad_amount, 'constant')
x_test = np.pad(x_test, pad_amount, 'constant')
y_test = np.pad(y_test, pad_amount, 'constant')
y_train = np.pad(y_train, pad_amount, 'constant')


vol_shape = x_train.shape[1:]
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]



#unet, disp field and sptail trans
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)



#generate batch data
train_generator = vxm_data_generator(x_train, y_train, batch_size=4)
in_sample, out_sample = next(train_generator)




hist = vxm_model.fit_generator(train_generator, epochs=500, steps_per_epoch=5, verbose=2);



#test
# create the validation data generator
val_generator = vxm_data_generator(x_test,y_test, batch_size = 1)
val_input, _ = next(val_generator)

val_pred = vxm_model.predict(val_input)

#visulaization
# visualize
images = [img[0, :, :, 0] for img in val_input + val_pred]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);

#Downsampling of displacement vector
downsample = 10
disp_vect = val_pred[1].squeeze()
disp_vect_1= disp_vect[:,:,0]
disp_vect_2= disp_vect[:,:,1]

a = skimage.measure.block_reduce(disp_vect_1,
                                 (downsample, downsample),
                                 np.mean)
b = skimage.measure.block_reduce(disp_vect_2,
                                 (downsample, downsample),
                                 np.mean)

ds_array = np.stack((a,b), axis=-1)
ds_array = np.reshape(ds_array, (1,ds_array.shape[0],ds_array.shape[1],ds_array.shape[2]))

ne.plot.flow(ds_array, width=5);
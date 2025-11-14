import tensorflow as tf
from voxelmorph.tf.layers import SpatialTransformer

from train_utils import split_by_index
import pyvista as pv
pv.set_jupyter_backend("trame")  # or "client"
import neurite as ne
import voxelmorph as vxm
MODEL_WEIGHTS  = "/home/kchand/results/finetune_two_stage_optuna/trial_0023/weights_stage2.h5"
TRAIN_HDF5     = "/home/kchand/input_data/all_samples_simple_structures_padded.h5"
MODEL_JSON     = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
# Explicit split by index
NUM_SAMPLES = 16
#TEST_IDX = [12, 13, 14, 15]
TEST_IDX = [14, 15]
VAL_IDX  = [6, 11]


train_h5, val_h5, test_h5 = split_by_index(TRAIN_HDF5, TEST_IDX, VAL_IDX, NUM_SAMPLES)


def compensate_mesh_with_sdf(cad_bin, phi, spacing=(0.015,0.015,0.015), k=1.0, iso=0.5):
    """
    Create compensation mesh from a binary CAD volume using SDF normals.
    Args
      cad_bin : (Z,Y,X) uint8/bool  -> 1 = solid
      phi     : (Z,Y,X,3) float32   -> displacement (XCT→CAD) in *voxel units*, defined on CAD grid
      spacing : (sz,sy,sx) mm/voxel
      k       : scalar gain (1.0 is a good start)
      iso     : isovalue for surface extraction
    Returns
      pv.PolyData (compensated mesh)
    """
    cad_bin = (cad_bin > 0).astype(np.uint8)
    sz, sy, sx = spacing

    # --- 1) Signed distance field (SDF) in mm (positive outside)
    din  = distance_transform_edt(cad_bin, sampling=spacing)          # inside dist (mm) to background
    dout = distance_transform_edt(1 - cad_bin, sampling=spacing)      # outside dist (mm) to solid
    sdf  = dout - din                                                 # +outside / -inside

    # SDF gradient (∇SDF gives outward normal direction). np.gradient with spacing -> derivatives per mm
    gz, gy, gx = np.gradient(sdf, sz, sy, sx)  # components in (Z,Y,X) order

    # --- 2) Extract surface mesh from volume in PyVista
    # PyVista expects X,Y,Z order; set a UniformGrid then contour at iso
    Z, Y, X = cad_bin.shape
    grid = pv.UniformGrid()
    grid.dimensions = (X, Y, Z)
    grid.spacing    = (sx, sy, sz)
    grid.origin     = (0.0, 0.0, 0.0)
    # attach scalars as POINT data (flatten in Fortran order)
    grid.point_data["cad"] = np.ascontiguousarray(
        cad_bin.transpose(2, 1, 0)
    ).ravel(order="F")

    iso = 0.5

    surf = grid.contour(isosurfaces=[iso], scalars="cad") # pv.PolyData with (x,y,z) vertices

    # --- 3) Sample SDF normal and φ at each vertex
    Vxyz = surf.points.copy()                                   # (N,3) in mm
    # convert vertex coords (x,y,z) -> voxel coords (z,y,x)
    Vvox = np.column_stack((Vxyz[:,2]/sz, Vxyz[:,1]/sy, Vxyz[:,0]/sx))

    # sample SDF gradient at vertex positions (trilinear)
    n_z = map_coordinates(gz, Vvox.T, order=1, mode="nearest")
    n_y = map_coordinates(gy, Vvox.T, order=1, mode="nearest")
    n_x = map_coordinates(gx, Vvox.T, order=1, mode="nearest")
    N = np.column_stack((n_x, n_y, n_z))       # convert to (x,y,z) component order

    # normalize normals
    N /= np.linalg.norm(N, axis=1, keepdims=True).clip(min=1e-8)

    # sample φ (voxel units) at vertices and convert to mm
    phi_x = map_coordinates(phi[...,2], Vvox.T, order=1, mode="nearest")  # phi[...,2] = dx
    phi_y = map_coordinates(phi[...,1], Vvox.T, order=1, mode="nearest")  # dy
    phi_z = map_coordinates(phi[...,0], Vvox.T, order=1, mode="nearest")  # dz
    PHI_vox = np.column_stack((phi_x, phi_y, phi_z))
    PHI_mm  = PHI_vox * np.array([sx, sy, sz])  # voxel units -> mm in (x,y,z)

    # --- 4) Move along normal by -k*(φ·n)
    phi_dot_n = np.sum(PHI_mm * N, axis=1, keepdims=True)  # mm
    disp = -k * phi_dot_n * N                              # (N,3) mm
    Vnew = Vxyz + disp

    # --- 5) Return compensated mesh
    comp = pv.PolyData(Vnew, surf.faces.copy())
    return comp


def show_overlaps(xct, cad, cad_comp, axis='z', slices=None, alpha=0.35,
                  cad_cmap='YlOrBr', xct_cmap='gray', comp_cmap='Reds',
                  vminmax='auto', titles=True):
    """
    Visualize overlaps:
      (1) XCT vs CAD
      (2) CAD_comp vs XCT

    Args:
      xct, cad, cad_comp : np.ndarray with shape (Z,Y,X) or (Z,Y,X,1)
      axis   : 'z' | 'y' | 'x'  -> slicing axis
      slices : list of slice indices; if None, picks 5 evenly spaced
      alpha  : overlay opacity for second image in each panel
      *_cmap : colormaps for base/overlay images
      vminmax: 'auto' or tuple (vmin, vmax) for intensity scaling
      titles : bool, add titles
    """
    xct = np.squeeze(xct).astype(np.float32)
    cad = np.squeeze(cad).astype(np.float32)
    cad_comp = np.squeeze(cad_comp).astype(np.float32)
    assert xct.shape == cad.shape == cad_comp.shape, "All volumes must have same shape"

    Z, Y, X = xct.shape
    ax_idx = {'z':0, 'y':1, 'x':2}[axis]
    n_ax = [Z, Y, X][ax_idx]
    if slices is None:
        if n_ax < 5:
            slices = list(range(n_ax))
        else:
            slices = np.linspace(int(0.1*n_ax), int(0.9*n_ax), 5).astype(int).tolist()

    # intensity window
    if vminmax == 'auto':
        # robust window from XCT (works for both CT-like and binary)
        vmin, vmax = np.percentile(xct, [1, 99])
    else:
        vmin, vmax = vminmax

    # figure: 2 rows (two comparisons) x N slices
    ncols = len(slices)
    fig, axes = plt.subplots(2, ncols, figsize=(3.8*ncols, 7), squeeze=False)

    def take_slice(vol, s):
        if axis == 'z':   return vol[s, :, :]
        if axis == 'y':   return vol[:, s, :]
        if axis == 'x':   return vol[:, :, s]

    for c, s in enumerate(slices):
        # Row 1: XCT vs CAD (nominal)
        a = axes[0, c]
        a.imshow(take_slice(xct, s), cmap=xct_cmap, vmin=vmin, vmax=vmax, origin='lower')
        a.imshow(take_slice(cad, s), cmap=cad_cmap, alpha=alpha, vmin=vmin, vmax=vmax, origin='lower')
        a.set_xticks([]); a.set_yticks([])
        if titles:
            a.set_title(f"{axis.upper()}={s}  |  XCT vs CAD")

        # Row 2: CAD_comp vs XCT
        b = axes[1, c]
        b.imshow(take_slice(xct, s), cmap=xct_cmap, vmin=vmin, vmax=vmax, origin='lower')
        b.imshow(take_slice(cad_comp, s), cmap=comp_cmap, alpha=alpha, vmin=vmin, vmax=vmax, origin='lower')
        b.set_xticks([]); b.set_yticks([])
        if titles:
            b.set_title(f"{axis.upper()}={s}  |  CAD_comp vs XCT")

    if titles:
        fig.suptitle("Overlaps: (top) XCT vs CAD   •   (bottom) CAD_comp vs XCT", y=0.98)
    plt.tight_layout()
    plt.show()


def fit_warp_to_svf(warp,
                    nb_steps=5,
                    iters=100,
                    min_delta=1e-5,
                    lr=0.1,
                    init='warp',
                    verbose=True):
    """
    Experimental: Get an SVF to a warp

    Args:
        warp (np array): warp of size [*vol_shape, ndims]
        nb_steps (int, optional): Number of integral scaling and squaring steps
        iters (int, optional): number of iterations to fit
        min_delta ([type], optional): minimum delta for convergence
        lr (float, optional): learning rate (step size). Defaults to 0.1.
        init (str, optional): initialization strategy, 'warp' of 'jecwt'
        verbose (bool, optional): Defaults to True.

    Returns:
        [type]: [description]

    Author: adalca
    """

    import tensorflow as tf
    import tensorflow.keras.layers as KL

    inp = KL.Input((1,))
    vel_layer = ne.layers.LocalParamWithInput(shape=warp.shape)
    vel_tensor = vel_layer(inp)
    disp_tensor = vxm.layers.VecInt(int_steps=nb_steps)(vel_tensor)
    model = tf.keras.Model(inp, disp_tensor)

    # initialize the velocity field
    if init == 'warp':  # with a warp
        vel_layer.set_weights([warp])
    else:  # with a jacobian trick
        jac = jacobian_determinant(warp)
        jac3 = np.stack([jac] * warp.shape[-1], -1)
        init_wts = 1.02 * warp + (1 - jac3) * 0.05
        vel_layer.set_weights([init_wts])

    # compile and run model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse')
    callback = tf.keras.callbacks.EarlyStopping('loss', min_delta=min_delta)
    zero = np.zeros((1,))
    warpk = warp[np.newaxis, ...]
    hist = model.fit(zero, warpk,
                     epochs=iters,
                     verbose=0,
                     callbacks=[callback])

    return vel_layer.get_weights()[0]


def invert_warp_via_velocity(warp,
                             nb_steps=5,
                             iters=100,
                             **kwargs):
    """
    Experimental invert_warp algorithm:
    Fit a velocity field to displacement field, then negate and integrate.

    This starts being more useful than scipy griddata when you have large volumes.
    This might change if tf implements a griddata, but owuld likey not be easily    differentiable?

    Searched quite a bit on how we could do this more principled (quicker approx to vel from disp):
    - Discussion: https://itk.org/pipermail/insight-users/2010-September/037977.html
    - maybe use tf.linalg.expm and tf.linalg.logm (matrix logarim) in an effort to have a
        scale-and-square equivalent of "logarithmic map", but couldn't fully understand
        how possible this is
    - https://en.wikipedia.org/wiki/Derivative_of_the_exponential_map

    TODO:
    - think of implementing this in a layer, e.g. https://github.com/cvxgrp/cvxpylayers

    Args:
        warp (np array): warp of size [*vol_shape, ndims]
        nb_steps (int, optional): Number of integral scaling and squaring steps
        iters (int, optional): number of iterations to fit
        kwargs for warp_to_svf_fit

    Returns:
        approximate inverse warp of same size as warp

    Author: adalca
    """

    if iters > 0:
        # do it via a model.
        # TODO: Should change this to a normal optimization loop
        vel = fit_warp_to_svf(warp,
                              nb_steps=nb_steps,
                              iters=iters,
                              **kwargs)

        vel = vel
    else:
        vel = warp

    # integrate the negative velocity to get inverse
    vel_tensor = tf.convert_to_tensor(vel, tf.float32)
    return vxm.utils.integrate_vec(-vel_tensor, nb_steps=nb_steps).numpy()

# Load model architecture
with open(MODEL_JSON , "r") as json_file:
    model_json = json_file.read()
vxm_model = tf.keras.models.model_from_json(model_json, 
                custom_objects={'SpatialTransformer': vxm.layers.SpatialTransformer, 
                'VxmDense': vxm.networks.VxmDense})

# Load model weights
vxm_model.load_weights(MODEL_WEIGHTS)


# Initialize the test generator
test_generator = test_data_generator(vxm_model, test_h5, patch_size=(128, 128, 128), stride=(64, 64, 64))
# Get the output for just one sample
reconstructed_moved, reconstructed_displacement, fixed_image, moving_image = next(test_generator)
# Plot the images
plot_images(fixed_image, moving_image, reconstructed_moved)
plot_3x3_images(fixed_image, moving_image, reconstructed_moved)



inv_warp_v = invert_warp_via_velocity(reconstructed_displacement)
# numpy arrays:
# cad_vol    : (Z, Y, X)         e.g. your fixed CAD volume
# inv_warp_v : (Z, Y, X, 3)      your inverted displacement field

# add batch + channel dims, cast to float32
cad_tf = tf.convert_to_tensor(fixed_image[None, ..., None], dtype=tf.float32)   # (1, Z, Y, X, 1)
flow_tf = tf.convert_to_tensor(inv_warp_v[None, ...],    dtype=tf.float32)  # (1, Z, Y, X, 3)

# create transformer (no size needed in TF version)
st = SpatialTransformer(interp_method='nearest', fill_value=0.0)  # 'nearest' also ok

# apply inverse field to CAD
cad_comp_tf = st([cad_tf, flow_tf])        # (1, Z, Y, X, 1)
cad_comp = cad_comp_tf.numpy()[0, ..., 0]  # back to (Z, Y, X)

show_overlaps(moving_image, fixed_image, cad_comp, axis='z')


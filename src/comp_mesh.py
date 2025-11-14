
import numpy as np
import pyvista as pv
from skimage import measure
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import neurite as ne

def overlay_mesh_slice_three_5(mesh1, mesh2, mesh3,
                               axis="z",
                               slice_value=None,
                               thickness_ratio=0.01,
                               labels=("mesh1", "mesh2", "mesh3"),
                               colors=("C0", "C1", "C2")):
    """
    Overlay 5 consecutive 2D slices of three meshes for visual comparison.

    mesh1, mesh2, mesh3 : pyvista.PolyData (same coordinate system)
    axis         : 'x', 'y', or 'z'  (direction of slicing)
    slice_value  : coordinate at which the middle slice is taken (None → middle)
    thickness_ratio : fractional thickness of each slab (e.g. 0.01 = 1%)
    labels       : legend labels for the three meshes
    colors       : matplotlib colors for the three meshes
    """

    pts1 = mesh1.points
    pts2 = mesh2.points
    pts3 = mesh3.points

    # Assuming pts = (z, y, x)
    axis_to_idx = {"z": 0, "y": 1, "x": 2}
    if axis not in axis_to_idx:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    idx = axis_to_idx[axis]

    # Choose center slice position (middle of that axis if not given)
    if slice_value is None:
        slice_value = 0.5 * (pts1[:, idx].min() + pts1[:, idx].max())

    axis_min = pts1[:, idx].min()
    axis_max = pts1[:, idx].max()
    thickness = (axis_max - axis_min) * thickness_ratio

    # Step between consecutive slices (here: 2 * thickness so slabs just touch)
    slice_step = 2.0 * thickness

    # Prepare 5 slice positions: center ± 2, ±1, 0
    slice_offsets = [-2, -1, 0, 1, 2]
    slice_positions = [slice_value + o * slice_step for o in slice_offsets]

    # No shared x/y to avoid aspect issues
    fig, axes = plt.subplots(1, 5, figsize=(5 * 3, 3))

    # placeholders; will be set in branches
    xlabel = ""
    ylabel = ""

    for ax, sv in zip(axes, slice_positions):
        mask1 = np.abs(pts1[:, idx] - sv) < thickness
        mask2 = np.abs(pts2[:, idx] - sv) < thickness
        mask3 = np.abs(pts3[:, idx] - sv) < thickness

        pts1_slice = pts1[mask1]
        pts2_slice = pts2[mask2]
        pts3_slice = pts3[mask3]

        if pts1_slice.size == 0 or pts2_slice.size == 0 or pts3_slice.size == 0:
            ax.set_title(f"{axis}={sv:.2f}\n(no points)")
            ax.axis("off")
            continue

        # Pick 2D coordinates to plot (still assuming (z,y,x))
        if axis == "z":
            x1, y1 = pts1_slice[:, 2], pts1_slice[:, 1]  # x vs y
            x2, y2 = pts2_slice[:, 2], pts2_slice[:, 1]
            x3, y3 = pts3_slice[:, 2], pts3_slice[:, 1]
            xlabel, ylabel = "X (voxel)", "Y (voxel)"
        elif axis == "y":
            x1, y1 = pts1_slice[:, 2], pts1_slice[:, 0]  # x vs z
            x2, y2 = pts2_slice[:, 2], pts2_slice[:, 0]
            x3, y3 = pts3_slice[:, 2], pts3_slice[:, 0]
            xlabel, ylabel = "X (voxel)", "Z (voxel)"
        else:  # axis == "x"
            x1, y1 = pts1_slice[:, 1], pts1_slice[:, 0]  # y vs z
            x2, y2 = pts2_slice[:, 1], pts2_slice[:, 0]
            x3, y3 = pts3_slice[:, 1], pts3_slice[:, 0]
            xlabel, ylabel = "Y (voxel)", "Z (voxel)"

        ax.scatter(x1, y1, s=1, alpha=0.4, label=labels[0], color=colors[0])
        ax.scatter(x2, y2, s=1, alpha=0.4, label=labels[1], color=colors[1])
        ax.scatter(x3, y3, s=1, alpha=0.4, label=labels[2], color=colors[2])

        ax.set_title(f"{axis}={sv:.2f}")
        ax.set_aspect("equal", adjustable="box")

    # Axes labels (only first to reduce clutter)
    axes[0].set_ylabel(ylabel)
    for ax in axes:
        ax.set_xlabel(xlabel)

    # Global legend
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right", markerscale=4)

    plt.tight_layout()
    plt.show()

def overlay_mesh_slice_three(mesh1, mesh2, mesh3,
                             axis="z",
                             slice_value=None,
                             thickness_ratio=0.01,
                             labels=("mesh1", "mesh2", "mesh3"),
                             colors=("C0", "C1", "C2")):
    """
    Overlay a 2D slice of three meshes for visual comparison.

    mesh1, mesh2, mesh3 : pyvista.PolyData (same coordinate system)
    axis         : 'x', 'y', or 'z'  (direction of slicing)
    slice_value  : coordinate at which to slice (None → middle of axis)
    thickness_ratio : fractional thickness of the slab (e.g. 0.01 = 1%)
    labels       : legend labels for the three meshes
    colors       : matplotlib colors for the three meshes
    """

    pts1 = mesh1.points
    pts2 = mesh2.points
    pts3 = mesh3.points

    # Assuming pts = (z, y, x)
    axis_to_idx = {"z": 0, "y": 1, "x": 2}
    if axis not in axis_to_idx:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    idx = axis_to_idx[axis]

    # Choose slice position (middle of that axis if not given)
    if slice_value is None:
        slice_value = 0.5 * (pts1[:, idx].min() + pts1[:, idx].max())

    thickness = (pts1[:, idx].max() - pts1[:, idx].min()) * thickness_ratio

    mask1 = np.abs(pts1[:, idx] - slice_value) < thickness
    mask2 = np.abs(pts2[:, idx] - slice_value) < thickness
    mask3 = np.abs(pts3[:, idx] - slice_value) < thickness

    pts1_slice = pts1[mask1]
    pts2_slice = pts2[mask2]
    pts3_slice = pts3[mask3]

    if pts1_slice.size == 0 or pts2_slice.size == 0 or pts3_slice.size == 0:
        print("No points in this slice for at least one mesh. Try increasing thickness_ratio.")
        return

    # Pick 2D coordinates to plot (still assuming (z,y,x))
    if axis == "z":
        x1, y1 = pts1_slice[:, 2], pts1_slice[:, 1]  # x vs y
        x2, y2 = pts2_slice[:, 2], pts2_slice[:, 1]
        x3, y3 = pts3_slice[:, 2], pts3_slice[:, 1]
        xlabel, ylabel = "X (voxel)", "Y (voxel)"
    elif axis == "y":
        x1, y1 = pts1_slice[:, 2], pts1_slice[:, 0]  # x vs z
        x2, y2 = pts2_slice[:, 2], pts2_slice[:, 0]
        x3, y3 = pts3_slice[:, 2], pts3_slice[:, 0]
        xlabel, ylabel = "X (voxel)", "Z (voxel)"
    else:  # axis == "x"
        x1, y1 = pts1_slice[:, 1], pts1_slice[:, 0]  # y vs z
        x2, y2 = pts2_slice[:, 1], pts2_slice[:, 0]
        x3, y3 = pts3_slice[:, 1], pts3_slice[:, 0]
        xlabel, ylabel = "Y (voxel)", "Z (voxel)"

    plt.figure(figsize=(6, 6))
    plt.scatter(x1, y1, s=1, alpha=0.4, label=labels[0], color=colors[0])
    plt.scatter(x2, y2, s=1, alpha=0.4, label=labels[1], color=colors[1])
    plt.scatter(x3, y3, s=1, alpha=0.4, label=labels[2], color=colors[2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.title(f"Overlay slice at {axis}={slice_value:.2f}")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.show()


def sample_phi_at_points(phi, points):
    """
    Sample 3D displacement field phi at arbitrary points.

    phi    : (Z, Y, X, 3) displacement field (in voxel units)
    points : (N, 3) coordinates in voxel index space
             (here we assume points[:,0]=z, points[:,1]=y, points[:,2]=x,
              which is what skimage.marching_cubes returns).

    Returns:
        disp : (N, 3) displacement vectors at those points
    """
    z = points[:, 0]
    y = points[:, 1]
    x = points[:, 2]

    disp = np.zeros_like(points, dtype=np.float32)
    for c in range(3):
        disp[:, c] = map_coordinates(
            phi[..., c],        # 3D component
            [z, y, x],          # index into (Z, Y, X)
            order=1,
            mode="nearest"
        )
    return disp

def compensation_warp_mesh_with_phi(mesh, phi, k=1.0):
    """
    Warp a mesh using displacement field phi with factor k.

    mesh : pyvista.PolyData, points in (z, y, x) voxel coords
    phi  : (Z, Y, X, 3) displacement field
    k    : scale factor (1.0 = original deformation)
    """
    verts = mesh.points.copy()           # (N,3): (z,y,x)
    disp  = sample_phi_at_points(phi, verts)   # (N,3)

    # Apply backward warp: x' = x + k * phi(x)
    verts_warped = verts - k * disp

    mesh_warped = pv.PolyData(verts_warped, mesh.faces)
    return mesh_warped

def forward_warp_xct_cad(mesh, phi, k=1.0):
    """
    Warp a mesh using displacement field phi with factor k.

    mesh : pyvista.PolyData, points in (z, y, x) voxel coords
    phi  : (Z, Y, X, 3) displacement field
    k    : scale factor (1.0 = original deformation)
    """
    verts = mesh.points.copy()           # (N,3): (z,y,x)
    disp  = sample_phi_at_points(phi, verts)   # (N,3)

    # Apply backward warp: x' = x + k * phi(x)
    verts_warped = verts + k * disp

    mesh_warped = pv.PolyData(verts_warped, mesh.faces)
    return mesh_warped


def overlay_slices(mesh1, mesh2, axis="z", outfile="overlay.png"):
    pts1 = mesh1.points
    pts2 = mesh2.points

    slice0 = 0.5 * (pts1[:,2].min() + pts1[:,2].max())
    thickness = (pts1[:,2].max() - pts1[:,2].min()) * 0.01

    mask1 = np.abs(pts1[:,2] - slice0) < thickness
    mask2 = np.abs(pts2[:,2] - slice0) < thickness

    plt.figure(figsize=(6,6))
    plt.scatter(pts1[mask1,0], pts1[mask1,1], s=1, alpha=0.3, label="orig")
    plt.scatter(pts2[mask2,0], pts2[mask2,1], s=1, alpha=0.3, label="warped")
    plt.legend()
    plt.axis("equal")
    plt.imshow()

def plot_mesh_slice(mesh, axis="z", slice_value=None, thickness_ratio=0.01):
    """
    Display a 2D plot of mesh vertices near a slice, without saving.

    mesh: pyvista.PolyData
    axis: 'x', 'y', or 'z'
    slice_value: value along axis (if None → middle of mesh)
    thickness_ratio: thickness of slice relative to mesh dimension
    """

    pts = mesh.points    # shape (N, 3)

    # Map axis name to coordinate index
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    idx = axis_to_idx[axis]

    # Compute default slice location (middle of axis)
    if slice_value is None:
        slice_value = 0.5 * (pts[:, idx].min() + pts[:, idx].max())

    # Compute absolute thickness
    thickness = (pts[:, idx].max() - pts[:, idx].min()) * thickness_ratio

    # Select vertices within this slice/slab
    mask = np.abs(pts[:, idx] - slice_value) < thickness
    pts_slice = pts[mask]

    if pts_slice.size == 0:
        print("No points found in this slice. Increase thickness_ratio.")
        return

    # Pick which coordinates to plot (2D)
    if axis == "z":
        xvals, yvals = pts_slice[:, 0], pts_slice[:, 1]   # X-Y plane
        xlabel, ylabel = "X", "Y"
    elif axis == "y":
        xvals, yvals = pts_slice[:, 0], pts_slice[:, 2]   # X-Z plane
        xlabel, ylabel = "X", "Z"
    elif axis == "x":
        xvals, yvals = pts_slice[:, 1], pts_slice[:, 2]   # Y-Z plane
        xlabel, ylabel = "Y", "Z"

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(xvals, yvals, s=1, alpha=0.3)
    plt.title(f"Mesh slice at {axis}={slice_value:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.show()


def xct_to_mesh(xct_vol, iso=0.5):
    """
    xct_vol: 3D numpy array (Z, Y, X)
    iso: isosurface level for marching cubes
    returns: PyVista PolyData
    """
    verts, faces, normals, _ = measure.marching_cubes(xct_vol, level=iso)
    # skimage: faces is (N, 3). PyVista wants (4, i0, i1, i2).
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    mesh = pv.PolyData(verts, faces)
    return mesh

def load_displacement_field(vtk_file_path):
    """Load displacement field from a VTK file and reshape it into (Dx, Dy, Dz, 3)."""
    grid = pv.read(vtk_file_path)
    if "displacement" not in grid.point_data:
        raise ValueError("Displacement field not found in the VTK file!")

    displacement = grid["displacement"]
    dims = grid.dimensions  # Get the grid dimensions

    # Reshape to (Dx, Dy, Dz, 3)
    displacement_field = displacement.reshape((dims[0], dims[1], dims[2], 3), order="F")

    displacement_field = np.array(displacement_field)
    return displacement_field

    
def load_vtk_as_image(filename):
    """
    Load a VTK file and extract the image data as a NumPy array.

    Parameters:
        filename (str): Path to the VTK file.

    Returns:
        np.ndarray: The loaded image in its original shape.
    """
    # Load the VTK file
    grid = pv.read(filename)

    # Extract the image data
    image_data = np.array(grid["image"])

    # Determine the original shape
    x_dim, y_dim, z_dim = grid.dimensions  # Get grid dimensions

    # Reshape the data back to its original 3D shape
    image_array = image_data.reshape((x_dim, y_dim, z_dim), order="F")  # Fortran order

    return image_array


disp_vtk_file_path = r"/home/kchand/results/test_results_simple_structure/sample_15_trial23/disp_field.vtk"
ct_file_path = r"/home/kchand/results/test_results_simple_structure/sample_15_trial23/moving_image.vtk"  # XCT (Moving)
cad_file_path = r"/home/kchand/results/test_results_simple_structure/sample_15_trial23/fixed_image.vtk"  # CAD (Fixed)
moved_ct_file_path = r"/home/kchand/results/test_results_simple_structure/sample_15_trial23/moved_image.vtk"  # XCT (Moved)
# Load displacement field
reconstructed_displacement = load_displacement_field(disp_vtk_file_path)
fixed_image = load_vtk_as_image(cad_file_path)
moving_image = load_vtk_as_image(ct_file_path)
reconstructed_moved = load_vtk_as_image(moved_ct_file_path)

iso_xct = filters.threshold_otsu(moving_image)
iso_cad = filters.threshold_otsu(fixed_image)

xct_mesh = xct_to_mesh(moving_image, iso=iso_xct)
cad_mesh = xct_to_mesh(fixed_image, iso = iso_cad)

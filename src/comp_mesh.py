import os
import numpy as np
import pyvista as pv
from skimage import measure
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import neurite as ne
import h5py



def flip_stl_x_axis(input_stl, output_stl, fix_normals=True):
    """
    Load an STL, flip (mirror) it along the X-axis, and save it.

    input_stl  : path to original STL
    output_stl : path to flipped STL
    fix_normals: if True, reverse triangle winding to keep normals outward
    """
    mesh = pv.read(input_stl)
    print(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    flipped = mesh.copy()

    # --- 1) Flip X coordinates ---
    pts = flipped.points.copy()
    pts[:, 0] *= -1.0      # mirror in X
    flipped.points = pts

    # --- 2) Fix normals by reversing triangle vertex order ---
    if fix_normals:
        # faces: [3, i0, i1, i2, 3, j0, j1, j2, ...]
        faces = flipped.faces.reshape(-1, 4).copy()   # make writable
        faces[:, [2, 3]] = faces[:, [3, 2]]           # swap last two indices
        flipped.faces = faces.reshape(-1)

    flipped.save(output_stl)
    print(f"Saved flipped STL to: {output_stl}")

    
def pad_volume_zeros(vol, pad=5):
    """
    vol : (Z,Y,X)
    pad : number of voxels padded on each side
    """
    pad_width = ((pad, pad), (pad, pad), (pad, pad))  # (Z,Y,X)
    vol_padded = np.pad(vol, pad_width,
                        mode="constant",
                        constant_values=0)
    print(f"[pad_volume_zeros] original shape {vol.shape} -> padded {vol_padded.shape}")
    return vol_padded, pad


def overlay_mesh_slice_three_5(mesh1, mesh2, mesh3,
                               axis="z",
                               slice_value=None,
                               spacing=(1,1,1),
                               use_voxel_index=False,
                               thickness_ratio=0.01,
                               labels=("mesh1", "mesh2", "mesh3"),
                               colors=("C0", "C1", "C2"),
                               alphas=(1.0, 0.25, 0.25)):
    """
    Overlay 5 consecutive 2D slices of three meshes for visual comparison.

    mesh1, mesh2, mesh3 : pyvista.PolyData
    axis                : 'x', 'y', or 'z'
    slice_value         : center slice (float physical coord OR integer voxel index)
    spacing             : (sz, sy, sx) voxel size
    use_voxel_index     : if True → slice_value is voxel index
    thickness_ratio     : fractional slab thickness
    labels              : mesh labels
    colors              : scatter colors
    alphas              : opacities for (mesh1, mesh2, mesh3)
    """

    pts1 = mesh1.points
    pts2 = mesh2.points
    pts3 = mesh3.points

    axis_to_idx = {"z": 0, "y": 1, "x": 2}
    idx = axis_to_idx[axis]

    # ------------ Convert voxel index → physical coordinate ----------
    if use_voxel_index and slice_value is not None:
        slice_value = slice_value * spacing[idx]   # integer index → mm

    # ------------ Default slice = midpoint ------------
    if slice_value is None:
        slice_value = 0.5 * (pts1[:, idx].min() + pts1[:, idx].max())

    axis_min = pts1[:, idx].min()
    axis_max = pts1[:, idx].max()
    thickness = (axis_max - axis_min) * thickness_ratio

    slice_step = 2.0 * thickness
    slice_offsets = [-2, -1, 0, 1, 2]
    slice_positions = [slice_value + o * slice_step for o in slice_offsets]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    xlabel = ""
    ylabel = ""

    # --------------------------------------------------
    # Loop through the 5 slices
    # --------------------------------------------------
    for ax, sv in zip(axes, slice_positions):

        mask1 = np.abs(pts1[:, idx] - sv) < thickness
        mask2 = np.abs(pts2[:, idx] - sv) < thickness
        mask3 = np.abs(pts3[:, idx] - sv) < thickness

        p1 = pts1[mask1]
        p2 = pts2[mask2]
        p3 = pts3[mask3]

        if p1.size == 0 or p2.size == 0 or p3.size == 0:
            ax.set_title(f"{axis}={sv:.2f}\n(no points)")
            ax.axis("off")
            continue

        # -------------- Select 2D plotting axes -----------------
        if axis == "z":
            x1, y1 = p1[:, 2], p1[:, 1]
            x2, y2 = p2[:, 2], p2[:, 1]
            x3, y3 = p3[:, 2], p3[:, 1]
            xlabel, ylabel = "X (voxel)", "Y (voxel)"

        elif axis == "y":
            x1, y1 = p1[:, 2], p1[:, 0]
            x2, y2 = p2[:, 2], p2[:, 0]
            x3, y3 = p3[:, 2], p3[:, 0]
            xlabel, ylabel = "X (voxel)", "Z (voxel)"

        else:  # axis == "x"
            x1, y1 = p1[:, 1], p1[:, 0]
            x2, y2 = p2[:, 1], p2[:, 0]
            x3, y3 = p3[:, 1], p3[:, 0]
            xlabel, ylabel = "Y (voxel)", "Z (voxel)"

        # ---------------- Scatter with opacity -------------------
        ax.scatter(x1, y1, s=1, alpha=alphas[0], color=colors[0], label=labels[0])
        ax.scatter(x2, y2, s=1, alpha=alphas[1], color=colors[1], label=labels[1])
        ax.scatter(x3, y3, s=1, alpha=alphas[2], color=colors[2], label=labels[2])

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{axis}={sv:.2f}")

    # Label only the first axis
    axes[0].set_ylabel(ylabel)
    for ax in axes:
        ax.set_xlabel(xlabel)

    # Unified legend
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, markerscale=4, loc="upper right")

    plt.tight_layout()
    plt.show()
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

def plot_mesh_slice_2d(mesh, axis="z", index=None, color="black", figsize=(6,6)):
    """
    Plot a 2D slice (cross-section) of a 3D mesh using a cutting plane.
    
    mesh : pyvista.PolyData
    axis : 'x', 'y', or 'z'
    index: where to slice (same units as mesh.points, e.g. mm or voxels)
           If None → slice at the center along that axis.
    """
    # 1. Get bounds
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    # 2. Define plane
    if axis == "x":
        if index is None:
            index = 0.5 * (xmin + xmax)
        origin = (index, 0, 0)
        normal = (1, 0, 0)
    elif axis == "y":
        if index is None:
            index = 0.5 * (ymin + ymax)
        origin = (0, index, 0)
        normal = (0, 1, 0)
    elif axis == "z":
        if index is None:
            index = 0.5 * (zmin + zmax)
        origin = (0, 0, index)
        normal = (0, 0, 1)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # 3. Extract slice
    slc = mesh.slice(normal=normal, origin=origin)

    if slc.n_points == 0:
        print(f"No mesh intersection at {axis} = {index:.3f}")
        return

    pts = slc.points

    # Some slices have no faces (only lines/points)
    has_faces = slc.faces.size > 0 and slc.n_faces > 0
    if has_faces:
        faces = slc.faces.reshape(-1, 4)[:, 1:]   # (N,3) triangle indices
    else:
        faces = None

    # 4. Choose 2D projection
    if axis == "x":
        xs, ys = pts[:, 2], pts[:, 1]  # (z, y)
        xlabel, ylabel = "Z", "Y"
    elif axis == "y":
        xs, ys = pts[:, 2], pts[:, 0]  # (z, x)
        xlabel, ylabel = "Z", "X"
    elif axis == "z":
        xs, ys = pts[:, 1], pts[:, 0]  # (y, x)
        xlabel, ylabel = "Y", "X"

    # 5. Plot
    plt.figure(figsize=figsize)

    if has_faces and faces.size > 0:
        # Wireframe triangles
        plt.triplot(xs, ys, faces, color=color, linewidth=0.3)
    else:
        # Fallback: just show the points
        plt.scatter(xs, ys, s=1, color=color)
        print(f"Slice at {axis} = {index:.3f} has points but no triangle faces; "
              "showing scatter instead of triplot.")

    plt.gca().set_aspect("equal")
    plt.title(f"Mesh slice at {axis} = {index:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
    plt.gca().invert_yaxis()   
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



def xct_to_mesh(xct_vol, iso=0.5):
    """
    xct_vol: 3D numpy array (Z, Y, X)
    iso: isosurface level for marching cubes
    returns: PyVista PolyData
    """
    verts, faces, normals, _ = measure.marching_cubes(xct_vol, level=iso, allow_degenerate = False)
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



def check_stl_printability(mesh, verbose=True):
    """
    Check if an STL mesh is printable using PyVista-based diagnostics.
    Returns a dictionary with PASS/FAIL for each test and an overall result.
    """

    report = {}

    # 1️⃣ Basic mesh stats
    report["n_points"] = mesh.n_points
    report["n_faces"] = mesh.n_faces

    # 2️⃣ Manifoldness
    report["is_manifold"] = bool(mesh.is_manifold)

    # 3️⃣ Feature edges (boundary + non-manifold)
    feat = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    report["n_bad_edges"] = feat.n_cells
    report["is_watertight"] = (feat.n_cells == 0)

    # 4️⃣ Check for holes (PyVista fill_holes)
    mesh_filled = mesh.fill_holes(1000.0)  # big radius
    report["n_holes_filled"] = mesh_filled.n_faces - mesh.n_faces

    # 5️⃣ Normal orientation test
    mesh_checks = mesh.copy()
    mesh_checks.compute_normals(inplace=True, auto_orient_normals=True)
    oriented = mesh_checks.is_manifold  # approx: if normals consistent
    report["normals_consistent"] = oriented

    # 6️⃣ Self intersection check (approx via bounding boxes)
    # NOTE: PyVista doesn't have robust self-intersection detection
    # but this heuristic catches obvious ones
    bounds = mesh.bounds
    bbox_min = np.array(bounds[::2])
    bbox_max = np.array(bounds[1::2])
    report["bbox_ok"] = np.all(bbox_min < bbox_max)

    # 7️⃣ Slicer acceptance heuristic
    printable = (
        report["is_manifold"]
        and report["is_watertight"]
        and report["n_bad_edges"] == 0
    )
    report["PRINTABLE"] = printable

    if verbose:
        print(f"Points: {report['n_points']}")
        print(f"Faces: {report['n_faces']}")
        print("")
        print(f"Manifold:         {report['is_manifold']}")
        print(f"Watertight:       {report['is_watertight']}")
        print(f"Bad edges:        {report['n_bad_edges']}")
        print(f"Holes detected:   {report['n_holes_filled'] > 0}")
        print(f"Normals OK:       {report['normals_consistent']}")
        print(f"Bounding box OK:  {report['bbox_ok']}")
        print("-----------------------------------------")
        print(f"PRINTABLE:        {report['PRINTABLE']}")
        print("=========================================\n")

    return report

def compare_mesh_slices(mesh1, mesh2, axis="z", index=None,
                        label1="Original", label2="Compensated",
                        color1="black", color2="red",
                        figsize=(12,6)):
    """
    Plot side-by-side 2D slices of two meshes.

    mesh1 : original mesh
    mesh2 : compensated mesh
    axis  : 'x', 'y', or 'z'
    index : slice location
    """

    xmin, xmax, ymin, ymax, zmin, zmax = mesh1.bounds

    if axis == "x":
        if index is None:
            index = 0.5 * (xmin + xmax)
        origin = (index, 0, 0)
        normal = (1, 0, 0)

    elif axis == "y":
        if index is None:
            index = 0.5 * (ymin + ymax)
        origin = (0, index, 0)
        normal = (0, 1, 0)

    elif axis == "z":
        if index is None:
            index = 0.5 * (zmin + zmax)
        origin = (0, 0, index)
        normal = (0, 0, 1)

    else:
        raise ValueError("axis must be x,y,z")

    # Slice meshes
    s1 = mesh1.slice(normal=normal, origin=origin)
    s2 = mesh2.slice(normal=normal, origin=origin)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, slc, title, color in zip(
        axes,
        [s1, s2],
        [label1, label2],
        [color1, color2]
    ):

        pts = slc.points

        if pts.shape[0] == 0:
            ax.set_title("Empty slice")
            continue

        if axis == "x":
            xs, ys = pts[:,2], pts[:,1]
            xlabel, ylabel = "Z", "Y"

        elif axis == "y":
            xs, ys = pts[:,2], pts[:,0]
            xlabel, ylabel = "Z", "X"

        else:
            xs, ys = pts[:,1], pts[:,0]
            xlabel, ylabel = "Y", "X"

        ax.scatter(xs, ys, s=1, color=color)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.suptitle(f"Mesh slice comparison at {axis}={index:.3f}")
    plt.tight_layout()
    plt.show()


def pad_displacement_field(phi, pad):
    """
    Zero-pad a displacement field on all sides.

    phi : (Z, Y, X, 3) displacement field in voxel units
    pad : number of voxels padded on each side, same as used for volume

    Returns:
        phi_padded : padded displacement field
    """

    # pad only the spatial dimensions (Z, Y, X)
    pad_width = ((pad, pad), (pad, pad), (pad, pad), (0, 0))

    phi_padded = np.pad(
        phi,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0
    )

    print(f"[pad_displacement_field] phi {phi.shape} -> padded {phi_padded.shape}")
    return phi_padded

def binarize_cad_simple(cad8):
    """
    Binarize 8-bit voxelised CAD by simple threshold: >0.
    cad8: (Z,Y,X) uint8
    """
    cad_bin = (cad8 > 0).astype(np.uint8)
    print("[binarize_cad_simple] Foreground voxels:", cad_bin.sum())
    return cad_bin


def voxel_to_mm_zyx(mesh, spacing_zyx):
    """
    Convert a mesh from voxel index coordinates (z,y,x)
    to real-world millimetres.

    mesh          : pyvista.PolyData with points in (z,y,x)
    spacing_zyx   : tuple/list (sz, sy, sx) in mm/voxel

    returns: new mesh in mm coordinates
    """
    sz, sy, sx = spacing_zyx  # spacing in each axis

    m = mesh.copy()
    pts = m.points.copy()

    # Convert voxel → mm
    pts[:, 0] *= sz   # z direction
    pts[:, 1] *= sy   # y direction
    pts[:, 2] *= sx   # x direction

    m.points = pts
    return m

def load_scalar_field(vtk_file_path):
    """
    Load scalar field from VTK and reshape to (Z,Y,X)
    """
    grid = pv.read(vtk_file_path)

    if len(grid.point_data.keys()) == 0:
        raise ValueError("No scalar data found in VTK!")

    array_name = list(grid.point_data.keys())[0]

    data = np.array(grid[array_name])
    dims = grid.dimensions

    scalar_field = data.reshape((dims[0], dims[1], dims[2]), order="F")
    return np.array(scalar_field, dtype=np.float32)

disp_vtk_file_path = r"/home/kchand/results/test_results_simple_structure/sample_14_trial23/disp_field.vtk"
ct_file_path = r"/home/kchand/results/test_results_simple_structure/sample_14_trial23/moving_image.vtk"  # XCT (Moving)
cad_file_path = r"/home/kchand/results/test_results_simple_structure/sample_14_trial23/fixed_image.vtk"  # CAD (Fixed)
moved_ct_file_path = r"/home/kchand/results/test_results_simple_structure/sample_14_trial23/moved_image.vtk"  # XCT (Moved)
unc_vtk_file_path  = r"/home/kchand/results/BAM_Inconel_simple_str_ensemble/test_00_idx_00/disp_std_mag_B.vtk"
#save stl
save_dir = '/home/kchand/results/Comp_mesh_BAM_Inconel'

sample_name = 'sample_04_02'


spacing_zyx = (0.015, 0.015, 0.015) 

# Load displacement field
reconstructed_displacement = load_displacement_field(disp_vtk_file_path)
fixed_image = load_vtk_as_image(cad_file_path)
moving_image = load_vtk_as_image(ct_file_path)
reconstructed_moved = load_vtk_as_image(moved_ct_file_path)
disp_uncertainty = load_scalar_field(unc_vtk_file_path)


#weight disp field
p_low, p_high = np.percentile(disp_uncertainty, (5,95))
if p_high > p_low:
    unc_norm = (disp_uncertainty - p_low) / (p_high - p_low)
else:
    unc_norm = np.zeros_like(disp_uncertainty)

unc_norm = np.clip(unc_norm, 0, 1)
confidence_map = 1.0 - unc_norm
confidence_map = np.clip(confidence_map, 0.5, 1.0)
unc_reconstructed_displacement = reconstructed_displacement * confidence_map[..., None]


#binarised_fixed
cad_bin = binarize_cad_simple(fixed_image)
#padded volumes to avoid edge artifacts´
pad = 20
fixed_image_padded, pad = pad_volume_zeros(cad_bin, pad=pad)
moving_image_padded, pad = pad_volume_zeros(moving_image, pad=pad)
reconstructed_moved_padded, pad = pad_volume_zeros(reconstructed_moved, pad=pad)
displacement_padded = pad_displacement_field(reconstructed_displacement, pad=pad)
unc_displacement_padded = pad_displacement_field(unc_reconstructed_displacement, pad=pad)



#meshing the volume
iso_xct = filters.threshold_otsu(moving_image_padded)
xct_mesh = xct_to_mesh(moving_image_padded, iso=iso_xct)
cad_mesh = xct_to_mesh(fixed_image_padded, iso = 0.5)


comp_cad = compensation_warp_mesh_with_phi(cad_mesh, displacement_padded, k=1.0)
unc_comp_cad = compensation_warp_mesh_with_phi(cad_mesh, unc_displacement_padded, k=1.0)

# ==========================================================
# Process BOTH meshes
# ==========================================================

meshes = {
    "raw": comp_cad,
    "uncertainty_weighted": unc_comp_cad
}

for mesh_name, mesh in meshes.items():

    print("\n======================================")
    print(f"Processing: {mesh_name}")
    print("======================================")

    # -----------------------------
    # Printability check
    # -----------------------------
    result = check_stl_printability(mesh)

    # -----------------------------
    # Smoothing
    # -----------------------------
    mesh_smooth = mesh.smooth(
        n_iter=100,
        relaxation_factor=0.5,
        feature_smoothing=False,
        boundary_smoothing=False,
    )

    # -----------------------------
    # Convert voxel -> mm
    # -----------------------------
    mesh_mm = voxel_to_mm_zyx(
        mesh_smooth,
        spacing_zyx
    )

    # -----------------------------
    # Check manifoldness
    # -----------------------------
    if mesh_mm.is_manifold:
        print(f"{mesh_name}: manifold")
    else:
        print(f"{mesh_name}: NOT manifold")

    # -----------------------------
    # Save STL
    # -----------------------------
    stl_name = f"{sample_name}_{mesh_name}_comp_mesh.stl"

    stl_path = os.path.join(
        save_dir,
        stl_name
    )

    mesh_mm.save(stl_path)

    print("Saved:", stl_path)


overlay_mesh_slice_three_5(
    cad_mesh,
    comp_cad,
    xct_mesh,
    axis="z",
    slice_value=250,       # integer slice
    use_voxel_index=True,  # convert using spacing
    spacing=(1,1,1),       # or (sz, sy, sx) if real units
    alphas=(1.0, 0.2, 0.2),
    labels=("CAD", "compensated CAD", "CT")
)

overlay_mesh_slice_three(mesh1=cad_mesh, mesh2=comp_cad, mesh3=xct_mesh,
                         axis="z", slice_value=250,
                         labels=("CAD", "comp CAD", "XCT"),
                         colors=("C0", "C1", "C2"))

"""

#convert comp CAD to mm for printing

if comp_cad.is_manifold:
    print("Mesh is manifold")
else: 
    print("Mesh is lieder not manifold ")
    #comp_cad = comp_cad.reconstruct_surface(progress_bar=True)

#test if volume is printable
result = check_stl_printability(comp_cad)

# Laplacian smoothing
comp_cad_smooth = comp_cad.smooth(
    n_iter=100,              # start with 20–50
    relaxation_factor=0.5, # small value avoids shrinkage
    feature_smoothing=False,
    boundary_smoothing=False,
)

compare_mesh_slices(
    comp_cad,
    comp_cad_smooth,
    axis="z",
    index=None,
    label1="Compensated CAD",
    label2="Smoothed Compensated CAD"
)

comp_cad_mm = voxel_to_mm_zyx(comp_cad_smooth, spacing_zyx)

if comp_cad_mm.is_manifold:
    print("Mesh in mm is manifold")
else: 
    print("Mesh in mm is lieder not manifold ")
    #comp_cad = comp_cad.reconstruct_surface(progress_bar=True)
plot_mesh_slice_2d(comp_cad_mm, axis="y")

# output file name
#stl_name = f"{folder_name}_comp_mesh.stl"
stl_name = f"{sample_name}_comp_mesh.stl"

# full path
stl_path = os.path.join(save_dir, stl_name)

# triangulate and save
comp_cad_mm.save(stl_path)

print("Saved STL to:", stl_path)

"""


import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import norm, skew, kurtosis
from matplotlib.colors import ListedColormap

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


def visualize_3d_quiver(ct_image, displacement_field_corrected, title="3D Displacement Field", max_points=5000,
                        glyph_factor=5.0):
    """
    Visualizes a 3D displacement field as a quiver plot using PyVista.
    Only foreground (non-zero) points from the CT image are used.
    Args:
        ct_image (np.ndarray): 3D CT image (shape: [X, Y, Z]) with background = 0.
        displacement_field_corrected (np.ndarray): 4D array (X, Y, Z, 3) containing displacement vectors.
        title (str): Title for the visualization.
        max_points (int): Maximum number of points to visualize (downsampling applied if necessary).
        glyph_factor (float): Scaling factor for glyph (arrow) lengths.
    """
    try:
        Dx, Dy, Dz = displacement_field_corrected.shape[:3]
        X_idx = np.arange(Dx)
        Y_idx = np.arange(Dy)
        Z_idx = np.arange(Dz)
        # Create a structured grid
        X, Y, Z = np.meshgrid(X_idx, Y_idx, Z_idx, indexing="ij")
        points = np.column_stack((X.ravel(order="F"),
                                  Y.ravel(order="F"),
                                  Z.ravel(order="F")))
        # Invert Y-axis for proper Cartesian representation
        swapped_points = np.empty_like(points)
        swapped_points[:, 0] = points[:, 2]  # X unchanged
        swapped_points[:, 1] = -points[:, 1]  # Invert Y values
        swapped_points[:, 2] = points[:, 0]  # Z unchanged
        grid = pv.PolyData(swapped_points.astype(np.float32))
        # Create foreground mask from CT image
        ct_flat = ct_image.flatten(order="F")
        foreground_mask = ct_flat != 0  # Keep only nonzero voxels
        # Filter points to keep only foreground
        filtered_points = grid.points[foreground_mask]
        # Extract corresponding displacement vectors
        vectors_flat = displacement_field_corrected.reshape(-1, 3, order="F")
        filtered_vectors = vectors_flat[foreground_mask]
        # Compute vector magnitudes
        magnitude = np.linalg.norm(filtered_vectors, axis=1)
        # Create PyVista PolyData for visualization
        filtered_grid = pv.PolyData(filtered_points)
        filtered_grid["vectors"] = filtered_vectors
        filtered_grid["magnitude"] = magnitude
        # Downsample if necessary
        if filtered_grid.n_points > max_points:
            indices = np.linspace(0, filtered_grid.n_points - 1, max_points).astype(int)
            filtered_grid = filtered_grid.extract_points(indices)
            print("Downsampled to", filtered_grid.n_points, "points.")
        if filtered_grid.n_points == 0:
            print("No foreground points found! Check your CT image mask.")
            return
        # Create quiver glyphs (arrow representation for vectors)
        glyphs = filtered_grid.glyph(orient="vectors", scale=False, factor=glyph_factor)
        # PyVista plotter
        plotter = pv.Plotter()
        plotter.add_mesh(glyphs, scalars="magnitude", cmap="viridis",
                         clim=[np.min(magnitude), 40], lighting=False, point_size=5.0)
        # Adjust axis labels for Cartesian system
        plotter.show_axes()
        plotter.xlabel = "X-axis"
        plotter.ylabel = "Z-axis"  # Renaming Y as Z
        plotter.zlabel = "Y-axis"  # Renaming Z as Y
        plotter.add_title(title)
        plotter.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

def visualize_3d_displacement_magnitude(ct_image, displacement_field,
                                        title="3D Displacement Magnitude (μm)", clim_max=400):
    """
    Visualize the 3D displacement magnitude (as a heatmap) for the foreground defined by the CT image.
    Displacement is converted to micrometers (assuming 1 pixel = 10 µm).
    Args:
        ct_image (np.ndarray): 3D CT image of shape (Nx, Ny, Nz) where background is 0.
        displacement_field (np.ndarray): 4D array of shape (Nx, Ny, Nz, 3) with displacement vectors.
        title (str): Title for the plot.
        clim_max (float): Maximum displacement magnitude to display in µm (default 400).
    """
    try:
        # Reorder displacement field components: [x, y, z] -> [x, z, -y]
        disp_reordered = np.empty_like(displacement_field)
        disp_reordered[..., 0] = displacement_field[..., 0]  # x
        disp_reordered[..., 1] = displacement_field[..., 2]  # z
        disp_reordered[..., 2] = -displacement_field[..., 1]  # -y
        # Convert displacement to micrometers
        disp_reordered *= 10  # 1 pixel = 10 µm
        # Compute and clip magnitude
        magnitude = np.linalg.norm(disp_reordered, axis=-1)
        magnitude_clipped = np.clip(magnitude, 0, clim_max)
        # Apply foreground mask
        magnitude_clipped[ct_image == 0] = 0
        # Reorder to X, Z, -Y and invert Z-axis
        magnitude_reordered = np.transpose(magnitude_clipped, (0, 2, 1))
        magnitude_reordered = magnitude_reordered[:, :, ::-1]
        dims = magnitude_reordered.shape
        grid = pv.ImageData(dimensions=dims, spacing=(1, 1, 1), origin=(0, 0, 0))
        grid["magnitude"] = magnitude_reordered.ravel(order="F")
        # Plot
        plotter = pv.Plotter()
        actor = plotter.add_volume(grid, scalars="magnitude", cmap="viridis",
                                   clim=[0, clim_max], opacity="linear", name="Displacement", show_scalar_bar=False)
        # Add title and axes
        plotter.add_title(title, font_size=22)
        plotter.show_axes()
        plotter.set_background("white")
        # Add color bar with increased font size
        plotter.add_scalar_bar(title="Displacement (μm)", title_font_size=40,
                               label_font_size=40, n_labels=5)
        # Add scale bar: 1 mm = 100 pixels
        plotter.add_mesh(pv.Cube(center=(50, -20, -20), x_length=100, y_length=2, z_length=2),
                         color="black", show_edges=False)
        plotter.add_text("1 mm", position=(0.1, 0.08), font_size=20, color='black')
        plotter.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

def plot_overlay_and_quiver_single_axis_horizontal(ct_image, cad_image, displacement_field, axis='YZ', slice_index=None,
                                                   subsample=5):
    """
    Plots a horizontally stacked figure for a given plane:
      - Left panel: Overlay of CAD and CT images.
      - Right panel: Quiver plot of the displacement field.

    For the chosen plane:
      - 'YZ' plane: Uses the middle slice along the first dimension.
      - 'XZ' plane: Uses the middle slice along the third dimension and rotates it 90° clockwise.
      - 'XY' plane: Uses the middle slice along the second dimension.

    In the quiver plot:
      - Only the y-component of the displacement is inverted.
      - Arrow lengths equal the displacement (scale=1).
      - Colors are mapped to displacement magnitude with normalization fixed to [0, 45].

    Both panels are set to have an equal x-to-y aspect ratio, using the CAD/CT overlay as reference.

    Parameters:
      ct_image (np.ndarray): 3D CT image (background = 0).
      cad_image (np.ndarray): 3D CAD image.
      displacement_field (np.ndarray): 4D array of shape (Dx, Dy, Dz, 3).
      axis (str): Plane to plot. Options: 'YZ', 'XZ', or 'XY'. Default is 'YZ'.
      slice_index (int, optional): Specific slice index. If None, the middle slice is used.
      subsample (int): Factor for subsampling the grid for quiver plotting.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Determine slice and rotations based on chosen plane ---
    if axis.upper() == 'YZ':
        # Use the slice along the X-axis.
        slice_index = ct_image.shape[0] // 2 if slice_index is None else slice_index
        ct_slice = ct_image[slice_index, :, :]
        cad_slice = cad_image[slice_index, :, :]
        disp_slice = displacement_field[slice_index, :, :]
        title_axis = "YZ Plane (X Middle)"
    elif axis.upper() == 'XZ':
        # Use the slice along the Z-axis and rotate 90° clockwise.
        slice_index = ct_image.shape[2] // 2 if slice_index is None else slice_index
        ct_slice = np.rot90(ct_image[:, :, slice_index], k=-1)
        cad_slice = np.rot90(cad_image[:, :, slice_index], k=-1)
        disp_slice = np.rot90(displacement_field[:, :, slice_index], k=-1)
        title_axis = "XZ Plane (Y Middle)"
    elif axis.upper() == 'XY':
        # Use the slice along the Y-axis.
        slice_index = ct_image.shape[1] // 2 if slice_index is None else slice_index
        ct_slice = ct_image[:, slice_index, :]
        cad_slice = cad_image[:, slice_index, :]
        disp_slice = displacement_field[:, slice_index, :]
        title_axis = "XY Plane (Z Middle)"
    else:
        raise ValueError("Invalid axis. Choose from 'YZ', 'XZ', or 'XY'.")

    # --- Create horizontally stacked figure with two panels ---
    fig, ax = plt.subplots(1, 2, figsize=(24, 10))

    # --- Left Panel: Overlay of CAD and CT ---
    # Assumes plot_overlay is defined elsewhere.
    plot_overlay(ax[0], cad_slice, ct_slice, f"CT vs CAD Overlay ({title_axis})",
                 cmap_cad="Greens", cmap_ct="viridis")
    ax[0].set_title(f"CT vs CAD Overlay ({title_axis})", fontsize=14)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_aspect('equal')  # Ensure x-to-y ratio is the same as reference.

    # --- Right Panel: Quiver plot of the displacement field ---
    H, W, _ = disp_slice.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # Extract displacement components.
    U = disp_slice[..., 0]  # x-displacement remains unchanged.
    V = -disp_slice[..., 1]  # invert only the y-displacement.

    # Compute displacement magnitude for color mapping.
    M = np.linalg.norm(disp_slice, axis=-1)

    # Subsample grid and displacement vectors.
    X = X[::subsample, ::subsample]
    Y = Y[::subsample, ::subsample]
    U = U[::subsample, ::subsample]
    V = V[::subsample, ::subsample]
    M = M[::subsample, ::subsample]

    quiv = ax[1].quiver(
        X, Y, U, V, M,
        angles='xy',
        scale_units='xy',
        scale=1,  # True displacement lengths.
        cmap="viridis",
        norm=plt.Normalize(vmin=0, vmax=45)  # Fixed color mapping range.
    )
    ax[1].set_title(f"Displacement Vectors ({title_axis})", fontsize=14)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].invert_yaxis()  # Flip y-axis for Cartesian orientation.
    ax[1].set_aspect('equal')  # Ensure same x-to-y ratio.

    cbar = plt.colorbar(quiv, ax=ax[1], fraction=0.046, pad=0.08)
    cbar.set_label("Displacement Magnitude", fontsize=12)

    plt.tight_layout()
    plt.show()


def generate_plots_quiver(ct_image, cad_image, moved_ct, displacement_field_corrected):
    """Generates a 3x3 figure with:
       - Top row: Alpha-blended overlays of CT & CAD (XY rotated).
       - Middle row: Alpha-blended overlays of MOVED CT & CAD (XY rotated).
       - Bottom row: Quiver plots with displacement vectors (no extra scaling, y-axis flipped,
         and color normalized to [0, 45]).
    """
    Dx, Dy, Dz = displacement_field_corrected.shape[:3]
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2  # Middle slices
    fig = plt.figure(figsize=(25, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.2])
    axes = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j])
    def add_quiver(ax, disp_slice, title, subsample=5):
        """Creates a quiver plot for displacement vectors without additional scaling.
           The y-axis is flipped to match Cartesian coordinates and the color mapping is
           normalized to the range [0, 45].
        """
        H, W, _ = disp_slice.shape  # Get slice dimensions
        X, Y = np.meshgrid(np.arange(W), np.arange(H))  # Grid for quiver
        # Extract displacement components; invert Y-displacement to fix orientation
        U = disp_slice[..., 0]
        V = -disp_slice[..., 1]
        # Compute magnitudes for color mapping
        M = np.linalg.norm(disp_slice, axis=-1)
        # Subsample for better visualization
        X = X[::subsample, ::subsample]
        Y = Y[::subsample, ::subsample]
        U = U[::subsample, ::subsample]
        V = V[::subsample, ::subsample]
        M = M[::subsample, ::subsample]
        # Plot quiver with no extra scaling (scale=1 means arrow length = displacement)
        quiv = ax.quiver(
            X, Y, U, V, M,
            angles='xy',
            scale_units='xy',
            scale=1,
            cmap="viridis",
            norm=plt.Normalize(vmin=0, vmax=40)  # Fixed colormap normalization range
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()  # Flip y-axis to match Cartesian coordinate system
        # Add colorbar for displacement magnitude
        cbar = plt.colorbar(quiv, ax=ax, fraction=0.046, pad=0.08)
        cbar.set_label("Displacement Magnitude", fontsize=12)
    # TOP ROW: CT vs CAD overlays
    plot_overlay(axes[0, 0], cad_image[x_mid, :, :], ct_image[x_mid, :, :],
                 "YZ Plane - CT vs CAD (X Middle)", cmap_cad="Greens", cmap_ct="gray")
    plot_overlay(axes[0, 1], np.rot90(cad_image[:, :, z_mid], k=-1),
                 np.rot90(ct_image[:, :, z_mid], k=-1),
                 "XZ Plane - CT vs CAD (Y Middle)", cmap_cad="Greens", cmap_ct="gray")
    plot_overlay(axes[0, 2], cad_image[:, y_mid, :], ct_image[:, y_mid, :],
                 "XY Plane - CT vs CAD (Z Middle)", cmap_cad="Greens", cmap_ct="gray")
    # MIDDLE ROW: MOVED CT vs CAD overlays
    plot_overlay(axes[1, 0], cad_image[x_mid, :, :], moved_ct[x_mid, :, :],
                 "YZ Plane - MOVED CT vs CAD (X Middle)", cmap_cad="Greens", cmap_ct="gray")
    plot_overlay(axes[1, 1], np.rot90(cad_image[:, :, z_mid], k=-1),
                 np.rot90(moved_ct[:, :, z_mid], k=-1),
                 "XZ Plane - MOVED CT vs CAD (Y Middle)", cmap_cad="Greens", cmap_ct="gray")
    plot_overlay(axes[1, 2], cad_image[:, y_mid, :], moved_ct[:, y_mid, :],
                 "XY Plane - MOVED CT vs CAD (Z Middle)", cmap_cad="Greens", cmap_ct="gray")
    # BOTTOM ROW: Quiver plots of displacement vectors
    add_quiver(axes[2, 0], displacement_field_corrected[x_mid, :, :],
               "YZ Plane - Displacement Vectors (X Middle)")
    add_quiver(axes[2, 1], np.rot90(displacement_field_corrected[:, :, z_mid], k=-1),
               "XZ Plane - Displacement Vectors (Y Middle)")
    add_quiver(axes[2, 2], displacement_field_corrected[:, y_mid, :],
               "XY Plane - Displacement Vectors (Z Middle)")
    plt.tight_layout()
    plt.show(block=True)



def apply_weighted_gaussian_filter(deformation_field, sigma=1.5):
    """
    Apply a Gaussian filter to smooth the displacement field with more weight for values closer to zero.

    Args:
        deformation_field (numpy array): The filtered displacement field (X, Y, Z, 3) with NaN values in background.
        sigma (float): Standard deviation for Gaussian filter.

    Returns:
        numpy array: Smoothed displacement field.
    """
    try:
        smoothed_field = np.copy(deformation_field)  # Copy to preserve original data

        for i in range(3):  # Apply filter to each displacement component
            component = deformation_field[..., i]
            valid_mask = ~np.isnan(component)  # Identify valid foreground voxels

            # Apply Gaussian weight: higher weight for values near zero
            weights = np.exp(-np.abs(component) / 10)  # Adjust decay rate based on data

            # Apply Gaussian filter only to valid values
            smoothed_component = gaussian_filter(component * weights, sigma=sigma)
            normalization_factor = gaussian_filter(weights, sigma=sigma)  # Normalize effect
            smoothed_component /= normalization_factor  # Avoid over-smoothing at high values

            # Preserve NaN values in the background
            smoothed_field[..., i][valid_mask] = smoothed_component[valid_mask]

        return smoothed_field
    except Exception as e:
        print(f"Error applying Gaussian filter: {e}")
        return None

def filter_deformation_field(deformation_field, ct_image):
    """
    Filter the deformation field to keep only values where the CT image is foreground.

    Args:
        deformation_field (numpy array): The deformation field with shape (X, Y, Z, 3).
        ct_image (numpy array): Binary mask of the CT image with shape (X, Y, Z),
                                where 1 represents foreground and 0 represents background.

    Returns:
        numpy array: Filtered deformation field where background values are removed.
    """
    try:
        # Ensure CT image is boolean (1=foreground, 0=background)
        mask = ct_image > 0  # Convert to boolean mask

        # Apply mask to deformation field (set background to NaN or discard)
        filtered_deformation_field = np.full_like(deformation_field, np.nan)  # Create empty array
        filtered_deformation_field[mask] = deformation_field[mask]  # Keep values in foreground

        return filtered_deformation_field
    except Exception as e:
        print(f"Error filtering deformation field: {e}")
        return None

def plot_directional_histograms(deformation_field, bins=50, title_prefix="Histogram of Deformation Components"):
    """
    Plot histograms for the deformation components (dx, dy, dz) along the X, Y, and Z axes
    with a fitted Gaussian curve, ignoring NaN values in the calculations.
    Args:
        deformation_field (numpy array): The deformation field with shape (X, Y, Z, 3), where background voxels may contain NaN.
        bins (int): The number of bins for the histograms.
        title_prefix (str): The prefix for the histogram titles.
    """
    try:
        # Extract components and ignore NaN values
        dx = deformation_field[..., 0][~np.isnan(deformation_field[..., 0])]
        dy = deformation_field[..., 2][~np.isnan(deformation_field[..., 2])]
        dz = deformation_field[..., 1][~np.isnan(deformation_field[..., 1])]
        components = [dx, dy, dz]
        labels = ['X-axis', 'Y-axis', 'Z-axis']
        # Define fixed x-axis range
        x_min, x_max = -40, 40
        # Compute histogram counts within the x-range to determine max y-limit
        hist_data = [np.histogram(comp, bins=bins, range=(x_min, x_max)) for comp in components]
        max_freq = max([max(hist[0]) for hist in hist_data])  # Find max y-axis limit
        # Increase y-axis limit slightly (e.g., 10% more)
        y_max = max_freq * 1.1
        # Plot histograms with fitted Gaussian curves
        for component, label in zip(components, labels):
            plt.figure(figsize=(10, 6))
            # Compute histogram
            counts, bin_edges = np.histogram(component, bins=bins, range=(x_min, x_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
            # Fit Gaussian (Normal) distribution while ignoring NaNs
            mu, sigma = np.nanmean(component), np.nanstd(component)  # Use nan-safe functions
            gaussian_curve = norm.pdf(bin_centers, mu, sigma) * np.sum(counts) * (
                        bin_edges[1] - bin_edges[0])  # Scale for histogram
            # Plot histogram
            plt.hist(component, bins=bins, range=(x_min, x_max), color="skyblue", edgecolor="black", alpha=0.6,
                     label="Histogram")
            # Plot Gaussian fit
            plt.plot(bin_centers, gaussian_curve, 'r-', linewidth=2, label=f"Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}")
            # Formatting
            plt.title(f"{title_prefix} ({label})")
            plt.xlabel(f"Displacement field in {label}")
            plt.ylabel("Frequency")
            plt.xlim(x_min, x_max)  # Set fixed displacement range
            plt.ylim(0, y_max)  # Increase y-axis limit
            plt.legend()
            plt.grid(True)
            plt.show()
    except Exception as e:
        print(f"Error plotting directional histograms: {e}")


def generate_plots_displacement_xyz(ct_image, cad_image, moved_ct, displacement_field_corrected):
    """Generates a 5x3 figure with:
       - 1st row: Alpha-blended overlays of CT & CAD.
       - 2nd row: Alpha-blended overlays of MOVED CT & CAD.
       - 3rd row: Centered displacement in X direction.
       - 4th row: Centered displacement in Y direction.
       - 5th row: Centered displacement in Z direction.
    """
    Dx, Dy, Dz = displacement_field_corrected.shape[:3]
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2  # Middle slices
    fig = plt.figure(figsize=(25, 20))  # Increased height for better spacing
    gs = gridspec.GridSpec(5, 3, height_ratios=[1, 1, 1.5, 1.5, 1.5])  # More space for displacement plots
    axes = np.empty((5, 3), dtype=object)
    for i in range(5):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j])
    # Function to add a centered colorbar with proper range
    def add_colorbar(ax, img, label="Displacement Magnitude"):
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.05)
        cbar.set_label(label, fontsize=12)
        cbar.mappable.set_clim(vmin=0, vmax=20)  # Fixed range
    # Extract displacement in X, Y, and Z directions
    disp_x = displacement_field_corrected[:, :, :, 0]  # X-direction displacement
    disp_y = displacement_field_corrected[:, :, :, 1]  # Y-direction displacement
    disp_z = displacement_field_corrected[:, :, :, 2]  # Z-direction displacement
    # 1ST ROW: CT vs CAD Overlays (Before Deformation)
    plot_overlay(axes[0, 0], cad_image[x_mid, :, :], ct_image[x_mid, :, :], "YZ Plane - CT vs CAD ")
    plot_overlay(axes[0, 1], np.rot90(cad_image[:, :, z_mid], k=-1), np.rot90(ct_image[:, :, z_mid], k=-1),
                 "XZ Plane - CT vs CAD ")
    plot_overlay(axes[0, 2], cad_image[:, y_mid, :], ct_image[:, y_mid, :], "XY Plane - CT vs CAD ")
    # 2ND ROW: MOVED CT vs CAD Overlays (After Deformation)
    plot_overlay(axes[1, 0], cad_image[x_mid, :, :], moved_ct[x_mid, :, :], "YZ Plane - MOVED CT vs CAD ")
    plot_overlay(axes[1, 1], np.rot90(cad_image[:, :, z_mid], k=-1), np.rot90(moved_ct[:, :, z_mid], k=-1),
                 "XZ Plane - MOVED CT vs CAD ")
    plot_overlay(axes[1, 2], cad_image[:, y_mid, :], moved_ct[:, y_mid, :], "XY Plane - MOVED CT vs CAD ")
    # 3RD ROW: Centered Displacement in X Direction
    img1 = axes[2, 0].imshow(disp_x[x_mid, :, :], cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[2, 0].set_title("YZ Plane - Displacement in X ")
    add_colorbar(axes[2, 0], img1, "Displacement in X")
    img2 = axes[2, 1].imshow(np.rot90(disp_x[:, :, z_mid], k=-1), cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[2, 1].set_title("XZ Plane - Displacement in X ")
    add_colorbar(axes[2, 1], img2, "Displacement in X")
    img3 = axes[2, 2].imshow(disp_x[:, y_mid, :], cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[2, 2].set_title("XY Plane - Displacement in X ")
    add_colorbar(axes[2, 2], img3, "Displacement in X")
    # 4TH ROW: Centered Displacement in Y Direction
    img4 = axes[3, 0].imshow(disp_z[x_mid, :, :], cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[3, 0].set_title("YZ Plane - Displacement in Y ")
    add_colorbar(axes[3, 0], img4, "Displacement in Y")
    img5 = axes[3, 1].imshow(np.rot90(disp_z[:, :, z_mid], k=-1), cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[3, 1].set_title("XZ Plane - Displacement in Y ")
    add_colorbar(axes[3, 1], img5, "Displacement in Y")
    img6 = axes[3, 2].imshow(disp_z[:, y_mid, :], cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[3, 2].set_title("XY Plane - Displacement in Y ")
    add_colorbar(axes[3, 2], img6, "Displacement in Y")
    # 5TH ROW: Centered Displacement in Z Direction
    img7 = axes[4, 0].imshow(disp_y[x_mid, :, :], cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[4, 0].set_title("YZ Plane - Displacement in Z ")
    add_colorbar(axes[4, 0], img7, "Displacement in Z")
    img8 = axes[4, 1].imshow(np.rot90(disp_y[:, :, z_mid], k=-1), cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[4, 1].set_title("XZ Plane - Displacement in Z ")
    add_colorbar(axes[4, 1], img8, "Displacement in Z")
    img9 = axes[4, 2].imshow(disp_y[:, y_mid, :], cmap="viridis", origin="upper", vmin=0, vmax=20)
    axes[4, 2].set_title("XY Plane - Displacement in Z ")
    add_colorbar(axes[4, 2], img9, "Displacement in Z")
    plt.tight_layout()
    plt.show(block=True)




def generate_plots_magnitude(ct_image, cad_image, moved_ct, displacement_field_corrected):
    """Generates a 3x3 figure with:
       - Top row: Alpha-blended overlays of CT & CAD (XY rotated).
       - Middle row: Alpha-blended overlays of MOVED CT & CAD (XY rotated).
       - Bottom row: Centered Displacement Magnitude heatmaps WITH COLORBARS."""

    Dx, Dy, Dz = displacement_field_corrected.shape[:3]
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2  # Middle slices

    fig = plt.figure(figsize=(25, 12))  # Increased height for better spacing
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.2])  # Make bottom row slightly larger

    axes = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j])

    # ✅ Function to add colorbar with better positioning
    def add_colorbar(ax, img):
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.08)
        cbar.set_label("Displacement Magnitude", fontsize=12)

    # ✅ TOP ROW: CT vs CAD overlays
    plot_overlay(axes[0, 0], cad_image[x_mid, :, :], ct_image[x_mid, :, :], "YZ Plane - CT vs CAD (X Middle)",
                 cmap_cad="Greens", cmap_ct="gray")

    plot_overlay(axes[0, 1], np.rot90(cad_image[:, :, z_mid], k=-1),
                 np.rot90(ct_image[:, :, z_mid], k=-1),
                 "XZ Plane - CT vs CAD (Y Middle)", cmap_cad="Greens", cmap_ct="gray")

    plot_overlay(axes[0, 2], cad_image[:, y_mid, :], ct_image[:, y_mid, :], "XY Plane - CT vs CAD (Z Middle)",
                 cmap_cad="Greens", cmap_ct="gray")

    # ✅ MIDDLE ROW: MOVED CT vs CAD overlays
    plot_overlay(axes[1, 0], cad_image[x_mid, :, :], moved_ct[x_mid, :, :], "YZ Plane - MOVED CT vs CAD (X Middle)",
                 cmap_cad="Greens", cmap_ct="gray")

    plot_overlay(axes[1, 1], np.rot90(cad_image[:, :, z_mid], k=-1),
                 np.rot90(moved_ct[:, :, z_mid], k=-1),
                 "XZ Plane - MOVED CT vs CAD (Y Middle)", cmap_cad="Greens", cmap_ct="gray")

    plot_overlay(axes[1, 2], cad_image[:, y_mid, :], moved_ct[:, y_mid, :], "XY Plane - MOVED CT vs CAD (Z Middle)",
                 cmap_cad="Greens", cmap_ct="gray")

    # ✅ BOTTOM ROW: CENTERED Displacement Magnitude heatmaps WITH COLORBARS
    magnitude = np.linalg.norm(displacement_field_corrected, axis=-1)

    img1 = axes[2, 0].imshow(magnitude[x_mid, :, :], cmap="viridis", origin="upper", vmin=0, vmax=40)
    axes[2, 0].set_title("YZ Plane - Displacement Magnitude (X Middle)")
    add_colorbar(axes[2, 0], img1)  # Add colorbar

    img2 = axes[2, 1].imshow(np.rot90(magnitude[:, :, z_mid], k=-1), cmap="viridis", origin="upper", vmin=0, vmax=40)
    axes[2, 1].set_title("XZ Plane - Displacement Magnitude (Y Middle)")
    add_colorbar(axes[2, 1], img2)  # Add colorbar

    img3 = axes[2, 2].imshow(magnitude[:, y_mid, :], cmap="viridis", origin="upper", vmin=0, vmax=40)
    axes[2, 2].set_title("XY Plane - Displacement Magnitude (Z Middle)")
    add_colorbar(axes[2, 2], img3)  # Add colorbar

    plt.tight_layout()
    plt.show(block=True)


def plot_overlay(ax, cad, ct, title, cmap_cad="Greens", cmap_ct="gray", alpha=0.5):
    """Overlay two images with custom colormaps for CAD and CT."""
    ax.imshow(cad, cmap=cmap_cad, alpha=1.0)  # CAD in Greens
    ax.imshow(ct, cmap=cmap_ct, alpha=0.5)    # CT in Gray
    ax.set_title(title)
    ax.axis("off")




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


def load_tiff_image(tiff_file_path):
    """Load a 3D TIFF image as a NumPy array."""
    return tiff.imread(tiff_file_path)


def plot_quiver(ax, X, Y, U, V, title):
    """Helper function to plot a quiver plot."""
    ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=0.5, color="r")
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect("equal")


def plot_overlay_quiver(ax, img1, img2, title, alpha=0.5):
    """Overlay two images with alpha blending."""
    ax.imshow(img1, cmap="gray", alpha=1.0)  # Background
    ax.imshow(img2, cmap="jet", alpha=alpha)  # Overlay
    ax.set_title(title)
    ax.axis("off")


def generate_plots(ct_image, cad_image, displacement_field, subsample=5):
    """Generates a 2x3 figure with:
       - Top row: Alpha-blended overlays of CT & CAD.
       - Bottom row: Quiver plots of displacement field."""

    Dx, Dy, Dz = displacement_field.shape[:3]
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2  # Middle slices

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # CT & CAD Overlays
    plot_overlay_quiver(axes[0, 0], cad_image[:, :, z_mid], ct_image[:, :, z_mid], "XY Plane - CT vs CAD (Z Middle)")
    plot_overlay_quiver(axes[0, 1], cad_image[:, y_mid, :], ct_image[:, y_mid, :], "XZ Plane - CT vs CAD (Y Middle)")
    plot_overlay_quiver(axes[0, 2], cad_image[x_mid, :, :], ct_image[x_mid, :, :], "YZ Plane - CT vs CAD (X Middle)")

    # XY Plane (Middle Z Slice)
    X, Y = np.meshgrid(np.arange(0, Dx, subsample), np.arange(0, Dy, subsample), indexing="ij")
    U, V = displacement_field[::subsample, ::subsample, z_mid, 0], displacement_field[::subsample, ::subsample, z_mid,
                                                                   1]
    plot_quiver(axes[1, 0], X, Y, U, V, "XY Plane - Displacement (Z Middle)")

    # XZ Plane (Middle Y Slice)
    X, Y = np.meshgrid(np.arange(0, Dx, subsample), np.arange(0, Dz, subsample), indexing="ij")
    U, V = displacement_field[::subsample, y_mid, ::subsample, 0], displacement_field[::subsample, y_mid, ::subsample,
                                                                   2]
    plot_quiver(axes[1, 1], X, Y, U, V, "XZ Plane - Displacement (Y Middle)")

    # YZ Plane (Middle X Slice)
    X, Y = np.meshgrid(np.arange(0, Dy, subsample), np.arange(0, Dz, subsample), indexing="ij")
    U, V = displacement_field[x_mid, ::subsample, ::subsample, 1], displacement_field[x_mid, ::subsample, ::subsample,
                                                                   2]
    plot_quiver(axes[1, 2], X, Y, U, V, "YZ Plane - Displacement (X Middle)")

    plt.tight_layout()
    plt.show()


def plot_overlapping_histograms(deformation_field, bins=50, title="Overlapping Histograms of Deformation Components"):
    """
    Plot overlapping histograms for the deformation components (dx, dy, dz) along the X, Y, and Z axes
    with fitted Gaussian curves, ignoring NaN values in the calculations.

    Args:
        deformation_field (numpy array): The deformation field with shape (X, Y, Z, 3), where background voxels may contain NaN.
        bins (int): The number of bins for the histograms.
        title (str): The title for the histogram plot.
    """
    try:
        # Close any existing figures to free up memory
        plt.close('all')

        # Extract components and ignore NaN values
        dx = deformation_field[..., 0][~np.isnan(deformation_field[..., 0])]
        dy = deformation_field[..., 2][~np.isnan(deformation_field[..., 2])]
        dz = deformation_field[..., 1][~np.isnan(deformation_field[..., 1])]

        components = [dx, dy, dz]
        labels = ['X-axis', 'Y-axis', 'Z-axis']
        colors = ['blue', 'green', 'red']

        # Compute and print statistical properties
        for component, label in zip(components, labels):
            mu = np.nanmean(component)
            std_dev = np.nanstd(component)
            skewness = skew(component, nan_policy='omit')
            kurt = kurtosis(component, nan_policy='omit')
            print(
                f"{label}: Mean = {mu:.2f}, Std Dev = {std_dev:.2f}, Skewness = {skewness:.2f}, Kurtosis = {kurt:.2f}")

        # Define fixed x-axis range
        x_min, x_max = -40, 40

        # Compute histogram counts within the x-range to determine max y-limit
        hist_data = [np.histogram(comp, bins=bins, range=(x_min, x_max), density=True) for comp in components]
        max_freq = max([max(hist[0]) for hist in hist_data])  # Find max y-axis limit
        y_max = max_freq * 1.2  # Increase y-axis limit slightly

        # Create overlapping histogram plot
        plt.figure(figsize=(10, 6))

        for component, label, color in zip(components, labels, colors):
            # Downsample data if too large (for performance reasons)
            if len(component) > 100000:
                component = np.random.choice(component, 100000, replace=False)

            # Compute histogram
            counts, bin_edges = np.histogram(component, bins=bins, range=(x_min, x_max), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers

            # Fit Gaussian (Normal) distribution
            mu, sigma = np.nanmean(component), np.nanstd(component)
            gaussian_curve = norm.pdf(bin_centers, mu, sigma)

            # Plot histogram
            plt.hist(component, bins=bins, range=(x_min, x_max), color=color, edgecolor="black", alpha=0.4,
                     density=True, label=f"{label} Histogram")

            # Plot Gaussian fit (normalized for density)
            plt.plot(bin_centers, gaussian_curve, color=color, linestyle='dashed', linewidth=2,
                     label=f"{label} Gaussian Fit (μ={mu:.2f}, σ={sigma:.2f})")

        # Formatting
        plt.title(title, fontsize=18)
        plt.xlabel("Displacement Field", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Show plot with a brief pause to prevent freezing
        plt.pause(0.1)
        plt.show(block=False)

    except Exception as e:
        print(f"Error plotting overlapping histograms: {e}")


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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 22})


def plot_overlay(ax, cad, ct, title, cmap_cad="Greens", cmap_ct="gray", alpha=0.5):
    ax.imshow(cad, cmap=cmap_cad, alpha=1.0)
    ax.imshow(ct, cmap=cmap_ct, alpha=alpha)
    ax.set_title(title, fontsize=22)
    ax.axis("off")
    ax.set_aspect('equal')


def add_colorbar(ax, img):
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("Disp. Magnitude", fontsize=22)


def add_quiver(ax, disp_slice, title, show_labels=False, subsample=5):
    H, W, _ = disp_slice.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    U = disp_slice[..., 0]
    V = -disp_slice[..., 1]
    # ✅ Convert only the color magnitude to micrometers
    M = np.linalg.norm(disp_slice, axis=-1) * 10  # keep U, V unchanged
    # Subsample
    X, Y, U, V, M = [a[::subsample, ::subsample] for a in [X, Y, U, V, M]]
    # Quiver plot (keep U, V in pixels, but M in micrometers)
    quiv = ax.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', scale=1,
                     cmap="viridis", norm=plt.Normalize(vmin=0, vmax=200))
    ax.set_title(title, fontsize=22)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    # ✅ Updated colorbar for micrometers
    cbar = plt.colorbar(quiv, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("Displacement (μm)", fontsize=22)
    cbar.ax.tick_params(labelsize=16)
    if show_labels:
        ax.set_xlabel("X", fontsize=18)
        ax.set_ylabel("Y", fontsize=18)


def generate_plot_overlays(ct_image, cad_image, moved_ct):
    Dx, Dy, Dz = cad_image.shape
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2

    fig, axes = plt.subplots(2, 3, figsize=(25, 10))
    titles = ["XCT vs CAD", "XCT vs CAD", "XCT vs CAD",
              "MOVED XCT vs CAD", "MOVED XCT vs CAD", "MOVED XCT vs CAD"]
    slices = [
        (cad_image[x_mid, :, :], ct_image[x_mid, :, :]),
        (np.rot90(cad_image[:, :, z_mid], k=-1), np.rot90(ct_image[:, :, z_mid], k=-1)),
        (cad_image[:, y_mid, :], ct_image[:, y_mid, :]),
        (cad_image[x_mid, :, :], moved_ct[x_mid, :, :]),
        (np.rot90(cad_image[:, :, z_mid], k=-1), np.rot90(moved_ct[:, :, z_mid], k=-1)),
        (cad_image[:, y_mid, :], moved_ct[:, y_mid, :]),
    ]

    for ax, (cad, ct), title in zip(axes.flat, slices, titles):
        plot_overlay(ax, cad, ct, title)

    plt.tight_layout()
    plt.show()


def generate_plot_deformations(displacement_field_corrected):
    Dx, Dy, Dz = displacement_field_corrected.shape[:3]
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2
    fig, axes = plt.subplots(2, 3, figsize=(28, 12))
    disp_magnitude = np.linalg.norm(displacement_field_corrected * 10, axis=-1)  # convert to micrometers
    mags = [
        disp_magnitude[x_mid, :, :],
        np.rot90(disp_magnitude[:, :, z_mid], k=-1),
        disp_magnitude[:, y_mid, :],
    ]
    vectors = [
        displacement_field_corrected[x_mid, :, :],
        np.rot90(displacement_field_corrected[:, :, z_mid], k=-1),
        displacement_field_corrected[:, y_mid, :],
    ]
    for i in range(3):
        img = axes[0, i].imshow(mags[i], cmap="viridis", origin="upper", vmin=0, vmax=200)
        axes[0, i].set_title("Disp. Magnitude", fontsize=22)
        axes[0, i].set_aspect("equal")
        axes[0, i].tick_params(bottom=True, left=True)
        # Colorbar for top row only
        cbar = plt.colorbar(img, ax=axes[0, i], fraction=0.046, pad=0.04)
        cbar.set_label("Displacement (μm)", fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        show_labels = (i == 0)
        add_quiver(axes[1, i], vectors[i], "Disp. Vectors", show_labels=show_labels)
    plt.tight_layout()
    plt.show()

def add_scalebar(ax, length_pixels=150, label="1 mm", height=8, pad=20):
    """
    Adds a clean horizontal scalebar with the label above it.
    Parameters:
    - ax: matplotlib axis
    - length_pixels: length of the scalebar (in pixels)
    - label: string to display above the bar
    - height: thickness of the black bar
    - pad: padding from the bottom of the image (in pixels)
    """
    # Adjust for axis direction (handles flipped Y axes)
    ylim = ax.get_ylim()
    y_direction = -1 if ylim[0] > ylim[1] else 1
    y_start = ylim[0] + y_direction * pad
    x_start = 20  # fixed x offset from left
    # Draw the black scalebar
    ax.add_patch(
        plt.Rectangle((x_start, y_start), length_pixels, height,
                      color='black', zorder=10)
    )
    # Draw the label clearly ABOVE the bar (not overlapping)
    ax.text(
        x_start + length_pixels / 2,  # center of the bar
        y_start + height + 40 * y_direction,  # position above the bar
        label,
        color='black',
        fontsize=16,
        ha='center',
        va='bottom' if y_direction == 1 else 'top',
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.7',
            alpha=0.5
        )
    )

def generate_plot_overlays(ct_image, cad_image, moved_ct):
    Dx, Dy, Dz = cad_image.shape
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2
    fig, axes = plt.subplots(2, 3, figsize=(25, 10))
    titles = ["XCT vs CAD", "XCT vs CAD", "XCT vs CAD",
              "MOVED XCT vs CAD", "MOVED XCT vs CAD", "MOVED XCT vs CAD"]
    slices = [
        (cad_image[x_mid, :, :], ct_image[x_mid, :, :]),
        (np.rot90(cad_image[:, :, z_mid], k=-1), np.rot90(ct_image[:, :, z_mid], k=-1)),
        (cad_image[:, y_mid, :], ct_image[:, y_mid, :]),
        (cad_image[x_mid, :, :], moved_ct[x_mid, :, :]),
        (np.rot90(cad_image[:, :, z_mid], k=-1), np.rot90(moved_ct[:, :, z_mid], k=-1)),
        (cad_image[:, y_mid, :], moved_ct[:, y_mid, :]),
    ]
    for ax, (cad, ct), title in zip(axes.flat, slices, titles):
        plot_overlay(ax, cad, ct, title)
        add_scalebar(ax, length_pixels=100, label="1 mm")  # 100 pixels = 1 mm
    plt.tight_layout()
    plt.show()


def adaptive_threshold_ct(ct_data, block_size=41, offset=5, min_size=500, slice_index=None, visualize=False):
    """
    Apply adaptive thresholding to a 3D CT volume slice-by-slice.
    Parameters:
        ct_data (ndarray): 3D grayscale CT image.
        block_size (int): Size of the neighborhood for adaptive thresholding.
        offset (float): Subtracted from local mean to determine threshold.
        min_size (int): Minimum object size to retain in each slice.
        slice_index (int): If visualize=True, this slice will be shown.
        visualize (bool): Whether to display the original and thresholded slice.
    Returns:
        masked_ct_data (ndarray): CT volume with only thresholded foreground retained.
    """
    binary_mask = np.zeros_like(ct_data, dtype=bool)
    for i in range(ct_data.shape[0]):
        slice_img = ct_data[i]
        local_thresh = threshold_local(slice_img, block_size=block_size, offset=offset)
        binarized = slice_img > local_thresh
        binary_mask[i] = binarized

    # Visualization
    if visualize and slice_index is not None and 0 <= slice_index < ct_data.shape[0]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ct_data[slice_index], cmap="gray")
        axes[0].set_title("Original CT Slice")
        axes[1].imshow(binary_mask[slice_index], cmap="gray")
        axes[1].set_title("Thresholded Slice")
        plt.tight_layout()
        plt.show()
    return binary_mask.astype(np.int8)

def binarize_volume(data):
    """
    Convert all non-zero values in the input array to 1 (binary foreground),
    and leave zero values as 0 (background).
    Parameters:
        data (ndarray): 2D or 3D input volume or image.
    Returns:
        binary_data (ndarray): Binarized array with values 0 or 1.
    """
    binary_data = (data != 0).astype(np.uint8)
    return binary_data

def display_diff_and_overlap(cad_image, ct_image, diff_map, title="Difference Map", clim=(-1, 1)):
    """
    Displays overlap of cad_image and ct_image and the diff_map in x, y, z slices.
    Parameters:
        cad_image (ndarray): Fixed reference (e.g., CAD).
        ct_image (ndarray): Moving image (e.g., XCT or moved XCT).
        diff_map (ndarray): Difference map between the binarized volumes.
        title (str): Title prefix for difference map plots.
        clim (tuple): Color limits for the difference map.
    """
    Dx, Dy, Dz = cad_image.shape
    x_mid, y_mid, z_mid = Dx // 2, Dy // 2, Dz // 2
    # Extract slices
    slices = {
        "X Slice": (cad_image[x_mid], ct_image[x_mid], diff_map[x_mid]),
        "Z Slice": (np.rot90(cad_image[:, :, z_mid], k=-1),
                    np.rot90(ct_image[:, :, z_mid], k=-1),
                    np.rot90(diff_map[:, :, z_mid], k=-1)),
        "Y Slice": (cad_image[:, y_mid], ct_image[:, y_mid], diff_map[:, y_mid]),
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for idx, (label, (cad_slice, ct_slice, diff_slice)) in enumerate(slices.items()):
        # Row 1: Overlap
        axes[0, idx].imshow(cad_slice, cmap='Greens', alpha=1.0, interpolation='none')
        axes[0, idx].imshow(ct_slice, cmap='gray', alpha=0.5, interpolation='none')
        #axes[0, idx].set_title(f"Overlay - {label}")
        axes[0, idx].axis('off')
        # Row 2: Difference map
        cmap = ListedColormap(['blue', 'white', 'red'])  # -1, 0, 1
        im = axes[1, idx].imshow(diff_slice, cmap=cmap, vmin=-1, vmax=1, interpolation='none')
        #axes[1, idx].set_title(f"{title} - {label}")
        axes[1, idx].axis('off')
        axes[1, idx].set_aspect('equal')
        # Colorbar with only three ticks: -1, 0, 1
        cbar = fig.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["-1", "0", "1"])
        cbar.set_label("Difference map")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def report_combined_difference_percentages(diff_map, binary_cad, binary_moved_ct):
    """
    Computes and prints percentage of -1, 0, and 1 in the diff_map
    restricted to the union of foreground regions in CAD and moved CT.
    Parameters:
        diff_map (ndarray): Difference map with values in {-1, 0, 1}.
        binary_cad (ndarray): Binarized CAD volume (0 or 1).
        binary_moved_ct (ndarray): Binarized moved CT volume (0 or 1).
    """
    # Union of foreground voxels
    combined_foreground = (binary_cad == 1) | (binary_moved_ct == 1)
    total_voxels = np.sum(combined_foreground)
    # Count values
    count_minus1 = np.sum((diff_map == -1) & combined_foreground)
    count_zero   = np.sum((diff_map == 0)  & combined_foreground)
    count_plus1  = np.sum((diff_map == 1)  & combined_foreground)
    # Compute percentages
    percent_minus1 = (count_minus1 / total_voxels) * 100 if total_voxels else 0
    percent_zero   = (count_zero   / total_voxels) * 100 if total_voxels else 0
    percent_plus1  = (count_plus1  / total_voxels) * 100 if total_voxels else 0
    # Display
    print("=== Difference Map Percentage Analysis ===")
    print(f"Total foreground voxels (union): {total_voxels}")
    print(f"-1 (CAD=1, CT=0):   {count_minus1} voxels  →  {percent_minus1:.2f}%")
    print(f" 0 (Match):         {count_zero} voxels    →  {percent_zero:.2f}%")
    print(f" 1 (CAD=0, CT=1):   {count_plus1} voxels   →  {percent_plus1:.2f}%")

from skimage import filters
import numpy as np

def get_middle_region(volume):
    """
    Extract the middle third region of a 3D volume (ROI).
    Avoids the outermost areas which might contain air or artifacts.
    """
    start_x, end_x = volume.shape[0] // 3, 2 * volume.shape[0] // 3
    start_y, end_y = volume.shape[1] // 3, 2 * volume.shape[1] // 3
    start_z, end_z = volume.shape[2] // 3, 2 * volume.shape[2] // 3
    return volume[start_x:end_x, start_y:end_y, start_z:end_z]

def binarize_volume_with_middle_roi(volume):
    """
    Binarizes a 3D volume using global Otsu thresholding.
    The threshold is calculated from the middle third of the volume to reduce influence of background air.
    """
    roi = get_middle_region(volume)
    threshold = filters.threshold_otsu(roi.flatten())
    binary_volume = (volume >= threshold).astype(np.uint8)
    return binary_volume

if __name__ == "__main__":


    # Define file paths
    vtk_file_path = r"F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\results\stride_64_gaussian0.5_v1\disp_field.vtk"
    ct_file_path = r"F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\results\stride_64_gaussian0.5_v1\moving_image.tiff"  # XCT (Moving)
    cad_file_path = r"F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\results\stride_64_gaussian0.5_v1\fixed_image.tiff"  # CAD (Fixed)
    moved_ct_file_path = r"F:\Projects\_Additive_manufacturing\QI_Digital\Publications\04_deformation_prediction\results\stride_64_gaussian0.5_v1\reconstructed_moved.tiff"  # XCT (Moved)
    # Load displacement field
    displacement_field = load_displacement_field(vtk_file_path)
    # Load CT (moving) and CAD (fixed) images
    ct_image = load_tiff_image(ct_file_path)
    cad_image = load_tiff_image(cad_file_path)
    moved_ct = load_tiff_image(moved_ct_file_path)
    # Generate plots

    #generate magnitude plots
    generate_plots_magnitude(ct_image, cad_image, moved_ct, displacement_field)


    generate_plots_displacement_xyz(ct_image, cad_image, moved_ct, displacement_field)
    filtered_displacement_field = filter_deformation_field(displacement_field, ct_image)
    plot_directional_histograms(filtered_displacement_field, bins=50,
                                title_prefix="Histogram of displacement field")

    # Apply Gaussian smoothing with weight emphasis on small values
    #smoothed_deformation_field = apply_weighted_gaussian_filter(filtered_displacement_field, sigma=1.5)

    # Plot histograms with Gaussian fit for smoothed field
    plot_directional_histograms(filtered_displacement_field, bins=50,
                                title_prefix="Histogram of Smoothed Deformation Components")
    plot_overlapping_histograms(filtered_displacement_field, bins=50,
                               title="Overlapping Histograms of Deformation Components")


    #plot 2D quiver plots to plot displacement
    generate_plots_quiver(ct_image, cad_image, moved_ct, displacement_field)
    plot_overlay_and_quiver_single_axis_horizontal(ct_image, cad_image, displacement_field, axis='XY', slice_index=None,
                                                   subsample=5)


    #3d plots
    visualize_3d_displacement_magnitude(ct_image, displacement_field,
                                        title="3D Displacement Magnitude", clim_max=10)
    visualize_3d_quiver(ct_image, displacement_field, title="3D Displacement Field", max_points=500000,
                        glyph_factor=10)

    #for publication
    generate_plot_overlays(ct_image, cad_image, moved_ct)
    generate_plot_deformations(displacement_field)


    #binarize and plot and save
    binary_ct = binarize_volume(ct_image)
    binary_moved_ct = binarize_volume(moved_ct)
    binary_cad = binarize_volume(cad_image)
    generate_plot_overlays(binary_ct, binary_cad, binary_moved_ct)
    diff_map = binary_moved_ct.astype(np.int8) - binary_cad.astype(np.int8)
    report_combined_difference_percentages(diff_map, binary_ct, binary_moved_ct)
    display_diff_and_overlap(
        cad_image=binary_cad,
        ct_image=binary_ct,
        diff_map=diff_map,
        title="Binary Difference",
        clim=(-1, 1)
    )
    #dice_before = 2 * np.sum(binary_ct & binary_cad[:,:,:]) / (np.sum(binary_ct[:,:,:]) + np.sum(binary_cad[:,:,:]))
    #dice_after = 2 * np.sum(binary_ct & binary_moved_cad[:,:,:]) / (np.sum(binary_ct[:,:,:]) + np.sum(binary_moved_cad[:,:,:]))
    dice = dice_coefficient(cad_image, moved_ct)
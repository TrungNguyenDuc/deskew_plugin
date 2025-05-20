<h1 align="center">
<img src="https://github.com/TheDeanLab/navigate-plugin-template/blob/main/plugin-icon.jpg" width="200" height="200"/>

OPM Analysis Navigate Plugin
	


# Advanced OPM Analysis Plugin for Navigate Software

This plugin provides a suite of tools for processing and analyzing Optical Projection Microscopy (OPM) and Light-Sheet Microscopy (LSM) data, including deskewing, decorrelation analysis for resolution estimation, and Point Spread Function (PSF) fitting. It is designed to be integrated into the **Navigate Software** https://github.com/TheDeanLab/navigate.

The plugin leverages GPU acceleration via CuPy for computationally intensive tasks, with CPU fallbacks for broader compatibility.

## Features

* **Comprehensive Data Processing:**
    * **Deskewing:** Corrects geometric distortions in light-sheet images arising from angled illumination.
        * User-configurable parameters: XY pixel size, Z-step, light-sheet angle, flip direction.
        * GPU-accelerated shear and rotation operations with chunked processing for large datasets.
        * Optional post-shear Gaussian smoothing on Y'c, X, and Z'c axes.
        * Option to save intermediate sheared volumes and final deskewed volumes.
    * **Resolution Estimation (Decorrelation Analysis):**
        * Calculates resolution and Signal-to-Noise Ratio (SNR) from Maximum Intensity Projections (MIPs) of the deskewed data (XY, XZ, YZ planes).
        * Outputs results in specified physical units (e.g., µm).
    * **PSF Fitting:**
        * Analyzes 3D images of beads or point-like structures to determine experimental Point Spread Function characteristics.
        * Fits Gaussian profiles to line profiles through identified particles to calculate FWHM in X, Y, and Z.
        * User-configurable parameters: ROI padding, ROI radius, intensity threshold for bead detection, R² fit quality threshold.
        * Optional GPU acceleration for PSF data preparation steps.
* **Interactive GUI:**
    * Intuitive tabbed interface for Deskewing, Decorrelation, and PSF Fitting parameters.
    * File browser for selecting 3D TIFF input stacks.
    * Real-time logging of processing steps and summary results.
    * Integrated display for:
        * Combined Orthographic Maximum Intensity Projections (MIPs) of the deskewed output.
        * PSF FWHM vs. Position plots.
    * Interactive display controls:
        * Zoom (mouse wheel) and Pan (mouse drag) for image/plot viewers.
        * Contrast adjustment (min/max display sliders, auto-contrast) for the MIP display.
* **Output:**
    * Deskewed 3D TIFF images.
    * Saved MIPs (individual planes and combined orthographic view) as TIFF and PNG.
    * Saved PSF analysis plots (thresholded MIP, FWHM vs. Position) as PNG.
    * Text file with deskewing parameters.
    * Detailed PSF fitting results potentially saved as NPZ files.
    * Log files and summary results displayed in the GUI.

## Requirements

* Python 3.9
* Navigate Software (Host Application - *please specify version if applicable*) https://github.com/TheDeanLab/navigate
* **Required Python Libraries:**
    * `numpy`
    * `scipy`
    * `tifffile`
    * `Pillow` (PIL)
    * `matplotlib`
    * `scikit-image`
    * `tkinter` (usually included with standard Python installations)
    * `cupy` (Optional, for GPU acceleration. Requires a compatible NVIDIA GPU and CUDA toolkit installed. The plugin will attempt to use CPU fallbacks if CuPy is not available or fails.)

## Installation

1.  Ensure all required Python libraries listed above are installed in your Python environment used by the Navigate Software.
    ```bash
    pip install numpy scipy tifffile Pillow matplotlib scikit-image
    # For GPU acceleration (optional, requires NVIDIA GPU & CUDA):
    # pip install cupy-cudaXXX # Replace XXX with your CUDA version, e.g., cupy-cuda11x or cupy-cuda12x
    ```
2.  Place the plugin files (`confocal_projection_frame.py` and `confocal_projection_controller.py`, and any other necessary modules) into the appropriate plugin directory for the Navigate Software. (Please provide specific instructions here based on how Navigate handles plugins).
3.  Follow the instructions to integrate OPM Analysis into Navigate https://thedeanlab.github.io/navigate/03_contributing/06_plugin/plugin_home.html

## How to Use

1.  **Launch Navigate Software** and open the OPM Analysis Plugin.
2.  **Select Input File:** Click the "Select TIFF File" button to choose your 3D OPM/LSM raw data stack.
3.  **Configure Parameters:**
    * **Deskewing Tab:**
        * Enter the correct XY pixel size (µm), Z stage step (µm) for your acquisition.
        * Set the Light Sheet Angle (degrees) and Flip Direction (+1 or -1) based on your microscope setup.
        * Optionally, enable and configure "Post-Shear Smoothing" parameters (Y'c, X, Z'c sigmas).
        * Choose output options (save intermediate, save final, save plots).
        * Advanced: Adjust GPU chunking parameters if needed for very large datasets or specific GPU memory constraints.
    * **Decorrelation Tab:**
        * Specify the "Units Label" (e.g., um, nm) for resolution reporting.
        * Choose whether to save decorrelation plots.
    * **PSF Fitting Tab:**
        * Adjust parameters for bead detection (Padding, ROI Radius, Intensity Threshold).
        * Set the R² threshold for qualifying good fits.
        * Choose whether to save PSF analysis plots and use GPU for preparation.
4.  **Run Analysis:** Click the "Run Full Analysis" button.
5.  **View Results:**
    * Processing steps and any errors will appear in the "Log" tab.
    * A summary of deskewing, decorrelation, and PSF fitting results will be shown in the "Summary Results" tab.
    * The "Combined MIP Output" area will display an orthographic projection of the deskewed image. You can adjust its contrast using the sliders and buttons below it.
    * If PSF analysis was run and plots generated, the "PSF FWHM Plot" tab will display the FWHM vs. Position scatter plot.
    * Zoom and pan are available for both the MIP and PSF plot displays.
6.  **Output Files:** Processed images, plots, and parameter notes will be saved in a sub-folder (e.g., `Deskewed_<filename>_angleXX`) created in the same directory as your input file.

## GPU Acceleration

* The plugin attempts to use CuPy for GPU acceleration of demanding tasks like image shearing, rotation, and zooming.
* If CuPy is not installed or a compatible GPU/CUDA environment is not found, the plugin will automatically attempt to fall back to CPU-based processing for most operations. CPU processing will be significantly slower for large datasets.
* **Note on Plugin Environment:** GPU memory availability might be more constrained when running as a plugin compared to a stand-alone application, as the host software also consumes resources. If you encounter GPU out-of-memory errors:
    * Try processing smaller datasets if possible.
    * Increase the number of chunks for GPU operations (e.g., "Shear Z-Chunks", "Zoom/Rot X-Chunks" in the Deskewing tab, or the internal chunking for the final Z-downsampling step).
    * Ensure the "Fallback to CPU" options are enabled to allow processing to complete, albeit more slowly.



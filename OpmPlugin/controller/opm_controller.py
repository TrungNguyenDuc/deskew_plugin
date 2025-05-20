# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:36:02 2025

@author: S233755
"""

# --- COMMON IMPORTS ---
import numpy as np
import cupyx.scipy.ndimage as ndi_cp
import cupy as cp  # CuPy for GPU acceleration
import tifffile
import os
import time
import math
from scipy.ndimage import rotate, zoom, gaussian_filter
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from scipy.fft import fftn, fftshift, ifftn, ifftshift
import scipy.signal.windows as ss
import gc
import threading
import queue
from pathlib import Path
import traceback
from scipy.ndimage import label as scipy_label
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter


import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox # Keep messagebox for controller-level errors if needed by host
import os
import time
import math
import gc
import threading
import queue # Keep for controller if any internal queuing is ever needed, though not used now
from pathlib import Path
import traceback

# Scientific and Image Processing Imports
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi_cp
import tifffile
from scipy.ndimage import rotate as scipy_rotate, zoom as scipy_zoom, gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import label as scipy_label_func # Renamed to avoid conflict
from PIL import Image, ImageOps, ImageTk # ImageTk might be for host display, PIL for loading/saving
import matplotlib
matplotlib.use('Agg') # Essential for non-GUI backend plotting
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from skimage.measure import label as skimage_label_rp, regionprops # For PSF
from skimage.exposure import rescale_intensity
from scipy.fft import fftn, fftshift, ifftn, ifftshift
import scipy.signal.windows as ss

from navigate.controller.sub_controllers.gui import GUIController # Assuming this is from your plugin framework


# ConfocalProjectionController.py
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import os
import time
import math
import gc
import threading
# Removed queue as direct calls/after might be used, or host provides solution
from pathlib import Path
import traceback

# Scientific and Image Processing Imports (already present and mostly fine)
import numpy as np
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi_cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    # Define dummy cp and ndi_cp if needed for CPU-only fallback,
    # or ensure functions gracefully handle CUPY_AVAILABLE being False.
    # For simplicity, processing functions should check CUPY_AVAILABLE.
    print("CuPy not found. GPU acceleration will be unavailable.")
    cp = None # Placeholder
    ndi_cp = None # Placeholder


import tifffile
from scipy.ndimage import rotate as scipy_rotate, zoom as scipy_zoom, gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import label as scipy_label_func
from PIL import Image, ImageOps, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from skimage.measure import label as skimage_label_rp, regionprops
from skimage.exposure import rescale_intensity
from scipy.fft import fftn, fftshift, ifftn, ifftshift
import scipy.signal.windows as ss

# Assuming this is from your plugin framework
from navigate.controller.sub_controllers.gui import GUIController

# Needed imports (should be at the top of your controller file or this section)
import numpy as np
import tifffile
import os # Used by Path for basename, stem
from pathlib import Path
import time
import gc
import traceback
import matplotlib.pyplot as plt # Already imported with Agg backend
from scipy.optimize import curve_fit
from skimage.measure import regionprops as skimage_regionprops
from skimage.exposure import rescale_intensity
# Conditionally import GPU libraries if you want to make GPU optional at a higher level

# >>> START OF COPIED/ADAPTED PROCESSING FUNCTIONS & HELPERS <<<
# All your processing functions (gpu_shear_image, perform_deskewing, etc.)
# should be here. They seem to be largely present in the provided controller.
# Key changes needed within them:
# 1. Ensure all logging uses `_log_process_message(log_adapter, ...)`
# 2. Ensure `_controller_clear_gpu_memory(log_adapter)` is used.
# 3. `show_..._plots` flags should mean "save plot file" and the path returned/logged.
#    The controller will then tell the view to display it.
# 4. `psf_plot_path_queue` related logic in `run_psf_fitting_analysis` should be removed.
#    Instead, `run_psf_fitting_analysis` should return the path to the saved plot (if any).

# --- Pillow Resampling Filter (version compatibility) ---
try:
    LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    try:
        LANCZOS_RESAMPLE = Image.LANCZOS
    except AttributeError:
        LANCZOS_RESAMPLE = Image.ANTIALIAS
        print("Warning: Using Image.ANTIALIAS for resizing.")
        
# --- GPU Memory Management ---
def clear_gpu_memory():
    """Clear CuPy memory to prevent fragmentation and release unused blocks."""
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        # Optional: Synchronize to ensure operations are complete, though free_all_blocks should handle this.
        # cp.cuda.Stream.null.synchronize() 
    except Exception as e:
        # This function should not fail catastrophically itself.
        # If log_queue is available globally or passed, can log here.
        print(f"Minor error during clear_gpu_memory: {e}")



def _controller_clear_gpu_memory(log_adapter_func=None):
    global CUPY_AVAILABLE, cp
    if not CUPY_AVAILABLE or cp is None: return
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except Exception as e:
        message = f"Minor error during _controller_clear_gpu_memory: {e}"
        if log_adapter_func:
            log_adapter_func(f"[{time.strftime('%H:%M:%S')}] [ERROR] {message}")
        else:
            print(message)

def _log_process_message(log_adapter_func, message, level="INFO"):
    formatted_message = f"[{time.strftime('%H:%M:%S')}] [{level}] {message}"
    if log_adapter_func:
        # Schedule the call to the log_adapter in the main GUI thread
        # This assumes log_adapter_func is self.view.log_message which is a Tkinter widget method.
        # If self.view has a 'master' or 'after' method:
        if hasattr(log_adapter_func, '__self__') and hasattr(log_adapter_func.__self__, 'after'): # Check if it's a bound method of a widget
            log_adapter_func.__self__.after(0, lambda fm=formatted_message: log_adapter_func(fm))
        else: # Fallback if direct call or can't schedule easily (less safe for cross-thread)
            try:
                log_adapter_func(formatted_message)
            except Exception as e_log:
                print(f"Fallback Log Failed ({type(e_log).__name__}): {formatted_message}")
                print(f"Original message: {message}")

    else:
        print(formatted_message)


def matlab_round(x):
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)

# --- gpu_shear_image (ensure it uses log_adapter and _controller_clear_gpu_memory) ---
# [This function is large, assume it's correctly adapted as per your provided controller snippet]
# Make sure CUPY_AVAILABLE and cp are checked at the beginning of GPU-specific blocks.
def gpu_shear_image(image_chunk_yxz,
                    original_zsize, cz_ref_0based_full, pixel_shift_z,
                    flip_direction, max_y_offset_full, z_chunk_offset,
                    log_adapter=None):
    global CUPY_AVAILABLE, cp
    if not CUPY_AVAILABLE or cp is None:
        _log_process_message(log_adapter, "gpu_shear_image: CuPy not available, cannot perform GPU shear.", "ERROR")
        raise RuntimeError("CuPy not available for GPU shear") # Or handle fallback differently

    image_gpu, shear_image_gpu = None, None
    # ... (rest of your gpu_shear_image logic, using cp, and _log_process_message)
    # Ensure cp.asarray, cp.zeros etc. are used.
    # --- The rest of the gpu_shear_image function from your controller ---
    if log_adapter:
        _log_process_message(log_adapter, f"gpu_shear_image ENTRY: chunk_offset={z_chunk_offset}, "
                                      f"chunk_shape={image_chunk_yxz.shape if image_chunk_yxz is not None else 'None'}, "
                                      f"orig_Z={original_zsize}, cz_ref={cz_ref_0based_full}, px_shift={pixel_shift_z:.4f}, "
                                      f"flip={flip_direction}, max_y_off={max_y_offset_full}", "DEBUG")
    try:
        if image_chunk_yxz is None or image_chunk_yxz.size == 0:
            if log_adapter: _log_process_message(log_adapter, "GPU Shear: Input chunk is empty or None. Returning calculated empty array.", "INFO")
            expected_y_dim = (image_chunk_yxz.shape[0] if image_chunk_yxz is not None and image_chunk_yxz.ndim > 0 else 0) + 2 * max_y_offset_full
            expected_x_dim = image_chunk_yxz.shape[1] if image_chunk_yxz is not None and image_chunk_yxz.ndim > 1 else 0
            expected_z_dim_chunk = image_chunk_yxz.shape[2] if image_chunk_yxz is not None and image_chunk_yxz.ndim > 2 else 0
            dtype_to_use = image_chunk_yxz.dtype if image_chunk_yxz is not None else np.float32
            if log_adapter: _log_process_message(log_adapter, f"GPU Shear: Empty chunk output shape=({expected_y_dim},{expected_x_dim},{expected_z_dim_chunk}), dtype={dtype_to_use}", "DEBUG")
            return np.zeros((expected_y_dim, expected_x_dim, expected_z_dim_chunk), dtype=dtype_to_use)

        current_chunk_ysize = image_chunk_yxz.shape[0]
        current_chunk_xsize = image_chunk_yxz.shape[1]
        current_chunk_zsize = image_chunk_yxz.shape[2]

        if log_adapter: _log_process_message(log_adapter, f"GPU Shear: cp.asarray on chunk with shape {image_chunk_yxz.shape}, dtype {image_chunk_yxz.dtype}", "DEBUG")
        image_gpu = cp.asarray(image_chunk_yxz)
        padded_ysize = current_chunk_ysize + 2 * max_y_offset_full
        if log_adapter: _log_process_message(log_adapter, f"GPU Shear: cp.zeros for shear_image_gpu with shape ({padded_ysize}, {current_chunk_xsize}, {current_chunk_zsize}), dtype {image_gpu.dtype}", "DEBUG")
        shear_image_gpu = cp.zeros((padded_ysize, current_chunk_xsize, current_chunk_zsize), dtype=image_gpu.dtype)

        for z_local_idx in range(current_chunk_zsize):
            z_global_idx = z_local_idx + z_chunk_offset
            y_offset_float = flip_direction * (z_global_idx - cz_ref_0based_full) * pixel_shift_z
            y_offset = int(matlab_round(y_offset_float))
            y_s_dest = y_offset + max_y_offset_full
            y_e_dest = y_s_dest + current_chunk_ysize
            y_s_dest_c = max(0, y_s_dest)
            y_e_dest_c = min(padded_ysize, y_e_dest)
            y_s_src_c = max(0, -y_s_dest)
            y_e_src_c = current_chunk_ysize - max(0, y_e_dest - padded_ysize)

            if log_adapter and z_local_idx < 5:
                _log_process_message(log_adapter, f"  z_loc={z_local_idx}, z_glob={z_global_idx}, y_off={y_offset} (from {y_offset_float:.2f})", "DEBUG")
                _log_process_message(log_adapter, f"    Dest: y_s_d={y_s_dest}, y_e_d={y_e_dest} -> Clip: y_s_d_c={y_s_dest_c}, y_e_d_c={y_e_dest_c}", "DEBUG")
                _log_process_message(log_adapter, f"    Src:  y_s_s_c={y_s_src_c}, y_e_s_c={y_e_src_c}", "DEBUG")

            if y_s_dest_c < y_e_dest_c and y_s_src_c < y_e_src_c:
                src_h = y_e_src_c - y_s_src_c
                dest_h = y_e_dest_c - y_s_dest_c
                if log_adapter and z_local_idx < 5:
                    _log_process_message(log_adapter, f"      src_h={src_h}, dest_h={dest_h}", "DEBUG")
                if src_h == dest_h and src_h > 0:
                    try:
                        dest_slice_for_assign = shear_image_gpu[y_s_dest_c:y_e_dest_c, :, z_local_idx]
                        src_slice_to_assign = image_gpu[y_s_src_c:y_e_src_c, :, z_local_idx]
                        if log_adapter and z_local_idx < 2:
                            _log_process_message(log_adapter, f"        ASSIGN PRE-CHECK: Dest slice shape: {dest_slice_for_assign.shape}, Src slice shape: {src_slice_to_assign.shape}", "DEBUG")
                            _log_process_message(log_adapter, f"                          Dest dtype: {dest_slice_for_assign.dtype}, Src dtype: {src_slice_to_assign.dtype}", "DEBUG")
                        if dest_slice_for_assign.shape != src_slice_to_assign.shape:
                            err_msg = (f"CRITICAL SHAPE MISMATCH at z_loc={z_local_idx}: "
                                       f"LHS {dest_slice_for_assign.shape} != RHS {src_slice_to_assign.shape}."
                                       f" Indices: dest=({y_s_dest_c}:{y_e_dest_c}), src=({y_s_src_c}:{y_e_src_c})")
                            if log_adapter: _log_process_message(log_adapter, err_msg, "ERROR")
                            raise ValueError(err_msg)
                        shear_image_gpu[y_s_dest_c:y_e_dest_c, :, z_local_idx] = src_slice_to_assign
                    except TypeError as te_assign:
                        if log_adapter:
                            _log_process_message(log_adapter, f"ASSIGNMENT TypeError at z_loc={z_local_idx}: {te_assign}", "CRITICAL")
                        raise
                    except Exception as e_assign:
                        if log_adapter: _log_process_message(log_adapter, f"ASSIGNMENT Error (non-TypeError) at z_loc={z_local_idx}: {type(e_assign).__name__} - {e_assign}", "CRITICAL")
                        raise
                elif log_adapter and src_h <= 0:
                     _log_process_message(log_adapter, f"  z_loc={z_local_idx}: Skipped assignment because src_h ({src_h}) <= 0.", "DEBUG")
                elif log_adapter:
                     _log_process_message(log_adapter, f"  z_loc={z_local_idx}: Skipped assignment because src_h ({src_h}) != dest_h ({dest_h}).", "WARNING")
            elif log_adapter and z_local_idx < 5:
                _log_process_message(log_adapter, f"  z_loc={z_local_idx}: Skipped assignment due to invalid clipped range.", "DEBUG")

        result_cpu = cp.asnumpy(shear_image_gpu)
        if log_adapter: _log_process_message(log_adapter, f"gpu_shear_image EXIT: Successfully processed chunk. Output shape {result_cpu.shape}", "DEBUG")
        return result_cpu
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError, AttributeError, TypeError, ValueError) as e: # type: ignore
        if log_adapter:
            _log_process_message(log_adapter, f"GPU Shear Main Error (chunk_offset={z_chunk_offset}): {type(e).__name__} - {e}", "ERROR")
            _log_process_message(log_adapter, traceback.format_exc(), "ERROR")
        raise
    finally:
        if image_gpu is not None: del image_gpu
        if shear_image_gpu is not None: del shear_image_gpu
        _controller_clear_gpu_memory(log_adapter)


# --- gpu_max_projection (ensure it uses log_adapter and _controller_clear_gpu_memory) ---
# [This function is large, assume it's correctly adapted as per your provided controller snippet]
def gpu_max_projection(image_stack, axis, log_adapter=None):
    global CUPY_AVAILABLE, cp
    if not CUPY_AVAILABLE or cp is None:
        _log_process_message(log_adapter, "gpu_max_projection: CuPy not available.", "ERROR")
        raise RuntimeError("CuPy not available for GPU max projection")

    image_gpu, mip_gpu = None, None
    try:
        image_gpu = cp.asarray(image_stack)
        mip_gpu = cp.max(image_gpu, axis=axis)
        result = cp.asnumpy(mip_gpu)
        return result
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as e: # type: ignore
        if log_adapter: _log_process_message(log_adapter, f"GPU MIP: CUDA Error - {type(e).__name__}: {e}. Re-raising.", "ERROR")
        raise
    finally:
        if image_gpu is not None: del image_gpu
        if mip_gpu is not None: del mip_gpu
        _controller_clear_gpu_memory(log_adapter)

# --- process_in_chunks_gpu (ensure it uses log_adapter and _controller_clear_gpu_memory) ---
# [This function is large, assume it's correctly adapted as per your provided controller snippet]
def process_in_chunks_gpu(input_array_cpu, gpu_function, chunk_axis, num_chunks,
                          log_adapter=None, fallback_to_cpu_func=None, **kwargs):
    global CUPY_AVAILABLE, cp, ndi_cp
    if not CUPY_AVAILABLE or cp is None or ndi_cp is None: # Check all required CuPy components
        _log_process_message(log_adapter, "process_in_chunks_gpu: CuPy not available. Attempting CPU fallback if provided.", "WARNING")
        if fallback_to_cpu_func:
            return fallback_to_cpu_func(input_array_cpu, **kwargs), False # Signal CPU was used
        else:
            raise RuntimeError("CuPy not available and no CPU fallback for chunked processing.")


    if input_array_cpu is None:
        if log_adapter: _log_process_message(log_adapter, "Chunked GPU: Input array is None. Cannot process.", "ERROR")
        raise ValueError("Input array to process_in_chunks_gpu cannot be None.")
    if input_array_cpu.size == 0:
        if log_adapter: _log_process_message(log_adapter, "Chunked GPU: Input array is empty. Returning copy.", "INFO")
        if fallback_to_cpu_func:
            try: return fallback_to_cpu_func(input_array_cpu.copy(), **kwargs), True # True because CPU func "succeeded" on empty
            except: return input_array_cpu.copy(), True
        return input_array_cpu.copy(), True

    if num_chunks <= 0: num_chunks = 1
    array_shape = input_array_cpu.shape
    if chunk_axis >= len(array_shape) or array_shape[chunk_axis] == 0:
        num_chunks = 1
    min_slices_per_chunk = 1 # You might want to make this configurable
    if num_chunks > 1 and array_shape[chunk_axis] / num_chunks < min_slices_per_chunk:
        num_chunks = max(1, int(array_shape[chunk_axis] / min_slices_per_chunk))
        if log_adapter: _log_process_message(log_adapter, f"Chunked GPU: Adjusted num_chunks to {num_chunks}.", "INFO")

    processed_chunks_cpu, all_gpu_chunks_succeeded = [], True
    if log_adapter: _log_process_message(log_adapter, f"Chunked GPU: Using {num_chunks} chunk(s) along axis {chunk_axis} for {gpu_function.__name__}.", "DEBUG")

    chunk_indices = np.array_split(np.arange(array_shape[chunk_axis]), num_chunks)
    for i, indices_in_chunk_axis in enumerate(chunk_indices):
        if not indices_in_chunk_axis.size: continue
        chunk_start, chunk_end = indices_in_chunk_axis[0], indices_in_chunk_axis[-1] + 1
        slicer = [slice(None)] * input_array_cpu.ndim; slicer[chunk_axis] = slice(chunk_start, chunk_end)
        current_chunk_cpu = input_array_cpu[tuple(slicer)] # type: ignore
        if log_adapter: _log_process_message(log_adapter, f"  Chunk {i+1}/{num_chunks} (axis {chunk_axis}: {chunk_start}-{chunk_end-1}) -> GPU", "DEBUG")
        chunk_gpu, processed_chunk_gpu = None, None
        try:
            if current_chunk_cpu.size == 0:
                if log_adapter: _log_process_message(log_adapter, f"  Chunk {i+1} is empty, processing accordingly.", "INFO")
                chunk_gpu = cp.asarray(current_chunk_cpu)
                processed_chunk_gpu = gpu_function(chunk_gpu, **kwargs) # gpu_function is e.g., ndi_cp.zoom
            else:
                chunk_gpu = cp.asarray(current_chunk_cpu)
                processed_chunk_gpu = gpu_function(chunk_gpu, **kwargs)
            processed_chunks_cpu.append(cp.asnumpy(processed_chunk_gpu))
        except Exception as e_gpu_chunk:
            if log_adapter: _log_process_message(log_adapter, f"  Chunk {i+1} GPU FAILED: {type(e_gpu_chunk).__name__} - {e_gpu_chunk}", "ERROR")
            all_gpu_chunks_succeeded = False; break
        finally:
            if chunk_gpu is not None: del chunk_gpu
            if processed_chunk_gpu is not None: del processed_chunk_gpu
            _controller_clear_gpu_memory(log_adapter)

    if all_gpu_chunks_succeeded and processed_chunks_cpu:
        if log_adapter: _log_process_message(log_adapter, "Chunked GPU: All chunks GPU OK. Stitching...", "DEBUG")
        try:
            if len(processed_chunks_cpu) > 1 and gpu_function is ndi_cp.rotate and kwargs.get('reshape', False): # type: ignore
                ref_shape = list(processed_chunks_cpu[0].shape)
                del ref_shape[chunk_axis] # type: ignore
                for ch_idx in range(1, len(processed_chunks_cpu)):
                    current_ch_shape = list(processed_chunks_cpu[ch_idx].shape)
                    del current_ch_shape[chunk_axis] # type: ignore
                    if current_ch_shape != ref_shape:
                        if log_adapter: _log_process_message(log_adapter, f"Chunked GPU: Inconsistent shapes for rotated chunks. Fallback.", "ERROR")
                        all_gpu_chunks_succeeded = False; break
            if not all_gpu_chunks_succeeded: pass # Will fall to CPU fallback section
            elif not processed_chunks_cpu: # No chunks processed
                 if log_adapter: _log_process_message(log_adapter, "Chunked GPU: No chunks to concatenate. Returning copy of input.", "INFO")
                 return input_array_cpu.copy(), True # GPU "succeeded" as no work needed
            else:
                result_array_cpu = np.concatenate(processed_chunks_cpu, axis=chunk_axis) # type: ignore
                if log_adapter: _log_process_message(log_adapter, f"Chunked GPU: Stitched shape: {result_array_cpu.shape}", "DEBUG")
                return result_array_cpu, True # True indicates GPU success
        except ValueError as e_concat:
            if log_adapter: _log_process_message(log_adapter, f"Chunked GPU: Concat FAILED: {e_concat}. GPU considered failed.", "CRITICAL")
            all_gpu_chunks_succeeded = False

    # Fallback if GPU failed or concatenation failed
    if fallback_to_cpu_func:
        if log_adapter: _log_process_message(log_adapter, "Chunked GPU: Falling back to CPU for entire operation.", "WARNING")
        try:
            return fallback_to_cpu_func(input_array_cpu, **kwargs), False # False indicates CPU was used
        except Exception as e_cpu_fallback:
            if log_adapter: _log_process_message(log_adapter, f"Chunked GPU: CPU fallback FAILED: {type(e_cpu_fallback).__name__} - {e_cpu_fallback}", "ERROR")
            raise
    else:
        # This means GPU processing failed and there was no fallback.
        _log_process_message(log_adapter, "Chunked GPU processing failed and no CPU fallback provided.", "ERROR")
        raise RuntimeError("Chunked GPU processing failed and no CPU fallback provided.")


# --- generate_and_save_mips_for_gui (ensure it uses log_adapter) ---
# [This function is large, assume it's correctly adapted as per your provided controller snippet]
# It should return the path to the *combined ortho MIP TIFF* for the view to display.
def generate_and_save_mips_for_gui(image_stack_yxz, output_folder_str, base_filename,
                                   show_matplotlib_plots_flag, log_adapter=None):
    # ... (Your existing generate_and_save_mips_for_gui logic from the controller)
    # Ensure all plt.show() are removed, and plots are saved if flag is True.
    # Return combined_mip_tiff_path_for_gui_display
    # --- The rest of the generate_and_save_mips_for_gui function from your controller ---
    _log_process_message(log_adapter, "--- Generating and Saving Maximum Intensity Projections (MIPs) ---", "INFO")
    output_folder = Path(output_folder_str)
    base_filename_noext = Path(base_filename).stem
    combined_mip_tiff_path_for_gui_display = None

    if image_stack_yxz is None or image_stack_yxz.ndim != 3 or image_stack_yxz.size == 0:
        _log_process_message(log_adapter, "Cannot generate MIPs. Input stack not 3D or is empty.", "WARNING")
        return None # Return None if no image can be generated

    y_prime_dim, x_dim, z_prime_dim = image_stack_yxz.shape
    _log_process_message(log_adapter, f"Input stack for MIPs (Y',X,Z'): {(y_prime_dim, x_dim, z_prime_dim)}, dtype: {image_stack_yxz.dtype}", "DEBUG")
    canvas_dtype = image_stack_yxz.dtype
    display_min_val, display_max_val = np.inf, -np.inf # Initialize correctly

    # --- Calculate MIPs ---
    mip_xy, mip_xz_orig, mip_yz_orig = None, None, None
    try:
        mip_xy = np.max(image_stack_yxz, axis=2) # Shape: (Y', X)
        if mip_xy.size > 0:
            display_min_val = min(display_min_val, np.min(mip_xy))
            display_max_val = max(display_max_val, np.max(mip_xy))
    except Exception as e: _log_process_message(log_adapter, f"Error XY MIP: {e}", "ERROR")

    try:
        mip_xz_orig = np.max(image_stack_yxz, axis=0) # Shape: (X, Z')
        if mip_xz_orig.size > 0:
            display_min_val = min(display_min_val, np.min(mip_xz_orig))
            display_max_val = max(display_max_val, np.max(mip_xz_orig))
    except Exception as e: _log_process_message(log_adapter, f"Error XZ MIP: {e}", "ERROR")

    try:
        mip_yz_orig = np.max(image_stack_yxz, axis=1) # Shape: (Y', Z')
        if mip_yz_orig.size > 0:
            display_min_val = min(display_min_val, np.min(mip_yz_orig))
            display_max_val = max(display_max_val, np.max(mip_yz_orig))
    except Exception as e: _log_process_message(log_adapter, f"Error YZ MIP: {e}", "ERROR")

    # --- Prepare MIPs for Display Orientation ---
    mip_xz_disp = mip_xz_orig.T if mip_xz_orig is not None else None # Shape: (Z', X)
    mip_yz_disp = mip_yz_orig.T if mip_yz_orig is not None else None # Shape: (Z', Y')

    _log_process_message(log_adapter, "Saving individual MIP TIFFs (original orientation)...", "DEBUG")
    for mip_data, plane_name in [(mip_xy, "XY"), (mip_xz_orig, "XZ_native"), (mip_yz_orig, "YZ_native")]:
        if mip_data is not None and mip_data.size > 0:
            save_path = output_folder / f"{base_filename_noext}_deskew_mip_{plane_name}.tif"
            try:
                save_mip = mip_data
                # Basic type casting for saving, adapt from original if more complex needed
                if not np.issubdtype(save_mip.dtype, np.unsignedinteger):
                    if np.issubdtype(save_mip.dtype, np.floating):
                        save_mip = rescale_intensity(save_mip, out_range=(0, 65535)).astype(np.uint16)
                    elif save_mip.min() >= 0:
                        save_mip = np.clip(save_mip, 0, 65535).astype(np.uint16)
                    else:
                        save_mip = save_mip.astype(np.float32) # Save as float if complex or negative
                elif save_mip.dtype != np.uint16 : # Already uint, but not uint16
                    save_mip = np.clip(save_mip, 0, 65535).astype(np.uint16)

                tifffile.imwrite(save_path, save_mip, imagej=True)
                _log_process_message(log_adapter, f"Saved {plane_name} MIP to: {save_path}", "DEBUG")
            except Exception as e_tif:
                _log_process_message(log_adapter, f"Error saving {plane_name} MIP: {e_tif}", "ERROR")

    if any(x is None for x in [mip_xy, mip_xz_disp, mip_yz_disp]):
        _log_process_message(log_adapter, "Skipping combined MIP: one or more display MIPs are invalid/empty.", "WARNING")
        return None

    # --- Dimensions for Combined Canvas ---
    h_xy, w_xy = mip_xy.shape
    h_xz_d, w_xz_d = mip_xz_disp.shape
    h_yz_d, w_yz_d = mip_yz_disp.shape
    # ... (Your dimension checks and warnings using _log_process_message) ...

    total_width = max(w_xz_d + w_yz_d, w_xy)
    total_height = h_xz_d + h_xy

    bg_value = 0 # Simplified background value
    if display_min_val == np.inf: display_min_val, display_max_val = 0, 1 # Handle all empty case

    combined_mip_array = np.full((total_height, total_width), fill_value=bg_value, dtype=canvas_dtype)
    try:
        combined_mip_array[0:h_xz_d, 0:w_xz_d] = mip_xz_disp
        if h_yz_d <= h_xz_d :
            combined_mip_array[0:h_yz_d, w_xz_d : w_xz_d + w_yz_d] = mip_yz_disp
        else:
            _log_process_message(log_adapter, f"Warning: YZ_disp Z' height ({h_yz_d}) > XZ_disp Z' height ({h_xz_d}). Cropping YZ_disp for combined view.", "WARNING")
            combined_mip_array[0:h_xz_d, w_xz_d : w_xz_d + w_yz_d] = mip_yz_disp[:h_xz_d, :]
        combined_mip_array[h_xz_d : h_xz_d + h_xy, 0:w_xy] = mip_xy
    except Exception as e_paste:
        _log_process_message(log_adapter, f"Error pasting MIPs onto canvas: {e_paste}", "ERROR")
        _log_process_message(log_adapter, traceback.format_exc(), "DEBUG")
        return None

    # Save combined MIP TIFF for GUI display
    combined_mip_tiff_path_for_gui_display = output_folder / f"{base_filename_noext}_deskew_mip_COMBINED_ORTHO.tif"
    try:
        save_array_for_display = combined_mip_array
        # Adapt type casting from original if needed
        if not np.issubdtype(save_array_for_display.dtype, np.floating):
            if save_array_for_display.dtype != np.uint16:
                try:
                    if save_array_for_display.min() >= 0 and save_array_for_display.max() <= 65535:
                        save_array_for_display = save_array_for_display.astype(np.uint16)
                    else:
                        save_array_for_display = rescale_intensity(save_array_for_display, out_range=(0,65535)).astype(np.uint16)
                except Exception: save_array_for_display = save_array_for_display.astype(np.float32)

        tifffile.imwrite(combined_mip_tiff_path_for_gui_display, save_array_for_display, imagej=True)
        _log_process_message(log_adapter, f"Saved COMBINED Ortho MIP TIFF to: {combined_mip_tiff_path_for_gui_display}", "INFO")
    except Exception as e_save_comb:
        _log_process_message(log_adapter, f"Error saving combined ortho MIP TIFF: {e_save_comb}", "ERROR")
        combined_mip_tiff_path_for_gui_display = None # Don't return bad path

    # --- Matplotlib Visualization (Save PNG) ---
    if show_matplotlib_plots_flag and combined_mip_array is not None and combined_mip_array.size > 0:
        # ... (Your Matplotlib plotting logic from original, ensuring plt.close() and saving to file) ...
        # Example:
        plt.style.use('default')
        dpi_target = 100
        fig_width_inches = max(6.0, min(15.0, total_width / dpi_target))
        fig_height_inches = max(5.0, min(12.0, (total_height / dpi_target) + 1.0))

        fig_comb, ax_comb = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
        current_display_vmin_plot = display_min_val if display_min_val != np.inf else 0
        current_display_vmax_plot = display_max_val if display_max_val != -np.inf else (current_display_vmin_plot + 1)
        if current_display_vmax_plot <= current_display_vmin_plot:
            current_display_vmax_plot = current_display_vmin_plot + 1

        im_comb = ax_comb.imshow(combined_mip_array, cmap='gray',
                                 vmin=current_display_vmin_plot, vmax=current_display_vmax_plot,
                                 interpolation='nearest', aspect='equal', origin='upper')
        ax_comb.set_title(f'Combined Ortho MIPs: {base_filename_noext}', fontsize=10, pad=15)
        ax_comb.axis('off')
        # ... (add text labels, lines, colorbar as in original) ...
        cbar = fig_comb.colorbar(im_comb, ax=ax_comb, shrink=0.7, aspect=20, pad=0.02, location='right') # type: ignore
        cbar.set_label('Max Intensity', size=9); cbar.ax.tick_params(labelsize=8) # type: ignore
        fig_comb.tight_layout(rect=[0, 0, 0.95, 0.95])


        plot_output_dir_viz = output_folder / "analysis_plots"
        plot_output_dir_viz.mkdir(parents=True, exist_ok=True)
        save_path_combined_png = plot_output_dir_viz / f"{base_filename_noext}_deskew_mip_COMBINED_ORTHO_viz.png"
        try:
            plt.savefig(save_path_combined_png, dpi=150, bbox_inches='tight')
            _log_process_message(log_adapter, f"Saved COMBINED Ortho MIP PNG (Viz) to: {save_path_combined_png}", "INFO")
        except Exception as e_save_png:
             _log_process_message(log_adapter, f"Error saving combined ortho MIP PNG: {e_save_png}", "ERROR")
        finally:
            plt.close(fig_comb)

    return str(combined_mip_tiff_path_for_gui_display) if combined_mip_tiff_path_for_gui_display else None


# --- perform_deskewing (ensure it uses log_adapter and all GPU flags) ---
# [This function is large, assume it's correctly adapted as per your provided controller snippet]
# It should return all necessary values:
# output_path_top_deskewed_tiff_str, final_dx_eff_um, final_dz_eff_um,
# output_folder_top_str, processed_file_name_str, combined_mip_display_path_str
# --- The rest of the perform_deskewing function from your controller ---
def perform_deskewing(full_file_path: str, dx_um: float, dz_um: float, angle_deg: float, flip_direction: int,
                      save_intermediate_shear: bool, show_deskew_plots: bool, # show_deskew_plots means save plots
                      log_adapter=None,
                      # GPU/Chunking parameters (now passed in)
                      num_z_chunks_for_gpu_shear: int = 4,
                      min_z_slices_per_chunk_gpu_shear: int = 32, # Added for consistency
                      gpu_shear_fallback_to_cpu_process: bool = True,
                      num_x_chunks_for_gpu_zoom_rotate: int = 4,
                      gpu_zoom_rotate_fallback_to_cpu: bool = True,
                      # Smoothing parameters
                      apply_post_shear_smoothing: bool = False,
                      smoothing_sigma_yc: float = 0.7,
                      smoothing_sigma_x: float = 0.0,  # Usually 0
                      smoothing_sigma_zc: float = 0.0,  # Usually 0
                      save_final_deskew: bool = True
                      ):
    _log_process_message(log_adapter, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", "INFO")
    _log_process_message(log_adapter, "%           Deskew & Rotate Light-Sheet Data (New Plugin Version)                   %", "INFO")
    _log_process_message(log_adapter, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", "INFO")

    start_time_total = time.time()
    # Initialize return values to ensure they are always defined
    output_path_top_deskewed_tiff_str=None
    final_dx_eff_um=dx_um # Default to input if something goes wrong early
    final_dz_eff_um=dx_um # Default to input dx if scaling fails (common isotropic target)
    output_folder_top_str=None
    processed_file_name_str= Path(full_file_path).name # Default
    combined_mip_display_path_str=None
    zsize_full = 0 # Initialize in case of early exit

    # Intermediate data variables (ensure they are defined for the finally block)
    image_stack=None; shear_image=None; scaled_shear_image_cpu=None;
    rot_top_shear_image = None; final_image_for_mips_and_save=None; final_image_to_save_zyx=None

    try:
        full_path_obj = Path(full_file_path)
        file_path_obj, actual_file_name_str, actual_file_stem_str = full_path_obj.parent, full_path_obj.name, full_path_obj.stem
        processed_file_name_str = actual_file_name_str # Will be output filename if saving
        _log_process_message(log_adapter, f'Processing file for deskewing: {full_file_path}', "INFO")

        # Create output folder based on input file's directory
        top_output_folder_name = f'Deskewed_{actual_file_stem_str}_angle{angle_deg}' # Example name
        output_folder_top = file_path_obj / top_output_folder_name
        output_folder_top.mkdir(parents=True, exist_ok=True)
        output_folder_top_str = str(output_folder_top)

        _log_process_message(log_adapter, "\n--- Reading and Preparing Image ---", "INFO"); tic_read = time.time()
        with tifffile.TiffFile(full_file_path) as tif: image_stack_raw = tif.asarray()

        # Transposition logic (copied from your original, ensure it's what you need for plugin context)
        if image_stack_raw.ndim < 3: raise ValueError("Input image must be at least 3D.")
        if image_stack_raw.ndim > 3:
            if image_stack_raw.shape[0] == 1 and image_stack_raw.ndim == 4: # Squeeze if first dim is singleton (e.g. time)
                image_stack_raw = np.squeeze(image_stack_raw, axis=0)
            if image_stack_raw.ndim > 3: # If still >3D, take first volume
                _log_process_message(log_adapter, f"Warning: Input is >3D (shape: {image_stack_raw.shape}). Using first 3D volume [0,...].", "WARNING")
                image_stack_raw = image_stack_raw[0,...]
        if image_stack_raw.ndim != 3: raise ValueError(f"Could not reduce input to 3D. Current ndim: {image_stack_raw.ndim}")

        s_in = image_stack_raw.shape
        # Heuristic for YXZ vs ZYX based on smallest dimension (often Z for raw data)
        # This assumes the processing pipeline expects YXZ internally
        z_threshold_factor = 0.8
        if s_in[0] < z_threshold_factor * s_in[1] and s_in[0] < z_threshold_factor * s_in[2]: # Smallest is dim 0 (likely ZYX)
            image_stack = image_stack_raw.transpose(1, 2, 0) # ZYX to YXZ
            _log_process_message(log_adapter, f"Input assumed ZYX {s_in}, transposed to YXZ {image_stack.shape}.", "DEBUG")
        elif s_in[2] < z_threshold_factor * s_in[0] and s_in[2] < z_threshold_factor * s_in[1]: # Smallest is dim 2 (likely YXZ)
            image_stack = image_stack_raw # Already YXZ
            _log_process_message(log_adapter, f"Input assumed YXZ {s_in}, no transpose needed.", "DEBUG")
        else: # Ambiguous or cubic, default to assuming YXZ and log warning
            _log_process_message(log_adapter, f"Input shape {s_in} is ambiguous or near-cubic. Assuming YXZ directly. Check data orientation if results are incorrect.", "WARNING")
            image_stack = image_stack_raw

        del image_stack_raw; gc.collect()
        toc_read = time.time()
        _log_process_message(log_adapter, f"Image read and prepared in {toc_read-tic_read:.2f}s. Internal YXZ shape: {image_stack.shape}, dtype: {image_stack.dtype}", "INFO")
        # Define output data type (e.g., for saving)
        out_dtype = np.uint16 # Or can be adaptive based on input dtype / user settings

        # --- Shear Transformation ---
        ysize, xsize, zsize_full = image_stack.shape
        new_dz_shear = dz_um * math.cos(math.radians(angle_deg)) # Effective Z-step along shear axis
        _log_process_message(log_adapter, "\n--- Applying Shear Transformation ---", "INFO"); tic_s = time.time()

        cz_ref_shear = math.floor(zsize_full / 2) # Center Z reference for shear
        pixel_shift_z_shear = new_dz_shear / dx_um if dx_um != 0 else 0 # Y-shift per Z-slice
        max_y_offset_shear = 0
        if zsize_full > 0 : # Calculate only if there are Z slices
             max_y_offset_shear = int(max(
                abs(matlab_round(flip_direction * (0 - cz_ref_shear) * pixel_shift_z_shear)),
                abs(matlab_round(flip_direction * ((zsize_full - 1) - cz_ref_shear) * pixel_shift_z_shear))
            ))
        padded_y_dim_shear = ysize + 2 * max_y_offset_shear
        _log_process_message(log_adapter, f"Shear params: cz_ref={cz_ref_shear}, px_shift_z={pixel_shift_z_shear:.4f}, max_y_offset={max_y_offset_shear}, padded_Y_dim={padded_y_dim_shear}", "DEBUG")

        gpu_shear_successful = False
        if zsize_full == 0: # Handle empty Z stack
            shear_image = np.zeros((padded_y_dim_shear, xsize, 0), dtype=image_stack.dtype)
            gpu_shear_successful = True # No actual GPU work, but "successful" for this case
        elif CUPY_AVAILABLE and cp is not None:
            effective_z_chunks_shear = num_z_chunks_for_gpu_shear
            if effective_z_chunks_shear > 0 and zsize_full / effective_z_chunks_shear < min_z_slices_per_chunk_gpu_shear:
                effective_z_chunks_shear = max(1, int(zsize_full / min_z_slices_per_chunk_gpu_shear))
            if effective_z_chunks_shear == 0 : effective_z_chunks_shear = 1
            _log_process_message(log_adapter, f"Attempting GPU shear with {effective_z_chunks_shear} Z-chunk(s).", "DEBUG")

            sheared_chunks_cpu_list = []
            all_gpu_shear_chunks_ok = True
            z_indices_for_chunks = np.array_split(np.arange(zsize_full), effective_z_chunks_shear)

            for i, chunk_z_indices in enumerate(z_indices_for_chunks):
                if not chunk_z_indices.size: continue
                z_start_chunk, z_end_chunk = chunk_z_indices[0], chunk_z_indices[-1] + 1
                _log_process_message(log_adapter, f"  Processing GPU Shear Z-chunk {i+1}/{effective_z_chunks_shear} (Z slices: {z_start_chunk}-{z_end_chunk-1})...", "DEBUG")
                try:
                    chunk_data_to_shear = image_stack[:, :, z_start_chunk:z_end_chunk]
                    sheared_chunk_cpu = gpu_shear_image(
                        chunk_data_to_shear, zsize_full, cz_ref_shear, pixel_shift_z_shear,
                        flip_direction, max_y_offset_shear, z_start_chunk, log_adapter
                    )
                    sheared_chunks_cpu_list.append(sheared_chunk_cpu)
                except Exception as e_gpu_chunk_shear:
                    _log_process_message(log_adapter, f"  GPU shear chunk {i+1} FAILED: {type(e_gpu_chunk_shear).__name__} - {e_gpu_chunk_shear}", "ERROR")
                    all_gpu_shear_chunks_ok = False
                    if not gpu_shear_fallback_to_cpu_process:
                        raise RuntimeError("GPU shear chunk failed, and no CPU fallback allowed.") from e_gpu_chunk_shear
                    break # Break from chunk loop to go to CPU fallback

            if all_gpu_shear_chunks_ok and sheared_chunks_cpu_list:
                try:
                    shear_image = np.concatenate(sheared_chunks_cpu_list, axis=2)
                    gpu_shear_successful = True
                    _log_process_message(log_adapter, "GPU shear completed successfully.", "INFO")
                except ValueError as e_concat_shear:
                    _log_process_message(log_adapter, f"Concatenation of sheared GPU chunks FAILED: {e_concat_shear}. Will attempt CPU fallback if enabled.", "CRITICAL")
                    gpu_shear_successful = False # Force CPU fallback
            elif not all_gpu_shear_chunks_ok: # GPU chunk failed
                gpu_shear_successful = False # Ensure it's false for CPU fallback logic
            else: # No chunks (e.g. zsize_full was small, or already handled if effective_z_chunks_shear=0 led here)
                _log_process_message(log_adapter, "No GPU shear chunks processed (this may be normal for small Z).", "DEBUG")
                # This path might mean we need to go to CPU if list is empty but should not be

            del sheared_chunks_cpu_list; gc.collect()
        else: # CUPY_AVAILABLE is False or cp is None
            _log_process_message(log_adapter, "CuPy not available. GPU shear will be skipped.", "INFO")
            # gpu_shear_successful remains False

        if not gpu_shear_successful:
            if not gpu_shear_fallback_to_cpu_process and zsize_full > 0:
                raise RuntimeError("GPU shear failed or was skipped, and CPU fallback is not allowed.")
            _log_process_message(log_adapter, "Performing shear transformation on CPU.", "WARNING")
            shear_image = np.zeros((padded_y_dim_shear, xsize, zsize_full), dtype=image_stack.dtype)
            if zsize_full > 0: # Standard CPU shear loop
                for z_idx in range(zsize_full):
                    y_offset_cpu = int(matlab_round(flip_direction * (z_idx - cz_ref_shear) * pixel_shift_z_shear))
                    y_s_dest_cpu = y_offset_cpu + max_y_offset_shear
                    y_e_dest_cpu = y_s_dest_cpu + ysize
                    # Clipping logic for source and destination
                    y_s_dest_clipped = max(0, y_s_dest_cpu)
                    y_e_dest_clipped = min(padded_y_dim_shear, y_e_dest_cpu)
                    y_s_src_clipped = max(0, -y_s_dest_cpu)
                    y_e_src_clipped = ysize - max(0, y_e_dest_cpu - padded_y_dim_shear)

                    if y_s_dest_clipped < y_e_dest_clipped and y_s_src_clipped < y_e_src_clipped and \
                       (y_e_src_clipped - y_s_src_clipped) == (y_e_dest_clipped - y_s_dest_clipped) > 0:
                        shear_image[y_s_dest_clipped:y_e_dest_clipped, :, z_idx] = \
                            image_stack[y_s_src_clipped:y_e_src_clipped, :, z_idx]
            _log_process_message(log_adapter, "CPU shear completed.", "INFO")

        del image_stack; image_stack = None; gc.collect()
        if shear_image is None: raise RuntimeError("Shear image is None after shear step. This should not happen.")
        toc_s = time.time(); _log_process_message(log_adapter, f"Shear transformation done in {toc_s-tic_s:.2f}s. Sheared image shape: {shear_image.shape}", "INFO")

        if save_intermediate_shear:
            # Check size before saving (e.g. < 4GB)
            if shear_image.nbytes / (1024**3) < 4.0:
                _log_process_message(log_adapter, "Saving intermediate sheared image...", "INFO")
                intermediate_shear_path = output_folder_top / f"INTERMEDIATE_{actual_file_stem_str}_sheared.tif"
                # Prepare for saving (dtype conversion, transpose)
                img_to_save_intermediate = shear_image
                if img_to_save_intermediate.dtype != out_dtype: # Example: np.uint16
                    img_to_save_intermediate = np.clip(img_to_save_intermediate.astype(np.float32), # Go via float for robust clip
                                                       np.iinfo(out_dtype).min, np.iinfo(out_dtype).max).astype(out_dtype)
                tifffile.imwrite(str(intermediate_shear_path), img_to_save_intermediate.transpose(2,0,1), # ZYX
                                 imagej=True, metadata={'axes': 'ZYX'},
                                 bigtiff=(img_to_save_intermediate.nbytes / (1024**3) > 3.9))
                _log_process_message(log_adapter, f"Intermediate sheared image saved to: {intermediate_shear_path}", "INFO")
            else:
                _log_process_message(log_adapter, f"Intermediate sheared image is large ({shear_image.nbytes/(1024**3):.2f}GB), skipping save.", "WARNING")


        # --- Rotation to Top View ---
        _log_process_message(log_adapter, "\n--- Rotating Sheared Image to Top View ---", "INFO"); tic_rot_s = time.time()
        # Z-scaling factor for rotation to make voxels isotropic in the new Z' (original Y-shear) dimension
        scale_factor_z_rot = abs(dz_um * math.sin(math.radians(angle_deg)) / dx_um) if dx_um != 0 else 1.0
        _log_process_message(log_adapter, f"Rotation params: Z-scaling factor for an isotropic Z' = {scale_factor_z_rot:.4f}", "DEBUG")

        _log_process_message(log_adapter, "Applying Z-scaling (Zoom) to sheared image...", "DEBUG")
        zoom_params = {'zoom': (1.0, 1.0, scale_factor_z_rot), 'order': 1, 'mode': 'constant', 'cval': 0.0, 'prefilter': True}
        # process_in_chunks_gpu expects (input_array_cpu, gpu_function, chunk_axis, num_chunks, ...)
        # gpu_function will be ndi_cp.zoom, fallback is scipy_zoom
        scaled_shear_image_cpu, zoom_gpu_used = process_in_chunks_gpu(
            shear_image, ndi_cp.zoom if CUPY_AVAILABLE and ndi_cp else scipy_zoom, # Pass the function itself
            1, # Chunk along X axis (axis 1 of YXZ')
            num_x_chunks_for_gpu_zoom_rotate,
            log_adapter=log_adapter,
            fallback_to_cpu_func=scipy_zoom if gpu_zoom_rotate_fallback_to_cpu else None,
            **zoom_params
        )
        zoom_source = "GPU (chunked)" if zoom_gpu_used and CUPY_AVAILABLE else "CPU"
        _log_process_message(log_adapter, f"Z-scaling (Zoom) completed using {zoom_source}. Scaled shape: {scaled_shear_image_cpu.shape}", "INFO")
        del shear_image; shear_image = None; gc.collect()
        if scaled_shear_image_cpu is None or scaled_shear_image_cpu.ndim != 3:
            # Allow empty Z if original Z was empty and scaling resulted in Z=0
            if not (scaled_shear_image_cpu is not None and scaled_shear_image_cpu.ndim==3 and scaled_shear_image_cpu.shape[2]==0 and zsize_full==0):
                raise RuntimeError("Z-scaled image is invalid (None or not 3D).")
        if scaled_shear_image_cpu.size > 0 and min(scaled_shear_image_cpu.shape) <= 0 : # If not empty, dimensions must be positive
             raise RuntimeError(f"Z-scaled image has a zero dimension: {scaled_shear_image_cpu.shape} but is not empty.")


        _log_process_message(log_adapter, "Calculating Bounding Box for rotation...", "DEBUG")
        min_row_bbox, max_row_bbox, min_col_bbox, max_col_bbox = 0, 0, 0, 0
        rotation_angle_deskew = -1 * flip_direction * angle_deg # Angle to rotate around X-axis (axis 1 of Y'XZ')

        if scaled_shear_image_cpu.size > 0:
            # MIP along X (axis 1 of Y'XZ') to get Y'Z' plane for BBox calculation
            scaled_mip_yz_for_bbox_cpu = None
            try:
                if CUPY_AVAILABLE and cp is not None:
                    scaled_mip_yz_for_bbox_cpu = gpu_max_projection(scaled_shear_image_cpu, axis=1, log_adapter=log_adapter)
                    _log_process_message(log_adapter, "GPU MIP for BBox calculation successful.", "DEBUG")
                else: raise RuntimeError("CuPy not available for MIP") # Force fallback
            except Exception as e_gpu_mip_bbox:
                _log_process_message(log_adapter, f"GPU MIP for BBox failed ({type(e_gpu_mip_bbox).__name__}). Falling back to CPU for MIP.", "WARNING")
                scaled_mip_yz_for_bbox_cpu = np.max(scaled_shear_image_cpu, axis=1)

            if scaled_mip_yz_for_bbox_cpu is None: raise RuntimeError("MIP for BBox is None.")
            # Rotate this 2D MIP (Y'Z') to find the bounds of the content
            # scipy_rotate axes are (row_axis_idx, col_axis_idx) from the input 2D array's perspective.
            # scaled_mip_yz_for_bbox_cpu has axes (Y', Z'). Rotation is around an axis perpendicular to this plane.
            rotated_mip_for_bbox = scipy_rotate(scaled_mip_yz_for_bbox_cpu, angle=rotation_angle_deskew,
                                                reshape=True, order=1, mode='constant', cval=0, prefilter=True)
            del scaled_mip_yz_for_bbox_cpu; gc.collect()

            # Find content bounds on the reshaped rotated MIP
            # These are row and column indices on the *rotated 2D MIP*
            min_row_bbox, max_row_bbox = 0, rotated_mip_for_bbox.shape[0]
            min_col_bbox, max_col_bbox = 0, rotated_mip_for_bbox.shape[1]
            if np.any(rotated_mip_for_bbox > 1e-9): # If there's any content
                rows_bbox, cols_bbox = np.where(rotated_mip_for_bbox > 1e-9)
                min_row_bbox, max_row_bbox = np.min(rows_bbox), np.max(rows_bbox) + 1
                min_col_bbox, max_col_bbox = np.min(cols_bbox), np.max(cols_bbox) + 1
            _log_process_message(log_adapter, f"Rotation BBox on Y'Z' projection (min_r,max_r,min_c,max_c): [{min_row_bbox}:{max_row_bbox}, {min_col_bbox}:{max_col_bbox}]", "DEBUG")
            del rotated_mip_for_bbox; gc.collect()
        else:
            _log_process_message(log_adapter, "Z-scaled image is empty, skipping BBox calculation for rotation.", "INFO")


        _log_process_message(log_adapter, f"Rotating Z-scaled stack by {rotation_angle_deskew:.2f} degrees around X-axis...", "DEBUG")
        # Rotation for 3D stack: axes=(0,2) means rotating Y'Z' planes around the X-axis (axis 1)
        # This corresponds to rotating the Y' (axis 0) and Z' (axis 2) axes of the scaled_shear_image_cpu (Y'XZ')
        rotation_params = {'angle': rotation_angle_deskew, 'axes': (0, 2), 'reshape': True,
                           'order': 1, 'mode': 'constant', 'cval': 0.0, 'prefilter': True}
        rotated_stack_cpu, rot_gpu_used = process_in_chunks_gpu(
            scaled_shear_image_cpu, ndi_cp.rotate if CUPY_AVAILABLE and ndi_cp else scipy_rotate, # Pass function
            1, # Chunk along X axis (axis 1)
            num_x_chunks_for_gpu_zoom_rotate,
            log_adapter=log_adapter,
            fallback_to_cpu_func=scipy_rotate if gpu_zoom_rotate_fallback_to_cpu else None,
            **rotation_params
        )
        rot_source = "GPU (chunked)" if rot_gpu_used and CUPY_AVAILABLE else "CPU"
        _log_process_message(log_adapter, f"Stack rotation completed using {rot_source}. Rotated shape: {rotated_stack_cpu.shape}", "INFO")
        del scaled_shear_image_cpu; scaled_shear_image_cpu = None; gc.collect()
        if rotated_stack_cpu is None: raise RuntimeError("Rotated stack is None.")
        # After rotation, axes are Y'' (new Y), X (original X), Z'' (new Z from original Z')
        # The BBox (min_row_bbox, etc.) was calculated on a 2D projection of Y'Z' that was rotated.
        # So, min_row_bbox/max_row_bbox correspond to Y'' axis (axis 0 of rotated_stack_cpu)
        # And min_col_bbox/max_col_bbox correspond to Z'' axis (axis 2 of rotated_stack_cpu)

        if rotated_stack_cpu.size > 0:
            # Crop the rotated 3D stack using the BBox
            # Ensure BBox indices are within the bounds of the rotated_stack_cpu
            crop_y_prime_start = max(0, min_row_bbox)
            crop_y_prime_end = min(rotated_stack_cpu.shape[0], max_row_bbox)
            crop_z_prime_start = max(0, min_col_bbox)
            crop_z_prime_end = min(rotated_stack_cpu.shape[2], max_col_bbox)

            if crop_y_prime_start < crop_y_prime_end and crop_z_prime_start < crop_z_prime_end:
                rot_top_shear_image = rotated_stack_cpu[crop_y_prime_start:crop_y_prime_end, :, crop_z_prime_start:crop_z_prime_end]
                _log_process_message(log_adapter, f"Cropped rotated stack to BBox. New Shape (Y'' X Z''): {rot_top_shear_image.shape}", "DEBUG")
            else:
                _log_process_message(log_adapter, "Warning: Invalid BBox crop for rotated stack. Using uncropped rotated stack.", "WARNING")
                rot_top_shear_image = rotated_stack_cpu # Use uncropped if BBox is bad
        else: # Rotated stack is empty
            rot_top_shear_image = rotated_stack_cpu # Keep the empty rotated stack

        if rotated_stack_cpu is not rot_top_shear_image : del rotated_stack_cpu # Free memory if cropped version is different
        gc.collect()
        if rot_top_shear_image is None or rot_top_shear_image.ndim != 3:
             if not (rot_top_shear_image is not None and rot_top_shear_image.ndim==3 and rot_top_shear_image.size==0 and zsize_full==0):
                raise RuntimeError("Cropped rotated image is invalid (None or not 3D).")
        if rot_top_shear_image.size > 0 and min(rot_top_shear_image.shape) <= 0:
            raise RuntimeError(f"Cropped rotated image has a zero dimension: {rot_top_shear_image.shape} but is not empty.")
        toc_rot_s = time.time(); _log_process_message(log_adapter, f"Rotation and cropping to top view done in {toc_rot_s-tic_rot_s:.2f}s. Shape (Y'' X Z''): {rot_top_shear_image.shape}", "INFO")

        # --- Post-Shear Smoothing (if enabled) ---
        if apply_post_shear_smoothing and rot_top_shear_image is not None and rot_top_shear_image.size > 0:
            if smoothing_sigma_yc > 0 or smoothing_sigma_x > 0 or smoothing_sigma_zc > 0:
                _log_process_message(log_adapter, f"Applying post-shear Gaussian smoothing to Y''XZ'' ({rot_top_shear_image.shape}) "
                                                  f"with sigmas: Y''={smoothing_sigma_yc}, X={smoothing_sigma_x}, Z''={smoothing_sigma_zc}", "INFO")
                tic_smooth = time.time()
                sigmas_for_filter_smoothing = (smoothing_sigma_yc, smoothing_sigma_x, smoothing_sigma_zc)
                original_dtype_smoothing = rot_top_shear_image.dtype
                smoothed_image_final_cpu = None
                gpu_smoothing_attempted = False
                gpu_smoothing_successful = False

                # Declare GPU arrays for finally block
                rot_top_shear_image_gpu_smooth = None; image_to_filter_gpu_smooth = None; smoothed_image_gpu_smooth = None

                if CUPY_AVAILABLE and cp is not None and ndi_cp is not None: # Try GPU smoothing
                    try:
                        _log_process_message(log_adapter, "  Smoothing: Attempting GPU Gaussian filter...", "DEBUG")
                        gpu_smoothing_attempted = True
                        rot_top_shear_image_gpu_smooth = cp.asarray(rot_top_shear_image)
                        image_to_filter_gpu_smooth = rot_top_shear_image_gpu_smooth
                        if not cp.issubdtype(rot_top_shear_image_gpu_smooth.dtype, cp.floating):
                            image_to_filter_gpu_smooth = rot_top_shear_image_gpu_smooth.astype(cp.float32)

                        smoothed_image_gpu_smooth = ndi_cp.gaussian_filter(
                            image_to_filter_gpu_smooth, sigma=sigmas_for_filter_smoothing,
                            order=0, mode='constant', cval=0.0
                        )
                        smoothed_image_float_cpu_smooth = cp.asnumpy(smoothed_image_gpu_smooth)

                        if not np.issubdtype(original_dtype_smoothing, np.floating):
                            if np.issubdtype(original_dtype_smoothing, np.integer):
                                min_val_dtype, max_val_dtype = np.iinfo(original_dtype_smoothing).min, np.iinfo(original_dtype_smoothing).max
                                smoothed_image_final_cpu = np.clip(smoothed_image_float_cpu_smooth, min_val_dtype, max_val_dtype).astype(original_dtype_smoothing)
                            else: smoothed_image_final_cpu = smoothed_image_float_cpu_smooth.astype(original_dtype_smoothing)
                        else: smoothed_image_final_cpu = smoothed_image_float_cpu_smooth.astype(original_dtype_smoothing)
                        gpu_smoothing_successful = True
                        _log_process_message(log_adapter, "  Post-shear smoothing on GPU successful.", "INFO")
                    except Exception as e_gpu_smooth:
                        _log_process_message(log_adapter, f"  GPU smoothing FAILED ({type(e_gpu_smooth).__name__}). Falling back to CPU.", "WARNING")
                        gpu_smoothing_successful = False
                    finally:
                        if rot_top_shear_image_gpu_smooth is not None: del rot_top_shear_image_gpu_smooth
                        if image_to_filter_gpu_smooth is not None and image_to_filter_gpu_smooth is not rot_top_shear_image_gpu_smooth : del image_to_filter_gpu_smooth
                        if smoothed_image_gpu_smooth is not None: del smoothed_image_gpu_smooth
                        if gpu_smoothing_attempted: _controller_clear_gpu_memory(log_adapter)

                if not gpu_smoothing_successful: # CPU Fallback for smoothing
                    _log_process_message(log_adapter, "  Applying post-shear smoothing on CPU...", "INFO")
                    image_to_filter_cpu_smooth = rot_top_shear_image
                    if not np.issubdtype(original_dtype_smoothing, np.floating):
                        image_to_filter_cpu_smooth = rot_top_shear_image.astype(np.float32)
                    smoothed_image_float_cpu_smooth = scipy_gaussian_filter(
                        image_to_filter_cpu_smooth, sigma=sigmas_for_filter_smoothing,
                        order=0, mode='constant', cval=0.0
                    )
                    if not np.issubdtype(original_dtype_smoothing, np.floating):
                        if np.issubdtype(original_dtype_smoothing, np.integer):
                             min_val_dtype, max_val_dtype = np.iinfo(original_dtype_smoothing).min, np.iinfo(original_dtype_smoothing).max
                             smoothed_image_final_cpu = np.clip(smoothed_image_float_cpu_smooth, min_val_dtype, max_val_dtype).astype(original_dtype_smoothing)
                        else: smoothed_image_final_cpu = smoothed_image_float_cpu_smooth.astype(original_dtype_smoothing)
                    else: smoothed_image_final_cpu = smoothed_image_float_cpu_smooth.astype(original_dtype_smoothing)
                    if image_to_filter_cpu_smooth is not rot_top_shear_image : del image_to_filter_cpu_smooth
                    if smoothed_image_float_cpu_smooth is not smoothed_image_final_cpu: del smoothed_image_float_cpu_smooth
                    _log_process_message(log_adapter, "  Post-shear smoothing on CPU successful.", "INFO")

                if smoothed_image_final_cpu is not None:
                    if rot_top_shear_image is not smoothed_image_final_cpu : del rot_top_shear_image
                    rot_top_shear_image = smoothed_image_final_cpu
                gc.collect()
                toc_smooth = time.time()
                _log_process_message(log_adapter, f"Post-shear smoothing completed in {toc_smooth - tic_smooth:.2f}s. New shape: {rot_top_shear_image.shape if rot_top_shear_image is not None else 'None'}", "INFO")
            else:
                _log_process_message(log_adapter, "Post-shear smoothing skipped: all sigmas are zero.", "INFO")
        elif apply_post_shear_smoothing:
            _log_process_message(log_adapter, "Post-shear smoothing skipped: image is None or empty.", "INFO")


        # --- Prepare and Save Final View ---
        _log_process_message(log_adapter, "\n--- Preparing Final Deskewed Output ***", "INFO")
        final_image_to_process=rot_top_shear_image
        out_gb=final_image_to_process.nbytes/(1024**3) if final_image_to_process.size>0 else 0; ds_f_z=1
        if out_gb>=12.0: ds_f_z=4
        elif out_gb>=8.0: ds_f_z=3
        elif out_gb>=4.0: ds_f_z=2
        final_dz_eff_um=dx_um*ds_f_z
        _controller_clear_gpu_memory()
        
        final_image_for_mips_and_save=final_image_to_process
        if ds_f_z>1 and final_image_to_process.shape[2]>1:
            _log_process_message(log_adapter,log_adapter, f"Deskew:Downsample Z' by 1/{ds_f_z}x. Size:{out_gb:.2f}GB")
            tgt_z=max(1,round(final_image_to_process.shape[2]/ds_f_z))
            zm_f=tgt_z/final_image_to_process.shape[2]
            
            if zm_f<1.0:
                # Determine chunking strategy based on available GPU memory
                gpu_chunks = 4  # Start with no chunking
                chunk_success = False
                
                
                # Try GPU downsampling first (in chunks if needed)
                try:
                    if gpu_chunks == 1:
                        # Try single GPU operation
                        final_image_gpu = cp.asarray(final_image_to_process)
                        downsampled_gpu = ndi_cp.zoom(final_image_gpu, zoom=(1.0, 1.0, zm_f), 
                                                     order=1, mode='constant', cval=0.0, prefilter=True)
                        final_image_for_mips_and_save = cp.asnumpy(downsampled_gpu)
                        del final_image_gpu, downsampled_gpu
                        
                        
                        _log_process_message(log_adapter, f"Deskew:GPU Z downsampling successful (single chunk). Shape={final_image_for_mips_and_save.shape}")
                        chunk_success = True
                    else:
                        # Process in chunks along Z axis
                        chunk_indices = np.array_split(np.arange(final_image_to_process.shape[2]), gpu_chunks)
                        processed_chunks = []
                        
                        for i, chunk_z_idx in enumerate(chunk_indices):
                            if not chunk_z_idx.size:
                                continue
                            
                            z_start, z_end = chunk_z_idx[0], chunk_z_idx[-1] + 1
                            _log_process_message(log_adapter,f"  Processing GPU chunk {i+1}/{gpu_chunks} (Z={z_start}-{z_end-1})...")
                            
                            try:
                                # Process this chunk on GPU
                                chunk_cpu = final_image_to_process[:, :, z_start:z_end]
                                chunk_gpu = cp.asarray(chunk_cpu)
                                downsampled_chunk_gpu = ndi_cp.zoom(chunk_gpu, zoom=(1.0, 1.0, zm_f),
                                                                   order=1, mode='constant', cval=0.0, prefilter=True)
                                processed_chunk_cpu = cp.asnumpy(downsampled_chunk_gpu)
                                processed_chunks.append(processed_chunk_cpu)
                                
                                # Clean up GPU memory
                                del chunk_gpu, downsampled_chunk_gpu
                                _controller_clear_gpu_memory()
                            except Exception as e_chunk:
                                _log_process_message(log_adapter,f"  GPU chunk {i+1} failed: {type(e_chunk).__name__} - {e_chunk}", "ERROR")
                                processed_chunks = []  # Clear partial results
                                raise  # Will trigger fallback to CPU
                        
                        if processed_chunks:
                            # Stitch chunks back together
                            final_image_for_mips_and_save = np.concatenate(processed_chunks, axis=2)
                            _log_process_message(log_adapter,f"Deskew:GPU Z downsampling successful ({gpu_chunks} chunks). Final shape={final_image_for_mips_and_save.shape}")
                            chunk_success = True
                        
                except Exception as e_gpu:
                    _log_process_message(log_adapter,f"Deskew:GPU downsampling failed ({type(e_gpu).__name__}), falling back to CPU", "WARNING")
                    _controller_clear_gpu_memory()
                
                # Fall back to CPU if GPU failed
                if not chunk_success:
                    _log_process_message(log_adapter,"Deskew:Attempting CPU downsampling...")
                    try:
                        final_image_for_mips_and_save = scipy_zoom(final_image_to_process, (1.0, 1.0, zm_f),
                                                                  order=1, mode='constant', cval=0.0, prefilter=True)
                        _log_process_message(log_adapter,f"Deskew:CPU Z downsampling complete. Shape={final_image_for_mips_and_save.shape}")
                    except Exception as e_cpu:
                        _log_process_message(log_adapter,f"Deskew:CPU downsampling failed! Keeping original size. Error: {e_cpu}", "ERROR")
                        ds_f_z = 1
                        final_dz_eff_um = dx_um
                        final_image_for_mips_and_save = final_image_to_process
            else: 
                _log_process_message(log_adapter,f"Deskew:Zoom factor {zm_f:.3f}>=1.0, skip Z downsample.")
                ds_f_z = 1
                final_dz_eff_um = dx_um
        # Prepare for saving: dtype conversion and transpose to ZYX (Z''Y''X)
        if final_image_for_mips_and_save is not None and final_image_for_mips_and_save.size > 0:
            final_image_to_save_astype = final_image_for_mips_and_save
            if np.issubdtype(out_dtype, np.integer):
                min_val_out, max_val_out = np.iinfo(out_dtype).min, np.iinfo(out_dtype).max
                if np.isnan(final_image_to_save_astype).any(): # Handle NaNs if present
                    final_image_to_save_astype = np.nan_to_num(final_image_to_save_astype.copy() if final_image_to_save_astype is final_image_for_mips_and_save else final_image_to_save_astype, nan=0)
                if final_image_to_save_astype.dtype != out_dtype:
                    final_image_to_save_astype = np.clip(final_image_to_save_astype.astype(np.float32), min_val_out, max_val_out).astype(out_dtype)
            elif final_image_to_save_astype.dtype != out_dtype: # For float output types
                final_image_to_save_astype = final_image_to_save_astype.astype(out_dtype)

            # Transpose Y'' X Z''  to  Z'' Y'' X  for ImageJ compatibility
            final_image_to_save_zyx = final_image_to_save_astype.transpose(2, 0, 1)
            _log_process_message(log_adapter, f"Final image prepared for saving (Z''Y''X): Shape {final_image_to_save_zyx.shape}, dtype {final_image_to_save_zyx.dtype}", "INFO")

            if save_final_deskew:
                # Use the original filename for the output, saved in the new subfolder
                output_path_final_tif_obj = output_folder_top / actual_file_name_str
                tifffile.imwrite(
                    str(output_path_final_tif_obj), final_image_to_save_zyx, imagej=True,
                    metadata={'axes': 'ZYX', 'spacing': final_dz_eff_um, 'unit': 'um', # Basic metadata
                              'fX': final_dx_eff_um, 'fY': final_dx_eff_um, 'fZ': final_dz_eff_um}, # More specific pixel sizes
                    resolution=(1.0/final_dx_eff_um, 1.0/final_dx_eff_um), # XY resolution in pixels/um
                    bigtiff=(final_image_to_save_zyx.nbytes / (1024**3) > 3.9)
                )
                output_path_top_deskewed_tiff_str = str(output_path_final_tif_obj)
                _log_process_message(log_adapter, f"Final deskewed image saved to: {output_path_top_deskewed_tiff_str}", "INFO")
            else:
                _log_process_message(log_adapter, "Saving of final deskewed image was not requested.", "INFO")
        else:
            _log_process_message(log_adapter, "Final image for saving is None or empty. Skipping save.", "WARNING")
            final_image_to_save_zyx = None # Ensure it's None if not processed

        # Write a note/parameters file
        deskew_note_path = output_folder_top / f"deskew_parameters_{actual_file_stem_str}.txt"
        try:
            with open(deskew_note_path, 'w') as f:
                f.write(f"--- Deskewing Parameters ---\n")
                f.write(f"Input File: {actual_file_name_str}\n")
                f.write(f"Original XY Pixel Size (dx_um): {dx_um}\n")
                f.write(f"Original Z Stage Step (dz_um): {dz_um}\n")
                f.write(f"Lightsheet Angle (angle_deg): {angle_deg}\n")
                f.write(f"Flip Direction: {flip_direction}\n")
                f.write(f"Applied Post-Shear Smoothing: {apply_post_shear_smoothing}\n")
                if apply_post_shear_smoothing:
                    f.write(f"  Smoothing Sigmas (Y''c, X, Z''c): {smoothing_sigma_yc}, {smoothing_sigma_x}, {smoothing_sigma_zc}\n")
                f.write(f"\n--- Output Details ---\n")
                f.write(f"Effective Output XY Pixel Size (um): {final_dx_eff_um:.4f}\n")
                f.write(f"Effective Output Z'' Pixel Size (um): {final_dz_eff_um:.4f}\n")
                if final_image_to_save_zyx is not None:
                    f.write(f"Final Deskewed Image Shape (Z''Y''X): {final_image_to_save_zyx.shape}\n")
                    f.write(f"Final Deskewed Image Dtype: {final_image_to_save_zyx.dtype}\n")
                if output_path_top_deskewed_tiff_str:
                    f.write(f"Saved Deskewed File: {output_path_top_deskewed_tiff_str}\n")
            _log_process_message(log_adapter, f"Deskewing parameters saved to: {deskew_note_path}", "INFO")
        except Exception as e_note:
            _log_process_message(log_adapter, f"Could not save deskew parameters note: {e_note}", "WARNING")


        # Generate MIPs from the *final_image_for_mips_and_save* (which is Y'' X Z'')
        # This image might have been downsampled in Z''
        if final_image_for_mips_and_save is not None and final_image_for_mips_and_save.size > 0:
            combined_mip_display_path_str = generate_and_save_mips_for_gui(
                final_image_for_mips_and_save, # Should be Y'' X Z''
                str(output_folder_top),
                actual_file_name_str, # Use original base name for MIP files
                show_deskew_plots, # This flag now means "save diagnostic MIP PNG"
                log_adapter
            )
            if combined_mip_display_path_str:
                _log_process_message(log_adapter, f"Combined MIP TIFF (for display/host app) generated: {combined_mip_display_path_str}", "INFO")
        else:
             _log_process_message(log_adapter, "No final image available for MIP generation after deskewing.", "WARNING")


    except Exception as e_pipeline_deskew:
        _log_process_message(log_adapter, f"[CRITICAL_ERROR] Deskewing pipeline error: {type(e_pipeline_deskew).__name__} - {e_pipeline_deskew}", "CRITICAL_ERROR")
        _log_process_message(log_adapter, traceback.format_exc(), "CRITICAL_ERROR")
        # Ensure all return values are set, even on error, to avoid unpacking issues in the caller
        # Return None for paths, and original/default values for pixel sizes if error occurs before they are set
        return None, dx_um, dx_um, None, Path(full_file_path).name, None

    finally:
        _log_process_message(log_adapter, "Deskewing function: Final cleanup.", "DEBUG")
        vars_to_del_deskew=['image_stack', 'shear_image', 'scaled_shear_image_cpu', 'rot_top_shear_image',
                            'final_image_for_mips_and_save', 'final_image_to_save_zyx'] # Add others if created
        for var_name_deskew in vars_to_del_deskew:
            if var_name_deskew in locals() and locals()[var_name_deskew] is not None:
                try: del locals()[var_name_deskew]
                except NameError: pass
        gc.collect()
        if CUPY_AVAILABLE and cp is not None: _controller_clear_gpu_memory(log_adapter)
        end_time_total_deskew = time.time()
        _log_process_message(log_adapter, f"Total deskewing processing time: {end_time_total_deskew - start_time_total:.2f} seconds.", "INFO")

    return output_path_top_deskewed_tiff_str, final_dx_eff_um, final_dz_eff_um, output_folder_top_str, processed_file_name_str, combined_mip_display_path_str


# --- Decorrelation Analysis Functions (copied from your controller, ensure log_adapter is used) ---
# [DECORR_POD_SIZE, _decorr_fft, _decorr_ifft, _decorr_masked_fft, decorr_apodise,
#  DecorrImageDecorr class, decorr_measure_single_image, _decorr_analyze_single_projection,
#  run_decorrelation_analysis]
# Ensure these functions are present and adapted. For brevity, I'll assume they are
# correctly pasted from your `confocal_projection_controller.py` file. The key is
# that `run_decorrelation_analysis` takes `log_adapter` and uses `_log_process_message`.
# Constants for Decorrelation
DECORR_POD_SIZE = 30
DECORR_POD_ORDER = 8
def _decorr_fft(image): return fftshift(fftn(fftshift(image)))
def _decorr_ifft(im_fft): return ifftshift(ifftn(ifftshift(im_fft)))
def _decorr_masked_fft(im, mask, size): return (mask * _decorr_fft(im)).ravel()[: size // 2]

def decorr_apodise(image, border, order=DECORR_POD_ORDER):
    nx, ny = image.shape
    sig_x = max(1, nx // 2 - border) # Ensure sigma is positive
    sig_y = max(1, ny // 2 - border)
    window_x = ss.general_gaussian(nx, order, sig_x)
    window_y = ss.general_gaussian(ny, order, sig_y)
    window = np.outer(window_x, window_y)
    return window * image

class DecorrImageDecorr:
    pod_size = DECORR_POD_SIZE
    pod_order = DECORR_POD_ORDER
    def __init__(self, image, pixel_size=1.0, square_crop=True):
        if not image.ndim == 2: raise ValueError("ImageDecorr expects a 2D image.")
        image_float = image.astype(np.float64, copy=False)
        self.image = decorr_apodise(image_float, self.pod_size, self.pod_order)
        self.pixel_size = float(pixel_size)
        nx_orig, ny_orig = self.image.shape

        if square_crop:
            n = min(nx_orig, ny_orig); n = n // 2 * 2 # Make even
            if n <= 0: raise ValueError("Image dim non-positive after square crop.")
            start_x = (nx_orig - n) // 2; start_y = (ny_orig - n) // 2
            self.image = self.image[start_x : start_x + n, start_y : start_y + n]
            self.size = n * n
            xx, yy = np.meshgrid(np.linspace(-1, 1, n, endpoint=True), np.linspace(-1, 1, n, endpoint=True))
        else:
            nx = nx_orig // 2 * 2; ny = ny_orig // 2 * 2
            if nx <= 0 or ny <= 0: raise ValueError("Image dim non-positive after non-square crop.")
            self.image = self.image[:nx, :ny]; self.size = nx * ny
            xx, yy = np.meshgrid(np.linspace(-1, 1, ny, endpoint=True), np.linspace(-1, 1, nx, endpoint=True))

        self.disk = xx**2 + yy**2
        self.mask0 = self.disk < 1.0
        im_fft0_raw = _decorr_fft(self.image)
        norm_factor = np.abs(im_fft0_raw)
        im_fft0_normalized = np.zeros_like(im_fft0_raw, dtype=np.complex128)
        np.divide(im_fft0_raw, norm_factor, out=im_fft0_normalized, where=norm_factor != 0)
        im_fft0_normalized[~np.isfinite(im_fft0_normalized)] = 0
        self.im_fft0 = im_fft0_normalized * self.mask0
        img_mean = self.image.mean(); img_std = self.image.std()
        if img_std < 1e-9: img_std = 1.0
        image_bar = (self.image - img_mean) / img_std
        im_fftk = _decorr_fft(image_bar) * self.mask0
        self.im_invk = _decorr_ifft(im_fftk).real
        self.im_fftr = _decorr_masked_fft(self.im_invk, self.mask0, self.size)
        res_max_corr = self.maximize_corcoef(self.im_fftr)
        self.snr0 = res_max_corr["snr"]; self.kc0 = res_max_corr["kc"]
        self.max_width = 2.0 / self.kc0 if self.kc0 > 1e-6 else 20.0
        self.max_width = min(self.max_width, min(self.image.shape) / 4.0, 50.0) # Add absolute cap
        self.kc = None; self.resolution = None

    def corcoef(self, radius, im_fftr_ref, c1_norm=None):
        mask_radius = self.disk < radius**2
        f_im_fft_masked_radius = (mask_radius * self.im_fft0).ravel()
        num_elements_to_take = self.size // 2
        if len(f_im_fft_masked_radius) < num_elements_to_take: num_elements_to_take = len(f_im_fft_masked_radius)
        f_im_fft_half = f_im_fft_masked_radius[:num_elements_to_take]
        c1_val_norm = np.linalg.norm(im_fftr_ref) if c1_norm is None else c1_norm
        c2_val_norm = np.linalg.norm(f_im_fft_half)
        if c1_val_norm * c2_val_norm < 1e-12: return 0.0
        correlation_sum = (im_fftr_ref * f_im_fft_half.conjugate()).real.sum()
        return correlation_sum / (c1_val_norm * c2_val_norm)

    def maximize_corcoef(self, im_fftr_to_test, r_min=0.0, r_max=1.0):
        c1_norm_local = np.linalg.norm(im_fftr_to_test)
        if c1_norm_local < 1e-9: return {"snr": 0.0, "kc": 0.0}
        def anti_cor(radius_param): return 1.0 - self.corcoef(radius_param, im_fftr_to_test, c1_norm=c1_norm_local)
        r_min_actual = max(1e-3, r_min); r_max_actual = min(1.0 - 1e-3, r_max)
        if r_min_actual >= r_max_actual: return {"snr": 0.0, "kc": r_min_actual if r_min_actual > 0 else 0.0}
        res = minimize_scalar(anti_cor, bounds=(r_min_actual, r_max_actual), method="bounded", options={"xatol": 1e-4})
        if not res.success or res.fun is None: return {"snr": 0.0, "kc": r_min_actual}
        final_snr = 1.0 - res.fun; final_kc = res.x
        if final_snr < 1e-3: final_kc = 0.0
        return {"snr": final_snr, "kc": final_kc}

    def compute_resolution(self):
        if self.snr0 < 1e-3 or self.kc0 < 1e-6:
            self.resolution = np.inf; self.kc = 0.0
            return None, {"snr": self.snr0, "kc": self.kc0}
        def filtered_decorr_cost(width_param, return_gm=True):
            sigma_val = max(width_param, 1e-6)
            f_im = self.im_invk - scipy_gaussian_filter(self.im_invk, sigma=sigma_val) # Use scipy
            f_im_fft = _decorr_masked_fft(f_im, self.mask0, self.size)
            res_corr = self.maximize_corcoef(f_im_fft)
            if return_gm:
                if res_corr["kc"] < 1e-6 or res_corr["snr"] < 1e-3: return 1.0
                return 1.0 - (res_corr["kc"] * res_corr["snr"]) ** 0.5
            return res_corr
        lower_bound_width = 0.15; upper_bound_width = self.max_width
        if upper_bound_width <= lower_bound_width:
            self.kc = self.kc0; self.resolution = 2.0 * self.pixel_size / self.kc if self.kc > 1e-6 else np.inf
            return None, {"snr": self.snr0, "kc": self.kc0}
        res_opt = minimize_scalar(filtered_decorr_cost, method="bounded", bounds=(lower_bound_width, upper_bound_width), options={"xatol": 1e-3})
        if not res_opt.success or res_opt.fun is None:
            self.kc = self.kc0; self.resolution = 2.0 * self.pixel_size / self.kc if self.kc > 1e-6 else np.inf
            return res_opt, {"snr": self.snr0, "kc": self.kc0}
        optimal_width = res_opt.x
        max_cor_at_optimal_width = filtered_decorr_cost(optimal_width, return_gm=False)
        self.kc = max_cor_at_optimal_width["kc"]
        self.resolution = 2.0 * self.pixel_size / self.kc if self.kc > 1e-6 else np.inf
        return res_opt, max_cor_at_optimal_width

def decorr_measure_single_image(image_2d, pixel_size_units, square_crop=True):
    if image_2d.ndim != 2: raise ValueError("Image must be 2D for decorr_measure_single_image.")
    if image_2d.shape[0] < DECORR_POD_SIZE * 2.5 or image_2d.shape[1] < DECORR_POD_SIZE * 2.5 : # Stricter size check
        # print(f"Decorrelation: Image too small ({image_2d.shape}) for reliable analysis. Min size approx {DECORR_POD_SIZE*2.5}x{DECORR_POD_SIZE*2.5}.")
        return {"SNR": np.nan, "resolution": np.nan, "kc": np.nan, "error": "Image too small"}
    imdecor_obj = DecorrImageDecorr(image_2d, pixel_size=pixel_size_units, square_crop=square_crop)
    _, _ = imdecor_obj.compute_resolution()
    return {"SNR": imdecor_obj.snr0, "resolution": imdecor_obj.resolution, "kc": imdecor_obj.kc} # Add kc

def _decorr_analyze_single_projection(
    projection_image, projection_name_full,
    pixel_size_y_display, pixel_size_x_display,
    pixel_size_for_resolution_scaling,
    display_axis_labels, resolution_description, units_label="units",
    save_decorr_plots=True, log_adapter=None, plot_output_dir=None ):

    analysis_output = {"resolution": np.nan, "SNR": np.nan, "kc": np.nan, "error": None}
    if projection_image is None or projection_image.size == 0:
        _log_process_message(log_adapter, f"Decorrelation: Projection image '{projection_name_full}' is empty. Skipping.", "WARNING")
        analysis_output["error"] = "Empty projection image"
        return analysis_output
    num_rows, num_cols = projection_image.shape
    x_max_phys = num_cols * pixel_size_x_display
    y_max_phys = num_rows * pixel_size_y_display

    if save_decorr_plots and plot_output_dir:
        try:
            fig_aspect = (y_max_phys / x_max_phys) if x_max_phys > 0 else 1.0
            fig_width = 7; fig_height = max(3, min(10, fig_width * fig_aspect))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            vmin_plot = np.percentile(projection_image, 1) if projection_image.size > 0 else 0
            vmax_plot = np.percentile(projection_image, 99) if projection_image.size > 0 else 1
            if vmax_plot <= vmin_plot: vmax_plot = vmin_plot + 1

            im = ax.imshow(projection_image, cmap='gray', extent=[0, x_max_phys, y_max_phys, 0], vmin=vmin_plot, vmax=vmax_plot)
            ax.set_title(projection_name_full, fontsize=10)
            ax.set_xlabel(f"Original {display_axis_labels[1]} ({units_label})", fontsize=9)
            ax.set_ylabel(f"Original {display_axis_labels[0]} ({units_label})", fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            plt.colorbar(im, ax=ax, label="Intensity", shrink=0.8).ax.tick_params(labelsize=8) # type: ignore
            plt.tight_layout()
            safe_proj_name = "".join(c if c.isalnum() else "_" for c in projection_name_full)
            plot_filename = Path(plot_output_dir) / f"decorr_mip_{safe_proj_name}.png"
            plt.savefig(plot_filename, dpi=150)
            _log_process_message(log_adapter, f"Saved decorrelation MIP plot to {plot_filename}", "INFO")
            plt.close(fig)
        except Exception as e_plot:
            _log_process_message(log_adapter, f"Error generating/saving decorr MIP plot for {projection_name_full}: {e_plot}", "ERROR")
            _log_process_message(log_adapter, traceback.format_exc(), "DEBUG")

    _log_process_message(log_adapter, f"Calculating Decorrelation for '{projection_name_full}' (pixel_size_for_res_scaling={pixel_size_for_resolution_scaling:.3f} {units_label})...", "INFO")
    try:
        if np.all(projection_image == projection_image[0,0]):
             _log_process_message(log_adapter, f"Warning (Decorrelation): Image '{projection_name_full}' is flat. Results may not be meaningful.", "WARNING")
        raw_results = decorr_measure_single_image(projection_image, pixel_size_for_resolution_scaling, square_crop=True)
        resolution_val = raw_results.get('resolution'); snr_val = raw_results.get('SNR'); kc_val = raw_results.get('kc')
        res_str = f"{resolution_val:.2f} {units_label}" if resolution_val is not None and np.isfinite(resolution_val) else str(resolution_val)
        snr_str = f"{snr_val:.2f}" if snr_val is not None and np.isfinite(snr_val) else "N/A"
        kc_str = f"{kc_val:.3f}" if kc_val is not None and np.isfinite(kc_val) else "N/A"
        _log_process_message(log_adapter, f"  Decorrelation ({resolution_description}): Resolution = {res_str}, SNR = {snr_str}, kc = {kc_str}", "INFO")
        analysis_output.update(raw_results)
    except ValueError as ve:
        _log_process_message(log_adapter, f"ValueError in decorr for '{projection_name_full}': {ve}", "ERROR")
        analysis_output["error"] = str(ve)
    except Exception as e_calc:
        _log_process_message(log_adapter, f"Error during decorr calculation for '{projection_name_full}': {e_calc}", "ERROR")
        _log_process_message(log_adapter, traceback.format_exc(), "DEBUG")
        analysis_output["error"] = str(e_calc)
    return analysis_output

def run_decorrelation_analysis(
    deskewed_tiff_path, stack_name_prefix,
    lateral_pixel_size_units, axial_pixel_size_units, units_label="units",
    show_decorr_plots=True, log_adapter=None, main_output_folder=None ):
    _log_process_message(log_adapter, "\n--- Starting Decorrelation Analysis ---", "INFO")
    plot_output_dir_decorr = None
    if main_output_folder and show_decorr_plots:
        plot_output_dir_decorr = Path(main_output_folder) / "analysis_plots"
        try: plot_output_dir_decorr.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            _log_process_message(log_adapter, f"Could not create plot dir for decorr: {e_mkdir}. Plots not saved.", "ERROR")
            plot_output_dir_decorr = None
    all_results = {}
    try:
        _log_process_message(log_adapter, f"Loading deskewed stack for decorrelation: '{deskewed_tiff_path}'", "DEBUG")
        image_stack = tifffile.imread(deskewed_tiff_path) # Assumed ZYX from deskew output
        _log_process_message(log_adapter, f"Loaded stack shape: {image_stack.shape}, dtype: {image_stack.dtype}", "DEBUG")
    except Exception as e_load:
        _log_process_message(log_adapter, f"Error loading deskewed TIFF for decorr: {e_load}", "CRITICAL_ERROR")
        return {"error": f"Failed to load TIFF: {e_load}"}
    if image_stack.ndim != 3:
        _log_process_message(log_adapter, f"Error (Decorrelation): Stack not 3D (shape: {image_stack.shape}).", "ERROR")
        return {"error": "Input stack not 3D"}
    if image_stack.size == 0:
        _log_process_message(log_adapter, "Error (Decorrelation): Stack is empty.", "ERROR")
        return {"error": "Input stack is empty"}

    nz, ny, nx = image_stack.shape # Assuming ZYX from perform_deskewing output file (ImageJ convention)
    _log_process_message(log_adapter, f"\n--- Analyzing Z-MIP (XY Plane) ---", "INFO")
    try:
        proj_xy = np.max(image_stack, axis=0) # Max along Z -> YX plane
        all_results["Z-MIP (XY Plane)"] = _decorr_analyze_single_projection(
            proj_xy, f"Z-MIP (XY) of {stack_name_prefix}", lateral_pixel_size_units, lateral_pixel_size_units,
            lateral_pixel_size_units, ("Y", "X"), "Lateral (in XY plane)", units_label,
            show_decorr_plots, log_adapter, plot_output_dir_decorr )
    except Exception as e_xy_mip: _log_process_message(log_adapter, f"Error in Z-MIP (XY) decorr: {e_xy_mip}", "ERROR"); all_results["Z-MIP (XY Plane)"] = {"error": str(e_xy_mip)}

    _log_process_message(log_adapter, f"\n--- Analyzing Y-MIP (XZ Plane) ---", "INFO")
    try:
        proj_xz = np.max(image_stack, axis=1) # Max along Y -> ZX plane
        all_results["Y-MIP (XZ Plane)"] = _decorr_analyze_single_projection(
            proj_xz, f"Y-MIP (XZ) of {stack_name_prefix}", axial_pixel_size_units, lateral_pixel_size_units,
            axial_pixel_size_units, ("Z", "X"), "Axial-like (Z in XZ plane)", units_label, # Pixel size for res scaling: axial for Z
            show_decorr_plots, log_adapter, plot_output_dir_decorr )
    except Exception as e_xz_mip: _log_process_message(log_adapter, f"Error in Y-MIP (XZ) decorr: {e_xz_mip}", "ERROR"); all_results["Y-MIP (XZ Plane)"] = {"error": str(e_xz_mip)}

    _log_process_message(log_adapter, f"\n--- Analyzing X-MIP (YZ Plane) ---", "INFO")
    try:
        proj_yz = np.max(image_stack, axis=2) # Max along X -> ZY plane
        all_results["X-MIP (YZ Plane)"] = _decorr_analyze_single_projection(
            proj_yz, f"X-MIP (YZ) of {stack_name_prefix}", axial_pixel_size_units, lateral_pixel_size_units, # Y-axis is Z, X-axis is Y
            axial_pixel_size_units, ("Z", "Y"), "Axial-like (Z in YZ plane)", units_label, # Pixel size for res scaling: axial for Z
            show_decorr_plots, log_adapter, plot_output_dir_decorr )
    except Exception as e_yz_mip: _log_process_message(log_adapter, f"Error in X-MIP (YZ) decorr: {e_yz_mip}", "ERROR"); all_results["X-MIP (YZ Plane)"] = {"error": str(e_yz_mip)}

    del image_stack; gc.collect()
    _log_process_message(log_adapter, "--- Decorrelation Analysis Finished ---", "INFO")
    return all_results

# --- PSF Fitting Analysis Functions (copied from your controller, ensure log_adapter is used) ---
# [psf_gaussian, psf_calculate_fwhm_from_fit, run_psf_fitting_analysis]
# Ensure these functions are present and adapted. `run_psf_fitting_analysis` should now
# return a tuple: (psf_results_dict, path_to_fwhm_plot_or_None)
def psf_gaussian(x, amplitude, center, sigma_related_width):
    return amplitude * np.exp(-((x - center) / sigma_related_width) ** 2)

def psf_calculate_fwhm_from_fit(popt_params, pixel_size_val_nm):
    width_parameter_pixels = popt_params[2]
    fwhm_pixels = width_parameter_pixels * 2.0 * np.sqrt(np.log(2.0))
    return fwhm_pixels * pixel_size_val_nm


# >>> END OF COPIED/ADAPTED PROCESSING FUNCTIONS & HELPERS <<<
# --- Main PSF Fitting Analysis Function ---
def run_psf_fitting_analysis(
    deskewed_tiff_path: str,
    base_file_name: str, # Base name for output files/plots
    pixel_size_z_nm: float,
    pixel_size_y_nm: float,
    pixel_size_x_nm: float,
    padding_pixels: int = 1,
    roi_radius_pixels: int = 15,
    intensity_threshold: float = 1000.0,
    fit_quality_r2_threshold: float = 0.85,
    show_psf_plots: bool = True,         # Means "save FWHM vs Pos plot"
    show_psf_threshold_plot: bool = True, # Means "save thresholded projection plot"
    log_adapter=None,                  # For logging messages
    main_output_folder=None,           # Base folder for saving plots and data
    use_gpu_for_psf_prep: bool = False  # Whether to use GPU for initial steps
):
    """
    Performs PSF fitting analysis on a deskewed 3D TIFF image.

    Returns:
        tuple: (psf_results_dict, path_to_fwhm_plot_or_None)
    """
    _log_process_message(log_adapter, f"\n--- Starting PSF Fitting Analysis for: {base_file_name} (GPU Prep: {use_gpu_for_psf_prep}) ---", "INFO")

    plot_output_dir_psf = None
    if main_output_folder and (show_psf_plots or show_psf_threshold_plot):
        plot_output_dir_psf = Path(main_output_folder) / "analysis_plots"
        try:
            plot_output_dir_psf.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            _log_process_message(log_adapter, f"Could not create plot directory for PSF: {e_mkdir}. Plots will not be saved.", "ERROR")
            plot_output_dir_psf = None # Disable plotting if dir creation fails
    
    final_fwhm_plot_path = None # Path to the saved FWHM vs Position plot
    psf_results_summary = {}    # To store summary statistics

    # Initialize variables that might be accessed in `finally`
    im_data_raw_cpu = None
    im_data_processed = None # This will hold the padded data, either CPU or GPU type
    # GPU specific intermediate arrays for cleanup
    im_data_gpu_intermediate = None
    im_bw_gpu_intermediate = None
    labeled_image_gpu_intermediate = None


    xp_module = np # Default to numpy
    label_function_to_use = scipy_label_func # Default to scipy label
    gpu_prep_actually_active = False # Flag to track if GPU path was successfully used

    try:
        _log_process_message(log_adapter, f"Loading deskewed stack for PSF: '{deskewed_tiff_path}'", "DEBUG")
        im_data_raw_cpu = tifffile.imread(deskewed_tiff_path).astype(np.float64) # Load as float64 for precision
        _log_process_message(log_adapter, f"Loaded stack shape: {im_data_raw_cpu.shape}, dtype: {im_data_raw_cpu.dtype}", "DEBUG")

        if im_data_raw_cpu.ndim != 3:
            _log_process_message(log_adapter, f"Error (PSF): Image not 3D. Dim: {im_data_raw_cpu.ndim}. Skipping.", "ERROR")
            return {"error": "Image not 3D"}, None
        if im_data_raw_cpu.size == 0:
            _log_process_message(log_adapter, "Error (PSF): Image is empty. Skipping.", "ERROR")
            return {"error": "Image is empty"}, None

        # --- GPU/CPU Preparation ---
        if use_gpu_for_psf_prep and CUPY_AVAILABLE:
            try:
                cp.cuda.Device(0).use() # Select default GPU
                xp_module = cp
                label_function_to_use = ndi_cp.label
                _log_process_message(log_adapter, "Attempting PSF data prep on GPU.", "INFO")
                im_data_gpu_intermediate = xp_module.asarray(im_data_raw_cpu)
                pad_width_gpu = [(padding_pixels, padding_pixels)] * im_data_gpu_intermediate.ndim
                im_data_processed = xp_module.pad(im_data_gpu_intermediate, pad_width_gpu, mode="constant", constant_values=0)
                gpu_prep_actually_active = True
                _log_process_message(log_adapter, f"PSF data transferred to GPU and padded. Shape: {im_data_processed.shape}", "DEBUG")
            except Exception as e_gpu_init:
                _log_process_message(log_adapter, f"PSF GPU Prep FAILED: {type(e_gpu_init).__name__} - {e_gpu_init}. Falling back to CPU.", "WARNING")
                xp_module = np; label_function_to_use = scipy_label_func; gpu_prep_actually_active = False
                if im_data_gpu_intermediate is not None: del im_data_gpu_intermediate; im_data_gpu_intermediate = None
                _controller_clear_gpu_memory(log_adapter) # Clear if GPU error occurred
        
        if not gpu_prep_actually_active: # CPU path for padding (if GPU failed or not requested)
            xp_module = np; label_function_to_use = scipy_label_func
            _log_process_message(log_adapter, "Using CPU for PSF data prep and padding.", "INFO")
            pad_width_cpu = [(padding_pixels, padding_pixels)] * im_data_raw_cpu.ndim
            im_data_processed = xp_module.pad(im_data_raw_cpu, pad_width_cpu, mode="constant", constant_values=0)
            _log_process_message(log_adapter, f"PSF data padded on CPU. Shape: {im_data_processed.shape}", "DEBUG")

        # Original raw CPU data can be deleted now
        if im_data_raw_cpu is not None: del im_data_raw_cpu; im_data_raw_cpu = None; gc.collect()

        # Thresholding (on GPU or CPU depending on im_data_processed type)
        im_bw_thresholded = im_data_processed > intensity_threshold
        if gpu_prep_actually_active: im_bw_gpu_intermediate = im_bw_thresholded # Keep ref for cleanup

        # Save thresholded MIP plot if requested
        if show_psf_threshold_plot and plot_output_dir_psf and im_data_processed.shape[0] > 1:
            try:
                # Ensure im_bw_thresholded is on CPU for plotting
                im_bw_cpu_for_plot = xp_module.asnumpy(im_bw_thresholded) if gpu_prep_actually_active else im_bw_thresholded.copy()
                
                fig_thresh, ax_thresh = plt.subplots(figsize=(6,6))
                ax_thresh.imshow(np.max(im_bw_cpu_for_plot, axis=0), cmap="gray", interpolation="nearest") # Max Z projection
                ax_thresh.set_title(f"PSF Thresholded Max Projection (Z): {base_file_name}", fontsize=10)
                ax_thresh.set_xlabel("X Pixel", fontsize=9); ax_thresh.set_ylabel("Y Pixel", fontsize=9)
                safe_base_name_plot = "".join(c if c.isalnum() else "_" for c in base_file_name)
                plot_filename_thresh = plot_output_dir_psf / f"psf_threshold_proj_{safe_base_name_plot}.png"
                plt.savefig(plot_filename_thresh, dpi=100); plt.close(fig_thresh)
                _log_process_message(log_adapter, f"Saved PSF threshold plot to {plot_filename_thresh}", "INFO")
                del im_bw_cpu_for_plot
            except Exception as e_plot_thresh:
                _log_process_message(log_adapter, f"Error saving PSF threshold plot: {e_plot_thresh}", "ERROR")
        
        # Labeling (on GPU or CPU)
        labeled_image, num_labels = label_function_to_use(im_bw_thresholded)
        if gpu_prep_actually_active: labeled_image_gpu_intermediate = labeled_image
        
        if num_labels == 0:
            _log_process_message(log_adapter, "PSF: No particles found after thresholding and labeling. Skipping fitting.", "INFO")
            return {"summary_stats": {"num_good_beads": 0}, "error": "No particles found"}, None
        _log_process_message(log_adapter, f"PSF: Found {num_labels} potential particles using {label_function_to_use.__name__}.", "INFO")

        # Initialize lists to store results for each fitted bead
        psf_results_lists = {k: [] for k in ["fwhmZ_nm", "fwhmY_nm", "fwhmX_nm", "zR2", "yR2", "xR2",
                                             "peak_intensity", "centroid_z_px", "centroid_y_px", "centroid_x_px"]}
        
        # --- ROI Extraction and Fitting Loop ---
        im_roi_gpu_loop = None # For finally block inside GPU loop

        if gpu_prep_actually_active and CUPY_AVAILABLE: # GPU-assisted ROI finding
            _log_process_message(log_adapter, "PSF: Iterating labels for GPU-based ROI property extraction.", "DEBUG")
            # Data for fitting needs to be on CPU eventually, so we transfer ROI or profiles
            im_data_cpu_for_fitting = xp_module.asnumpy(im_data_processed) # Get full image on CPU once for ROIs

            for label_idx in range(1, num_labels + 1): # CuPy labels are 1-based
                try:
                    current_label_mask_gpu = (labeled_image == label_idx) # labeled_image is GPU array
                    if not xp_module.any(current_label_mask_gpu): continue
                    
                    # Get centroid from GPU mask
                    coords_gpu = xp_module.argwhere(current_label_mask_gpu)
                    centroid_gpu_float = xp_module.mean(coords_gpu.astype(xp_module.float32), axis=0)
                    centroid_gpu_int = xp_module.round(centroid_gpu_float).astype(xp_module.int32)
                    del coords_gpu, current_label_mask_gpu # Free GPU mem
                    
                    centroid_px_zyx_cpu = cp.asnumpy(centroid_gpu_int) # Transfer centroid to CPU
                    del centroid_gpu_int, centroid_gpu_float

                    # ROI definition on CPU using CPU centroid
                    start_idx = centroid_px_zyx_cpu - roi_radius_pixels
                    end_idx   = centroid_px_zyx_cpu + roi_radius_pixels + 1 # +1 for slicing upper bound

                    # Boundary checks ( crucial for ROI extraction)
                    if np.any(start_idx < 0) or \
                       np.any(end_idx > np.array(im_data_cpu_for_fitting.shape)): # Use shape of CPU image
                        # _log_process_message(log_adapter, f"PSF Bead {label_idx}: ROI out of bounds. Skipping.", "DEBUG")
                        continue
                    
                    # Extract ROI from the CPU copy of im_data_processed
                    im_roi_cpu_loop = im_data_cpu_for_fitting[
                        start_idx[0]:end_idx[0],
                        start_idx[1]:end_idx[1],
                        start_idx[2]:end_idx[2]
                    ]

                    if im_roi_cpu_loop.size == 0 or np.all(im_roi_cpu_loop == 0):
                        continue
                    
                    # Find peak within the CPU ROI
                    try:
                        max_intensity_in_roi_idx_zyx_cpu = np.unravel_index(np.argmax(im_roi_cpu_loop), im_roi_cpu_loop.shape)
                    except ValueError: # Handles empty or all-NaN ROIs if that occurs
                        continue 
                    peak_val_cpu = float(im_roi_cpu_loop[max_intensity_in_roi_idx_zyx_cpu])

                    # Extract profiles (already on CPU)
                    line_z_prof_cpu = im_roi_cpu_loop[:, max_intensity_in_roi_idx_zyx_cpu[1], max_intensity_in_roi_idx_zyx_cpu[2]]
                    line_y_prof_cpu = im_roi_cpu_loop[max_intensity_in_roi_idx_zyx_cpu[0], :, max_intensity_in_roi_idx_zyx_cpu[2]]
                    line_x_prof_cpu = im_roi_cpu_loop[max_intensity_in_roi_idx_zyx_cpu[0], max_intensity_in_roi_idx_zyx_cpu[1], :]
                    
                    # --- Common fitting logic (CPU-based) ---
                    # (This part is identical to the CPU path's fitting logic)
                    def normalize_profile(line_prof):
                        min_v, max_v = np.min(line_prof), np.max(line_prof)
                        return (line_prof - min_v) / (max_v - min_v) if (max_v - min_v) > 1e-9 else np.zeros_like(line_prof)

                    line_z_norm = normalize_profile(line_z_prof_cpu)
                    line_y_norm = normalize_profile(line_y_prof_cpu)
                    line_x_norm = normalize_profile(line_x_prof_cpu)

                    fwhm_z, fwhm_y, fwhm_x = np.nan, np.nan, np.nan
                    r2_z, r2_y, r2_x = np.nan, np.nan, np.nan
                    coords_z, coords_y, coords_x = np.arange(line_z_norm.size), np.arange(line_y_norm.size), np.arange(line_x_norm.size)
                    max_fit_width_pixels = roi_radius_pixels + 5.0 # Max sigma_related_width in pixels

                    def fit_line_psf_profile(coords_axis, line_norm_axis, pixel_size_nm_axis):
                        fwhm_val, r2_val = np.nan, np.nan
                        if len(line_norm_axis) < 4: return fwhm_val, r2_val # Not enough points
                        
                        peak_idx_float = float(np.argmax(line_norm_axis))
                        # Ensure peak is somewhat centered for better fit initialization
                        if not (0.25 * len(coords_axis) < peak_idx_float < 0.75 * len(coords_axis)):
                            peak_idx_float = float(len(coords_axis) // 2)
                        
                        p0_fit = [1.0, peak_idx_float, 1.7] # Initial guess: [Amplitude, Center, Sigma_related_width]
                        bounds_fit = ([0.0, 0.0, 0.1], [1.5, float(len(coords_axis) - 1), max_fit_width_pixels])
                        
                        try:
                            popt, _ = curve_fit(psf_gaussian, coords_axis, line_norm_axis, p0=p0_fit, bounds=bounds_fit, maxfev=5000)
                            fit_line_pred = psf_gaussian(coords_axis, *popt)
                            ss_res = np.sum((line_norm_axis - fit_line_pred) ** 2)
                            ss_tot = np.sum((line_norm_axis - np.mean(line_norm_axis)) ** 2)
                            r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else (1.0 if ss_res < 1e-9 else 0.0)
                            if r2_val > -np.inf : # Check for valid R2 before FWHM calc
                                fwhm_val = psf_calculate_fwhm_from_fit(popt, pixel_size_nm_axis)
                        except (RuntimeError, ValueError): # Fit failed
                            pass 
                        return fwhm_val, r2_val

                    if im_data_cpu_for_fitting.shape[0] > 1 + 2*padding_pixels : # Check if it's truly 3D considering padding
                        fwhm_z, r2_z = fit_line_psf_profile(coords_z, line_z_norm, pixel_size_z_nm)
                    fwhm_y, r2_y = fit_line_psf_profile(coords_y, line_y_norm, pixel_size_y_nm)
                    fwhm_x, r2_x = fit_line_psf_profile(coords_x, line_x_norm, pixel_size_x_nm)

                    if not(np.isnan(r2_x) and np.isnan(r2_y) and np.isnan(r2_z)): # If at least one fit was successful
                        psf_results_lists["fwhmZ_nm"].append(fwhm_z); psf_results_lists["fwhmY_nm"].append(fwhm_y); psf_results_lists["fwhmX_nm"].append(fwhm_x)
                        psf_results_lists["zR2"].append(r2_z); psf_results_lists["yR2"].append(r2_y); psf_results_lists["xR2"].append(r2_x)
                        psf_results_lists["peak_intensity"].append(peak_val_cpu)
                        psf_results_lists["centroid_z_px"].append(centroid_px_zyx_cpu[0])
                        psf_results_lists["centroid_y_px"].append(centroid_px_zyx_cpu[1])
                        psf_results_lists["centroid_x_px"].append(centroid_px_zyx_cpu[2])
                
                except Exception as e_loop_gpu:
                    _log_process_message(log_adapter, f"Error processing bead {label_idx} (GPU path): {e_loop_gpu}", "ERROR")
                    _log_process_message(log_adapter, traceback.format_exc(), "DEBUG")
                finally:
                    # GPU arrays created inside loop were already deleted
                    pass
            if 'im_data_cpu_for_fitting' in locals(): del im_data_cpu_for_fitting

        else: # CPU path for regionprops (if GPU not used or not available)
            _log_process_message(log_adapter, "PSF: Using SciPy regionprops for CPU-based particle analysis.", "DEBUG")
            # Ensure im_data_processed and labeled_image are NumPy arrays for skimage
            im_data_cpu_rp = xp_module.asnumpy(im_data_processed) if gpu_prep_actually_active else im_data_processed
            labeled_image_cpu_rp = xp_module.asnumpy(labeled_image) if gpu_prep_actually_active else labeled_image
            
            # skimage.measure.regionprops expects integer label image
            if not np.issubdtype(labeled_image_cpu_rp.dtype, np.integer):
                labeled_image_cpu_rp = labeled_image_cpu_rp.astype(np.int32)

            stats_cpu = skimage_regionprops(labeled_image_cpu_rp, intensity_image=im_data_cpu_rp)
            _log_process_message(log_adapter, f"PSF: Found {len(stats_cpu)} particles from regionprops.", "INFO")

            for particle_idx, particle in enumerate(stats_cpu):
                try:
                    centroid_px_zyx_cpu_rp = np.round(particle.centroid).astype(int) # Z, Y, X order
                    
                    start_idx_rp = centroid_px_zyx_cpu_rp - roi_radius_pixels
                    end_idx_rp = centroid_px_zyx_cpu_rp + roi_radius_pixels + 1
                    
                    if np.any(start_idx_rp < 0) or np.any(end_idx_rp > np.array(im_data_cpu_rp.shape)):
                        # _log_process_message(log_adapter, f"PSF Bead (CPU) {particle_idx+1}: ROI out of bounds. Centroid: {centroid_px_zyx_cpu_rp}. Skipping.", "DEBUG")
                        continue
                        
                    im_roi_cpu_loop_rp = im_data_cpu_rp[
                        start_idx_rp[0]:end_idx_rp[0],
                        start_idx_rp[1]:end_idx_rp[1],
                        start_idx_rp[2]:end_idx_rp[2]
                    ]

                    if im_roi_cpu_loop_rp.size == 0 or np.all(im_roi_cpu_loop_rp == 0):
                        continue
                    
                    try:
                        max_idx_cpu_rp = np.unravel_index(np.argmax(im_roi_cpu_loop_rp), im_roi_cpu_loop_rp.shape)
                    except ValueError: continue
                    peak_val_cpu_rp = float(im_roi_cpu_loop_rp[max_idx_cpu_rp])

                    line_z_prof_cpu_rp = im_roi_cpu_loop_rp[:, max_idx_cpu_rp[1], max_idx_cpu_rp[2]]
                    line_y_prof_cpu_rp = im_roi_cpu_loop_rp[max_idx_cpu_rp[0], :, max_idx_cpu_rp[2]]
                    line_x_prof_cpu_rp = im_roi_cpu_loop_rp[max_idx_cpu_rp[0], max_idx_cpu_rp[1], :]
                    
                    # --- Common fitting logic (identical to GPU path's fitting part) ---
                    def normalize_profile_rp(line_prof): # Renamed to avoid scope issues if this else block is complex
                        min_v, max_v = np.min(line_prof), np.max(line_prof)
                        return (line_prof - min_v) / (max_v - min_v) if (max_v - min_v) > 1e-9 else np.zeros_like(line_prof)

                    line_z_norm_rp = normalize_profile_rp(line_z_prof_cpu_rp)
                    line_y_norm_rp = normalize_profile_rp(line_y_prof_cpu_rp)
                    line_x_norm_rp = normalize_profile_rp(line_x_prof_cpu_rp)

                    fwhm_z_rp, fwhm_y_rp, fwhm_x_rp = np.nan, np.nan, np.nan
                    r2_z_rp, r2_y_rp, r2_x_rp = np.nan, np.nan, np.nan
                    coords_z_rp, coords_y_rp, coords_x_rp = np.arange(line_z_norm_rp.size), np.arange(line_y_norm_rp.size), np.arange(line_x_norm_rp.size)
                    max_fit_width_pixels_rp = roi_radius_pixels + 5.0

                    def fit_line_psf_profile_rp(coords_axis, line_norm_axis, pixel_size_nm_axis): # Renamed
                        fwhm_val, r2_val = np.nan, np.nan
                        if len(line_norm_axis) < 4: return fwhm_val, r2_val
                        peak_idx_float = float(np.argmax(line_norm_axis))
                        if not (0.25 * len(coords_axis) < peak_idx_float < 0.75 * len(coords_axis)):
                            peak_idx_float = float(len(coords_axis) // 2)
                        p0_fit = [1.0, peak_idx_float, 1.7]; bounds_fit = ([0.0, 0.0, 0.1], [1.5, float(len(coords_axis) - 1), max_fit_width_pixels_rp])
                        try:
                            popt, _ = curve_fit(psf_gaussian, coords_axis, line_norm_axis, p0=p0_fit, bounds=bounds_fit, maxfev=5000)
                            fit_line_pred = psf_gaussian(coords_axis, *popt); ss_res = np.sum((line_norm_axis - fit_line_pred) ** 2); ss_tot = np.sum((line_norm_axis - np.mean(line_norm_axis)) ** 2)
                            r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else (1.0 if ss_res < 1e-9 else 0.0)
                            if r2_val > -np.inf: fwhm_val = psf_calculate_fwhm_from_fit(popt, pixel_size_nm_axis)
                        except (RuntimeError, ValueError): pass
                        return fwhm_val, r2_val

                    if im_data_cpu_rp.shape[0] > 1 + 2*padding_pixels:
                        fwhm_z_rp, r2_z_rp = fit_line_psf_profile_rp(coords_z_rp, line_z_norm_rp, pixel_size_z_nm)
                    fwhm_y_rp, r2_y_rp = fit_line_psf_profile_rp(coords_y_rp, line_y_norm_rp, pixel_size_y_nm)
                    fwhm_x_rp, r2_x_rp = fit_line_psf_profile_rp(coords_x_rp, line_x_norm_rp, pixel_size_x_nm)

                    if not(np.isnan(r2_x_rp) and np.isnan(r2_y_rp) and np.isnan(r2_z_rp)):
                        psf_results_lists["fwhmZ_nm"].append(fwhm_z_rp); psf_results_lists["fwhmY_nm"].append(fwhm_y_rp); psf_results_lists["fwhmX_nm"].append(fwhm_x_rp)
                        psf_results_lists["zR2"].append(r2_z_rp); psf_results_lists["yR2"].append(r2_y_rp); psf_results_lists["xR2"].append(r2_x_rp)
                        psf_results_lists["peak_intensity"].append(peak_val_cpu_rp)
                        psf_results_lists["centroid_z_px"].append(centroid_px_zyx_cpu_rp[0])
                        psf_results_lists["centroid_y_px"].append(centroid_px_zyx_cpu_rp[1])
                        psf_results_lists["centroid_x_px"].append(centroid_px_zyx_cpu_rp[2])
                except Exception as e_loop_cpu:
                    _log_process_message(log_adapter, f"Error processing bead {particle_idx+1} (CPU path): {e_loop_cpu}", "ERROR")
                    _log_process_message(log_adapter, traceback.format_exc(), "DEBUG")
            if 'im_data_cpu_rp' in locals(): del im_data_cpu_rp
            if 'labeled_image_cpu_rp' in locals(): del labeled_image_cpu_rp


        # Convert result lists to NumPy arrays
        final_psf_results_data = {"file_name": base_file_name}
        for key, lst_data in psf_results_lists.items():
            final_psf_results_data[key] = np.array(lst_data if lst_data else [], dtype=np.float64)

        # Filtering and summary statistics (on CPU arrays)
        if final_psf_results_data["fwhmX_nm"].size > 0:
            # R2 filter: If R2 is NaN (fit failed), consider it as not passing the threshold.
            # If R2 is valid, check against threshold.
            q_x = np.where(~np.isnan(final_psf_results_data["xR2"]), final_psf_results_data["xR2"] >= fit_quality_r2_threshold, False)
            q_y = np.where(~np.isnan(final_psf_results_data["yR2"]), final_psf_results_data["yR2"] >= fit_quality_r2_threshold, False)
            q_z = np.where(~np.isnan(final_psf_results_data["zR2"]), final_psf_results_data["zR2"] >= fit_quality_r2_threshold, False)
            
            # Has FWHM: At least one FWHM must be a valid number (not NaN)
            has_any_fwhm = ~ ( np.isnan(final_psf_results_data["fwhmX_nm"]) & \
                               np.isnan(final_psf_results_data["fwhmY_nm"]) & \
                               np.isnan(final_psf_results_data["fwhmZ_nm"]) )
            
            good_indices = np.where(has_any_fwhm & q_x & q_y & q_z)[0]
            num_good_beads = len(good_indices)
            _log_process_message(log_adapter, f"PSF: Total beads processed for fitting: {final_psf_results_data['fwhmX_nm'].size}. Good beads after R2 filter: {num_good_beads}", "INFO")

            if num_good_beads > 0:
                final_psf_results_data["filtered_fwhmX_nm"] = final_psf_results_data["fwhmX_nm"][good_indices]
                final_psf_results_data["filtered_fwhmY_nm"] = final_psf_results_data["fwhmY_nm"][good_indices]
                final_psf_results_data["filtered_fwhmZ_nm"] = final_psf_results_data["fwhmZ_nm"][good_indices]
                final_psf_results_data["filtered_peak_intensity"] = final_psf_results_data["peak_intensity"][good_indices]
                final_psf_results_data["filtered_centroid_x_um"] = final_psf_results_data["centroid_x_px"][good_indices] * pixel_size_x_nm / 1000.0
                final_psf_results_data["filtered_centroid_y_um"] = final_psf_results_data["centroid_y_px"][good_indices] * pixel_size_y_nm / 1000.0
                final_psf_results_data["filtered_centroid_z_um"] = final_psf_results_data["centroid_z_px"][good_indices] * pixel_size_z_nm / 1000.0
                
                psf_results_summary = {
                    "num_good_beads": num_good_beads,
                    "mean_fwhm_x_nm": np.nanmean(final_psf_results_data["filtered_fwhmX_nm"]),
                    "std_fwhm_x_nm":  np.nanstd(final_psf_results_data["filtered_fwhmX_nm"]),
                    "mean_fwhm_y_nm": np.nanmean(final_psf_results_data["filtered_fwhmY_nm"]),
                    "std_fwhm_y_nm":  np.nanstd(final_psf_results_data["filtered_fwhmY_nm"]),
                    "mean_fwhm_z_nm": np.nanmean(final_psf_results_data["filtered_fwhmZ_nm"]),
                    "std_fwhm_z_nm":  np.nanstd(final_psf_results_data["filtered_fwhmZ_nm"])
                }
                _log_process_message(log_adapter, f"PSF Summary (nm): X: {psf_results_summary['mean_fwhm_x_nm']:.1f} ± {psf_results_summary['std_fwhm_x_nm']:.1f}, "
                                              f"Y: {psf_results_summary['mean_fwhm_y_nm']:.1f} ± {psf_results_summary['std_fwhm_y_nm']:.1f}, "
                                              f"Z: {psf_results_summary['mean_fwhm_z_nm']:.1f} ± {psf_results_summary['std_fwhm_z_nm']:.1f} ({num_good_beads} beads)", "INFO")
            else:
                psf_results_summary = {"num_good_beads": 0}
            final_psf_results_data["summary_stats"] = psf_results_summary

            # Save FWHM vs Position plot if requested and good beads exist
            if show_psf_plots and plot_output_dir_psf and num_good_beads > 0:
                fig_fwhm_plot = None
                try:
                    fig_fwhm_plot, axs_fwhm = plt.subplots(3, 1, figsize=(8, 10), sharex=True) # Slightly smaller
                    plot_data_map_fwhm = {
                        "X": {"fwhm": final_psf_results_data["filtered_fwhmX_nm"], "pos": final_psf_results_data["filtered_centroid_x_um"], "label": "X FWHM vs X-pos"},
                        "Y": {"fwhm": final_psf_results_data["filtered_fwhmY_nm"], "pos": final_psf_results_data["filtered_centroid_x_um"], "label": "Y FWHM vs X-pos"},
                        "Z": {"fwhm": final_psf_results_data["filtered_fwhmZ_nm"], "pos": final_psf_results_data["filtered_centroid_x_um"], "label": "Z FWHM vs X-pos"}
                    }
                    peak_intensities_filt = final_psf_results_data.get("filtered_peak_intensity", np.array([]))
                    any_subplot_has_data = False

                    for i, (axis_key, data_dict_fwhm) in enumerate(plot_data_map_fwhm.items()):
                        ax = axs_fwhm[i]
                        fwhm_d, pos_d = data_dict_fwhm["fwhm"], data_dict_fwhm["pos"]
                        valid_mask_plot = ~np.isnan(fwhm_d) & ~np.isnan(pos_d)
                        
                        if peak_intensities_filt.size == fwhm_d.size: # Ensure alignment for coloring
                             valid_mask_plot &= ~np.isnan(peak_intensities_filt)
                        
                        valid_fwhm_p = fwhm_d[valid_mask_plot]
                        valid_pos_p = pos_d[valid_mask_plot]

                        if valid_fwhm_p.size > 0:
                            any_subplot_has_data = True
                            sc = None
                            if peak_intensities_filt.size == fwhm_d.size and valid_fwhm_p.size > 0:
                                valid_intensity_p = peak_intensities_filt[valid_mask_plot]
                                vmin_calc_p = np.percentile(valid_intensity_p, 5) if valid_intensity_p.size > 0 else None
                                vmax_calc_p = np.percentile(valid_intensity_p, 95) if valid_intensity_p.size > 0 else None
                                sc = ax.scatter(valid_pos_p, valid_fwhm_p, c=valid_intensity_p, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5, vmin=vmin_calc_p, vmax=vmax_calc_p, s=20)
                                cbar = fig_fwhm_plot.colorbar(sc, ax=ax, label='Peak Intensity', aspect=10, pad=0.03, fraction=0.046)
                                cbar.ax.tick_params(labelsize=7)
                            else:
                                ax.scatter(valid_pos_p, valid_fwhm_p, alpha=0.7, edgecolors='k', linewidths=0.5, c='blue', s=20)
                            
                            y_upper_limit_p = np.nanmax(valid_fwhm_p) * 1.2 if np.any(np.isfinite(valid_fwhm_p)) and np.nanmax(valid_fwhm_p) > 0 else (pixel_size_x_nm * 2) # Fallback upper
                            ax.set_ylim(bottom=0, top=max(50, y_upper_limit_p)) # Ensure a minimum sensible range
                        else:
                            ax.text(0.5, 0.5, 'No valid data points', ha='center', va='center', transform=ax.transAxes, fontsize=9)
                        
                        ax.set_title(f"{data_dict_fwhm['label']}", fontsize=9)
                        ax.set_xlabel("X Position (µm)" if i == 2 else "", fontsize=8)
                        ax.set_ylabel("FWHM (nm)", fontsize=8)
                        ax.tick_params(axis='both', which='major', labelsize=7)
                        ax.grid(True, linestyle=':', alpha=0.6)

                    if not any_subplot_has_data:
                        _log_process_message(log_adapter, "PSF Plot: No valid data points for any FWHM subplot.", "WARNING")
                        if fig_fwhm_plot: plt.close(fig_fwhm_plot)
                    else:
                        fig_fwhm_plot.suptitle(f"PSF FWHM Analysis: {base_file_name}", fontsize=11, y=0.99)
                        fig_fwhm_plot.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
                        safe_base_name_plot = "".join(c if c.isalnum() else "_" for c in base_file_name)
                        plot_filename_fwhm = plot_output_dir_psf / f"psf_fwhm_vs_pos_{safe_base_name_plot}.png"
                        plt.savefig(plot_filename_fwhm, dpi=150)
                        _log_process_message(log_adapter, f"Saved PSF FWHM plot to {plot_filename_fwhm}", "INFO")
                        final_fwhm_plot_path = str(plot_filename_fwhm)
                        plt.close(fig_fwhm_plot)
                except Exception as e_plot_fwhm:
                    _log_process_message(log_adapter, f"Error during PSF FWHM plot generation: {e_plot_fwhm}", "ERROR")
                    _log_process_message(log_adapter, traceback.format_exc(), "DEBUG")
                    if fig_fwhm_plot and plt.fignum_exists(fig_fwhm_plot.number): plt.close(fig_fwhm_plot)
            elif num_good_beads == 0:
                 _log_process_message(log_adapter, "PSF FWHM Plot: Skipped, no good beads found.", "INFO")
        else: # No beads processed from the start
            _log_process_message(log_adapter, "PSF: No beads initially processed or found for summary statistics or plotting.", "INFO")
            final_psf_results_data["summary_stats"] = {"num_good_beads": 0}

        return final_psf_results_data, final_fwhm_plot_path

    except Exception as e_psf_pipeline:
        _log_process_message(log_adapter, f"CRITICAL Error in PSF Fitting Pipeline: {type(e_psf_pipeline).__name__} - {e_psf_pipeline}", "CRITICAL_ERROR")
        _log_process_message(log_adapter, traceback.format_exc(), "CRITICAL_ERROR")
        return {"error": str(e_psf_pipeline), "summary_stats": {"num_good_beads": 0}}, None
    finally:
        # Cleanup GPU arrays if they were used
        if im_data_gpu_intermediate is not None: del im_data_gpu_intermediate
        if im_bw_gpu_intermediate is not None: del im_bw_gpu_intermediate
        if labeled_image_gpu_intermediate is not None: del labeled_image_gpu_intermediate
        if 'im_data_processed' in locals() and im_data_processed is not None and gpu_prep_actually_active:
            del im_data_processed # This was a cp.ndarray
        
        if gpu_prep_actually_active and CUPY_AVAILABLE:
            _controller_clear_gpu_memory(log_adapter)
        gc.collect()
        _log_process_message(log_adapter, "PSF Fitting Analysis function finished.", "DEBUG")



class OpmController(GUIController):
    def __init__(self, view, parent_controller=None):
        super().__init__(view, parent_controller) # type: ignore
        self.view = view # view is an instance of ConfocalProjectionFrame
        self.analysis_thread = None
        # The view's log_message method is used as the adapter.
        # It needs to be thread-safe or scheduled by _log_process_message.
        self.log_adapter = self.view.log_message

        # Bind UI elements from the view
        # The frame now creates self.browse_btn and self.analyze_button directly
        if hasattr(self.view, 'browse_btn'):
            self.view.browse_btn.configure(command=self.browse_file)
        if hasattr(self.view, 'analyze_button'):
            self.view.analyze_button.configure(command=self.start_analysis_thread)

        _log_process_message(self.log_adapter, "Advanced OPM Analysis Controller initialized!", "INFO")

    def browse_file(self):
        new_file_path = filedialog.askopenfilename(
            title="Select 3D TIFF Stack",
            filetypes=[("TIFF Files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if new_file_path:
            # Update the view's Tkinter variable for file_path
            if "file_path" in self.view.vars:
                self.view.vars["file_path"].set(new_file_path)
                _log_process_message(self.log_adapter, f"File selected: {new_file_path}", "INFO")
                # Clear previous displays when a new file is selected
                self.view.display_mip_image(None)
                self.view.display_psf_plot_image(None)
                try:
                    self.view.results_text_area.configure(state='normal')
                    self.view.results_text_area.delete(1.0, tk.END)
                    self.view.results_text_area.configure(state='disabled')
                except Exception: pass # Ignore if widget not ready
            else:
                _log_process_message(self.log_adapter, f"File selected but view variable not found: {new_file_path}", "WARNING")

    def start_analysis_thread(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            _log_process_message(self.log_adapter, "Analysis is already in progress. Please wait.", "WARNING")
            # Optionally show a messagebox if view has master
            if hasattr(self.view, 'root') and self.view.root:
                 messagebox.showwarning("Busy", "An analysis is already in progress.", parent=self.view.root)
            return

        current_gui_params = self.view.get_variables()
        file_to_analyze = current_gui_params.get("file_path", "")

        if not file_to_analyze or not os.path.exists(file_to_analyze):
            _log_process_message(self.log_adapter, "Error: Please select a valid TIFF file first.", "ERROR")
            if hasattr(self.view, 'root') and self.view.root:
                 messagebox.showerror("File Error", "Please select a valid TIFF file.", parent=self.view.root)
            return

        # Validate essential parameters (example)
        try:
            float(current_gui_params.get("deskew_dx_um", 0.1)) # Try to convert
            # Add more critical parameter validations if needed
        except ValueError as ve:
            _log_process_message(self.log_adapter, f"Invalid parameter format: {ve}", "ERROR")
            if hasattr(self.view, 'root') and self.view.root:
                 messagebox.showerror("Parameter Error", f"Invalid parameter value: {ve}", parent=self.view.root)
            return


        _log_process_message(self.log_adapter, f"Starting analysis for: {os.path.basename(file_to_analyze)}...", "IMPORTANT")
        if hasattr(self.view, 'analyze_button'):
            self.view.analyze_button.config(state=tk.DISABLED)
        if hasattr(self.view, 'browse_btn'):
             self.view.browse_btn.config(state=tk.DISABLED)


        self.analysis_thread = threading.Thread(
            target=self._execute_analysis_pipeline,
            args=(file_to_analyze, current_gui_params,), # Pass file_path separately for clarity
            daemon=True
        )
        self.analysis_thread.start()
        if hasattr(self.view, 'after'):
            self.view.after(100, self._check_analysis_thread_status)


    def _check_analysis_thread_status(self):
        try:
            if self.analysis_thread and self.analysis_thread.is_alive():
                if hasattr(self.view, 'after'): self.view.after(250, self._check_analysis_thread_status)
            else:
                if hasattr(self.view, 'analyze_button') and self.view.analyze_button.winfo_exists():
                    self.view.analyze_button.config(state=tk.NORMAL)
                if hasattr(self.view, 'browse_btn') and self.view.browse_btn.winfo_exists():
                    self.view.browse_btn.config(state=tk.NORMAL)

                if self.analysis_thread is not None: # Check if it was a thread that just finished
                    _log_process_message(self.log_adapter, "Analysis thread has completed.", "INFO")
                self.analysis_thread = None
        except Exception as e_check:
            print(f"Error in _check_analysis_thread_status: {e_check}")
            if hasattr(self.view, 'analyze_button') and self.view.analyze_button.winfo_exists():
                try: self.view.analyze_button.config(state=tk.NORMAL)
                except: pass
            if hasattr(self.view, 'browse_btn') and self.view.browse_btn.winfo_exists():
                try: self.view.browse_btn.config(state=tk.NORMAL)
                except: pass
            self.analysis_thread = None


    def _execute_analysis_pipeline(self, current_file_path, gui_params_dict):
        """
        Orchestrates the full analysis pipeline. Runs in a separate thread.
        gui_params_dict: A dictionary of parameters fetched from self.view.get_variables().
        """
        # This structure is similar to your original OPMAnalysisApp._run_full_analysis_orchestrator
        # but adapted to get params from gui_params_dict and use self.view for updates.
        deskewed_tiff_path_out = None
        combined_mip_path_out = None
        psf_plot_path_out = None
        summary_text_parts = []

        try:
            _log_process_message(self.log_adapter, "===== ANALYSIS PIPELINE STARTED =====", "IMPORTANT")

            # --- 1. Deskewing ---
            _log_process_message(self.log_adapter, "\n\n--- STEP 1: DESKEWING ---", "INFO")
            # Fetch all deskew params from gui_params_dict
            deskew_results = perform_deskewing(
                full_file_path=current_file_path,
                dx_um=gui_params_dict.get("deskew_dx_um"),
                dz_um=gui_params_dict.get("deskew_dz_um"),
                angle_deg=gui_params_dict.get("deskew_angle_deg"),
                flip_direction=gui_params_dict.get("deskew_flip_direction"),
                save_intermediate_shear=gui_params_dict.get("deskew_save_intermediate_shear"),
                show_deskew_plots=gui_params_dict.get("deskew_show_plots"), # Save plots
                log_adapter=self.log_adapter,
                num_z_chunks_for_gpu_shear=gui_params_dict.get("deskew_num_z_chunks_gpu_shear"),
                gpu_shear_fallback_to_cpu_process=gui_params_dict.get("deskew_gpu_shear_fallback_to_cpu"),
                num_x_chunks_for_gpu_zoom_rotate=gui_params_dict.get("deskew_num_x_chunks_gpu_zoom_rotate"),
                gpu_zoom_rotate_fallback_to_cpu=gui_params_dict.get("deskew_gpu_zoom_rotate_fallback_to_cpu"),
                apply_post_shear_smoothing=gui_params_dict.get("deskew_apply_smoothing"),
                smoothing_sigma_yc=gui_params_dict.get("deskew_smoothing_sigma_yc"),
                smoothing_sigma_x=gui_params_dict.get("deskew_smoothing_sigma_x"),
                smoothing_sigma_zc=gui_params_dict.get("deskew_smoothing_sigma_zc"),
                save_final_deskew=gui_params_dict.get("deskew_save_final_deskew")
            )

            if not deskew_results or deskew_results[0] is None:
                _log_process_message(self.log_adapter, "Deskewing failed or produced no output TIFF. Halting.", "CRITICAL_ERROR")
                summary_text_parts.append("Deskewing Failed. Pipeline Halted.")
            else:
                deskewed_tiff_path_out, final_dx_um, final_dz_um_eff, deskew_output_folder, \
                processed_file_name, combined_mip_path_out = deskew_results

                _log_process_message(self.log_adapter, f"Deskewing complete. Output: {deskewed_tiff_path_out}", "INFO")
                summary_text_parts.append(f"--- Deskewing Summary ---")
                summary_text_parts.append(f"Output File: {os.path.basename(deskewed_tiff_path_out if deskewed_tiff_path_out else 'N/A')}")
                summary_text_parts.append(f"Effective XY Pixel Size: {final_dx_um:.3f} um")
                summary_text_parts.append(f"Effective Z'' Pixel Size: {final_dz_um_eff:.3f} um")


                if hasattr(self.view, 'display_mip_image') and hasattr(self.view, 'after'):
                    self.view.after(0, lambda p=combined_mip_path_out: self.view.display_mip_image(p))


                # --- 2. Decorrelation Analysis ---
                if deskewed_tiff_path_out: # Only proceed if deskewing was successful
                    _log_process_message(self.log_adapter, "\n\n--- STEP 2: DECORRELATION ANALYSIS ---", "INFO")
                    decorr_res_dict = run_decorrelation_analysis(
                        deskewed_tiff_path=deskewed_tiff_path_out,
                        stack_name_prefix=os.path.splitext(processed_file_name)[0],
                        lateral_pixel_size_units=final_dx_um,
                        axial_pixel_size_units=final_dz_um_eff, # Use effective Z'' for axial-like
                        units_label=gui_params_dict.get("decorr_units_label"),
                        show_decorr_plots=gui_params_dict.get("decorr_show_plots"),
                        log_adapter=self.log_adapter,
                        main_output_folder=deskew_output_folder
                    )
                    summary_text_parts.append(f"\n--- Decorrelation Analysis ({gui_params_dict.get('decorr_units_label')}) ---")
                    if decorr_res_dict and not decorr_res_dict.get("error"):
                        for view_name, data in decorr_res_dict.items():
                            res_val = data.get('resolution', 'N/A')
                            snr_val = data.get('SNR', 'N/A')
                            res_s = f"{res_val:.2f}" if isinstance(res_val, (int, float)) and np.isfinite(res_val) else str(res_val)
                            snr_s = f"{snr_val:.2f}" if isinstance(snr_val, (int, float)) and np.isfinite(snr_val) else str(snr_val)
                            summary_text_parts.append(f"  {view_name}: Res = {res_s}, SNR = {snr_s}")
                    else:
                        summary_text_parts.append(f"  Decorrelation analysis failed or no results. Error: {decorr_res_dict.get('error', 'Unknown')}")


                # --- 3. PSF Fitting Analysis ---
                if deskewed_tiff_path_out: # Only proceed if deskewing was successful
                    _log_process_message(self.log_adapter, "\n\n--- STEP 3: PSF FITTING ANALYSIS ---", "INFO")
                    # Effective pixel sizes in nm for PSF
                    psf_pix_z_nm = final_dz_um_eff * 1000.0
                    psf_pix_y_nm = final_dx_um * 1000.0
                    psf_pix_x_nm = final_dx_um * 1000.0

                    psf_results, psf_plot_path_out = run_psf_fitting_analysis(
                        deskewed_tiff_path=deskewed_tiff_path_out,
                        base_file_name=os.path.splitext(processed_file_name)[0],
                        pixel_size_z_nm=psf_pix_z_nm,
                        pixel_size_y_nm=psf_pix_y_nm,
                        pixel_size_x_nm=psf_pix_x_nm,
                        padding_pixels=gui_params_dict.get("psf_padding_pixels"),
                        roi_radius_pixels=gui_params_dict.get("psf_roi_radius_pixels"),
                        intensity_threshold=gui_params_dict.get("psf_intensity_threshold"),
                        fit_quality_r2_threshold=gui_params_dict.get("psf_fit_quality_r2_threshold"),
                        show_psf_plots=gui_params_dict.get("psf_show_plots"), # Save plots
                        show_psf_threshold_plot=gui_params_dict.get("psf_show_threshold_plot"),
                        log_adapter=self.log_adapter,
                        main_output_folder=deskew_output_folder,
                        use_gpu_for_psf_prep=gui_params_dict.get("psf_use_gpu_for_prep")
                    )
                    summary_text_parts.append(f"\n--- PSF Fitting Analysis (nm) ---")
                    if psf_results and not psf_results.get("error"):
                        psf_summary = psf_results.get("summary_stats", {})
                        summary_text_parts.append(f"  Num good beads: {psf_summary.get('num_good_beads', 'N/A')}")
                        for dim in ['x', 'y', 'z']:
                            mean_fwhm = psf_summary.get(f'mean_fwhm_{dim}_nm', np.nan)
                            std_fwhm = psf_summary.get(f'std_fwhm_{dim}_nm', np.nan)
                            summary_text_parts.append(f"  Mean FWHM {dim.upper()}: {mean_fwhm:.2f} ± {std_fwhm:.2f}")
                        # Save detailed PSF data if needed
                        if deskew_output_folder and psf_results:
                             psf_npz_path = Path(deskew_output_folder) / f"{os.path.splitext(processed_file_name)[0]}_psf_details.npz"
                             try: np.savez_compressed(psf_npz_path, **psf_results)
                             except Exception as e_npz: _log_process_message(self.log_adapter, f"Error saving PSF NPZ: {e_npz}", "ERROR")

                    else:
                        summary_text_parts.append(f"  PSF analysis failed or no results. Error: {psf_results.get('error', 'Unknown') if psf_results else 'Unknown'}")

                    if hasattr(self.view, 'display_psf_plot_image') and hasattr(self.view, 'after') and psf_plot_path_out:
                        self.view.after(0, lambda p=psf_plot_path_out: self.view.display_psf_plot_image(p))
                    elif hasattr(self.view, 'display_psf_plot_image') and hasattr(self.view, 'after'): # No plot generated
                         self.view.after(0, lambda: self.view.display_psf_plot_image(None))


            _log_process_message(self.log_adapter, "\n\n===== ANALYSIS PIPELINE SUCCESSFULLY FINISHED =====", "IMPORTANT")

        except Exception as e_pipeline:
            error_full_traceback = traceback.format_exc()
            critical_error_msg = f"CRITICAL ERROR in analysis pipeline: {type(e_pipeline).__name__} - {e_pipeline}\n{error_full_traceback}"
            _log_process_message(self.log_adapter, critical_error_msg, "CRITICAL_ERROR")
            summary_text_parts.append(f"\n\nPIPELINE CRASHED:\n{type(e_pipeline).__name__}: {e_pipeline}")
            # Signal view to clear/show error for displays
            if hasattr(self.view, 'display_mip_image') and hasattr(self.view, 'after'):
                self.view.after(0, lambda: self.view.display_mip_image(None)) # None indicates error/no image
            if hasattr(self.view, 'display_psf_plot_image') and hasattr(self.view, 'after'):
                 self.view.after(0, lambda: self.view.display_psf_plot_image(None))
        finally:
            gc.collect()
            if CUPY_AVAILABLE and cp is not None:
                _controller_clear_gpu_memory(self.log_adapter)

            # Update results display in the main thread
            final_summary = "\n".join(summary_text_parts)
            if hasattr(self.view, 'update_results_display') and hasattr(self.view, 'after'):
                self.view.after(0, lambda s=final_summary: self.view.update_results_display(s))
            else: # Fallback if view doesn't have the method or after
                _log_process_message(self.log_adapter, "--- FINAL SUMMARY ---", "INFO")
                _log_process_message(self.log_adapter, final_summary, "INFO")

            # _check_analysis_thread_status will re-enable the button via the main Tkinter loop polling
# OPM_frame.py
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os # For path operations
from pathlib import Path # For path operations
import traceback
# confocal_projection_frame.py
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os # For path operations
from pathlib import Path # For path operations
import traceback
import time # For logging timestamp, if _log_to_gui_if_possible is enhanced

# Placeholder for LabelInput if not available from navigate.view
class LabelInput(ttk.Frame):
    def __init__(self, parent, label, input_class, input_var, input_args=None, label_args=None, **kwargs):
        super().__init__(parent, **kwargs)
        input_args = input_args or {}
        label_args = label_args or {}

        self.label = ttk.Label(self, text=label, **label_args)
        self.label.pack(side=tk.LEFT, padx=(0, 5))

        self.input_var = input_var
        self.input_widget = input_class(self, textvariable=input_var, **input_args)
        if isinstance(self.input_widget, ttk.Entry) and "width" not in input_args:
             self.input_widget.config(width=10) # Default width for entries
        self.input_widget.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def get_variable(self):
        return self.input_var

    def get(self):
        try:
            return self.input_var.get()
        except tk.TclError: # Handle cases where variable might be uninitialized (e.g., DoubleVar with empty string)
            if isinstance(self.input_var, tk.DoubleVar): return 0.0
            if isinstance(self.input_var, tk.IntVar): return 0
            return ""


    def set(self, value):
        self.input_var.set(value)

# --- Pillow Resampling Filter (version compatibility) ---
try:
    LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    try:
        LANCZOS_RESAMPLE = Image.LANCZOS # type: ignore
    except AttributeError:
        LANCZOS_RESAMPLE = Image.ANTIALIAS # type: ignore
        print("Warning: Using Image.ANTIALIAS for resizing.")


class OpmFrame(ttk.Frame):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.root = root 

        self.vars = {}
        # File Path
        self.vars["file_path"] = tk.StringVar()

        # Deskewing
        self.vars["deskew_dx_um"] = tk.DoubleVar(value=0.122)
        self.vars["deskew_dz_um"] = tk.DoubleVar(value=0.2)
        self.vars["deskew_angle_deg"] = tk.DoubleVar(value=45.0)
        self.vars["deskew_flip_direction"] = tk.IntVar(value=1)
        self.vars["deskew_save_intermediate_shear"] = tk.BooleanVar(value=False)
        self.vars["deskew_save_final_deskew"] = tk.BooleanVar(value=True)
        self.vars["deskew_show_plots"] = tk.BooleanVar(value=True) 
        self.vars["deskew_apply_smoothing"] = tk.BooleanVar(value=False)
        self.vars["deskew_smoothing_sigma_yc"] = tk.DoubleVar(value=0.7)
        self.vars["deskew_smoothing_sigma_x"] = tk.DoubleVar(value=0.0)
        self.vars["deskew_smoothing_sigma_zc"] = tk.DoubleVar(value=0.0)
        self.vars["deskew_num_z_chunks_gpu_shear"] = tk.IntVar(value=4)
        self.vars["deskew_min_z_slices_per_chunk_gpu_shear"] = tk.IntVar(value=32)
        self.vars["deskew_gpu_shear_fallback_to_cpu"] = tk.BooleanVar(value=True)
        self.vars["deskew_num_x_chunks_gpu_zoom_rotate"] = tk.IntVar(value=4)
        self.vars["deskew_gpu_zoom_rotate_fallback_to_cpu"] = tk.BooleanVar(value=True)

        # Decorrelation
        self.vars["decorr_units_label"] = tk.StringVar(value="um")
        self.vars["decorr_show_plots"] = tk.BooleanVar(value=True)

        # PSF Fitting
        self.vars["psf_padding_pixels"] = tk.IntVar(value=1)
        self.vars["psf_roi_radius_pixels"] = tk.IntVar(value=15)
        self.vars["psf_intensity_threshold"] = tk.DoubleVar(value=1000.0)
        self.vars["psf_fit_quality_r2_threshold"] = tk.DoubleVar(value=0.85)
        self.vars["psf_show_plots"] = tk.BooleanVar(value=True)
        self.vars["psf_show_threshold_plot"] = tk.BooleanVar(value=True)
        self.vars["psf_use_gpu_for_prep"] = tk.BooleanVar(value=False)

        self.vars["display_min_slider"] = tk.DoubleVar(value=0)
        self.vars["display_max_slider"] = tk.DoubleVar(value=255)
        self.vars["display_actual_min_max_label"] = tk.StringVar(value="Img Min/Max: N/A")

        self.pack(fill=tk.BOTH, expand=True) 

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_pane_frame = ttk.Frame(main_pane, padding="5")
        main_pane.add(left_pane_frame, weight=1) 

        file_frame = ttk.LabelFrame(left_pane_frame, text="Input File", padding="5")
        file_frame.pack(fill=tk.X, pady=(0,10), anchor="n")
        self.browse_btn = ttk.Button(file_frame, text="Select TIFF File") 
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.vars["file_path"], width=40, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        param_notebook = ttk.Notebook(left_pane_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True, pady=5, anchor="n")


        deskew_tab = ttk.Frame(param_notebook, padding="10")
        param_notebook.add(deskew_tab, text="Deskewing")
        self._create_deskew_params_ui(deskew_tab)

        decorr_tab = ttk.Frame(param_notebook, padding="10")
        param_notebook.add(decorr_tab, text="Decorrelation")
        self._create_decorr_params_ui(decorr_tab)

        psf_tab = ttk.Frame(param_notebook, padding="10")
        param_notebook.add(psf_tab, text="PSF Fitting")
        self._create_psf_params_ui(psf_tab)

        self.analyze_button = ttk.Button(left_pane_frame, text="Run Full Analysis") 
        self.analyze_button.pack(pady=10, fill=tk.X, ipady=5, anchor="s") 
        style = ttk.Style(); style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))
        self.analyze_button.configure(style="Accent.TButton")
        
        left_pane_frame.rowconfigure(1, weight=1) 
        left_pane_frame.columnconfigure(0, weight=1)


        right_pane_outer = ttk.Frame(main_pane) 
        main_pane.add(right_pane_outer, weight=3) 
        right_pane_outer.rowconfigure(0, weight=3) 
        right_pane_outer.rowconfigure(1, weight=2) 
        right_pane_outer.columnconfigure(0, weight=1)

        top_right_pane = ttk.PanedWindow(right_pane_outer, orient=tk.VERTICAL)
        top_right_pane.grid(row=0, column=0, sticky="nsew")

        image_display_frame = ttk.LabelFrame(top_right_pane, text="Combined MIP Output (Scroll=Zoom, Drag=Pan)", padding="5")
        top_right_pane.add(image_display_frame, weight=3) 
        self.image_label = ttk.Label(image_display_frame, text="Combined MIP from Deskewing will appear here", relief="groove", anchor="center", justify="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self._bind_interactive_display_events(self.image_label, "mip")

        adj_frame = ttk.LabelFrame(top_right_pane, text="MIP Display Adjustments", padding="5")
        top_right_pane.add(adj_frame, weight=1) 
        self._create_display_adjustments_ui(adj_frame)

        output_notebook = ttk.Notebook(right_pane_outer)
        output_notebook.grid(row=1, column=0, sticky="nsew", pady=(5,0))

        log_tab = ttk.Frame(output_notebook, padding="5")
        output_notebook.add(log_tab, text="Log")
        self.log_text_area = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, width=80, height=10, state="disabled")
        self.log_text_area.pack(fill=tk.BOTH, expand=True)

        results_tab = ttk.Frame(output_notebook, padding="5")
        output_notebook.add(results_tab, text="Summary Results")
        self.results_text_area = scrolledtext.ScrolledText(results_tab, wrap=tk.WORD, width=80, height=10, state="disabled")
        self.results_text_area.pack(fill=tk.BOTH, expand=True)

        psf_plot_tab = ttk.Frame(output_notebook, padding="5")
        output_notebook.add(psf_plot_tab, text="PSF FWHM Plot")
        self.psf_plot_label = ttk.Label(psf_plot_tab, text="PSF FWHM vs. Position plot will appear here.\n(Scroll=Zoom, Drag=Pan)", relief="groove", anchor="center", justify="center")
        self.psf_plot_label.pack(fill=tk.BOTH, expand=True)
        self._bind_interactive_display_events(self.psf_plot_label, "psf_plot")

        self.mip_original_pil_image = None; self.mip_original_image_data_np = None
        self.mip_image_data_min_val = 0; self.mip_image_data_max_val = 255
        self.mip_current_zoom_level = 1.0; self.mip_view_rect_on_original = None
        self.mip_pan_start_pos = None; self.mip_pan_start_view_offset = None; self.mip_tk_image = None
        self.psf_plot_original_pil_image = None; self.psf_plot_current_zoom_level = 1.0
        self.psf_plot_view_rect_on_original = None; self.psf_plot_pan_start_pos = None
        self.psf_plot_pan_start_view_offset = None; self.psf_tk_plot_image = None

    def _create_deskew_params_ui(self, parent_frame):
        parent_frame.columnconfigure(1, weight=1) 
        r = 0
        LabelInput(parent_frame, "XY Pixel Size (µm):", ttk.Entry, self.vars["deskew_dx_um"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=2, pady=2); r+=1
        LabelInput(parent_frame, "Z Stage Step (µm):", ttk.Entry, self.vars["deskew_dz_um"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=2, pady=2); r+=1
        LabelInput(parent_frame, "LS Angle (°):", ttk.Entry, self.vars["deskew_angle_deg"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=2, pady=2); r+=1

        flip_frame = ttk.Frame(parent_frame)
        ttk.Label(flip_frame, text="Flip Direction:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(flip_frame, text="+1", variable=self.vars["deskew_flip_direction"], value=1).pack(side=tk.LEFT)
        ttk.Radiobutton(flip_frame, text="-1", variable=self.vars["deskew_flip_direction"], value=-1).pack(side=tk.LEFT)
        flip_frame.grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=2); r+=1

        ttk.Checkbutton(parent_frame, text="Apply Post-Shear Smoothing", variable=self.vars["deskew_apply_smoothing"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=2); r+=1
        LabelInput(parent_frame, "  └─ Smooth Y'c Sigma:", ttk.Entry, self.vars["deskew_smoothing_sigma_yc"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=(20,2), pady=1); r+=1
        LabelInput(parent_frame, "  └─ Smooth X Sigma:", ttk.Entry, self.vars["deskew_smoothing_sigma_x"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=(20,2), pady=1); r+=1
        LabelInput(parent_frame, "  └─ Smooth Z'c Sigma:", ttk.Entry, self.vars["deskew_smoothing_sigma_zc"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=(20,2), pady=1); r+=1

        ttk.Label(parent_frame, text="--- Output Options ---").grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=(8,2)); r+=1
        ttk.Checkbutton(parent_frame, text="Save Intermediate Sheared Img", variable=self.vars["deskew_save_intermediate_shear"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=1); r+=1
        ttk.Checkbutton(parent_frame, text="Save Final Deskewed Img", variable=self.vars["deskew_save_final_deskew"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=1); r+=1
        ttk.Checkbutton(parent_frame, text="Save Deskew Plots (MIP PNGs)", variable=self.vars["deskew_show_plots"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=1); r+=1

        ttk.Label(parent_frame, text="--- GPU/Chunking (Advanced) ---").grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=(8,2)); r+=1
        LabelInput(parent_frame, "Shear Z-Chunks (GPU):", ttk.Entry, self.vars["deskew_num_z_chunks_gpu_shear"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=2, pady=1); r+=1
        LabelInput(parent_frame, "Min Z slices/chunk (Shear):", ttk.Entry, self.vars["deskew_min_z_slices_per_chunk_gpu_shear"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=2, pady=1); r+=1
        ttk.Checkbutton(parent_frame, text="Fallback Shear to CPU if GPU fails", variable=self.vars["deskew_gpu_shear_fallback_to_cpu"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=1); r+=1
        LabelInput(parent_frame, "Zoom/Rot X-Chunks (GPU):", ttk.Entry, self.vars["deskew_num_x_chunks_gpu_zoom_rotate"]).grid(row=r, column=0, columnspan=2,sticky=tk.EW, padx=2, pady=1); r+=1
        ttk.Checkbutton(parent_frame, text="Fallback Zoom/Rot to CPU if GPU fails", variable=self.vars["deskew_gpu_zoom_rotate_fallback_to_cpu"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=1); r+=1

    def _create_decorr_params_ui(self, parent_frame):
        parent_frame.columnconfigure(1, weight=1)
        r=0
        LabelInput(parent_frame, "Units Label (e.g., um, nm):", ttk.Entry, self.vars["decorr_units_label"]).grid(row=r, column=0,columnspan=2, sticky=tk.EW, padx=2, pady=2); r+=1
        ttk.Checkbutton(parent_frame, text="Save Decorrelation Plots", variable=self.vars["decorr_show_plots"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=2); r+=1

    def _create_psf_params_ui(self, parent_frame):
        parent_frame.columnconfigure(1, weight=1)
        r=0
        LabelInput(parent_frame, "Padding for ROI (px):", ttk.Entry, self.vars["psf_padding_pixels"]).grid(row=r, column=0,columnspan=2, sticky=tk.EW, padx=2, pady=2); r+=1
        LabelInput(parent_frame, "ROI Radius from Centroid (px):", ttk.Entry, self.vars["psf_roi_radius_pixels"]).grid(row=r, column=0,columnspan=2, sticky=tk.EW, padx=2, pady=2); r+=1
        LabelInput(parent_frame, "Intensity Threshold for Beads:", ttk.Entry, self.vars["psf_intensity_threshold"]).grid(row=r, column=0,columnspan=2, sticky=tk.EW, padx=2, pady=2); r+=1
        LabelInput(parent_frame, "Fit Quality R² Threshold (0-1):", ttk.Entry, self.vars["psf_fit_quality_r2_threshold"]).grid(row=r, column=0,columnspan=2, sticky=tk.EW, padx=2, pady=2); r+=1
        ttk.Checkbutton(parent_frame, text="Save PSF FWHM vs. Position Plot", variable=self.vars["psf_show_plots"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=2); r+=1
        ttk.Checkbutton(parent_frame, text="Save PSF Thresholded MIP Plot", variable=self.vars["psf_show_threshold_plot"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=2); r+=1
        ttk.Checkbutton(parent_frame, text="Use GPU for PSF Prep (if available)", variable=self.vars["psf_use_gpu_for_prep"]).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=2); r+=1

    def _create_display_adjustments_ui(self, parent_frame):
        parent_frame.columnconfigure(1, weight=1) 
        ttk.Label(parent_frame, text="Min Display:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=1)
        self.min_display_slider = ttk.Scale(parent_frame, from_=0, to=65535, orient=tk.HORIZONTAL, variable=self.vars["display_min_slider"], length=150, command=self._on_display_sliders_change)
        self.min_display_slider.grid(row=0, column=1, sticky=tk.EW, padx=2, pady=1)
        self.min_display_val_label = ttk.Label(parent_frame, text="0", width=7, anchor='w')
        self.min_display_val_label.grid(row=0, column=2, sticky=tk.W, padx=2)

        ttk.Label(parent_frame, text="Max Display:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=1)
        self.max_display_slider = ttk.Scale(parent_frame, from_=0, to=65535, orient=tk.HORIZONTAL, variable=self.vars["display_max_slider"], length=150, command=self._on_display_sliders_change)
        self.max_display_slider.grid(row=1, column=1, sticky=tk.EW, padx=2, pady=1)
        self.max_display_val_label = ttk.Label(parent_frame, text="255", width=7, anchor='w')
        self.max_display_val_label.grid(row=1, column=2, sticky=tk.W, padx=2)

        self.vars["display_min_slider"].trace_add("write", lambda *args: self.min_display_val_label.config(text=f"{self.vars['display_min_slider'].get():.0f}"))
        self.vars["display_max_slider"].trace_add("write", lambda *args: self.max_display_val_label.config(text=f"{self.vars['display_max_slider'].get():.0f}"))

        button_frame = ttk.Frame(parent_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(8,2), sticky="ew")
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)

        self.auto_contrast_button = ttk.Button(button_frame, text="Auto Contrast", command=self._auto_adjust_contrast_mip, width=12)
        self.auto_contrast_button.grid(row=0, column=0, sticky='w', padx=2)
        self.reset_display_button = ttk.Button(button_frame, text="Reset Display", command=self._reset_display_params_mip, width=12)
        self.reset_display_button.grid(row=0, column=1, sticky='e', padx=2)

        ttk.Label(parent_frame, textvariable=self.vars["display_actual_min_max_label"]).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(8,2))

    def get_variables(self):
        # Ensure all variables are fetched, handling potential TclErrors if a var is not properly set
        fetched_vars = {}
        for key, var_obj in self.vars.items():
            try:
                fetched_vars[key] = var_obj.get()
            except tk.TclError:
                # Provide a default or log an error if a variable is not properly initialized
                # For numeric types, a common default is 0 or 0.0
                if isinstance(var_obj, tk.DoubleVar): fetched_vars[key] = 0.0
                elif isinstance(var_obj, tk.IntVar): fetched_vars[key] = 0
                elif isinstance(var_obj, tk.BooleanVar): fetched_vars[key] = False
                else: fetched_vars[key] = ""
                self._log_to_gui_if_possible(f"Warning: TclError getting variable '{key}'. Using default.", "WARNING")
        return fetched_vars


    def get_widgets(self):
        return { "browse_btn": self.browse_btn, "analyze_button": self.analyze_button }

    def log_message(self, message_with_timestamp_and_level):
        try:
            self.log_text_area.configure(state="normal")
            self.log_text_area.insert(tk.END, message_with_timestamp_and_level + "\n")
            self.log_text_area.see(tk.END)
            self.log_text_area.configure(state="disabled")
            if hasattr(self.root, 'update_idletasks'): # Ensure root is a Tkinter widget
                self.root.update_idletasks() 
        except tk.TclError: print(f"Log (widget destroyed?): {message_with_timestamp_and_level}")
        except Exception as e: print(f"Error logging to GUI: {e}\nOriginal message: {message_with_timestamp_and_level}")

    def update_results_display(self, summary_text):
        try:
            self.results_text_area.configure(state='normal')
            self.results_text_area.delete(1.0, tk.END)
            self.results_text_area.insert(tk.END, summary_text)
            self.results_text_area.configure(state='disabled')
        except tk.TclError: print(f"Results (widget destroyed?): {summary_text}")
        except Exception as e: print(f"Error updating results display: {e}")

    def display_mip_image(self, image_path):
        self._clear_displayed_image("mip")
        if image_path and os.path.exists(str(image_path)): 
            self._setup_initial_image_for_display(str(image_path), "mip", autofit=True)
        elif image_path is None: self.image_label.configure(text="Deskew MIP failed, skipped, or no image.", image='')
        else: self.image_label.configure(text=f"MIP image not found:\n{image_path}", image='')

    def _update_displayed_image_portion(self, display_type="mip"):
        target_label=self.image_label if display_type=="mip" else self.psf_plot_label; original_pil_image=getattr(self,f"{display_type}_original_pil_image")
        original_np_data=getattr(self,f"{display_type}_original_image_data_np",None); view_rect=getattr(self,f"{display_type}_view_rect_on_original"); tk_image_attr=f"{display_type}_tk_image"
        if original_pil_image is None or view_rect is None: return
        try:
            target_label.update_idletasks(); lbl_w,lbl_h=target_label.winfo_width(),target_label.winfo_height()
            # CORRECTED LINE:
            if lbl_w<=1:lbl_w=max(200,int(self.root.winfo_width()*0.3)); 
            if lbl_h<=1:lbl_h=max(200,int(self.root.winfo_height()*0.3))

            crop_box_f=view_rect; h_orig_pil,w_orig_pil=original_pil_image.height,original_pil_image.width
            crop_x0=max(0,int(crop_box_f[0]));crop_y0=max(0,int(crop_box_f[1]));crop_x1=min(w_orig_pil,int(crop_box_f[2]));crop_y1=min(h_orig_pil,int(crop_box_f[3]))
            if crop_x0>=crop_x1 or crop_y0>=crop_y1: self._handle_display_error(f"Invalid PIL crop: [{crop_y0}:{crop_y1}, {crop_x0}:{crop_x1}]",display_type); return
            cropped_pil_image=original_pil_image.crop((crop_x0,crop_y0,crop_x1,crop_y1)); display_img_pil=None
            if display_type=="mip" and original_np_data is not None:
                cropped_np_array=original_np_data[crop_y0:crop_y1,crop_x0:crop_x1]
                if cropped_np_array.size==0: self._handle_display_error("MIP Cropped NumPy empty.",display_type);return
                min_d,max_d=self.vars["display_min_slider"].get(),self.vars["display_max_slider"].get()
                if min_d>=max_d:max_d=min_d+1e-9
                try:
                    from skimage.exposure import rescale_intensity as skimage_rescale # Import locally if not global
                    adjusted_array_float = skimage_rescale(cropped_np_array, in_range=(min_d, max_d), out_range=(0.0, 255.0))
                except ImportError: 
                    scaled_np=(cropped_np_array-min_d)/((max_d-min_d) if (max_d-min_d) != 0 else 1e-9)*255.0 # type: ignore
                    adjusted_array_float = scaled_np
                adjusted_array_8bit=np.clip(adjusted_array_float,0,255).astype(np.uint8)
                try: display_img_pil=Image.fromarray(adjusted_array_8bit)
                except Exception as e_pil: self._handle_display_error(f"Err convert contrast MIP to PIL:{e_pil}",display_type);return
            else:
                display_img_pil=cropped_pil_image
                if display_img_pil.mode=='RGBA':display_img_pil=display_img_pil.convert('RGB')
                elif display_img_pil.mode not in ['L','RGB']:display_img_pil=display_img_pil.convert('L')
            crop_w_pil,crop_h_pil=display_img_pil.size
            if crop_w_pil==0 or crop_h_pil==0: self._handle_display_error("PIL for display zero dim.",display_type);return
            img_asp=crop_w_pil/crop_h_pil if crop_h_pil != 0 else 1.0
            lbl_asp=lbl_w/lbl_h if lbl_h != 0 else 1.0
            if img_asp>lbl_asp:new_w_d,new_h_d=lbl_w,int(lbl_w/img_asp)if img_asp!=0 else 0
            else:new_h_d,new_w_d=lbl_h,int(lbl_h*img_asp)
            new_w_d,new_h_d=max(1,new_w_d),max(1,new_h_d)
            resized_img_d=display_img_pil.resize((new_w_d,new_h_d),LANCZOS_RESAMPLE);new_photo_d=ImageTk.PhotoImage(resized_img_d)
            target_label.configure(image=new_photo_d,text="");setattr(self,tk_image_attr,new_photo_d)
        except Exception as e_upd:self._handle_display_error(f"Err update {display_type} display:{e_upd}\n{traceback.format_exc()}",display_type)

    def _on_display_sliders_change(self, *args):
        min_val_s = self.vars["display_min_slider"].get()
        max_val_s = self.vars["display_max_slider"].get()
        if min_val_s > max_val_s:
            # If min slider moved past max, set max to min.
            # If max slider moved past min, set min to max.
            # This simple logic might cause them to "chase" each other if not careful.
            # A robust way: if min changed and is > max, set max = min. If max changed and is < min, set min = max.
            # For simplicity, we'll just ensure min <= max after any change.
             self.vars["display_min_slider"].set(min(min_val_s, max_val_s))
             self.vars["display_max_slider"].set(max(min_val_s, max_val_s))

        if self.mip_original_pil_image: self._update_displayed_image_portion("mip")

    def _reset_display_params_mip(self):
        if self.mip_original_image_data_np is not None and self.mip_original_image_data_np.size > 0:
            min_v,max_v=self.mip_image_data_min_val,self.mip_image_data_max_val
            if max_v<=min_v:max_v=min_v+1.
            self.min_display_slider.config(from_=min_v,to=max_v);self.max_display_slider.config(from_=min_v,to=max_v)
            self.vars["display_min_slider"].set(min_v);self.vars["display_max_slider"].set(max_v)
        else:
            self.min_display_slider.config(from_=0,to=255);self.max_display_slider.config(from_=0,to=255)
            self.vars["display_min_slider"].set(0);self.vars["display_max_slider"].set(255)
        if self.mip_original_pil_image: self._update_displayed_image_portion("mip")

    def _auto_adjust_contrast_mip(self):
        if self.mip_original_image_data_np is None or self.mip_view_rect_on_original is None: self._log_to_gui_if_possible("AutoContrast:No MIP loaded/view undefined.","WARNING");return
        try:
            x0,y0,x1,y1=map(int,self.mip_view_rect_on_original); current_view_np=self.mip_original_image_data_np[y0:y1,x0:x1]
            if current_view_np.size==0:self._log_to_gui_if_possible("AutoContrast:Cropped MIP view empty.","WARNING");return
            finite_view=current_view_np[np.isfinite(current_view_np)]
            if finite_view.size==0:low_p,high_p=self.mip_image_data_min_val,self.mip_image_data_max_val
            else:low_p,high_p=np.percentile(finite_view,(0.3,99.7)) # type: ignore
            if high_p<=low_p:
                min_f,max_f= (np.min(finite_view) if finite_view.size>0 else self.mip_image_data_min_val), (np.max(finite_view) if finite_view.size>0 else self.mip_image_data_max_val)
                if max_f>min_f:low_p,high_p=min_f,max_f
                else:high_p=low_p+1.0 # Ensure high_p is float if low_p is
            glob_min,glob_max=self.min_display_slider.cget("from"),self.min_display_slider.cget("to")
            low_p,high_p=max(glob_min,float(low_p)),min(glob_max,float(high_p))
            if high_p<=low_p:high_p=low_p+1.0
            self.vars["display_min_slider"].set(low_p);self.vars["display_max_slider"].set(high_p)
            self._log_to_gui_if_possible(f"AutoContrast:Set display [{low_p:.1f}-{high_p:.1f}]","INFO");self._update_displayed_image_portion("mip")
        except Exception as e_ac:self._log_to_gui_if_possible(f"Err auto contrast:{e_ac}\n{traceback.format_exc()}","ERROR")

    def display_psf_plot_image(self, image_path):
        self._clear_displayed_image("psf_plot")
        if image_path and os.path.exists(str(image_path)): self._setup_initial_image_for_display(str(image_path), "psf_plot", autofit=True)
        elif image_path is None: self.psf_plot_label.configure(text="No PSF FWHM plot or no path provided.", image='')
        else: self.psf_plot_label.configure(text=f"PSF plot not found:\n{image_path}", image='')


    def _clear_displayed_image(self, display_type="mip"):
        target_label = self.image_label if display_type == "mip" else self.psf_plot_label
        default_text = "MIP appears here" if display_type == "mip" else "PSF Plot appears here"
        
        # Get the current PIL image object for the specified display type
        current_pil_image_attr_name = f"{display_type}_original_pil_image"
        current_pil_image = getattr(self, current_pil_image_attr_name, None)

        if current_pil_image:
            try:
                current_pil_image.close() # Explicitly close the file handle
                self._log_to_gui_if_possible(f"Closed previous {display_type.upper()} image handle.", "DEBUG")
            except Exception as e_close:
                self._log_to_gui_if_possible(f"Warning: Error closing previous {display_type.upper()} image: {e_close}", "WARNING")
        
        if target_label and hasattr(target_label, 'winfo_exists') and target_label.winfo_exists(): # Check if widget exists
            target_label.configure(image='', text=default_text)
        
        setattr(self, f"{display_type}_tk_image", None)
        setattr(self, f"{display_type}_original_pil_image", None)
        setattr(self, f"{display_type}_current_zoom_level", 1.0)
        setattr(self, f"{display_type}_view_rect_on_original", None)
        
        if display_type == "mip":
            setattr(self, f"{display_type}_original_image_data_np", None)
            setattr(self, f"{display_type}_image_data_min_val", 0)
            setattr(self, f"{display_type}_image_data_max_val", 255)
            if hasattr(self, 'min_display_slider'): # Check if contrast sliders exist
                self.min_display_slider.config(from_=0, to=65535) # Reset full range
                self.max_display_slider.config(from_=0, to=65535)
            self.vars["display_min_slider"].set(0)
            self.vars["display_max_slider"].set(255)
            self.vars["display_actual_min_max_label"].set("Img Min/Max: N/A")

    def _setup_initial_image_for_display(self, image_path, display_type="mip", autofit=False):
        target_label = self.image_label if display_type == "mip" else self.psf_plot_label
        original_pil_attr = f"{display_type}_original_pil_image"
        original_np_attr = f"{display_type}_original_image_data_np" # Only for MIP
        min_val_attr = f"{display_type}_image_data_min_val" # Only for MIP
        max_val_attr = f"{display_type}_image_data_max_val" # Only for MIP
        zoom_attr = f"{display_type}_current_zoom_level"
        view_rect_attr = f"{display_type}_view_rect_on_original"
        
        # --- Crucial Fix: Close any existing image before opening a new one ---
        existing_pil_image = getattr(self, original_pil_attr, None)
        if existing_pil_image:
            try:
                existing_pil_image.close()
                self._log_to_gui_if_possible(f"Closed existing {display_type.upper()} image before loading new one.", "DEBUG")
            except Exception as e_close:
                self._log_to_gui_if_possible(f"Warning: Error closing existing {display_type.upper()} image: {e_close}", "WARNING")
            setattr(self, original_pil_attr, None)
            if display_type == "mip":
                 setattr(self, original_np_attr, None)
        # --- End of Fix ---

        try:
            self._log_to_gui_if_possible(f"GUI: Loading {display_type.upper()} image: {image_path}", "DEBUG")
            if not image_path or not os.path.exists(str(image_path)): # Ensure image_path is a string for Path check
                 self._handle_display_error(f"Path invalid or file does not exist: '{image_path}'", display_type)
                 return
            if not str(image_path).lower().endswith((".tif", ".tiff", ".png")): 
                self._handle_display_error(f"Expected TIFF/PNG, got: {os.path.basename(str(image_path))}", display_type)
                return
            
            # Use context manager for opening to ensure it's closed if an error occurs during processing here
            with Image.open(str(image_path)) as pil_img_opened:
                pil_img_opened.load()
                # Since we need to store the PIL image for zoom/pan, we make a copy.
                # The 'pil_img_opened' from 'with' context will be closed upon exiting this block.
                pil_image_to_store = pil_img_opened.copy() 
            
            setattr(self, original_pil_attr, pil_image_to_store)

            if display_type == "mip":
                np_data = np.array(pil_image_to_store) # Use the in-memory copy
                setattr(self, original_np_attr, np_data)
                if np_data.size > 0:
                    min_val=float(np.min(np_data)); max_val=float(np.max(np_data))
                    if max_val <= min_val: max_val = min_val + 1.0
                    setattr(self, min_val_attr, min_val); setattr(self, max_val_attr, max_val)
                    self.vars["display_actual_min_max_label"].set(f"Img Min/Max: {min_val:.0f}/{max_val:.0f}")
                    if hasattr(self, 'min_display_slider'): # Check if widgets exist
                        self.min_display_slider.config(from_=min_val, to=max_val)
                        self.max_display_slider.config(from_=min_val, to=max_val)
                    self.vars["display_min_slider"].set(min_val)
                    self.vars["display_max_slider"].set(max_val)
                else: 
                    self._log_to_gui_if_possible("Loaded MIP image array is empty.", "WARNING")
                    # Set default contrast ranges if data is empty
                    setattr(self, min_val_attr, 0); setattr(self, max_val_attr, 255)
                    if hasattr(self, 'min_display_slider'):
                        self.min_display_slider.config(from_=0, to=65535)
                        self.max_display_slider.config(from_=0, to=65535)
                    self.vars["display_min_slider"].set(0); self.vars["display_max_slider"].set(255)
                    self.vars["display_actual_min_max_label"].set("Img Min/Max: N/A (empty)")


            if autofit: 
                self._autofit_image_to_label(getattr(self, original_pil_attr), target_label, zoom_attr, view_rect_attr)
            else: 
                setattr(self, zoom_attr, 1.0)
                img_w, img_h = getattr(self, original_pil_attr).size
                setattr(self, view_rect_attr, (0, 0, img_w, img_h))
            
            self._update_displayed_image_portion(display_type)

        except FileNotFoundError:
             self._handle_display_error(f"File not found: '{image_path}'", display_type)
        except Exception as e: 
            self._handle_display_error(f"Error loading initial {display_type} image: {e}\n{traceback.format_exc()}", display_type)



    def _autofit_image_to_label(self, pil_img, disp_lbl, zoom_attr_str, view_rect_attr_str):
        if pil_img is None or disp_lbl is None:return
        disp_lbl.update_idletasks();lbl_w,lbl_h=disp_lbl.winfo_width(),disp_lbl.winfo_height()
        # CORRECTED LINE:
        if lbl_w<=1:lbl_w=max(200,int(self.root.winfo_width()*0.3)); 
        if lbl_h<=1:lbl_h=max(200,int(self.root.winfo_height()*0.3))
        img_w,img_h=pil_img.size
        if img_w==0 or img_h==0:return 
        sf=min(lbl_w/img_w if img_w>0 else 1,lbl_h/img_h if img_h>0 else 1); sf=min(1.,sf)if sf>0 else 1. 
        setattr(self,zoom_attr_str,sf);setattr(self,view_rect_attr_str,(0,0,img_w,img_h))

    def _bind_interactive_display_events(self, target_label, display_type_prefix):
        target_label.bind("<MouseWheel>", lambda e, dtp=display_type_prefix: self._on_mouse_wheel(e, dtp))
        target_label.bind("<Button-4>", lambda e, dtp=display_type_prefix: self._on_mouse_wheel(e, dtp))
        target_label.bind("<Button-5>", lambda e, dtp=display_type_prefix: self._on_mouse_wheel(e, dtp))
        target_label.bind("<ButtonPress-1>", lambda e, dtp=display_type_prefix: self._on_button_press(e, dtp))
        target_label.bind("<B1-Motion>", lambda e, dtp=display_type_prefix: self._on_mouse_drag(e, dtp))
        target_label.bind("<ButtonRelease-1>", lambda e, dtp=display_type_prefix: self._on_button_release(e, dtp))

    def _on_mouse_wheel(self, event, display_type):
        orig_img=getattr(self,f"{display_type}_original_pil_image");curr_zoom=getattr(self,f"{display_type}_current_zoom_level");view_rect=getattr(self,f"{display_type}_view_rect_on_original")
        target_lbl=self.image_label if display_type=="mip" else self.psf_plot_label
        if orig_img is None or view_rect is None:return
        zoom_inc=1.2; zoom_f=1/zoom_inc if event.num==5 or event.delta<0 else zoom_inc if event.num==4 or event.delta>0 else 1
        if zoom_f == 1: return
        new_zoom=curr_zoom*zoom_f; new_zoom=max(0.01,min(50.,new_zoom))
        if abs(new_zoom-curr_zoom)<1e-6:return
        x0_o,y0_o,x1_o,y1_o=view_rect;target_lbl.update_idletasks();lbl_w,lbl_h=target_lbl.winfo_width(),target_lbl.winfo_height()
        if lbl_w<=1 or lbl_h<=1:return
        mx_l,my_l=event.x,event.y;curr_vw_w,curr_vw_h=(x1_o-x0_o),(y1_o-y0_o)
        if curr_vw_w == 0 or curr_vw_h == 0 : return # Avoid division by zero
        focus_x_o=x0_o+(mx_l/lbl_w)*curr_vw_w; focus_y_o=y0_o+(my_l/lbl_h)*curr_vw_h
        new_vw_w=orig_img.width/new_zoom; new_vw_h=orig_img.height/new_zoom
        new_x0=focus_x_o-(mx_l/lbl_w)*new_vw_w; new_y0=focus_y_o-(my_l/lbl_h)*new_vw_h
        new_x1=new_x0+new_vw_w; new_y1=new_y0+new_vw_h
        orig_w,orig_h=orig_img.width,orig_img.height
        if new_x0<0:new_x1-=new_x0;new_x0=0
        if new_y0<0:new_y1-=new_y0;new_y0=0
        if new_x1>orig_w:new_x0-=(new_x1-orig_w);new_x1=orig_w
        if new_y1>orig_h:new_y0-=(new_y1-orig_h);new_y1=orig_h
        new_x0=max(0,new_x0);new_y0=max(0,new_y0);new_x1=min(orig_w,new_x1);new_y1=min(orig_h,new_y1)
        if new_x1<=new_x0:new_x1=new_x0+1e-3 # type: ignore
        if new_y1<=new_y0:new_y1=new_y0+1e-3 # type: ignore
        setattr(self,f"{display_type}_current_zoom_level",new_zoom);setattr(self,f"{display_type}_view_rect_on_original",(new_x0,new_y0,new_x1,new_y1))
        self._update_displayed_image_portion(display_type)

    def _on_button_press(self, event, display_type):
        orig_img=getattr(self,f"{display_type}_original_pil_image");view_rect=getattr(self,f"{display_type}_view_rect_on_original")
        target_lbl=self.image_label if display_type=="mip" else self.psf_plot_label
        if orig_img is None or view_rect is None:return
        setattr(self,f"{display_type}_pan_start_pos",(event.x,event.y));setattr(self,f"{display_type}_pan_start_view_offset",(view_rect[0],view_rect[1]))
        try:target_lbl.config(cursor="fleur")
        except tk.TclError:pass

    def _on_mouse_drag(self, event, display_type):
        orig_img=getattr(self,f"{display_type}_original_pil_image");pan_start=getattr(self,f"{display_type}_pan_start_pos");view_rect=getattr(self,f"{display_type}_view_rect_on_original");pan_offset=getattr(self,f"{display_type}_pan_start_view_offset")
        target_lbl=self.image_label if display_type=="mip" else self.psf_plot_label
        if orig_img is None or pan_start is None or view_rect is None or pan_offset is None:return
        dx_s,dy_s=event.x-pan_start[0],event.y-pan_start[1];target_lbl.update_idletasks();lbl_w,lbl_h=target_lbl.winfo_width(),target_lbl.winfo_height()
        if lbl_w<=1 or lbl_h<=1:return
        curr_vw_w,curr_vw_h=max(1,view_rect[2]-view_rect[0]),max(1,view_rect[3]-view_rect[1])
        px_o_per_s_x=curr_vw_w/lbl_w if lbl_w > 0 else 0
        px_o_per_s_y=curr_vw_h/lbl_h if lbl_h > 0 else 0
        dx_o_v,dy_o_v=dx_s*px_o_per_s_x,dy_s*px_o_per_s_y
        new_x0_p,new_y0_p=pan_offset[0]-dx_o_v,pan_offset[1]-dy_o_v
        orig_w,orig_h=orig_img.width,orig_img.height
        new_x0_p=max(0,min(new_x0_p,orig_w-curr_vw_w if orig_w > curr_vw_w else 0))
        new_y0_p=max(0,min(new_y0_p,orig_h-curr_vw_h if orig_h > curr_vw_h else 0))
        new_x1_p,new_y1_p=new_x0_p+curr_vw_w,new_y0_p+curr_vw_h
        setattr(self,f"{display_type}_view_rect_on_original",(new_x0_p,new_y0_p,new_x1_p,new_y1_p));self._update_displayed_image_portion(display_type)

    def _on_button_release(self, event, display_type):
        target_lbl=self.image_label if display_type=="mip" else self.psf_plot_label
        setattr(self,f"{display_type}_pan_start_pos",None);setattr(self,f"{display_type}_pan_start_view_offset",None)
        try:target_lbl.config(cursor="")
        except tk.TclError:pass

    def _handle_display_error(self, error_message, display_type="mip"):
        target_lbl=self.image_label if display_type=="mip" else self.psf_plot_label
        default_text="Error displaying MIP." if display_type=="mip" else "Error displaying PSF plot."
        self._log_to_gui_if_possible(f"GUI {display_type.upper()} Display Error: {error_message}", "ERROR")
        try: target_lbl.configure(text=f"{default_text}\nCheck logs.",image='')
        except tk.TclError:pass
        setattr(self,f"{display_type}_original_pil_image",None)
        if display_type=="mip":setattr(self,f"{display_type}_original_image_data_np",None)

    def _log_to_gui_if_possible(self, message, level="INFO"):
        # This is a direct call, assumes it's safe or controller handles thread safety via log_adapter
        # The controller's _log_process_message now attempts to use self.view.after for thread safety.
        if hasattr(self, 'log_message') and callable(self.log_message):
            # Construct the full formatted message if the frame's log_message expects it
            # However, the controller's _log_process_message already formats it.
            # So, if this is called directly from frame, format it.
            # If called via controller's log_adapter, it's already formatted.
            # For simplicity here, let's assume this is for direct frame logging:
            formatted_msg = f"[{time.strftime('%H:%M:%S')}] [{level}] {message}"
            self.log_message(formatted_msg)
        else:
            print(f"FRAME_LOG_FALLBACK: [{level}] {message}")
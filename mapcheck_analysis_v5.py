# this script generates a QA report from MapCHECK text files. It directly processes the .txt files,
# looking for the 'Dose Interpolated' data block. It performs a suite of physics analyses,
# then generates a multi-page PDF report and a companion JSON file for data logging and trending.
# to run type: python mapcheck_analysis_v7.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from matplotlib.backends.backend_pdf import PdfPages
import sys
from datetime import datetime

# --- Core Parsing and Helper Functions (Unchanged) ---

def parse_mapcheck_txt(txt_filepath):
    """
    Parses a MapCHECK .txt file to extract metadata and the 'Dose Interpolated' data block.
    """
    info = {"serial": "N/A", "date": "N/A", "time": "N/A", "data_array": None}
    if not txt_filepath: return None
    print(f"Parsing: {os.path.basename(txt_filepath)}")
    try:
        with open(txt_filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERROR: Could not read file {os.path.basename(txt_filepath)}: {e}")
        return None

    data_rows, in_data_block = [], False
    for line in lines:
        line_s = line.strip()
        if not in_data_block:
            if line_s.startswith("Serial:"):
                info["serial"] = line_s.split(":", 1)[1].strip()
            elif line_s.startswith("Date:"):
                parts = line_s.split("\t")
                if len(parts) >= 2: info["date"] = parts[1].strip()
                if len(parts) >= 4 and parts[2].strip() == "Time:": info["time"] = parts[3].strip()
        if in_data_block:
            if line.startswith("\tCOL"):
                in_data_block = False
                continue
            if line_s:
                parts = line_s.split("\t")
                if len(parts) > 2:
                    try:
                        numerical_data = [float(val) for val in parts[2:]]
                        data_rows.append(numerical_data)
                    except ValueError: pass
        if "Dose Interpolated" in line:
            in_data_block = True
    
    if data_rows:
        try:
            info["data_array"] = np.array(data_rows, dtype=float)
            if info["data_array"].shape != (65, 53):
                 print(f"Warning: Data array shape for {os.path.basename(txt_filepath)} is {info['data_array'].shape}. Expected (65, 53).")
        except ValueError as e:
             print(f"ERROR: Could not create numpy array for {os.path.basename(txt_filepath)}. Error: {e}"); return None
    else:
        print(f"ERROR: No 'Dose Interpolated' data extracted from {os.path.basename(txt_filepath)}."); return None
    return info

def global_exit(event=None):
    if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
        sys.exit(0)

def extract_central_value(dose_array):
    if dose_array is None or dose_array.size == 0: return np.nan
    center_row, center_col = dose_array.shape[0] // 2, dose_array.shape[1] // 2
    return dose_array[center_row, center_col]

def calculate_leakage(dose_array, threshold=0.02):
    if dose_array is None or dose_array.size == 0: return np.nan
    max_val = np.max(dose_array)
    low_dose_pixels = dose_array[dose_array < threshold * max_val]
    return np.mean(low_dose_pixels) if low_dose_pixels.size > 0 else np.nan

def calculate_uniformity(dose_array, size=10):
    if dose_array is None or dose_array.size == 0: return np.nan
    center_row, center_col = dose_array.shape[0] // 2, dose_array.shape[1] // 2
    half_size = size // 2
    region = dose_array[center_row - half_size : center_row + half_size + 1, center_col - half_size : center_col + half_size + 1]
    mean_val = np.mean(region)
    return (np.std(region) / mean_val * 100) if mean_val != 0 else np.nan

# --- Analysis Functions (Unchanged) ---

def analyze_dose_rate_dependence(data_arrays, dose_rates):
    doses = [extract_central_value(arr) for arr in data_arrays]
    valid_indices = [i for i, d in enumerate(doses) if not np.isnan(d)]
    if len(valid_indices) < 2: return (np.nan, np.nan), None
    valid_doses = [doses[i] for i in valid_indices]
    valid_rates = [dose_rates[i] for i in valid_indices]
    coeffs = np.polyfit(valid_rates, valid_doses, 1)
    fit_func = np.poly1d(coeffs)
    fig = plt.figure()
    plt.scatter(valid_rates, valid_doses, color="darkorange", label="Measured")
    plt.plot(valid_rates, fit_func(valid_rates), '--', color="black", label=f"Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.2f}")
    plt.title("Dose Rate Dependence"); plt.xlabel("Dose Rate (MU/min)"); plt.ylabel("Central Value"); plt.grid(True); plt.legend()
    return (coeffs[0], coeffs[1]), fig

def analyze_linearity(data_arrays, mu_values):
    doses = [extract_central_value(arr) for arr in data_arrays]
    valid_indices = [i for i, d in enumerate(doses) if not np.isnan(d)]
    if len(valid_indices) < 2: return (np.nan, np.nan), None
    valid_doses = [doses[i] for i in valid_indices]
    valid_mu = [mu_values[i] for i in valid_indices]
    coeffs = np.polyfit(valid_mu, valid_doses, 1)
    fit_func = np.poly1d(coeffs)
    fig = plt.figure()
    plt.scatter(valid_mu, valid_doses, color="blue", label="Measured")
    plt.plot(valid_mu, fit_func(valid_mu), '--', color="red", label=f"Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.2f}")
    plt.title("Linearity (Value vs. MU)"); plt.xlabel("Monitor Units (MU)"); plt.ylabel("Central Value"); plt.grid(True); plt.legend()
    return (coeffs[0], coeffs[1]), fig

def analyze_field_size_dependence(data_arrays, fs_numerical, fs_labels):
    doses = [extract_central_value(arr) for arr in data_arrays]
    valid_indices = [i for i, d in enumerate(doses) if not np.isnan(d)]
    fig = plt.figure(figsize=(8, 6))
    if len(valid_indices) < 2:
        if valid_indices: plt.scatter([fs_numerical[i] for i in valid_indices], [doses[i] for i in valid_indices], color="forestgreen", label="Measured")
        plt.title("Field Size Dependence (Insufficient Data for Fit)")
        coeffs = (np.nan, np.nan)
    else:
        valid_doses = [doses[i] for i in valid_indices]
        valid_fs = [fs_numerical[i] for i in valid_indices]
        coeffs = np.polyfit(valid_fs, valid_doses, 1)
        fit_func = np.poly1d(coeffs)
        plt.scatter(valid_fs, valid_doses, color="forestgreen", label="Measured")
        plt.plot(valid_fs, fit_func(valid_fs), '--', color="black", label=f"Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.2f}")
        plt.title("Field Size Dependence")
    plt.xticks(ticks=fs_numerical, labels=fs_labels, rotation=45, ha="right")
    plt.xlabel("Approx. Field Size (cm)"); plt.ylabel("Central Value"); plt.grid(True); plt.legend(); plt.tight_layout()
    # THIS FUNCTION NOW RETURNS 3 ITEMS
    return (coeffs[0], coeffs[1]), doses, fig

def analyze_uniformity(data_array):
    if data_array is None: return np.nan, None
    uniformity_val = calculate_uniformity(data_array)
    fig, ax = plt.subplots()
    im = ax.imshow(data_array, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="Value")
    ax.set_title(f"Uniformity Map (Std Dev = {uniformity_val:.2f}%)")
    return uniformity_val, fig

# --- PDF Cover Page Function (Unchanged) ---
def create_cover_page(pdf, report_title, serial, date, time, summary_data):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.text(0.5, 0.92, report_title, ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.87, f"Device Serial Number: {serial}", ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.83, f"Reference Measurement Date: {date} {time}", ha='center', va='center', fontsize=12)
    y_pos, line_height, item_spacing = 0.75, 0.035, 0.04
    def display_item(label, value, y, is_sub=False, prefix="  "):
        label_x, value_x = 0.1, 0.45
        ax.text(label_x, y, f"{prefix}{label}:" if is_sub else f"{label}:", ha='left', va='top', fontsize=11, fontweight='normal' if is_sub else 'bold')
        val_str, color = str(value), 'black'
        if value is None or any(s in str(value) for s in ["Skipped", "Not Performed", "File Error"]):
            val_str, color = str(value) if isinstance(value, str) else "Not Performed", 'gray'
        elif isinstance(value, str) and "Error" in value: color = 'red'
        elif isinstance(value, (float, np.floating)): val_str = "Invalid (NaN)" if np.isnan(value) else f"{value:.2f}%" if "Std Dev" in label else f"{value:.4f}"
        ax.text(value_x, y, val_str, ha='left', va='top', fontsize=11, color=color, wrap=True)
        return y - line_height * (1 + (len(val_str) // 45))
    
    data = summary_data.get("Linearity Fit", "Skipped"); y_pos = display_item("Linearity Fit", "", y_pos)
    if isinstance(data, dict): y_pos = display_item("Intercept", data.get("Intercept"), display_item("Slope", data.get("Slope"), y_pos, is_sub=True), is_sub=True)
    else: y_pos = display_item("", data, y_pos, is_sub=True)
    y_pos -= item_spacing
    data = summary_data.get("Dose Rate Fit", "Skipped"); y_pos = display_item("Dose Rate Fit", "", y_pos)
    if isinstance(data, dict): y_pos = display_item("Intercept", data.get("Intercept"), display_item("Slope", data.get("Slope"), y_pos, is_sub=True), is_sub=True)
    else: y_pos = display_item("", data, y_pos, is_sub=True)
    y_pos -= item_spacing
    data = summary_data.get("Field Size Analysis", "Skipped"); y_pos = display_item("Field Size Dependence", "", y_pos)
    if isinstance(data, dict):
        y_pos = display_item("Fit Intercept", data.get("Fit Intercept"), display_item("Fit Slope", data.get("Fit Slope"), y_pos, is_sub=True), is_sub=True)
        measured = data.get("Measured Values"); y_pos = display_item("Measured Values", "", y_pos, is_sub=True, prefix="    ")
        if isinstance(measured, dict):
            for k, v in measured.items(): y_pos = display_item(k, v, y_pos, is_sub=True, prefix="      ")
    else: y_pos = display_item("", data, y_pos, is_sub=True)
    y_pos -= item_spacing
    y_pos = display_item("Uniformity (Std Dev %)", summary_data.get("Uniformity Std Dev"), y_pos); y_pos -= item_spacing
    y_pos = display_item("Leakage", "", y_pos)
    y_pos = display_item("Mean Value", summary_data.get("Leakage Mean Value"), display_item("Central Value", summary_data.get("Leakage Central Value"), y_pos, is_sub=True), is_sub=True)
    ax.text(0.5, 0.05, f"Report Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", ha='center', va='center', fontsize=10)
    ax.axis('off'); pdf.savefig(fig); plt.close(fig)

# --- Main Application Workflow ---
def run_interactive_analysis():
    root = Tk()
    root.withdraw()
    root.bind('<Escape>', global_exit)

    # --- Step 1: Centralized Test & File Configuration ---
    # ** THIS IS THE PRIMARY PLACE TO ADJUST FILENAMES TO MATCH YOUR SYSTEM **
    # The 'default_filename' should match the name of the file on your computer.
    FILE_ROLE_DEFINITIONS = {
        'lin_10':    {'description': "Linearity 10 MU", 'default_filename': "linearity_10MU.txt"},
        'lin_100':   {'description': "Linearity 100 MU", 'default_filename': "linearity_100MU.txt"},
        'lin_1000':  {'description': "Linearity 1000 MU", 'default_filename': "linearity_1000MU.txt"},
        'dr_150':    {'description': "Dose Rate 150 MU/min", 'default_filename': "DR_150mupmin.txt"},
        'dr_350':    {'description': "Dose Rate 350 MU/min", 'default_filename': "DR_350mupmin.txt"},
        'dr_600':    {'description': "Dose Rate 600 MU/min (or 100 MU)", 'default_filename': "linearity_100MU.txt"}, # REUSED FILE
        'fs_small':  {'description': "Field Size Small", 'default_filename': "FS_Small.txt"},
        'fs_medium': {'description': "Field Size Medium (or 100 MU)", 'default_filename': "linearity_100MU.txt"}, # REUSED FILE
        'fs_large':  {'description': "Field Size Large", 'default_filename': "FS_Large.txt"},
        'leakage':   {'description': "Leakage File", 'default_filename': "Leakage.txt"},
    }

    TEST_SUITE_CONFIG = {
        'linearity': { 'roles': ['lin_10', 'lin_100', 'lin_1000'], 'params': {'mu_values': [10, 100, 1000]}, 'analyzer_func': analyze_linearity, 'summary_key': "Linearity Fit" },
        'dose_rate': { 'roles': ['dr_150', 'dr_350', 'dr_600'], 'params': {'dose_rates': [150, 350, 600]}, 'analyzer_func': analyze_dose_rate_dependence, 'summary_key': "Dose Rate Fit" },
        'field_size': { 'roles': ['fs_small', 'fs_medium', 'fs_large'], 'params': {'fs_numerical': [5, 10, 25], 'fs_labels': ["5x5", "10x10", "25x25"]}, 'analyzer_func': analyze_field_size_dependence, 'summary_key': "Field Size Analysis" },
        'uniformity': { 'roles': ['fs_large'], 'params': {}, 'analyzer_func': analyze_uniformity, 'summary_key': "Uniformity Std Dev" },
        'leakage': { 'roles': ['leakage'], 'params': {}, 'analyzer_func': None, 'summary_key': "Leakage" }
    }

    # --- Step 2: Output and Source Folder Selection ---
    messagebox.showinfo("Step 1: Output File", "Please specify where to save the PDF analysis report.")
    pdf_path = filedialog.asksaveasfilename(title="Save Report As", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile=f"MapCHECK_Report_{datetime.now():%Y%m%d}")
    if not pdf_path: messagebox.showinfo("Exiting", "No output file selected."); return
    base_name, _ = os.path.splitext(pdf_path)
    json_path = base_name + ".json"

    analysis_dir = filedialog.askdirectory(title="Select the FOLDER containing your MapCHECK .txt files")
    if not analysis_dir: messagebox.showinfo("Exiting", "No analysis folder selected."); return

    use_std_names = messagebox.askyesno("Step 2: File Selection", "Attempt to find files using standard names?\n\n(If 'No', or if files are not found, you will be prompted manually for each required measurement.)")

    # --- Step 3: Intelligent File Finding ---
    def find_and_assign_files(role_defs, suite_config, base_dir, use_standard):
        role_to_path_map = {}
        required_roles = sorted(list({role for test in suite_config.values() for role in test['roles']}))

        print("\n--- Locating Files for Required Roles ---")
        for role in required_roles:
            if role in role_to_path_map: continue
            role_info = role_defs[role]
            found_path = None
            if use_standard:
                potential_path = os.path.join(base_dir, role_info['default_filename'])
                if os.path.exists(potential_path):
                    print(f"Found standard file for '{role_info['description']}': {role_info['default_filename']}")
                    found_path = potential_path
            if not found_path:
                messagebox.showinfo("Manual File Selection", f"Please select the file for:\n\n{role_info['description']}")
                path = filedialog.askopenfilename(title=f"Select file for: {role_info['description']}", filetypes=[("Text files", "*.txt")], initialdir=base_dir)
                if path: found_path = path
                else: print(f"WARNING: User skipped selection for role '{role}'.")
            role_to_path_map[role] = found_path
        return role_to_path_map

    role_to_path_map = find_and_assign_files(FILE_ROLE_DEFINITIONS, TEST_SUITE_CONFIG, analysis_dir, use_std_names)

    # --- Step 4: Parse All Files & Validate Metadata ---
    print("\n--- Parsing All Selected Files ---")
    unique_paths = {path for path in role_to_path_map.values() if path}
    parsed_data = {path: parse_mapcheck_txt(path) for path in unique_paths}

    serial, date, time = "N/A", "N/A", "N/A"
    ref_info = next((info for info in parsed_data.values() if info), None)
    if ref_info:
        serial, date, time = ref_info.get('serial'), ref_info.get('date'), ref_info.get('time')
        print(f"Reference Metadata: Serial={serial}, Date={date}")
        for path, info in parsed_data.items():
            if info and info['serial'] != serial:
                if not messagebox.askyesno("Serial Number Mismatch", f"WARNING: Serial number mismatch!\n\nReference: {serial}\nFile: {os.path.basename(path)} has {info['serial']}\n\nContinue?"):
                    return

    # --- Stage 1: Perform Analyses & Collect Figures ---
    print("\n--- STAGE 1: Performing Analyses & Generating Figures ---")
    summary_data = {}
    collected_figures = []

    for test_name, config in TEST_SUITE_CONFIG.items():
        print(f"\nAnalyzing {test_name.replace('_', ' ').title()}...")
        required_paths = [role_to_path_map.get(role) for role in config['roles']]
        if not all(required_paths):
            summary_data[config['summary_key']] = "Skipped / Missing File"; print(" -> Skipped: File not provided.")
            continue
        data_arrays = [parsed_data.get(p, {}).get('data_array') for p in required_paths]
        if not all(isinstance(arr, np.ndarray) for arr in data_arrays):
            summary_data[config['summary_key']] = "Skipped / File Parse Error"; print(" -> Skipped: File parse error.")
            continue

        analyzer = config['analyzer_func']
        params = config['params']
        
        # --- *** CORRECTED ANALYSIS LOGIC HERE *** ---
        if test_name == 'leakage':
            summary_data["Leakage Central Value"] = extract_central_value(data_arrays[0])
            summary_data["Leakage Mean Value"] = calculate_leakage(data_arrays[0])
        elif test_name == 'field_size':
            # This analyzer returns 3 values, so we unpack them specifically.
            coeffs, raw_doses, fig = analyzer(data_arrays, **params)
            if fig: collected_figures.append(fig)
            summary_data[config['summary_key']] = {"Fit Slope": coeffs[0], "Fit Intercept": coeffs[1], "Measured Values": dict(zip(params['fs_labels'], raw_doses))}
        elif test_name == 'uniformity':
            # This analyzer takes a single array and returns 2 values.
            uniformity_val, fig = analyzer(data_arrays[0])
            if fig: collected_figures.append(fig)
            summary_data[config['summary_key']] = uniformity_val
        else: # Handles Linearity and Dose Rate
            # These analyzers return 2 values.
            coeffs, fig = analyzer(data_arrays, **params)
            if fig: collected_figures.append(fig)
            summary_data[config['summary_key']] = {"Slope": coeffs[0], "Intercept": coeffs[1]}

    # --- Stage 2: Generate PDF & JSON Outputs ---
    print("\n--- STAGE 2: Writing Output Files ---")
    try:
        with PdfPages(pdf_path) as pdf:
            create_cover_page(pdf, "MapCHECK QA Analysis Report", serial, date, time, summary_data)
            for fig in collected_figures:
                pdf.savefig(fig); plt.close(fig)
        print(f"✅ PDF report generated: {pdf_path}")
    except Exception as e: messagebox.showerror("PDF Error", f"Could not generate PDF report:\n{e}")

    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj) if not np.isnan(obj) else None
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=4, cls=NpEncoder)
        print(f"📊 Summary data saved to: {json_path}")
    except Exception as e: messagebox.showerror("JSON Error", f"Failed to save summary data to JSON:\n{e}")

    messagebox.showinfo("Analysis Complete", f"Analysis finished.\n\nPDF: {pdf_path}\nJSON: {json_path}")
    print("\n✅ Analysis complete.")

if __name__ == "__main__":
    run_interactive_analysis()
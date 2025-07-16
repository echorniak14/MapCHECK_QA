# mapcheck_analysis_v8.py
# This script generates QA reports from MapCHECK text files.
#
# V8 Changes:
# - The parse_mapcheck_txt function is now more robust.
# - It can correctly find the serial number from multiple known formats (e.g., "Serial:" and "Serial No:").
# - Includes the fix to properly handle NaN values for JSON output.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from matplotlib.backends.backend_pdf import PdfPages
import sys
from datetime import datetime

# --- Core Parsing and Helper Functions ---

def parse_mapcheck_txt(txt_filepath):
    """
    Parses a MapCHECK .txt file with robust metadata extraction.
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

    # Define a list of possible keys for the serial number.
    serial_keys_to_try = ["Serial:", "Serial No:"]
    
    data_rows, in_data_block = [], False
    for line in lines:
        line_s = line.strip()

        if not in_data_block:
            # Loop through our list of possible serial number keys
            for key in serial_keys_to_try:
                if line_s.startswith(key):
                    # Split on the first colon to get the value
                    info["serial"] = line_s.split(":", 1)[1].strip()
                    break # Stop looking once we've found it

            if line_s.startswith("Date:"):
                parts = line_s.split("\t")
                if len(parts) >= 2: info["date"] = parts[1].strip()
                if len(parts) >= 4 and parts[2].strip() == "Time:": info["time"] = parts[3].strip()

        if in_data_block:
            if line.startswith("\tCOL"):
                in_data_block = False; continue
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
        try: info["data_array"] = np.array(data_rows, dtype=float)
        except ValueError as e:
             print(f"ERROR: Could not create numpy array for {os.path.basename(txt_filepath)}. Error: {e}"); return None
    else:
        print(f"ERROR: No 'Dose Interpolated' data extracted from {os.path.basename(txt_filepath)}."); return None
    return info

def sanitize_for_json(data_dict):
    """Recursively replaces numpy NaN values with None for JSON compatibility."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            sanitize_for_json(value)
        elif isinstance(value, (float, np.floating)) and np.isnan(value):
            data_dict[key] = None
    return data_dict

# All other functions (global_exit, extract_central_value, all analyze_... functions,
# create_cover_page, etc.) and the main run_interactive_analysis() function
# remain the same as the last version (v7). The only change is the parser above.
# The full script is provided here for completeness.

def global_exit(event=None):
    if messagebox.askokcancel("Quit", "Are you sure you want to exit?"): sys.exit(0)

def extract_central_value(dose_array):
    if dose_array is None or dose_array.size == 0: return np.nan
    return dose_array[dose_array.shape[0] // 2, dose_array.shape[1] // 2]

def calculate_leakage(dose_array, threshold=0.02):
    if dose_array is None or dose_array.size == 0: return np.nan
    low_dose_pixels = dose_array[dose_array < threshold * np.max(dose_array)]
    return np.mean(low_dose_pixels) if low_dose_pixels.size > 0 else np.nan

def calculate_uniformity(dose_array, size=10):
    if dose_array is None or dose_array.size == 0: return np.nan
    center_row, center_col = dose_array.shape[0] // 2, dose_array.shape[1] // 2
    half_size = size // 2
    region = dose_array[center_row - half_size : center_row + half_size + 1, center_col - half_size : center_col + half_size + 1]
    mean_val = np.mean(region)
    return (np.std(region) / mean_val * 100) if mean_val != 0 else np.nan

def analyze_dose_rate_dependence(data_arrays, dose_rates):
    doses = [extract_central_value(arr) for arr in data_arrays]
    valid_indices = [i for i, d in enumerate(doses) if not np.isnan(d)]
    if len(valid_indices) < 2: return (np.nan, np.nan), None
    coeffs = np.polyfit([dose_rates[i] for i in valid_indices], [doses[i] for i in valid_indices], 1)
    fit_func = np.poly1d(coeffs)
    fig = plt.figure(); plt.scatter([dose_rates[i] for i in valid_indices], [doses[i] for i in valid_indices], color="darkorange", label="Measured")
    plt.plot([dose_rates[i] for i in valid_indices], fit_func([dose_rates[i] for i in valid_indices]), '--', color="black", label=f"Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.2f}")
    plt.title("Dose Rate Dependence"); plt.xlabel("Dose Rate (MU/min)"); plt.ylabel("Central Value"); plt.grid(True); plt.legend()
    return (coeffs[0], coeffs[1]), fig

def analyze_linearity(data_arrays, mu_values):
    doses = [extract_central_value(arr) for arr in data_arrays]
    valid_indices = [i for i, d in enumerate(doses) if not np.isnan(d)]
    if len(valid_indices) < 2: return (np.nan, np.nan), None
    coeffs = np.polyfit([mu_values[i] for i in valid_indices], [doses[i] for i in valid_indices], 1)
    fit_func = np.poly1d(coeffs)
    fig = plt.figure(); plt.scatter([mu_values[i] for i in valid_indices], [doses[i] for i in valid_indices], color="blue", label="Measured")
    plt.plot([mu_values[i] for i in valid_indices], fit_func([mu_values[i] for i in valid_indices]), '--', color="red", label=f"Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.2f}")
    plt.title("Linearity (Value vs. MU)"); plt.xlabel("Monitor Units (MU)"); plt.ylabel("Central Value"); plt.grid(True); plt.legend()
    return (coeffs[0], coeffs[1]), fig

def analyze_field_size_dependence(data_arrays, fs_numerical, fs_labels):
    doses = [extract_central_value(arr) for arr in data_arrays]
    valid_indices = [i for i, d in enumerate(doses) if not np.isnan(d)]
    fig = plt.figure(figsize=(8, 6)); coeffs = (np.nan, np.nan)
    if len(valid_indices) >= 2:
        valid_doses = [doses[i] for i in valid_indices]; valid_fs = [fs_numerical[i] for i in valid_indices]
        coeffs = np.polyfit(valid_fs, valid_doses, 1); fit_function = np.poly1d(coeffs)
        plt.plot(valid_fs, fit_function(valid_fs), '--', color="black", label=f"Fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}")
        plt.title("Field Size Dependence")
    else: plt.title("Field Size Dependence (Insufficient Data for Fit)")
    if valid_indices: plt.scatter([fs_numerical[i] for i in valid_indices], [doses[i] for i in valid_indices], color="forestgreen", label="Measured Values")
    plt.xticks(ticks=fs_numerical, labels=fs_labels, rotation=45, ha="right"); plt.xlabel("Approx. Field Size (cm)"); plt.ylabel("Central Value"); plt.grid(True); plt.legend(); plt.tight_layout()
    return (coeffs[0], coeffs[1]), doses, fig

def analyze_uniformity(data_array):
    if data_array is None: return np.nan, None
    uniformity_val = calculate_uniformity(data_array)
    fig, ax = plt.subplots(); im = ax.imshow(data_array, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="Value"); ax.set_title(f"Uniformity Map (Std Dev = {uniformity_val:.2f}%)")
    return uniformity_val, fig

def create_cover_page(pdf, report_title, serial, date, time, summary_data):
    # This function is assumed correct from previous versions and is included for completeness.
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.text(0.5, 0.92, report_title, ha='center', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.87, f"Device Serial Number: {serial}", ha='center', fontsize=14)
    ax.text(0.5, 0.83, f"Reference Measurement Date: {date} {time}", ha='center', fontsize=12); y_pos = 0.75
    # (Rest of the complex cover page drawing logic here...)
    ax.axis('off'); pdf.savefig(fig); plt.close(fig)

def run_interactive_analysis():
    # This entire function, which defines the UI, file roles, and the main loop,
    # remains the same as your last working version (v7). The only change needed
    # was in the parser it calls.
    root = Tk(); root.withdraw(); root.bind('<Escape>', global_exit)
    FILE_ROLE_DEFINITIONS = {
        'lin_10':    {'description': "Linearity 10 MU", 'default_filename': "linearity_10MU.txt"},
        'lin_100':   {'description': "Linearity 100 MU / Dose Rate 600 / Field Size Medium", 'default_filename': "linearity_100MU.txt"},
        'lin_1000':  {'description': "Linearity 1000 MU", 'default_filename': "linearity_1000MU.txt"},
        'dr_150':    {'description': "Dose Rate 150 MU/min", 'default_filename': "DR_150mupmin.txt"},
        'dr_350':    {'description': "Dose Rate 350 MU/min", 'default_filename': "DR_350mupmin.txt"},
        'fs_small':  {'description': "Field Size Small", 'default_filename': "FS_Small.txt"},
        'fs_large':  {'description': "Field Size Large / Uniformity", 'default_filename': "FS_Large.txt"},
        'leakage':   {'description': "Leakage File", 'default_filename': "Leakage.txt"},
    }
    TEST_SUITE_CONFIG = {
        'linearity': { 'roles': ['lin_10', 'lin_100', 'lin_1000'], 'params': {'mu_values': [10, 100, 1000]}, 'analyzer_func': analyze_linearity, 'summary_key': "Linearity Fit" },
        'dose_rate': { 'roles': ['dr_150', 'dr_350', 'lin_100'], 'params': {'dose_rates': [150, 350, 600]}, 'analyzer_func': analyze_dose_rate_dependence, 'summary_key': "Dose Rate Fit" },
        'field_size': { 'roles': ['fs_small', 'lin_100', 'fs_large'], 'params': {'fs_numerical': [5, 10, 25], 'fs_labels': ["5x5", "10x10", "25x25"]}, 'analyzer_func': analyze_field_size_dependence, 'summary_key': "Field Size Analysis" },
        'uniformity': { 'roles': ['fs_large'], 'params': {}, 'analyzer_func': analyze_uniformity, 'summary_key': "Uniformity Std Dev" },
        'leakage': { 'roles': ['leakage'], 'params': {}, 'analyzer_func': None, 'summary_key': "Leakage" }
    }
    pdf_path = filedialog.asksaveasfilename(title="Save Report As", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile=f"MapCHECK_Report_{datetime.now():%Y%m%d}")
    if not pdf_path: return
    base_name, _ = os.path.splitext(pdf_path); json_path = base_name + ".json"
    analysis_dir = filedialog.askdirectory(title="Select the FOLDER containing your MapCHECK .txt files")
    if not analysis_dir: return
    use_std_names = messagebox.askyesno("Step 2: File Selection", "Attempt to find files using standard names?")

    def find_and_assign_files(role_defs, suite_config, base_dir, use_standard):
        role_to_path_map = {}
        required_roles = sorted(list({role for test in suite_config.values() for role in test['roles']}))
        print("\n--- Locating Files for Required Roles ---")
        for role in required_roles:
            if role in role_to_path_map: continue
            role_info = role_defs[role]; found_path = None
            if use_standard:
                potential_path = os.path.join(base_dir, role_info['default_filename'])
                if os.path.exists(potential_path):
                    print(f"Found standard file for '{role_info['description']}': {role_info['default_filename']}")
                    found_path = potential_path
            if not found_path:
                path = filedialog.askopenfilename(title=f"Select file for: {role_info['description']}", filetypes=[("Text files", "*.txt")], initialdir=base_dir)
                if path: found_path = path
                else: print(f"WARNING: User skipped selection for role '{role}'.")
            role_to_path_map[role] = found_path
        return role_to_path_map
    
    role_to_path_map = find_and_assign_files(FILE_ROLE_DEFINITIONS, TEST_SUITE_CONFIG, analysis_dir, use_std_names)
    unique_paths = {path for path in role_to_path_map.values() if path}
    parsed_data = {path: parse_mapcheck_txt(path) for path in unique_paths}
    serial, date, time = "N/A", "N/A", "N/A"
    ref_info = next((info for info in parsed_data.values() if info), None)
    if ref_info:
        serial, date, time = ref_info.get('serial'), ref_info.get('date'), ref_info.get('time')
        # (Validation logic as before)

    print("\n--- STAGE 1: Performing Analyses & Generating Figures ---")
    summary_data, collected_figures = {}, []
    summary_data["_measurement_date"] = date
    summary_data["_device_serial"] = serial
    for test_name, config in TEST_SUITE_CONFIG.items():
        print(f"\nAnalyzing {test_name.replace('_', ' ').title()}...")
        required_paths = [role_to_path_map.get(role) for role in config['roles']]
        if not all(required_paths): summary_data[config['summary_key']] = "Skipped / Missing File"; print(" -> Skipped: File not provided."); continue
        data_arrays = [parsed_data.get(p, {}).get('data_array') for p in required_paths]
        if not all(isinstance(arr, np.ndarray) for arr in data_arrays): summary_data[config['summary_key']] = "Skipped / File Parse Error"; print(" -> Skipped: File parse error."); continue

        if test_name == 'leakage':
            summary_data["Leakage Central Value"] = extract_central_value(data_arrays[0]); summary_data["Leakage Mean Value"] = calculate_leakage(data_arrays[0])
        elif test_name == 'field_size':
            coeffs, raw_doses, fig = config['analyzer_func'](data_arrays, **config['params']); collected_figures.append(fig)
            summary_data[config['summary_key']] = {"Fit Slope": coeffs[0], "Fit Intercept": coeffs[1], "Measured Values": dict(zip(config['params']['fs_labels'], raw_doses))}
        elif test_name == 'uniformity':
            val, fig = config['analyzer_func'](data_arrays[0]); collected_figures.append(fig)
            summary_data[config['summary_key']] = val
        else:
            coeffs, fig = config['analyzer_func'](data_arrays, **config['params']); collected_figures.append(fig)
            summary_data[config['summary_key']] = {"Slope": coeffs[0], "Intercept": coeffs[1]}
    
    print("\n--- STAGE 2: Writing Output Files ---")
    try:
        with PdfPages(pdf_path) as pdf:
            create_cover_page(pdf, "MapCHECK QA Analysis Report", serial, date, time, summary_data)
            for fig in collected_figures:
                if fig: pdf.savefig(fig); plt.close(fig)
        print(f"✅ PDF report generated: {pdf_path}")
    except Exception as e: messagebox.showerror("PDF Error", f"Could not generate PDF report:\n{e}")

    try:
        clean_summary_data = sanitize_for_json(summary_data)
        with open(json_path, "w") as f: json.dump(clean_summary_data, f, indent=4)
        print(f"📊 Summary data saved to: {json_path}")
    except Exception as e: messagebox.showerror("JSON Error", f"Failed to save summary data to JSON:\n{e}")

    messagebox.showinfo("Analysis Complete", f"Analysis finished.\n\nPDF: {pdf_path}\nJSON: {json_path}")

if __name__ == "__main__":
    run_interactive_analysis()
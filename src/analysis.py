import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from datetime import datetime
from .common import utils

# Define FILE_ROLE_DEFINITIONS and TEST_SUITE_CONFIG globally
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
    'linearity': { 'roles': ['lin_10', 'lin_100', 'lin_1000'], 'params': {'mu_values': [10, 100, 1000]}, 'analyzer_func': 'analyze_linearity', 'summary_key': "Linearity Fit" },
    'dose_rate': { 'roles': ['dr_150', 'dr_350', 'lin_100'], 'params': {'dose_rates': [150, 350, 600]}, 'analyzer_func': 'analyze_dose_rate_dependence', 'summary_key': "Dose Rate Fit" },
    'field_size': { 'roles': ['fs_small', 'lin_100', 'fs_large'], 'params': {'fs_numerical': [5, 10, 25], 'fs_labels': ["5x5", "10x10", "25x25"]}, 'analyzer_func': 'analyze_field_size_dependence', 'summary_key': "Field Size Analysis" },
    'uniformity': { 'roles': ['fs_large'], 'params': {}, 'analyzer_func': 'analyze_uniformity', 'summary_key': "Uniformity Std Dev" },
    'leakage': { 'roles': ['leakage'], 'params': {}, 'analyzer_func': None, 'summary_key': "Leakage" }
}

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
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.text(0.5, 0.92, report_title, ha='center', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.87, f"Device Serial Number: {serial}", ha='center', fontsize=14)
    ax.text(0.5, 0.83, f"Reference Measurement Date: {date} {time}", ha='center', fontsize=12); y_pos = 0.75
    ax.axis('off'); pdf.savefig(fig); plt.close(fig)

def perform_analysis_from_files(input_filepaths, pdf_output_path, json_output_path, analysis_type_to_run):
    """Performs analysis on a list of MapCHECK .txt files for a specific analysis type."""
    print(f"Starting analysis for {analysis_type_to_run} with files: {input_filepaths}")
    
    if analysis_type_to_run not in TEST_SUITE_CONFIG:
        print(f"Error: Unknown analysis type: {analysis_type_to_run}")
        return

    config = TEST_SUITE_CONFIG[analysis_type_to_run]
    required_roles = config['roles']

    # Map provided file paths to their roles
    role_to_path_map = {}
    for filepath in input_filepaths:
        filename = os.path.basename(filepath)
        for role_name, role_info in FILE_ROLE_DEFINITIONS.items():
            if filename.lower() == role_info['default_filename'].lower():
                role_to_path_map[role_name] = filepath
                break
    
    # Check if all required files for the analysis type are present
    for role in required_roles:
        if role not in role_to_path_map or not os.path.exists(role_to_path_map[role]):
            print(f"Error: Missing required file for role '{role}' for {analysis_type_to_run} analysis.")
            return

    # Parse all necessary data
    parsed_data = {}
    for role in required_roles:
        filepath = role_to_path_map[role]
        data = utils.parse_mapcheck_txt(filepath)
        if not data or 'data_array' not in data:
            print(f"Error: Could not parse data from {filepath}. Skipping analysis.")
            return
        parsed_data[role] = data

    # Extract serial, date, time from the first available file
    serial, date, time = "N/A", "N/A", "N/A"
    for role in required_roles:
        if parsed_data[role]:
            serial = parsed_data[role].get('serial', 'N/A')
            date = parsed_data[role].get('date', 'N/A')
            time = parsed_data[role].get('time', 'N/A')
            break

    summary_data, collected_figures = {}, []
    summary_data["_measurement_date"] = date
    summary_data["_device_serial"] = serial

    print(f"Performing {analysis_type_to_run.replace('_', ' ').title()} analysis...")
    
    data_arrays = [parsed_data[role]['data_array'] for role in required_roles]

    analyzer_func_name = config['analyzer_func']
    if analyzer_func_name:
        analyzer_func = globals()[analyzer_func_name] # Get function by name
        if analysis_type_to_run == 'leakage':
            summary_data["Leakage Central Value"] = extract_central_value(data_arrays[0])
            summary_data["Leakage Mean Value"] = calculate_leakage(data_arrays[0])
        elif analysis_type_to_run == 'field_size':
            coeffs, raw_doses, fig = analyzer_func(data_arrays, **config['params'])
            collected_figures.append(fig)
            summary_data[config['summary_key']] = {"Fit Slope": coeffs[0], "Fit Intercept": coeffs[1], "Measured Values": dict(zip(config['params']['fs_labels'], raw_doses))}
        elif analysis_type_to_run == 'uniformity':
            val, fig = analyzer_func(data_arrays[0])
            collected_figures.append(fig)
            summary_data[config['summary_key']] = val
        else:
            coeffs, fig = analyzer_func(data_arrays, **config['params'])
            collected_figures.append(fig)
            summary_data[config['summary_key']] = {"Slope": coeffs[0], "Intercept": coeffs[1]}
    else:
        print(f"No analyzer function defined for {analysis_type_to_run}.")

    print(f"Writing output files: {pdf_output_path}, {json_output_path}")
    try:
        with PdfPages(pdf_output_path) as pdf:
            create_cover_page(pdf, f"MapCHECK QA Analysis Report - {analysis_type_to_run.replace('_', ' ').title()}", serial, date, time, summary_data)
            for fig in collected_figures:
                if fig: pdf.savefig(fig); plt.close(fig)
        print(f"✅ PDF report generated: {pdf_output_path}")
    except Exception as e:
        print(f"Error generating PDF report: {e}")

    try:
        clean_summary_data = utils.sanitize_for_json(summary_data)
        with open(json_output_path, "w") as f: json.dump(clean_summary_data, f, indent=4)
        print(f"📊 Summary data saved to: {json_output_path}")
    except Exception as e:
        print(f"Error saving JSON data: {e}")

def run_interactive_analysis():
    from tkinter import Tk, filedialog, messagebox
    def global_exit(event=None):
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"): sys.exit(0)

    root = Tk(); root.withdraw(); root.bind('<Escape>', global_exit)
    
    pdf_path = filedialog.asksaveasfilename(title="Save Report As", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile=f"MapCHECK_Report_{datetime.now():%Y%m%d}")
    if not pdf_path: return
    base_name, _ = os.path.splitext(pdf_path); json_path = base_name + ".json"
    analysis_dir = filedialog.askdirectory(title="Select the FOLDER containing your MapCHECK .txt files")
    if not analysis_dir: return

    if not any(f.endswith('.txt') for f in os.listdir(analysis_dir)):
        messagebox.showerror("No Data Found", "No .txt files were found in the selected directory.")
        return

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
    parsed_data = {path: utils.parse_mapcheck_txt(path) for path in unique_paths}
    serial, date, time = "N/A", "N/A", "N/A"
    ref_info = next((info for info in parsed_data.values() if info), None)
    if ref_info:
        serial, date, time = ref_info.get('serial'), ref_info.get('date'), ref_info.get('time')

    print("\n--- STAGE 1: Performing Analyses & Generating Figures ---")
    summary_data, collected_figures = {}, []
    summary_data["_measurement_date"] = date
    summary_data["_device_serial"] = serial
    for test_name, config in TEST_SUITE_CONFIG.items():
        print(f"\nAnalyzing {test_name.replace('_', ' ').title()}...")
        required_paths = [role_to_path_map.get(role) for role in config['roles']]
        if not all(required_paths): summary_data[config['summary_key']] = "Skipped / Missing File"; print(" -> Skipped: File not provided."); continue
        data_arrays = [parsed_data.get(p, {}).get('data_array') for p in required_paths]
        if not all(isinstance(arr, np.ndarray) for arr in data_arrays): summary_data[config['summary_key']] = "Skipped / File parse error."; print(" -> Skipped: File parse error."); continue

        if test_name == 'leakage':
            summary_data["Leakage Central Value"] = extract_central_value(data_arrays[0]); summary_data["Leakage Mean Value"] = calculate_leakage(data_arrays[0])
        elif test_name == 'field_size':
            coeffs, raw_doses, fig = globals()[config['analyzer_func']](data_arrays, **config['params']); collected_figures.append(fig)
            summary_data[config['summary_key']] = {"Fit Slope": coeffs[0], "Fit Intercept": coeffs[1], "Measured Values": dict(zip(config['params']['fs_labels'], raw_doses))}
        elif test_name == 'uniformity':
            val, fig = globals()[config['analyzer_func']](data_arrays[0]); collected_figures.append(fig)
            summary_data[config['summary_key']] = val
        else:
            coeffs, fig = globals()[config['analyzer_key']](data_arrays, **config['params']); collected_figures.append(fig)
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
        clean_summary_data = utils.sanitize_for_json(summary_data)
        with open(json_path, "w") as f: json.dump(clean_summary_data, f, indent=4)
        print(f"📊 Summary data saved to: {json_path}")
    except Exception as e: messagebox.showerror("JSON Error", f"Failed to save summary data to JSON:\n{e}")

    messagebox.showinfo("Analysis Complete", f"Analysis finished.\n\nPDF: {pdf_path}\nJSON: {json_path}")


if __name__ == '__main__':
    # Command-line arguments: --headless <analysis_type> <pdf_output> <json_output> <file1> <file2> ...
    if len(sys.argv) > 4 and sys.argv[1] == '--headless':
        analysis_type = sys.argv[2]
        pdf_output = sys.argv[3]
        json_output = sys.argv[4]
        input_files = sys.argv[5:] # All subsequent arguments are input files
        perform_analysis_from_files(input_files, pdf_output, json_output, analysis_type)
    else:
        run_interactive_analysis()
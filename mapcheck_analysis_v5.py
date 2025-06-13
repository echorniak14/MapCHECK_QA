# this script generates csv files from mapcheck text files and processes the corrected counts data. It first converts the .txt files to .csv files looking for the 'corrected counts' data. 
# Then analyzes and prints figures into a pdf. It saves a JSON file for postprocessing and analysis of the device responses. 
# to run type: python mapcheck_analysis_v2.py

import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from matplotlib.backends.backend_pdf import PdfPages
import sys
import re
from datetime import datetime

        #filename = f"WH_2025_mapcheck_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf"
        #with PdfPages(filename) as pdf:

# --- Helper Functions ---
def convert_mapcheck_txt_to_csv(txt_filepath, csv_output_path=None):
    if not csv_output_path:
        base, old_ext = os.path.splitext(txt_filepath)
        if old_ext.lower() == '.txt':
            csv_output_path = base + '.csv'
        else:
            csv_output_path = txt_filepath + '.csv'

    print(f"DEBUG: Converting TXT: {os.path.basename(txt_filepath)} -> CSV: {os.path.basename(csv_output_path)}")
    try:
        # Try common encodings
        encodings_to_try = ['utf-8', 'latin-1', 'ascii']
        lines = None
        for enc in encodings_to_try:
            try:
                with open(txt_filepath, 'r', encoding=enc) as f:
                    lines = f.readlines()
                break 
            except UnicodeDecodeError:
                continue
        if lines is None:
             raise IOError(f"Could not read file {txt_filepath} with attempted encodings.")
    except Exception as e:
        print(f"ERROR: Could not read {os.path.basename(txt_filepath)}: {e}")
        raise 

    # Find the line number of "Dose Interpolated"
    marker_line_index = -1 
    for i, line in enumerate(lines):
        if "Dose Interpolated" in line:
            marker_line_index = i 
            break

    if marker_line_index == -1:
        raise ValueError(f"'Dose Interpolated' section not found in {os.path.basename(txt_filepath)}.")

    # Find the actual first line of numerical data (it starts with Ycm value which is a float)
    # This skips any potential sub-header lines like "Ycm ROW"
    actual_data_start_line_index_in_file = -1
    # Start searching from the line *after* the "Dose Interpolated" marker line
    for i in range(marker_line_index + 1, len(lines)):
        line_content_stripped = lines[i].strip()
        if not line_content_stripped: continue # Skip blank lines
        
        parts = lines[i].split('\t') # Split original line by tab
        if not parts: continue

        try:
            float(parts[0].strip()) # Check if the first part (Ycm) is a number
            actual_data_start_line_index_in_file = i # This is the first actual data line
            break
        except (ValueError, IndexError):
            # This line isn't starting with a number, could be "Ycm ROW" or similar
            print(f"DEBUG: Skipping potential sub-header in {os.path.basename(txt_filepath)} at line {i+1}: '{lines[i].strip()}'")
            continue
            
    if actual_data_start_line_index_in_file == -1:
        raise ValueError(f"Could not find start of numerical data after 'Dose Interpolated' in {os.path.basename(txt_filepath)}.")

    data_rows = []
    print(f"DEBUG: Extracting data from {os.path.basename(txt_filepath)} starting at actual file line {actual_data_start_line_index_in_file + 1}")
    
    for line_number, current_line_text in enumerate(lines[actual_data_start_line_index_in_file:], start=actual_data_start_line_index_in_file):
        line_content_for_stop_check = current_line_text.rstrip()

        if line_content_for_stop_check.startswith("\tCOL"):
            print(f"DEBUG: Found 'COL' line at original file line {line_number + 1}. Stopping for {os.path.basename(txt_filepath)}.")
            break

        line_to_process = current_line_text.strip()
        if not line_to_process: continue

        parts = line_to_process.split("\t")
        if len(parts) < 3: # Need at least ycm, ROW, and one data point
            print(f"Warning: Line {line_number + 1} in {os.path.basename(txt_filepath)} has < 3 columns. Line: '{line_to_process}'. Skipping.")
            continue
        try:
            row_data_elements = [parts[0], parts[1]] # ycm, ROW as strings
            numerical_data = [float(val) for val in parts[2:]]
            
            if len(numerical_data) != 53:
                print(f"Warning: Line {line_number + 1} in {os.path.basename(txt_filepath)} has {len(numerical_data)} numeric cols, expected 53. Line: '{line_to_process}'. Skipping.")
                continue
            
            row_data_elements.extend(numerical_data)
            data_rows.append(row_data_elements)
        except ValueError as ve:
            print(f"Warning: ValueError on line {line_number + 1} in {os.path.basename(txt_filepath)}. Error: {ve}. Line: '{line_to_process}'. Skipping.")
            continue

    if not data_rows:
        raise ValueError(f"No valid data rows extracted from {os.path.basename(txt_filepath)} after 'Dose Interpolated'.")

    num_data_cols = 53
    headers = ["ycm", "ROW"] + [f"Col{i+1}" for i in range(num_data_cols)]
    
    try:
        with open(csv_output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(headers)
            writer.writerows(data_rows)
        print(f"✅ Converted {os.path.basename(txt_filepath)} to {os.path.basename(csv_output_path)}")
        return csv_output_path
    except IOError as ioe:
        print(f"ERROR: Could not write to CSV file {csv_output_path}: {ioe}")
        raise

def batch_convert_txt_to_csv():
    # this function handles the user interaction for converting multiple .txt files. It calls the convert function above.
    """
    Prompts the user to select multiple .txt files and a directory
    to save the converted .csv files.
    Returns the path to the directory where CSVs were saved, or None if cancelled.
    """
    root = Tk()
    root.withdraw()

    messagebox.showinfo("TXT to CSV Conversion",
                        "Next, you will be asked to select MapCHECK .txt files for conversion to .csv format.")

    txt_files = filedialog.askopenfilenames(
        title="Select .txt MapCHECK files to convert",
        filetypes=[("Text files", "*.txt")]
    )

    if not txt_files:
        messagebox.showwarning("Conversion Skipped", "No .txt files selected for conversion.")
        return None

    # Suggest an output directory, perhaps a subfolder in the first txt file's dir
    # or prompt for a new one. For simplicity, let's prompt for a dedicated output dir.
    initial_dir_suggestion = os.path.dirname(txt_files[0]) if txt_files else os.getcwd()
    csv_output_directory = filedialog.askdirectory(
        title="Select a FOLDER to save converted .csv files",
        initialdir=initial_dir_suggestion
    )

    if not csv_output_directory:
        messagebox.showwarning("Conversion Skipped", "No output directory selected for .csv files.")
        return None

    converted_count = 0
    failed_count = 0
    print("\n--- Starting TXT to CSV Conversion ---")
    for txt_file in txt_files:
        base_name = os.path.basename(txt_file)
        # Make CSV name cleaner, matching your analysis script's original expectation
        # (assuming it was something like 'leakage.csv', 'dose_rate_low.csv')
        # For now, let's just use the original name with .csv
        csv_name = os.path.splitext(base_name)[0] + ".csv"
        csv_path = os.path.join(csv_output_directory, csv_name)

        try:
            convert_mapcheck_txt_to_csv(txt_file, csv_path) # Use your existing function
            converted_count += 1
        except Exception as e:
            messagebox.showerror("Conversion Error", f"Could not convert {base_name}:\n{e}")
            print(f"❌ Error converting {base_name}: {e}")
            failed_count += 1

    summary_message = f"Conversion Complete:\n\nSuccessfully converted: {converted_count} file(s).\nFailed to convert: {failed_count} file(s)."
    if converted_count > 0:
        summary_message += f"\n\nConverted CSV files are saved in:\n{csv_output_directory}"
    
    messagebox.showinfo("Conversion Summary", summary_message)
    print(summary_message)

    if converted_count > 0:
        return csv_output_directory # Return the directory where files were saved
    else:
        return None # Indicate no files were successfully converted or saved

def batch_convert_selected_txt_to_csv():
    """
    Prompts user to select multiple .txt files and an output directory.
    Converts files, saving CSVs with the same base name as their TXT counterparts.
    Returns the path to the directory where CSVs were saved, or None.
    """
    root = Tk()
    root.withdraw()

    messagebox.showinfo("TXT to CSV Conversion",
                        "You will be asked to select ALL MapCHECK .txt files you intend to use for this analysis session. They will be converted to .csv format.")

    txt_files = filedialog.askopenfilenames(
        title="Select ALL .txt MapCHECK files for conversion",
        filetypes=[("Text files", "*.txt")]
    )

    if not txt_files:
        messagebox.showwarning("Conversion Skipped", "No .txt files selected for conversion.")
        return None

    # Suggest an output directory, perhaps a subfolder in the first txt file's dir
    # or prompt for a new one.
    initial_dir_suggestion = os.path.dirname(txt_files[0]) if txt_files else os.getcwd()
    csv_output_directory = filedialog.askdirectory(
        title="Select a FOLDER to save ALL converted .csv files",
        initialdir=initial_dir_suggestion
    )

    if not csv_output_directory:
        messagebox.showwarning("Conversion Skipped", "No output directory selected for .csv files.")
        return None

    converted_count = 0
    failed_count = 0
    print("\n--- Starting TXT to CSV Conversion ---")
    for txt_file in txt_files:
        base_name_txt = os.path.basename(txt_file)
        csv_name = os.path.splitext(base_name_txt)[0] + ".csv" # Same name, new extension
        csv_path = os.path.join(csv_output_directory, csv_name)

        try:
            # This calls your existing single-file conversion function
            convert_mapcheck_txt_to_csv(txt_file, csv_path)
            converted_count += 1
        except Exception as e:
            messagebox.showerror("Conversion Error", f"Could not convert {base_name_txt}:\n{e}")
            print(f"❌ Error converting {base_name_txt}: {e}")
            failed_count += 1
    
    summary_message = f"Conversion Complete:\n\nSuccessfully converted: {converted_count} file(s).\nFailed to convert: {failed_count} file(s)."
    if converted_count > 0:
        summary_message += f"\n\nConverted CSV files are saved in:\n{csv_output_directory}"
    
    messagebox.showinfo("Conversion Summary", summary_message)
    print(summary_message)

    return csv_output_directory if converted_count > 0 else None

def global_exit(event=None): # event=None allows calling it without an event
    print("Escape key pressed. Exiting script.")
    if messagebox.askokcancel("Quit", "Are you sure you want to exit the application?"):
        sys.exit(0) # Clean exit

def extract_info_from_txt(txt_filepath):
    """
    Extracts Serial Number, Date, and Time from a MapCHECK .txt file.
    Returns:
        tuple: (serial_number, measurement_date, measurement_time)
               Returns "N/A" for any field not found.
    """
    serial_number = "N/A"
    measurement_date = "N/A"
    measurement_time = "N/A"
    
    try:
        # Try common encodings
        encodings_to_try = ['utf-8', 'latin-1', 'ascii']
        lines = None
        for enc in encodings_to_try:
            try:
                with open(txt_filepath, 'r', encoding=enc) as f:
                    lines = f.readlines()
                break 
            except UnicodeDecodeError:
                continue
        if lines is None:
             raise IOError(f"Could not read file {txt_filepath} with attempted encodings.")

        for line in lines:
            line_s = line.strip()
            if line_s.startswith("Serial:"):
                parts = line_s.split(":")
                if len(parts) > 1:
                    serial_number = parts[1].strip()
            elif line_s.startswith("Date:"): # Example: "Date:\t05/06/25\tTime:\t19:29:14"
                parts = line_s.split("\t") # Split by tab
                if len(parts) >= 2:
                    measurement_date = parts[1].strip()
                if len(parts) >= 4 and parts[2].strip() == "Time:":
                    measurement_time = parts[3].strip()
            
            # Stop if all found to be slightly more efficient, though header is usually small
            if serial_number != "N/A" and measurement_date != "N/A" and measurement_time != "N/A":
                break
                
    except FileNotFoundError:
        print(f"Warning: TXT file for info extraction not found: {txt_filepath}")
    except Exception as e:
        print(f"Warning: Error reading info from {txt_filepath}: {e}")
        
    return serial_number, measurement_date, measurement_time

def load_mapcheck_csv(filepath):
    print(f"DEBUG: Attempting to load CSV: {filepath}") # Add this for confirmation
    try:
        # Be explicit about the delimiter and header row
        df = pd.read_csv(filepath, header=0, delimiter=',')

        # Now, iloc indexing will be on the data rows directly.
        # The columns "ycm" and "ROW" are columns 0 and 1 from the CSV.
        # Actual dose data starts from the 3rd column (index 2).
        # We expect 53 columns of dose data.
        # So, pandas columns at indices 2 through 2+53-1 = 54.
        # The slicing df.iloc[:, 2:55] means:
        #   :   all rows (after header is handled by pandas)
        #   2:55 select columns from index 2 up to (but not including) index 55. This gives 53 columns.

        # Check if DataFrame has enough columns before trying to slice
        if df.shape[1] < 55: # Need at least 55 columns to slice up to index 54
            print(f"ERROR: DataFrame from {os.path.basename(filepath)} has only {df.shape[1]} columns. Expected at least 55 to extract data.")
            return None

        data_block = df.iloc[:, 2:55].astype(float).to_numpy()
        
        expected_rows, expected_cols = 65, 53
        # Check actual number of data rows extracted (might be less than 65 if file is short)
        if data_block.shape[0] < expected_rows:
            print(f"WARNING: Loaded data for {os.path.basename(filepath)} has {data_block.shape[0]} rows, expected at least {expected_rows} for full block.")
            # You might decide if this is acceptable or an error depending on requirements
            # For now, if it's less, it will just use what it has.

        if data_block.shape[1] != expected_cols:
            print(f"ERROR: Extracted data block from {os.path.basename(filepath)} has {data_block.shape[1]} columns, expected {expected_cols}.")
            return None

        print(f"DEBUG: Successfully loaded data ({data_block.shape}) from {os.path.basename(filepath)}")
        return data_block
        
    except pd.errors.EmptyDataError:
        print(f"ERROR: CSV file {os.path.basename(filepath)} is empty.")
        return None
    except pd.errors.ParserError as pe:
        print(f"ERROR: Pandas parsing error for {os.path.basename(filepath)}: {pe}")
        return None
    except ValueError as ve: # Handles errors from .astype(float) if non-numeric data slips through
        print(f"ERROR: ValueError during data conversion (e.g., non-numeric data) for {os.path.basename(filepath)}: {ve}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error loading or processing CSV file {os.path.basename(filepath)} in load_mapcheck_csv: {e}")
        return None

def create_cover_page(pdf, report_title, serial_number, summary_data_dict):
    """
    Creates a cover page for the PDF report with summary information.
    Args:
        pdf (PdfPages object): The PDF object to save the page to.
        report_title (str): The main title for the report.
        serial_number (str): The device serial number.
        summary_data_dict (dict): Dictionary containing analysis results.
    """
    fig, ax = plt.subplots(figsize=(8.27, 11.69)) # A4 size in inches

    # --- Page Title ---
    ax.text(0.5, 0.90, report_title, ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.85, f"MapCHECK Device Serial Number: {serial_number}", ha='center', va='center', fontsize=14)

    # --- Summary Data ---
    y_pos = 0.75
    line_height = 0.05

    def add_summary_text(label, value, y):
        ax.text(0.1, y, f"{label}:", ha='left', va='top', fontsize=12, fontweight='bold')
        if isinstance(value, dict): # For nested results like linearity
            val_str = ""
            for k, v in value.items():
                if isinstance(v, float):
                    val_str += f"\n    {k}: {v:.3f}"
                else:
                    val_str += f"\n    {k}: {v}"
            ax.text(0.35, y, val_str.lstrip('\n'), ha='left', va='top', fontsize=12, multialignment='left')
            return y - (line_height * (val_str.count('\n') + 1))
        elif isinstance(value, float):
            ax.text(0.35, y, f"{value:.3f}", ha='left', va='top', fontsize=12)
        else:
            ax.text(0.35, y, str(value), ha='left', va='top', fontsize=12)
        return y - line_height

    # Linearity
    if "Linearity" in summary_data_dict and isinstance(summary_data_dict["Linearity"], dict):
        y_pos = add_summary_text("Linearity Fit", summary_data_dict["Linearity"], y_pos)
    elif "Linearity" in summary_data_dict: # If it's an error string
        y_pos = add_summary_text("Linearity", summary_data_dict["Linearity"], y_pos)
    else:
        y_pos = add_summary_text("Linearity", "Not Performed/Data Missing", y_pos)
    y_pos -= line_height # Extra space

    # Dose Rate Dependence
    if "Dose Rate Dependence" in summary_data_dict and isinstance(summary_data_dict["Dose Rate Dependence"], dict):
        y_pos = add_summary_text("Dose Rate Dependence Fit", summary_data_dict["Dose Rate Dependence"], y_pos)
    elif "Dose Rate Dependence" in summary_data_dict:
        y_pos = add_summary_text("Dose Rate Dependence", summary_data_dict["Dose Rate Dependence"], y_pos)
    else:
        y_pos = add_summary_text("Dose Rate Dependence", "Not Performed/Data Missing", y_pos)
    y_pos -= line_height

    # Uniformity
    uni_val = summary_data_dict.get("Uniformity Std Dev", "Not Performed/Data Missing")
    if isinstance(uni_val, float):
        y_pos = add_summary_text("Uniformity (Std Dev %)", f"{uni_val:.2f}%", y_pos)
    else:
        y_pos = add_summary_text("Uniformity (Std Dev %)", str(uni_val), y_pos)
    y_pos -= line_height

    # Add other key values as needed (e.g., Leakage, Field Size main values)
    # Leakage
    lc_dose = summary_data_dict.get("Leakage Central Dose", "N/A")
    lm_dose = summary_data_dict.get("Leakage Mean Dose", "N/A")
    leakage_text = f"Central: {lc_dose:.3f}" if isinstance(lc_dose, float) else f"Central: {lc_dose}"
    leakage_text += f"\n    Mean: {lm_dose:.3f}" if isinstance(lm_dose, float) else f"\n    Mean: {lm_dose}"
    y_pos = add_summary_text("Leakage", {"Values": leakage_text.replace("    ","")}, y_pos) # Use dict for multi-line
    y_pos -= line_height


    # You can add more data from summary_data_dict here following the pattern

    # --- Date and Time ---
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.5, 0.1, f"Report Generated: {current_time}", ha='center', va='center', fontsize=10)

    ax.axis('off') # Turn off axis lines and ticks
    pdf.savefig(fig)
    plt.close(fig)

# ---- Analysis Functions

def extract_central_value(dose_array):
    center_row = dose_array.shape[0] // 2
    center_col = dose_array.shape[1] // 2
    return dose_array[center_row, center_col]

def calculate_leakage(dose_array, threshold=0.02):
    max_val = np.max(dose_array)
    low_dose_pixels = dose_array[dose_array < threshold * max_val]
    if len(low_dose_pixels) == 0:
        return np.nan  # Or you could return 0 if you prefer.    
    return np.mean(low_dose_pixels)

def calculate_uniformity(dose_array, size=10):
    center_row = dose_array.shape[0] // 2
    center_col = dose_array.shape[1] // 2
    half_size = size // 2
    region = dose_array[
        center_row - half_size : center_row + half_size + 1,
        center_col - half_size : center_col + half_size + 1
    ]
    return np.std(region) / np.mean(region) * 100

# Analyze

def analyze_dose_rate_dependence(filepaths, dose_rates, pdf=None):
    doses = [extract_central_value(load_mapcheck_csv(f)) for f in filepaths]
    if any(d is None for d in doses): # Check if any file failed to load
        print("Error: Could not load data for all dose rate files. Skipping dose rate analysis.")
        # Return NaN or raise an error, or return default coeffs that indicate failure
        return np.nan, np.nan

    coeffs = np.polyfit(dose_rates, doses, 1)
    fit_poly_function = np.poly1d(coeffs)

    fig = plt.figure()
    plt.scatter(dose_rates, doses, color="darkorange", label="Measured Doses")
    plt.plot(dose_rates, fit_poly_function(dose_rates), '--', color="black", label=f"Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
    plt.title("Dose Rate Dependence")
    plt.xlabel("Dose Rate (MU/min)")
    plt.ylabel("Central Dose")
    plt.grid(True)
    plt.legend()
    if pdf:
        pdf.savefig()
    # plt.close()
    return coeffs[0], coeffs[1], fig

def compare_dose_values(filepaths, labels, title, pdf=None):
    doses = []
    for file in filepaths:
        dose = extract_central_value(load_mapcheck_csv(file))
        doses.append(dose)
        print(f"{file}: Central dose = {dose:.2f}")
    
    fig = plt.figure()
    bars = plt.bar(labels, doses, color="skyblue")
    plt.title(title)
    plt.ylabel("Central Dose (arb. units)")
    plt.grid(True)
    
    # Annotate each bar with its central dose value
    for bar, dose in zip(bars, doses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{dose:.2f}",
                 ha="center", va="bottom", fontsize=10, color="black")
    
    if pdf:
        pdf.savefig()
    else:
        plt.show()

    return doses, fig

def analyze_linearity(filepaths, mu_values, pdf=None):
    doses = []
    for file in filepaths:
        dose = extract_central_value(load_mapcheck_csv(file))
        doses.append(dose)
        print(f"{file}: Central dose = {dose:.2f}")

    # Fit a linear regression
    coeffs = np.polyfit(mu_values, doses, 1)
    fit_line = np.poly1d(coeffs)(mu_values)

    # Generate the plot
    fig = plt.figure()
    plt.scatter(mu_values, doses, color="blue", label="Measured Doses")
    plt.plot(mu_values, fit_line, color="red", linestyle="--", label=f"Fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}")
    plt.title("Linearity (Dose vs. MU)")
    plt.xlabel("Monitor Units (MU)")
    plt.ylabel("Central Dose (arb. units)")
    plt.grid(True)
    plt.legend()

    if pdf:
        pdf.savefig()
    # plt.close()

    return coeffs, fig

def analyze_uniformity(filepath, pdf=None):
    dose_array = load_mapcheck_csv(filepath)
    uniformity = calculate_uniformity(dose_array)
    print(f"Uniformity Std Dev (%): {uniformity:.2f}")

    fig = plt.figure()
    plt.imshow(dose_array, cmap="viridis", origin="lower")
    plt.colorbar(label="Dose (arb. units)")
    plt.title(f"Uniformity Map (Std Dev = {uniformity:.2f}%)")

    if pdf:
        pdf.savefig()
    else:
        plt.show()
    
    # plt.close()
    return uniformity, fig

def analyze_field_size_dependence(filepaths, field_size_numerical_values, plot_labels): # Removed pdf
    doses = []
    for f_path in filepaths:
        data_array = load_mapcheck_csv(f_path)
        if data_array is None:
            print(f"Error loading data from {os.path.basename(f_path)} for field size analysis.")
            doses.append(np.nan) # Keep list length consistent for zip later
            continue # Continue to process other files if possible
        doses.append(extract_central_value(data_array))

    # Prepare data for fitting (filter out NaNs from failed loads)
    valid_indices = [i for i, d in enumerate(doses) if not np.isnan(d)]
    
    if len(valid_indices) < 2: # Need at least 2 points for a linear fit
        print("Warning: Not enough valid data points for field size dependence fit. Plotting points only.")
        fig = plt.figure(figsize=(8, 6))
        if valid_indices: # If there's at least one point
            plt.scatter([field_size_numerical_values[i] for i in valid_indices],
                        [doses[i] for i in valid_indices],
                        color="forestgreen", label="Measured Values")
        plt.xticks(ticks=field_size_numerical_values, labels=plot_labels, rotation=45, ha="right")
        plt.title("Field Size Dependence (Insufficient Data for Fit)")
        plt.xlabel("Approx. Field Size (cm)")
        plt.ylabel("Central Value (Counts/Dose)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return (np.nan, np.nan), doses, fig # Return NaN coeffs, original doses, and figure

    valid_doses = [doses[i] for i in valid_indices]
    valid_fs_values = [field_size_numerical_values[i] for i in valid_indices]
    # valid_plot_labels = [plot_labels[i] for i in valid_indices] # Not directly used for fitting

    coeffs = np.polyfit(valid_fs_values, valid_doses, 1)
    fit_function = np.poly1d(coeffs)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(valid_fs_values, valid_doses, color="forestgreen", label="Measured Values")
    plt.plot(valid_fs_values, fit_function(valid_fs_values), '--', color="black",
             label=f"Fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}")
    plt.xticks(ticks=field_size_numerical_values, labels=plot_labels, rotation=45, ha="right")
    plt.title("Field Size Dependence")
    plt.xlabel("Approx. Field Size (cm)")
    plt.ylabel("Central Value (Counts/Dose)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return (coeffs[0], coeffs[1]), doses, fig

# --- Main Script ---
def select_files(prompt_text, multiple=True):
    print(f"\n👉 {prompt_text}")
    root = Tk()
    root.withdraw()
    if multiple:
        files = filedialog.askopenfilenames(
            title=prompt_text, filetypes=[("CSV files", "*.csv")]
        )
    else:
        files = filedialog.askopenfilename(
            title=prompt_text, filetypes=[("CSV files", "*.csv")]
        )
    return files

def run_interactive_analysis():
    root = Tk()
    root.withdraw()
    root.bind('<Escape>', global_exit) # Assumes global_exit is defined

    csv_directory_for_analysis = os.getcwd()
    source_txt_directory_guess = csv_directory_for_analysis 

    # --- Step 1: Optional TXT to CSV Conversion ---
    if messagebox.askyesno("Step 1: Convert TXT to CSV (Optional)",
                           "Do you want to convert .txt MapCHECK files to .csv format now?"):
        converted_csv_path = batch_convert_selected_txt_to_csv() 
        if converted_csv_path:
            csv_directory_for_analysis = converted_csv_path
            source_txt_directory_guess = converted_csv_path 
            messagebox.showinfo("Conversion Successful",
                                f"Converted CSVs are in: {csv_directory_for_analysis}\nFiles will be looked for here first if using standard naming.")
        else:
            messagebox.showwarning("Conversion Problem", "Conversion skipped/failed. Select directory for existing .csv files.")
            csv_directory_for_analysis = filedialog.askdirectory(title="Select FOLDER containing your .csv analysis files")
            if not csv_directory_for_analysis: messagebox.showerror("Exiting", "No CSV directory. Exiting."); return
            source_txt_directory_guess = csv_directory_for_analysis
    else:
        messagebox.showinfo("Step 1: Select CSV Directory", "Please select FOLDER containing pre-existing .csv files.")
        csv_directory_for_analysis = filedialog.askdirectory(title="Select FOLDER containing your .csv analysis files")
        if not csv_directory_for_analysis: messagebox.showerror("Exiting", "No CSV directory. Exiting."); return
        source_txt_directory_guess = csv_directory_for_analysis
        
    print(f"ℹ️ Using CSV files from directory: {csv_directory_for_analysis}")
    print(f"ℹ️ Will attempt to find corresponding .txt files for Serial/Date in: {source_txt_directory_guess}")

    # --- Step 2: Output File Setup ---
    messagebox.showinfo("Step 2: Output File Setup", "Next, please specify where to save the PDF analysis report.")
    file_path = filedialog.asksaveasfilename(
        title="Save Analysis Report As", defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")], initialfile="WH_MapCHECK_Analysis"
    )
    if not file_path: messagebox.showinfo("Exiting", "No PDF output file. Exiting."); return
    base_name = os.path.splitext(file_path)[0]
    pdf_filename = base_name + ".pdf"
    json_filename = base_name + "_data.json"

    # --- Define Standardized Filenames (for "Dose Interpolated" CSVs, no suffix) ---
    file_conventions = {
        "leakage": "Leakage.csv",
        "dose_rate": ["DR_150mupmin.csv", "DR_350mupmin.csv", "linearity_100MU.csv"], # ADJUST 3rd FILE
        "field_size": ["FS_small.csv", "linearity_100MU.csv", "FS_large.csv"],    # ADJUST MIDDLE FILE
        "linearity": ["linearity_10MU.csv", "linearity_100MU.csv", "linearity_1000MU.csv"],
        "uniformity": "FS_large.csv" # Or your preferred uniformity file name
    }
    mu_values_for_linearity = [10, 100, 1000]
    dose_rates_for_analysis = [150, 350, 600] 
    fs_plot_labels_for_analysis = ["5x5", "10x10", "25x25"] # Adjusted to match your example
    fs_numerical_values_for_fit = [5, 10, 25]

    use_standard_names = messagebox.askyesno("Step 3: File Selection Method",
                                             "Attempt to find analysis files using standard names?\n\n"
                                             f"(Example Leakage: {file_conventions['leakage']})\n"
                                             "If 'No', or if files are not found, you will be prompted manually.")

    summary_data = {}
    all_selected_csv_paths = [] 
    collected_figures = [] # <--- Initialize list for figures
    print("\n--- Locating Files & Performing Analyses ---")

    # --- Nested Helper Functions: find_file, find_or_prompt_files, find_or_prompt_single_file ---
    # (These should be the corrected versions from our previous successful run)
    # ... PASTE THEM HERE ...
    def find_file(basename, file_description, initial_dir):
        if not basename: return None 
        path = os.path.join(initial_dir, basename) 
        if not os.path.exists(path):
            return None
        print(f"Found standard file for {file_description}: {os.path.basename(path)}")
        return path

    def find_or_prompt_files(basenames_list_from_convention, description_plural, prompt_title_manual, num_expected, initial_dir_for_dialog):
        paths = []
        attempted_standard_and_failed = False
        all_found_standard = False 

        if use_standard_names:
            temp_paths = []
            all_found_standard = True 
            for i, bn in enumerate(basenames_list_from_convention):
                p = find_file(bn, f"{description_plural} file #{i+1}", csv_directory_for_analysis)
                if p is None:
                    all_found_standard = False
                    print(f"⚠️ Standard name file NOT FOUND for {description_plural} file #{i+1}: {bn}")
                    break
                temp_paths.append(p)
            
            if all_found_standard and len(temp_paths) == num_expected:
                paths = temp_paths
                print(f"Using standard {description_plural} files: {[os.path.basename(p) for p in paths if p]}")
            elif all_found_standard and len(temp_paths) != num_expected:
                 messagebox.showwarning("Configuration Error", f"Std. name convention for {description_plural} has {len(temp_paths)} items, expected {num_expected}. Select manually.")
                 attempted_standard_and_failed = True
                 paths = [] 
            else: 
                messagebox.showwarning("Standard Names Not Found", f"Could not find all standard files for {description_plural}. Please select them manually.")
                attempted_standard_and_failed = True
                paths = [] 
        
        if not paths and (not use_standard_names or attempted_standard_and_failed) : 
            messagebox.showinfo(prompt_title_manual, f"Please select {num_expected} .csv files for {description_plural}.")
            selected_files_tuple = filedialog.askopenfilenames(
                title=prompt_title_manual, filetypes=[("CSV files", "*.csv")], initialdir=initial_dir_for_dialog
            )
            paths = list(selected_files_tuple)

        if len(paths) != num_expected:
            msg = (f"{description_plural} analysis requires {num_expected} files. {len(paths)} provided.")
            if messagebox.askretrycancel("File Selection Error", msg + "\n\nRetry selection? (Cancel to skip this test)"):
                selected_files_tuple = filedialog.askopenfilenames(title=prompt_title_manual, filetypes=[("CSV files", "*.csv")], initialdir=initial_dir_for_dialog)
                paths = list(selected_files_tuple)
                if len(paths) != num_expected:
                    messagebox.showerror("File Selection Error", f"Still incorrect. Skipping {description_plural}.")
                    print(f"⚠️ Still incorrect. Skipping {description_plural}.")
                    return []
            else:
                print(f"⚠️ User skipped {description_plural} analysis.")
                return []
        print(f"Using {description_plural} files manually selected (or confirmed): {[os.path.basename(p) for p in paths if p]}")
        all_selected_csv_paths.extend(paths) 
        return paths

    def find_or_prompt_single_file(basename_from_convention, description, prompt_title_manual, initial_dir_for_dialog):
        path = None
        attempted_standard_and_failed = False
        if use_standard_names:
            path = find_file(basename_from_convention, description, csv_directory_for_analysis)
            if path:
                print(f"Using standard {description} file: {os.path.basename(path)}")
                all_selected_csv_paths.append(path) 
                return path
            else:
                messagebox.showwarning("Standard Name Not Found", f"Std file for {description} ('{basename_from_convention}') not found. Select manually.")
                attempted_standard_and_failed = True
        
        if not use_standard_names or attempted_standard_and_failed:
            messagebox.showinfo(prompt_title_manual, f"Please select the .csv file for {description}.")
            path = filedialog.askopenfilename(title=prompt_title_manual, filetypes=[("CSV files", "*.csv")], initialdir=initial_dir_for_dialog)
        
        if not path:
            if messagebox.askokcancel("File Not Selected", f"{description} file not selected. Skip this test? (Cancel to exit script)"):
                print(f"⚠️ {description} analysis skipped.")
                return None
            else:
                print("❌ User chose to exit.")
                global_exit() 
                return None 
        print(f"Using {description} file manually: {os.path.basename(path)}")
        all_selected_csv_paths.append(path)
        return path
    # --- End of nested helper function definitions ---

    # --- Get File Paths for Analysis ---
    leakage_file_path = find_or_prompt_single_file(file_conventions["leakage"], "Leakage", "LEAKAGE: Select CSV File", csv_directory_for_analysis)
    dr_file_paths = find_or_prompt_files(file_conventions["dose_rate"], "Dose Rate", f"DOSE RATE: Select {len(dose_rates_for_analysis)} CSVs (Order: {', '.join(map(str, dose_rates_for_analysis))} MU/min)", len(dose_rates_for_analysis), csv_directory_for_analysis)
    fs_file_paths = find_or_prompt_files(file_conventions["field_size"], "Field Size", f"FIELD SIZE: Select {len(fs_plot_labels_for_analysis)} CSVs (Order: {', '.join(fs_plot_labels_for_analysis)})", len(fs_plot_labels_for_analysis), csv_directory_for_analysis)
    lin_file_paths = find_or_prompt_files(file_conventions["linearity"], "Linearity", f"LINEARITY: Select {len(mu_values_for_linearity)} CSVs (Order: {', '.join(map(str, mu_values_for_linearity))} MU)", len(mu_values_for_linearity), csv_directory_for_analysis)
    
    uniformity_convention_name = file_conventions.get("uniformity", "uniformity.csv") 
    uniformity_file_path = None
    if use_standard_names:
        uniformity_file_path = find_file(uniformity_convention_name, "Uniformity", csv_directory_for_analysis)
        if not uniformity_file_path and uniformity_convention_name == file_conventions["field_size"][-1] and fs_file_paths and len(fs_file_paths) == len(file_conventions["field_size"]):
            if fs_file_paths[-1] is not None: 
                uniformity_file_path = fs_file_paths[-1]
                print(f"ℹ️ Using already located file for Uniformity (from Field Size): {os.path.basename(uniformity_file_path)}")
    if not uniformity_file_path:
        temp_uniformity_path = find_or_prompt_single_file(None, "Uniformity", "UNIFORMITY: Select CSV File", csv_directory_for_analysis)
        if temp_uniformity_path: uniformity_file_path = temp_uniformity_path


    # --- Extract Serial Number and Date ---
    first_serial, first_date, first_time = "N/A", "N/A", "N/A"
    reference_txt_for_info_path = None # The .txt path from which info was extracted

    # Try to find a .txt file from the collected CSV paths to get info
    # Use source_txt_directory_guess (where original TXTs were or where CSVs are)
    if all_selected_csv_paths:
        for csv_p in all_selected_csv_paths:
            if csv_p: # If a valid path
                base_csv_name = os.path.splitext(os.path.basename(csv_p))[0]
                # Construct potential original TXT filename (assuming no suffix like _DoseInterpolated in CSV name)
                potential_txt_name = base_csv_name + ".txt" 
                
                # Check in the directory where CSVs are, and also in the original TXT source dir if different
                possible_txt_locations = [
                    os.path.join(csv_directory_for_analysis, potential_txt_name),
                    os.path.join(source_txt_directory_guess, potential_txt_name)
                ]
                for ref_txt_path_candidate in possible_txt_locations:
                    if os.path.exists(ref_txt_path_candidate):
                        reference_txt_for_info_path = ref_txt_path_candidate
                        break
                if reference_txt_for_info_path:
                    break # Found a reference TXT

    if reference_txt_for_info_path:
        print(f"DEBUG: Using TXT file for initial info: {reference_txt_for_info_path}")
        first_serial, first_date, first_time = extract_info_from_txt(reference_txt_for_info_path)
        print(f"Extracted from {os.path.basename(reference_txt_for_info_path)}: Serial={first_serial}, Date={first_date}, Time={first_time}")

        # Validate serial numbers across all other selected files
        for csv_p in all_selected_csv_paths:
            if csv_p:
                base_csv_name = os.path.splitext(os.path.basename(csv_p))[0]
                potential_txt_name = base_csv_name + ".txt"
                other_txt_path_candidate = os.path.join(source_txt_directory_guess, potential_txt_name)
                if not os.path.exists(other_txt_path_candidate) and os.path.dirname(csv_p) != source_txt_directory_guess:
                    other_txt_path_candidate = os.path.join(os.path.dirname(csv_p), potential_txt_name)

                if os.path.exists(other_txt_path_candidate) and other_txt_path_candidate != reference_txt_for_info_path:
                    current_serial, _, _ = extract_info_from_txt(other_txt_path_candidate)
                    if current_serial != "N/A" and first_serial != "N/A" and current_serial != first_serial:
                        warning_msg = (f"SERIAL NUMBER MISMATCH!\n"
                                       f"Reference file ({os.path.basename(reference_txt_for_info_path)}) serial: {first_serial}\n"
                                       f"File ({os.path.basename(other_txt_path_candidate)}) serial: {current_serial}\n"
                                       "Proceed with analysis?")
                        print(f"⚠️ {warning_msg.replace('\n', ' ')}")
                        if not messagebox.askyesno("Serial Number Mismatch", warning_msg):
                            messagebox.showinfo("Exiting", "Analysis cancelled due to serial number mismatch.")
                            return
                        break # Stop checking after first mismatch if user agrees to proceed
    else:
        print("⚠️ Could not find a reference .txt file to extract serial number and date.")
        # Optionally prompt for manual input
        # first_serial = simpledialog.askstring("Input", "Enter Device Serial Number:", parent=root) or "N/A"
        # first_date = simpledialog.askstring("Input", "Enter Measurement Date (MM/DD/YY):", parent=root) or "N/A"
        # first_time = "" # Or ask for time too


    # --- STAGE 1: Perform Analyses & Collect Figures ---
    print("\n--- Stage 1: Performing Analyses & Generating Figures (In Memory) ---")
    # (Leakage data collection - assuming it populates summary_data correctly)
    if leakage_file_path:
        print("\nAnalyzing Leakage...")
        try:
            data_array = load_mapcheck_csv(leakage_file_path)
            if data_array is not None:
                summary_data["Leakage Central Value (Counts/Dose)"] = extract_central_value(data_array)
                leakage_mean_val = calculate_leakage(data_array)
                summary_data["Leakage Mean Value (Counts/Dose)"] = float(leakage_mean_val) if not np.isnan(leakage_mean_val) else "Invalid (NaN)"
            else: summary_data["Leakage Analysis"] = f"Error loading data from {os.path.basename(leakage_file_path)}"
        except Exception as e: print(f"❌ Error in Leakage: {e}"); summary_data["Leakage Analysis"] = f"Error: {e}"
    else:
        summary_data["Leakage Analysis"] = "Skipped"
        summary_data["Leakage Central Value (Counts/Dose)"] = "Skipped"
        summary_data["Leakage Mean Value (Counts/Dose)"] = "Skipped"

    if dr_file_paths:
        print("\nAnalyzing Dose Rate Dependence...")
        try:
            (dr_slope, dr_intercept), fig_dr = analyze_dose_rate_dependence(dr_file_paths, dose_rates_for_analysis)
            if fig_dr: collected_figures.append(fig_dr)
            summary_data["Dose Rate Dependence Fit"] = {"Slope (value per MU/min)": dr_slope, "Intercept (value)": dr_intercept}
        except Exception as e: print(f"❌ Error in DR: {e}"); summary_data["Dose Rate Dependence Fit"] = f"Error: {e}"
    else: summary_data["Dose Rate Dependence Fit"] = "Skipped"
        
    if fs_file_paths:
        print("\nAnalyzing Field Size Dependence...")
        try:
            (fs_slope, fs_intercept), field_doses_raw, fig_fs = analyze_field_size_dependence(fs_file_paths, fs_numerical_values_for_fit, fs_plot_labels_for_analysis)
            if fig_fs: collected_figures.append(fig_fs)
            summary_data["Field Size Dependence Analysis"] = {
                "Fit Slope (value/cm)": fs_slope, "Fit Intercept (value)": fs_intercept,
                "Measured Values": dict(zip(fs_plot_labels_for_analysis, [fd if not (isinstance(fd, float) and np.isnan(fd)) else "Load Error" for fd in field_doses_raw]))
            }
        except Exception as e: print(f"❌ Error in FS: {e}"); summary_data["Field Size Dependence Analysis"] = f"Error: {e}"
    else: summary_data["Field Size Dependence Analysis"] = "Skipped"
        
    if lin_file_paths:
        print("\nAnalyzing Linearity...")
        try:
            lin_coeffs, fig_lin = analyze_linearity(lin_file_paths, mu_values_for_linearity)
            if fig_lin: collected_figures.append(fig_lin)
            if lin_coeffs is not None:
                summary_data["Linearity Fit"] = {"Slope (value/MU)": lin_coeffs[0], "Intercept (value)": lin_coeffs[1]}
            else: summary_data["Linearity Fit"] = "Analysis Error"
        except Exception as e: print(f"❌ Error in Linearity: {e}"); summary_data["Linearity Fit"] = f"Error: {e}"
    else: summary_data["Linearity Fit"] = "Skipped"

    if uniformity_file_path:
        print("\nAnalyzing Uniformity...")
        try:
            uniformity_std, fig_uni = analyze_uniformity(uniformity_file_path)
            if fig_uni: collected_figures.append(fig_uni)
            summary_data["Uniformity (Std Dev %)"] = uniformity_std
        except Exception as e: print(f"❌ Error in Uniformity: {e}"); summary_data["Uniformity (Std Dev %)"] = f"Error: {e}"
    else: summary_data["Uniformity (Std Dev %)"] = "Skipped"

    # --- STAGE 2: Generate PDF ---
    print("\n--- Generating PDF Report ---")
    try:
        with PdfPages(pdf_filename) as pdf:
            create_cover_page(pdf, "MapCHECK Analysis Report", first_serial, first_date, first_time, summary_data)
            for fig in collected_figures:
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig) 
        print(f"✅ PDF report generated: {pdf_filename}")
    except Exception as e:
        messagebox.showerror("PDF Generation Error", f"Could not generate PDF report: {e}")
        print(f"❌ Error generating PDF: {e}")

    # --- Save summary data to JSON ---
    # ... (JSON saving code - ensure keys in summary_data are consistent with cover page expectations) ...
    try:
        with open(json_filename, "w") as json_file:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj) if not np.isnan(obj) else None
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, np.bool_): return bool(obj)
                    return super(NpEncoder, self).default(obj)
            json.dump(summary_data, json_file, indent=4, cls=NpEncoder)
        print(f"📊 Summary data saved to: {json_filename}")
    except Exception as e:
        messagebox.showerror("JSON Save Error", f"Failed to save summary data to JSON:\n{e}")
        print(f"❌ Error saving JSON file: {e}")

    print(f"\n✅ Analysis complete. PDF saved to: {pdf_filename}") #This line might be redundant if pdf generation succeeded
    messagebox.showinfo("Analysis Complete", f"Analysis finished.\nPDF: {pdf_filename}\nJSON: {json_filename}")

def create_cover_page(pdf, report_title, serial_number, measurement_date, measurement_time, summary_data_dict):
    fig, ax = plt.subplots(figsize=(8.27, 11.69)) # A4 size

    ax.text(0.5, 0.92, report_title, ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.87, f"Device Serial Number: {serial_number}", ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.83, f"Measurement Date: {measurement_date} {measurement_time}", ha='center', va='center', fontsize=12)

    y_pos = 0.75
    line_height = 0.035 
    item_spacing = 0.04 # Reduced spacing a bit

    def display_item(label, value, y_start, is_sub_item=False, sub_label_prefix="  "):
        current_y = y_start
        label_x = 0.1
        value_x = 0.45
        
        if is_sub_item:
            label_text = f"{sub_label_prefix}{label}:"
            label_fontweight = 'normal'
        else:
            label_text = f"{label}:"
            label_fontweight = 'bold'
        
        ax.text(label_x, current_y, label_text, ha='left', va='top', fontsize=11, fontweight=label_fontweight)

        val_str = ""
        if value is None or "Skipped" in str(value) or "Not Performed" in str(value) or "not found" in str(value):
            val_str = str(value) if isinstance(value, str) else "Not Performed / Data Missing"
            ax.text(value_x, current_y, val_str, ha='left', va='top', fontsize=11, color='gray', wrap=True)
            current_y -= line_height * (1 + (len(val_str) // 45 if isinstance(val_str, str) else 0)) # Adjusted wrap estimate
        elif isinstance(value, str) and "Error" in value:
            val_str = value
            ax.text(value_x, current_y, val_str, ha='left', va='top', fontsize=11, color='red', wrap=True)
            current_y -= line_height * (1 + (len(val_str) // 45))
        elif isinstance(value, str):
            val_str = value
            ax.text(value_x, current_y, val_str, ha='left', va='top', fontsize=11, wrap=True)
            current_y -= line_height * (1 + (len(val_str) // 45))
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value): val_str = "Invalid (NaN)"
            elif "Std Dev" in label: val_str = f"{value:.2f}%" # Specific for uniformity
            else: val_str = f"{value:.3f}"
            ax.text(value_x, current_y, val_str, ha='left', va='top', fontsize=11)
            current_y -= line_height
        else: # Other types like int
            val_str = str(value)
            ax.text(value_x, current_y, val_str, ha='left', va='top', fontsize=11)
            current_y -= line_height
        return current_y

    # Linearity
    linearity_fit_data = summary_data_dict.get("Linearity Fit", "Skipped")
    y_pos = display_item("Linearity Fit", "", y_pos) # Main label
    if isinstance(linearity_fit_data, dict):
        y_pos = display_item("Slope (value/MU)", linearity_fit_data.get("Slope (value/MU)"), y_pos, is_sub_item=True)
        y_pos = display_item("Intercept (value)", linearity_fit_data.get("Intercept (value)"), y_pos, is_sub_item=True)
    else:
        y_pos = display_item("", linearity_fit_data, y_pos, is_sub_item=True) # Show skipped/error
    y_pos -= item_spacing
    
    # Dose Rate Dependence
    dr_fit_data = summary_data_dict.get("Dose Rate Dependence Fit", "Skipped")
    y_pos = display_item("Dose Rate Dependence Fit", "", y_pos)
    if isinstance(dr_fit_data, dict):
        y_pos = display_item("Slope (value per MU/min)", dr_fit_data.get("Slope (value per MU/min)"), y_pos, is_sub_item=True)
        y_pos = display_item("Intercept (value)", dr_fit_data.get("Intercept (value)"), y_pos, is_sub_item=True)
    else:
        y_pos = display_item("", dr_fit_data, y_pos, is_sub_item=True)
    y_pos -= item_spacing

    # Field Size Dependence
    fs_analysis_data = summary_data_dict.get("Field Size Dependence Analysis", "Skipped")
    y_pos = display_item("Field Size Dependence", "", y_pos)
    if isinstance(fs_analysis_data, dict):
        y_pos = display_item("Fit Slope (value/cm)", fs_analysis_data.get("Fit Slope (value/cm)"), y_pos, is_sub_item=True)
        y_pos = display_item("Fit Intercept (value)", fs_analysis_data.get("Fit Intercept (value)"), y_pos, is_sub_item=True)
        measured_values = fs_analysis_data.get("Measured Values")
        if isinstance(measured_values, dict):
            y_pos = display_item("Measured Values", "", y_pos, is_sub_item=True, sub_label_prefix="    ") # Indented further
            for fs_label, fs_val in measured_values.items():
                y_pos = display_item(fs_label, fs_val, y_pos, is_sub_item=True, sub_label_prefix="      ")
    else:
        y_pos = display_item("", fs_analysis_data, y_pos, is_sub_item=True)
    y_pos -= item_spacing

    # Uniformity
    y_pos = display_item("Uniformity (Std Dev %)", summary_data_dict.get("Uniformity (Std Dev %)"), y_pos)
    y_pos -= item_spacing
    
    # Leakage
    y_pos = display_item("Leakage", "", y_pos)
    y_pos = display_item("Central Value (Counts/Dose)", summary_data_dict.get("Leakage Central Value (Counts/Dose)"), y_pos, is_sub_item=True)
    y_pos = display_item("Mean Value (Counts/Dose)", summary_data_dict.get("Leakage Mean Value (Counts/Dose)"), y_pos, is_sub_item=True)
    y_pos -= item_spacing

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.5, 0.05, f"Report Generated: {current_time_str}", ha='center', va='center', fontsize=10)

    ax.axis('off')
    pdf.savefig(fig)
    plt.close(fig) # Close the cover page figure

if __name__ == "__main__":
    run_interactive_analysis()

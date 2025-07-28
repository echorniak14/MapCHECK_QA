import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tkinter import messagebox

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

    serial_keys_to_try = ["Serial:", "Serial No:"]
    
    data_rows, in_data_block = [], False
    for line in lines:
        line_s = line.strip()

        if not in_data_block:
            for key in serial_keys_to_try:
                if line_s.startswith(key):
                    info["serial"] = line_s.split(":", 1)[1].strip()
                    break

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

def parse_date_robustly(date_str):
    """Tries to parse a date string using a list of common formats."""
    if not isinstance(date_str, str): return None
    common_formats = ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%d-%b-%Y', '%Y-%m-%d %H:%M:%S']
    for fmt in common_formats:
        try: return datetime.strptime(date_str, fmt)
        except ValueError: continue
    return None

def load_all_json_data(directory):
    """Finds and parses all '.json' files, extracting necessary data."""
    all_qa_data = []
    if not directory: return []
    print(f"--- Searching for JSON files in: {directory} ---")
    filenames = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not filenames:
        print("-> No files ending with '.json' were found.")
        return []
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            
            date_str = data.get("_measurement_date")
            serial_num = data.get("_device_serial")

            if not date_str or not serial_num or serial_num == "N/A":
                print(f" -> Skipping {filename}: Missing date or serial number.")
                continue

            measurement_date = parse_date_robustly(date_str)
            if measurement_date is None: continue
            
            record = {
                "date": measurement_date, "serial_number": serial_num,
                "linearity_slope": data.get("Linearity Fit", {}).get("Slope"),
                "dose_rate_slope": data.get("Dose Rate Fit", {}).get("Slope"),
                "fs_slope": data.get("Field Size Analysis", {}).get("Fit Slope"),
                "uniformity": data.get("Uniformity Std Dev"),
                "leakage_mean": data.get("Leakage Mean Value")
            }
            all_qa_data.append(record)
        except Exception:
            print(f" -> Skipping {filename} due to an error:")
    
    all_qa_data.sort(key=lambda x: x["date"])
    return all_qa_data

def load_external_data_and_get_tests(csv_path):
    """
    Loads the raw external data from the specified TQA CSV format.
    It identifies and returns the raw data and a list of unique test types.
    """
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["Test", "Report Date-Time", "Value"]
        if not all(col in df.columns for col in required_cols):
            messagebox.showerror("CSV Format Error", f"The selected CSV is missing one or more required columns: {required_cols}")
            return None, None

        df.rename(columns={"Report Date-Time": "date"}, inplace=True)
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        available_tests = df["Test"].unique().tolist()
        
        return df, available_tests

    except Exception as e:
        messagebox.showerror("CSV Error", f"Could not read or parse the CSV file.\nError: {e}")
        return None, None

def parse_metadata_from_file(file_path):
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('Model:'):
                metadata['Model'] = line.split(':')[1].strip()
            elif line.strip().startswith('Serial:'):
                metadata['Serial'] = line.split(':')[1].strip()
            elif line.strip().startswith('Software Version:'):
                metadata['Software Version'] = line.split(':')[1].strip()
            elif line.strip().startswith('Calibration File:'):
                metadata['Calibration File'] = line.split(':')[1].strip()
    return metadata

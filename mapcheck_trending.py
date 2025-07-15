# mapcheck_trending_v4.py
# This script analyzes a collection of MapCHECK JSON data files to visualize longitudinal trends.
#
# V4 Changes:
# - CORRECTED the file finding logic to look for any file ending in '.json'.
#   This aligns it with the output from the analysis script.
# - Reverted from diagnostic mode to clean output.

import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import Tk, filedialog, messagebox
from datetime import datetime
import traceback

def parse_date_robustly(date_str):
    """
    Tries to parse a date string using a list of common formats.
    """
    if not isinstance(date_str, str):
        return None
    common_formats = ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%d-%b-%Y']
    for fmt in common_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def load_all_json_data(directory):
    """
    Finds and parses all '.json' files in a given directory.
    """
    all_qa_data = []
    if not directory:
        return []

    print(f"--- Searching for JSON files in: {directory} ---")
    
    # --- THIS IS THE CORRECTED LINE ---
    # We now look for ANY file ending in .json, not the more specific _data.json
    filenames = [f for f in os.listdir(directory) if f.endswith(".json")]
    
    if not filenames:
        print("-> No files ending with '.json' were found.")
        return []

    for filename in filenames:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            date_str = data.get("_measurement_date")
            if not date_str or date_str == "N/A":
                print(f" -> Skipping {filename}: Missing '_measurement_date' key.")
                continue

            measurement_date = parse_date_robustly(date_str)
            if measurement_date is None:
                print(f" -> Skipping {filename}: Unknown date format for '{date_str}'.")
                continue
            
            record = {
                "date": measurement_date, "serial_number": data.get("_device_serial", "N/A"),
                "linearity_slope": data.get("Linearity Fit", {}).get("Slope"),
                "dose_rate_slope": data.get("Dose Rate Fit", {}).get("Slope"),
                "fs_slope": data.get("Field Size Analysis", {}).get("Fit Slope"),
                "uniformity": data.get("Uniformity Std Dev"),
                "leakage_mean": data.get("Leakage Mean Value")
            }
            all_qa_data.append(record)

        except Exception:
            print(f" -> Skipping {filename} due to an unexpected error during processing:")
            traceback.print_exc(limit=1)

    all_qa_data.sort(key=lambda x: x["date"])
    return all_qa_data

# The get_user_choice and plot_trend functions remain the same.
def get_user_choice(available_metrics):
    print("\nPlease choose a parameter to trend:")
    menu = {str(i): key for i, (key, name) in enumerate(available_metrics.items(), 1)}
    for i, (key, name) in enumerate(available_metrics.items(), 1):
        print(f"  {i}. {name}")
    print("  Q. Quit")
    while True:
        choice = input("Enter your choice (1, 2, etc.): ").strip().upper()
        if choice == 'Q': return None
        if choice in menu: return menu[choice]
        print("Invalid choice. Please try again.")

def plot_trend(records, metric_key, metric_name):
    dates = [rec["date"] for rec in records if rec.get(metric_key) is not None]
    values = [rec[metric_key] for rec in records if rec.get(metric_key) is not None]
    if len(dates) < 2:
        messagebox.showwarning("Not Enough Data", f"Cannot create a trend plot for '{metric_name}'.\nNeed at least two valid data points.")
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dates, values, marker='o', linestyle='-', label=metric_name)
    ax.set_title(f"Trend Analysis for {metric_name}", fontsize=16)
    ax.set_xlabel("Measurement Date", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    root = Tk()
    root.withdraw()
    messagebox.showinfo("Select Data Folder", "Please select the folder containing your MapCHECK JSON files.")
    json_folder = filedialog.askdirectory(title="Select Folder with JSON files")
    if not json_folder:
        print("No folder selected. Exiting."); return

    all_data = load_all_json_data(json_folder)
    
    if not all_data:
        messagebox.showerror("No Data Found", "No valid JSON data files could be processed.\n\nPlease check the console output for details on why files were skipped.")
        return
        
    print(f"\n--- Successfully loaded {len(all_data)} data point(s). Ready for analysis. ---")
    TRENDABLE_METRICS = {
        "linearity_slope": "Linearity Slope", "dose_rate_slope": "Dose Rate Slope",
        "fs_slope": "Field Size Fit Slope", "uniformity": "Uniformity (Std Dev %)",
        "leakage_mean": "Mean Leakage Value",
    }
    while True:
        metric_to_plot = get_user_choice(TRENDABLE_METRICS)
        if metric_to_plot is None:
            print("Exiting application."); break
        plot_trend(all_data, metric_to_plot, TRENDABLE_METRICS[metric_to_plot])

if __name__ == "__main__":
    main()
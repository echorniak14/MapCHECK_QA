# mapcheck_comparison_v1.py
#
# PURPOSE:
# This script is specifically for COMPARING QA trends across multiple MapCHECK devices.
# It loads data from a folder of JSONs and generates plots where each device
# is represented by its own color-coded line, making it easy to spot differences.
#
# This is a companion to the main trending script, which shows the overall trend of all devices combined.

import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import Tk, filedialog, messagebox
from datetime import datetime
import traceback

def parse_date_robustly(date_str):
    """Tries to parse a date string using a list of common formats."""
    if not isinstance(date_str, str): return None
    common_formats = ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%d-%b-%Y']
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
            serial_num = data.get("_device_serial") # Serial number is essential for comparison

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
            traceback.print_exc(limit=1)
    
    all_qa_data.sort(key=lambda x: x["date"])
    return all_qa_data

def get_user_choice(available_metrics):
    """Displays a menu and gets the user's choice of which metric to plot."""
    print("\nPlease choose a parameter to compare across devices:")
    menu = {str(i): key for i, (key, name) in enumerate(available_metrics.items(), 1)}
    for i, (key, name) in enumerate(available_metrics.items(), 1):
        print(f"  {i}. {name}")
    print("  Q. Quit")
    while True:
        choice = input("Enter your choice (1, 2, etc.): ").strip().upper()
        if choice == 'Q': return None
        if choice in menu: return menu[choice]
        print("Invalid choice. Please try again.")

def plot_device_comparison(records, metric_key, metric_name):
    """
    Generates a trend plot, creating separate, color-coded lines for each device.
    """
    # Find all unique serial numbers in the dataset
    serial_numbers = sorted(list({rec["serial_number"] for rec in records}))

    fig, ax = plt.subplots(figsize=(12, 7))
    has_data_to_plot = False

    for serial in serial_numbers:
        # Filter records for the current device
        device_records = [rec for rec in records if rec["serial_number"] == serial]
        
        # Extract dates and values for this device, skipping None values
        dates = [rec["date"] for rec in device_records if rec.get(metric_key) is not None]
        values = [rec[metric_key] for rec in device_records if rec.get(metric_key) is not None]

        if len(dates) >= 1:
            has_data_to_plot = True
            # Plot with a marker if only one point, line if multiple points
            linestyle = '-' if len(dates) > 1 else ''
            ax.plot(dates, values, marker='o', linestyle=linestyle, label=f"Device: {serial}")

    if not has_data_to_plot:
        messagebox.showwarning("Not Enough Data", f"Cannot create a plot for '{metric_name}'.\nNo valid data points were found for any device.")
        plt.close(fig) # Close the empty plot
        return

    # --- Formatting the Plot ---
    ax.set_title(f"Device Comparison for {metric_name}", fontsize=16)
    ax.set_xlabel("Measurement Date", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend() # The legend is crucial for telling devices apart
    plt.tight_layout()
    plt.show()

def main():
    root = Tk()
    root.withdraw()

    messagebox.showinfo("Select Data Folder", "Please select the folder containing your MapCHECK JSON files for comparison.")
    json_folder = filedialog.askdirectory(title="Select Folder with JSON files")
    if not json_folder:
        print("No folder selected. Exiting.")
        return

    all_data = load_all_json_data(json_folder)

    if not all_data:
        messagebox.showerror("No Data Found", "No valid JSON files could be processed. Please check console for details.");
        return
        
    print(f"\n--- Successfully loaded {len(all_data)} data point(s) from {len(set(rec['serial_number'] for rec in all_data))} device(s). ---")
    
    TRENDABLE_METRICS = {
        "linearity_slope": "Linearity Slope",
        "dose_rate_slope": "Dose Rate Slope",
        "fs_slope": "Field Size Fit Slope",
        "uniformity": "Uniformity (Std Dev %)",
        "leakage_mean": "Mean Leakage Value",
    }
    
    while True:
        metric_to_plot = get_user_choice(TRENDABLE_METRICS)
        
        if metric_to_plot is None:
            print("Exiting application.")
            break
        
        plot_device_comparison(all_data, metric_to_plot, TRENDABLE_METRICS[metric_to_plot])

if __name__ == "__main__":
    main()
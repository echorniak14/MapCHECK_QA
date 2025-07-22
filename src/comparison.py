import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import Tk, filedialog, messagebox
from .common import utils

def plot_device_comparison(records, metric_key, metric_name):
    """
    Generates a trend plot, creating separate, color-coded lines for each device.
    """
    serial_numbers = sorted(list({rec["serial_number"] for rec in records}))

    fig, ax = plt.subplots(figsize=(12, 7))
    has_data_to_plot = False

    for serial in serial_numbers:
        device_records = [rec for rec in records if rec["serial_number"] == serial]
        
        dates = [rec["date"] for rec in device_records if rec.get(metric_key) is not None]
        values = [rec[metric_key] for rec in device_records if rec.get(metric_key) is not None]

        if len(dates) >= 1:
            has_data_to_plot = True
            linestyle = '-' if len(dates) > 1 else ''
            ax.plot(dates, values, marker='o', linestyle=linestyle, label=f"Device: {serial}")

    if not has_data_to_plot:
        messagebox.showwarning("Not Enough Data", f"Cannot create a plot for '{metric_name}'.\nNo valid data points were found for any device.")
        plt.close(fig)
        return

    ax.set_title(f"Device Comparison for {metric_name}", fontsize=16)
    ax.set_xlabel("Measurement Date", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

def run_comparison(json_folder):
    root = Tk()
    root.withdraw()

    if not json_folder:
        messagebox.showinfo("Select Data Folder", "Please select the folder containing your MapCHECK JSON files for comparison.")
        json_folder = filedialog.askdirectory(title="Select Folder with JSON files")
        if not json_folder:
            print("No folder selected. Exiting.")
            return

    all_data = utils.load_all_json_data(json_folder)

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
        metric_to_plot = utils.get_user_choice("Please choose a parameter to compare across devices:", TRENDABLE_METRICS)
        
        if metric_to_plot is None:
            print("Exiting application.")
            break
        
        plot_device_comparison(all_data, metric_to_plot, TRENDABLE_METRICS[metric_to_plot])

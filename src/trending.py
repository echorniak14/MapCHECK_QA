import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import Tk, filedialog, messagebox
from .common import utils

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

def run_trending(json_folder):
    root = Tk()
    root.withdraw()
    if not json_folder:
        messagebox.showinfo("Select Data Folder", "Please select the folder containing your MapCHECK JSON files.")
        json_folder = filedialog.askdirectory(title="Select Folder with JSON files")
        if not json_folder:
            print("No folder selected. Exiting."); return

    all_data = utils.load_all_json_data(json_folder)
    
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
        metric_to_plot = utils.get_user_choice("Please choose a parameter to trend:", TRENDABLE_METRICS)
        if metric_to_plot is None:
            print("Exiting application."); break
        plot_trend(all_data, metric_to_plot, TRENDABLE_METRICS[metric_to_plot])
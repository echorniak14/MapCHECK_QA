# mapcheck_trending.py
# This script analyzes a collection of MapCHECK JSON data files to visualize longitudinal trends.
#
# Workflow:
# 1. Prompts the user to select a directory containing the JSON output files.
# 2. Reads all JSON files, extracts key QA metrics and measurement dates.
# 3. Asks the user which metric they would like to plot.
# 4. Generates and displays a time-series plot for the selected metric.

import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import Tk, filedialog, messagebox
from datetime import datetime

def load_all_json_data(directory):
    """
    Finds and parses all '_data.json' files in a given directory.

    Args:
        directory (str): The path to the folder containing JSON files.

    Returns:
        list: A sorted list of dictionaries, where each dictionary represents
              a single JSON report's data. Returns an empty list if no valid
              data is found.
    """
    all_qa_data = []
    if not directory:
        return []

    print(f"Searching for JSON files in: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith("_data.json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # --- Data Extraction ---
                    # Use the new _measurement_date field for trending.
                    date_str = data.get("_measurement_date")
                    if not date_str or date_str == "N/A":
                        print(f"Skipping {filename}: Does not contain a valid '_measurement_date' field.")
                        continue

                    # Convert date string (e.g., 'MM/DD/YY') to a Python datetime object
                    measurement_date = datetime.strptime(date_str, '%m/%d/%y')
                    
                    # Flatten the nested data into a single-level dictionary for easier access
                    record = {
                        "date": measurement_date,
                        "serial_number": data.get("_device_serial", "N/A"),
                        "linearity_slope": data.get("Linearity Fit", {}).get("Slope"),
                        "dose_rate_slope": data.get("Dose Rate Fit", {}).get("Slope"),
                        "fs_slope": data.get("Field Size Analysis", {}).get("Fit Slope"),
                        "uniformity": data.get("Uniformity Std Dev"),
                        "leakage_mean": data.get("Leakage Mean Value")
                    }
                    all_qa_data.append(record)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not process file {filename}. Error: {e}")

    # Sort the data chronologically, which is crucial for plotting
    all_qa_data.sort(key=lambda x: x["date"])
    return all_qa_data

def get_user_choice(available_metrics):
    """
    Displays a menu and gets the user's choice of which metric to plot.

    Args:
        available_metrics (dict): A dictionary mapping metric keys to descriptive names.

    Returns:
        str: The key of the selected metric, or None if the user quits.
    """
    print("\nPlease choose a parameter to trend:")
    menu = {}
    for i, (key, name) in enumerate(available_metrics.items(), 1):
        menu[str(i)] = key
        print(f"  {i}. {name}")
    
    print("  Q. Quit")

    while True:
        choice = input("Enter your choice (1, 2, etc.): ").strip().upper()
        if choice == 'Q':
            return None
        if choice in menu:
            return menu[choice]
        print("Invalid choice. Please try again.")

def plot_trend(records, metric_key, metric_name):
    """
    Generates and displays a trend plot for a given metric.

    Args:
        records (list): The list of sorted data records.
        metric_key (str): The key for the data to be plotted (e.g., 'linearity_slope').
        metric_name (str): The user-friendly name for the plot's title and y-axis.
    """
    # Extract dates and values, skipping any missing data points
    dates = [rec["date"] for rec in records if rec.get(metric_key) is not None]
    values = [rec[metric_key] for rec in records if rec.get(metric_key) is not None]

    if len(dates) < 2:
        messagebox.showwarning("Not Enough Data", f"Cannot create a trend plot for '{metric_name}'.\nNeed at least two valid data points.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dates, values, marker='o', linestyle='-', label=metric_name)

    # --- Formatting the Plot ---
    # Set titles and labels
    ax.set_title(f"Trend Analysis for {metric_name}", fontsize=16)
    ax.set_xlabel("Measurement Date", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    
    # Improve date formatting on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45) # Rotate dates to prevent overlap

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to orchestrate the trending analysis workflow."""
    root = Tk()
    root.withdraw()

    messagebox.showinfo("Select Data Folder", "Please select the folder containing your MapCHECK '_data.json' files.")
    json_folder = filedialog.askdirectory(title="Select Folder with JSON files")

    if not json_folder:
        print("No folder selected. Exiting.")
        return

    all_data = load_all_json_data(json_folder)

    if not all_data:
        messagebox.showerror("No Data Found", "No valid JSON data files were found in the selected directory.\nMake sure you've run the analysis script and that JSONs have a '_measurement_date' field.")
        return
        
    print(f"\nSuccessfully loaded and parsed {len(all_data)} data files.")
    
    # Define the metrics available for trending
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
        
        plot_trend(all_data, metric_to_plot, TRENDABLE_METRICS[metric_to_plot])


if __name__ == "__main__":
    main()
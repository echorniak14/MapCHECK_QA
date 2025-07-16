# mapcheck_correlation_v2_flexible_csv.py
# This script correlates MapCHECK QA parameters with an external data source.
#
# V2 Changes:
# - Adapts to the specific CSV format exported from TQA/Image Owl.
# - Automatically detects all available test types from the CSV.
# - Prompts the user to select which test type they wish to analyze.

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from datetime import datetime
import numpy as np

# --- Data Loading and Aggregation Functions ---

def parse_date_robustly(date_str):
    """Tries to parse a date string using a list of common formats."""
    if not isinstance(date_str, str): return None
    # Add the new format from your CSV to the list of formats to try
    common_formats = ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S']
    for fmt in common_formats:
        try: return datetime.strptime(date_str, fmt)
        except ValueError: continue
    return None

def load_and_aggregate_json_data(directory):
    """Loads all MapCHECK JSON data and aggregates it by year and month."""
    all_records = []
    filenames = [f for f in os.listdir(directory) if f.endswith(".json")]
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            measurement_date = parse_date_robustly(data.get("_measurement_date"))
            if measurement_date is None: continue
            record = {
                "date": measurement_date,
                "uniformity": data.get("Uniformity Std Dev"),
                "fs_slope": data.get("Field Size Analysis", {}).get("Fit Slope"),
                "linearity_slope": data.get("Linearity Fit", {}).get("Slope"),
                "dose_rate_slope": data.get("Dose Rate Fit", {}).get("Slope"),
            }
            all_records.append(record)
        except Exception: pass
    if not all_records: return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_mapcheck_df = df.groupby('year_month').mean(numeric_only=True).reset_index()
    return monthly_mapcheck_df

# <<< MODIFIED SECTION: This function is completely new to handle the TQA format >>>
def load_external_data_and_get_tests(csv_path):
    """
    Loads the raw external data from the specified TQA CSV format.
    It identifies and returns the raw data and a list of unique test types.
    """
    try:
        df = pd.read_csv(csv_path)
        # Check for the required columns from your export
        required_cols = ["Test", "Report Date-Time", "Value"]
        if not all(col in df.columns for col in required_cols):
            messagebox.showerror("CSV Format Error", f"The selected CSV is missing one or more required columns: {required_cols}")
            return None, None

        # Rename columns to a standard format for easier processing
        df.rename(columns={"Report Date-Time": "date"}, inplace=True)
        
        # Convert date column to datetime objects
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        # Find all unique tests available in the file
        available_tests = df["Test"].unique().tolist()
        
        return df, available_tests

    except Exception as e:
        messagebox.showerror("CSV Error", f"Could not read or parse the CSV file.\nError: {e}")
        return None, None
# <<< END OF MODIFIED SECTION >>>

def get_user_choice(prompt, options):
    """Generic function to get a user's choice from a list."""
    print(f"\n{prompt}")
    menu = {str(i): opt for i, opt in enumerate(options, 1)}
    for i, opt in menu.items(): print(f"  {i}. {opt.replace('_', ' ').title()}")
    print("  Q. Quit")
    while True:
        choice = input("Enter your choice: ").strip().upper()
        if choice == 'Q': return None
        if choice in menu: return menu[choice]
        print("Invalid choice.")

def main():
    root = Tk(); root.withdraw()

    # 1. Load MapCHECK Data (No change here)
    messagebox.showinfo("Step 1/3", "Select the folder containing your MapCHECK JSON files.")
    json_folder = filedialog.askdirectory(title="Select Folder with JSON files")
    if not json_folder: return
    mapcheck_df_monthly = load_and_aggregate_json_data(json_folder)
    if mapcheck_df_monthly.empty:
        messagebox.showerror("Error", "No valid MapCHECK data could be loaded."); return

    # 2. Load External CSV Data
    messagebox.showinfo("Step 2/3", "Select the CSV file exported from your QA software.")
    csv_path = filedialog.askopenfilename(title="Select External Data CSV", filetypes=[("CSV Files", "*.csv")])
    if not csv_path: return
    
    # <<< MODIFIED SECTION: New workflow for handling the TQA format >>>
    external_df_raw, available_tests = load_external_data_and_get_tests(csv_path)
    if external_df_raw is None: return

    # 3. Let user choose which specific test they want to analyze from the CSV
    chosen_test = get_user_choice("Please choose which test from the CSV you want to analyze:", available_tests)
    if chosen_test is None: return

    # Filter the raw external data to only include the test the user selected
    selected_test_df = external_df_raw[external_df_raw["Test"] == chosen_test].copy()
    
    # Now perform the monthly aggregation on this filtered data
    selected_test_df['year_month'] = selected_test_df['date'].dt.to_period('M')
    external_df_monthly = selected_test_df.groupby('year_month')[['Value']].mean().reset_index()
    
    # Rename the "Value" column to be descriptive (e.g., "neck_pass_rate")
    clean_col_name = f"{chosen_test.lower().replace(' ', '_')}_pass_rate"
    external_df_monthly.rename(columns={"Value": clean_col_name}, inplace=True)
    # <<< END OF MODIFIED SECTION >>>

    # 4. Merge the two monthly data sources
    merged_df = pd.merge(mapcheck_df_monthly, external_df_monthly, on='year_month', how='inner')

    if merged_df.empty:
        messagebox.showerror("No Matching Data", "No data with matching months was found between the MapCHECK data and the external CSV file."); return
    
    print("\n--- Successfully merged monthly data. Ready for correlation analysis. ---")
    print(merged_df.to_string()) # Use to_string() to ensure the whole table prints

    # 5. Let the user choose which parameters to correlate
    mapcheck_params = [col for col in mapcheck_df_monthly.columns if col != 'year_month']
    
    while True:
        mc_param = get_user_choice("Choose the MapCHECK parameter (X-axis):", mapcheck_params)
        if mc_param is None: break
        
        # The Y-axis choice is now simple, as we only have one pass rate column
        ext_param = clean_col_name 

        plot_df = merged_df[[mc_param, ext_param]].dropna()
        if len(plot_df) < 2:
            messagebox.showwarning("Not Enough Data", "Not enough matching monthly data points to create a correlation plot."); continue

        # 6. Generate Scatter Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(plot_df[mc_param], plot_df[ext_param], alpha=0.7, s=80)
        
        z = np.polyfit(plot_df[mc_param], plot_df[ext_param], 1)
        p = np.poly1d(z)
        plt.plot(plot_df[mc_param], p(plot_df[mc_param]), "r--", label=f"Fit (y={z[0]:.2f}x+{z[1]:.2f})")

        plt.title(f"Monthly Correlation: {chosen_test} Pass Rate vs. {mc_param.replace('_', ' ').title()}", fontsize=16)
        plt.xlabel(f"Average Monthly MapCHECK {mc_param.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel(f"Average Monthly {ext_param.replace('_', ' ').title()}", fontsize=12)
        plt.grid(True); plt.legend(); plt.show()

if __name__ == "__main__":
    main()
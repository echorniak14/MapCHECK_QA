import os
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
import numpy as np
from .common import utils

def load_and_aggregate_json_data(directory):
    """Loads all MapCHECK JSON data and aggregates it by year and month."""
    all_records = []
    filenames = [f for f in os.listdir(directory) if f.endswith(".json")]
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            measurement_date = utils.parse_date_robustly(data.get("_measurement_date"))
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

def run_correlation(json_folder):
    root = Tk(); root.withdraw()

    if not json_folder:
        messagebox.showinfo("Step 1/3", "Select the folder containing your MapCHECK JSON files.")
        json_folder = filedialog.askdirectory(title="Select Folder with JSON files")
        if not json_folder: return
    mapcheck_df_monthly = load_and_aggregate_json_data(json_folder)
    if mapcheck_df_monthly.empty:
        messagebox.showerror("Error", "No valid MapCHECK data could be loaded."); return

    messagebox.showinfo("Step 2/3", "Select the CSV file exported from your QA software.")
    csv_path = filedialog.askopenfilename(title="Select External Data CSV", filetypes=[("CSV Files", "*.csv")])
    if not csv_path: return
    
    external_df_raw, available_tests = utils.load_external_data_and_get_tests(csv_path)
    if external_df_raw is None: return

    chosen_test = utils.get_user_choice("Please choose which test from the CSV you want to analyze:", available_tests)
    if chosen_test is None: return

    selected_test_df = external_df_raw[external_df_raw["Test"] == chosen_test].copy()
    
    selected_test_df['year_month'] = selected_test_df['date'].dt.to_period('M')
    external_df_monthly = selected_test_df.groupby('year_month')[['Value']].mean().reset_index()
    
    clean_col_name = f"{chosen_test.lower().replace(' ', '_')}_pass_rate"
    external_df_monthly.rename(columns={"Value": clean_col_name}, inplace=True)

    merged_df = pd.merge(mapcheck_df_monthly, external_df_monthly, on='year_month', how='inner')

    if merged_df.empty:
        messagebox.showerror("No Matching Data", "No data with matching months was found between the MapCHECK data and the external CSV file."); return
    
    print("\n--- Successfully merged monthly data. Ready for correlation analysis. ---")
    print(merged_df.to_string())

    mapcheck_params = [col for col in mapcheck_df_monthly.columns if col != 'year_month']
    
    while True:
        mc_param = utils.get_user_choice("Choose the MapCHECK parameter (X-axis):", mapcheck_params)
        if mc_param is None: break
        
        ext_param = clean_col_name

        plot_df = merged_df[[mc_param, ext_param]].dropna()
        if len(plot_df) < 2:
            messagebox.showwarning("Not Enough Data", "Not enough matching monthly data points to create a correlation plot."); continue

        plt.figure(figsize=(10, 8))
        plt.scatter(plot_df[mc_param], plot_df[ext_param], alpha=0.7, s=80)
        
        z = np.polyfit(plot_df[mc_param], plot_df[ext_param], 1)
        p = np.poly1d(z)
        plt.plot(plot_df[mc_param], p(plot_df[mc_param]), "r--", label=f"Fit (y={z[0]:.2f}x+{z[1]:.2f})")

        plt.title(f"Monthly Correlation: {chosen_test} Pass Rate vs. {mc_param.replace('_', ' ').title()}", fontsize=16)
        plt.xlabel(f"Average Monthly MapCHECK {mc_param.replace('_', ' ').title()}", fontsize=12)
        plt.ylabel(f"Average Monthly {ext_param.replace('_', ' ').title()}", fontsize=12)
        plt.grid(True); plt.legend(); plt.show()
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .common import utils

TRENDABLE_METRICS = {
    "linearity_slope": "Linearity Slope",
    "dose_rate_slope": "Dose Rate Slope",
    "fs_slope": "Field Size Fit Slope",
    "uniformity": "Uniformity (Std Dev %)",
    "leakage_mean": "Mean Leakage Value",
}

def generate_trend_plot(records, metric_key, metric_name, output_path):
    dates = [rec["date"] for rec in records if rec.get(metric_key) is not None]
    values = [rec[metric_key] for rec in records if rec.get(metric_key) is not None]

    if len(dates) < 2:
        print(f"Not enough data for '{metric_name}'. Skipping plot.")
        return False

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
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Generated trend plot: {output_path}")
    return True

def generate_web_trend_plot(device_name, metric_key, static_dir):
    json_folder = os.path.join('data', device_name)
    all_data = utils.load_all_json_data(json_folder)
    
    if not all_data or len(all_data) < 2:
        return None, f"Not enough historical data to generate a trend plot for {TRENDABLE_METRICS.get(metric_key, 'the selected metric')}."

    metric_name = TRENDABLE_METRICS.get(metric_key)
    if not metric_name:
        return None, f"Invalid metric key: {metric_key}"

    plot_filename = f"trend_{device_name}_{metric_key}.png"
    output_path = os.path.join(static_dir, plot_filename)

    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    success = generate_trend_plot(all_data, metric_key, metric_name, output_path)
    
    if success:
        return plot_filename, None
    else:
        return None, f"Failed to generate trend plot for {metric_name}."



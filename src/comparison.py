import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .common import utils
from .trending import TRENDABLE_METRICS

def generate_web_comparison_plot(device_names, metric_key, static_dir):
    all_data = []
    for device_name in device_names:
        json_folder = os.path.join('data', device_name)
        device_data = utils.load_all_json_data(json_folder)
        for record in device_data:
            record['device_name'] = device_name  # Add device name to each record
        all_data.extend(device_data)

    if not all_data:
        return None, "No data found for the selected devices."

    metric_name = TRENDABLE_METRICS.get(metric_key)
    if not metric_name:
        return None, f"Invalid metric key: {metric_key}"

    plot_filename = f"comparison_{'_'.join(device_names)}_{metric_key}.png"
    output_path = os.path.join(static_dir, plot_filename)

    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # The plot_device_comparison function needs to be adapted to work without tkinter
    # and to save the plot to a file.
    success = plot_device_comparison_web(all_data, metric_key, metric_name, output_path)

    if success:
        return plot_filename, None
    else:
        return None, f"Failed to generate comparison plot for {metric_name}."

def plot_device_comparison_web(records, metric_key, metric_name, output_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    has_data_to_plot = False

    device_names = sorted(list({rec["device_name"] for rec in records}))

    for device_name in device_names:
        device_records = [rec for rec in records if rec["device_name"] == device_name]
        
        dates = [rec["date"] for rec in device_records if rec.get(metric_key) is not None]
        values = [rec[metric_key] for rec in device_records if rec.get(metric_key) is not None]

        if len(dates) >= 1:
            has_data_to_plot = True
            linestyle = '-' if len(dates) > 1 else ''
            ax.plot(dates, values, marker='o', linestyle=linestyle, label=f"Device: {device_name}")

    if not has_data_to_plot:
        plt.close(fig)
        return False

    ax.set_title(f"Device Comparison for {metric_name}", fontsize=16)
    ax.set_xlabel("Measurement Date", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return True



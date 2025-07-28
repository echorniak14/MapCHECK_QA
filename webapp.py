from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import subprocess
from datetime import datetime
from src.analysis import TEST_SUITE_CONFIG, FILE_ROLE_DEFINITIONS
from src.trending import generate_web_trend_plot, TRENDABLE_METRICS
from src.comparison import generate_web_comparison_plot
from src.common import utils
from src.common.utils import parse_metadata_from_file

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
DATA_DIR = 'data'
STATIC_DIR = 'static'

@app.route('/')
def index():
    devices = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    return render_template('index.html', devices=devices)

@app.route('/add_device', methods=['POST'])
def add_device():
    device_name = request.form['device_name']
    sample_file = request.files['sample_file']

    # Basic sanitization to prevent directory traversal issues
    device_name = "".join(c for c in device_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if device_name and sample_file.filename != '':
        device_path = os.path.join(DATA_DIR, device_name)
        if not os.path.exists(device_path):
            os.makedirs(device_path)
        
        # Save the sample file and extract metadata
        sample_file_path = os.path.join(device_path, sample_file.filename)
        sample_file.save(sample_file_path)
        
        metadata = parse_metadata_from_file(sample_file_path)
        
        metadata_path = os.path.join(device_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=4)
            
        os.remove(sample_file_path) # Remove the sample file after processing

    return redirect(url_for('index'))


@app.route('/device/<device_name>')
def device(device_name):
    device_path = os.path.join(DATA_DIR, device_name)
    if not os.path.exists(device_path):
        return redirect(url_for('index'))
    
    # Load metadata
    metadata = {}
    metadata_path = os.path.join(device_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            import json
            metadata = json.load(f)
            
    history = []
    for item in os.listdir(device_path):
        if item.endswith(".pdf"):
            try:
                # Expecting format YYYY-MM-DD_AnalysisType.pdf
                parts = item.replace('.pdf', '').split('_')
                date_str = parts[0]
                analysis_type = '_'.join(parts[1:])
                history.append({
                    "date": date_str,
                    "analysis_type": analysis_type,
                    "pdf_name": item,
                    "json_name": item.replace('.pdf', '.json')
                })
            except IndexError:
                # Handle files that don't match the expected format
                pass 
    # Sort history by date, newest first
    history.sort(key=lambda x: x['date'], reverse=True)

    return render_template('device.html', device_name=device_name, history=history, metadata=metadata)

@app.route('/device/<device_name>/<filename>')
def serve_file(device_name, filename):
    device_path = os.path.join(DATA_DIR, device_name)
    return send_from_directory(device_path, filename)

@app.route('/device/<device_name>/upload')
def upload_page(device_name):
    return render_template('upload.html', 
                           device_name=device_name, 
                           analysis_types=TEST_SUITE_CONFIG, 
                           file_roles=FILE_ROLE_DEFINITIONS)


@app.route('/device/<device_name>/run_analysis', methods=['POST'])
def run_analysis(device_name):
    files = request.files.getlist('files')
    device_path = os.path.join(DATA_DIR, device_name)
    analysis_type = request.form.get('analysis_type')

    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('upload_page', device_name=device_name))

    if not analysis_type:
        flash('No analysis type selected')
        return redirect(url_for('upload_page', device_name=device_name))

    # Save the uploaded file temporarily
    temp_files = []
    for file in files:
        if file.filename.lower().endswith('.txt'):
            temp_file_path = os.path.join(device_path, file.filename)
            file.save(temp_file_path)
            temp_files.append(temp_file_path)
        else:
            flash(f'Error: Invalid file type for {file.filename}. Only .txt files are accepted.', 'error')
            # Clean up already saved files
            for temp_file in temp_files:
                os.remove(temp_file)
            return redirect(url_for('upload_page', device_name=device_name))

    today_str = datetime.now().strftime('%Y-%m-%d')
    output_base = f"{today_str}_{analysis_type}"
    output_pdf_path = os.path.join(device_path, f"{output_base}.pdf")
    output_json_path = os.path.join(device_path, f"{output_base}.json")

    command = [
        'python',
        '-m', 'src.analysis',
        '--headless',
        analysis_type,
        output_pdf_path,
        output_json_path
    ] + temp_files

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        flash(f'Successfully processed {len(temp_files)} files for {analysis_type} analysis.', 'success')
    except subprocess.CalledProcessError as e:
        flash(f'Error processing files: {e.stderr}', 'error')
    finally:
        # Clean up the temporary raw files
        for temp_file in temp_files:
            os.remove(temp_file)

    return redirect(url_for('device', device_name=device_name))

@app.route('/device/<device_name>/trends', methods=['GET', 'POST'])
def trends_page(device_name):
    device_path = os.path.join(DATA_DIR, device_name)
    if not os.path.exists(device_path):
        return redirect(url_for('index'))

    plot_url = None
    selected_metric = None
    selected_metric_name = None

    if request.method == 'POST':
        selected_metric = request.form.get('metric')
        if selected_metric and selected_metric in TRENDABLE_METRICS:
            plot_filename, error = generate_web_trend_plot(device_name, selected_metric, STATIC_DIR)
            if error:
                flash(error, 'warning')
            if plot_filename:
                plot_url = url_for('static', filename=plot_filename)
                selected_metric_name = TRENDABLE_METRICS[selected_metric]

    return render_template('trends.html', 
                           device_name=device_name, 
                           metrics=TRENDABLE_METRICS, 
                           selected_metric=selected_metric,
                           selected_metric_name=selected_metric_name,
                           plot_url=plot_url)

@app.route('/device/<device_name>/trends/generate', methods=['POST'])
def generate_trend_plot_route(device_name):
    metric_to_plot = request.form.get('metric')
    return redirect(url_for('trends_page', device_name=device_name, metric=metric_to_plot))


@app.route('/device/<device_name>/rename', methods=['POST'])
def rename_device(device_name):
    new_device_name = request.form.get('new_device_name')
    if not new_device_name:
        flash('New device name cannot be empty.', 'error')
        return redirect(url_for('device', device_name=device_name))

    # Sanitize the new device name
    new_device_name = "".join(c for c in new_device_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not new_device_name:
        flash('Invalid device name. Only letters, numbers, spaces, underscores, and hyphens are allowed.', 'error')
        return redirect(url_for('device', device_name=device_name))

    old_device_path = os.path.join(DATA_DIR, device_name)
    new_device_path = os.path.join(DATA_DIR, new_device_name)

    if os.path.exists(new_device_path):
        flash(f'A device named "{new_device_name}" already exists.', 'error')
        return redirect(url_for('device', device_name=device_name))

    try:
        os.rename(old_device_path, new_device_path)
        flash(f'Device "{device_name}" has been renamed to "{new_device_name}".', 'success')
        return redirect(url_for('device', device_name=new_device_name))
    except OSError as e:
        flash(f'Error renaming device: {e}', 'error')
        return redirect(url_for('device', device_name=device_name))

@app.route('/device/<device_name>/delete', methods=['POST'])
def delete_device(device_name):
    password = request.form.get('password')
    if password != 'mbpcc1':
        flash('Incorrect password.', 'error')
        return redirect(url_for('device', device_name=device_name))

    device_path = os.path.join(DATA_DIR, device_name)
    if os.path.exists(device_path):
        import shutil
        shutil.rmtree(device_path)
        flash(f'Device "{device_name}" has been permanently deleted.', 'success')
    else:
        flash('Device not found.', 'error')

    return redirect(url_for('index'))

@app.route('/compare', methods=['GET', 'POST'])
def compare_page():
    all_devices = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    plot_url = None
    selected_devices = []
    selected_metric = None
    selected_metric_name = None

    if request.method == 'POST':
        selected_devices = request.form.getlist('devices')
        selected_metric = request.form.get('metric')

        if selected_devices and selected_metric and selected_metric in TRENDABLE_METRICS:
            plot_filename, error = generate_web_comparison_plot(selected_devices, selected_metric, STATIC_DIR)
            if error:
                flash(error, 'warning')
            if plot_filename:
                plot_url = url_for('static', filename=plot_filename)
                selected_metric_name = TRENDABLE_METRICS[selected_metric]

    return render_template('compare.html',
                           all_devices=all_devices,
                           selected_devices=selected_devices,
                           metrics=TRENDABLE_METRICS,
                           selected_metric=selected_metric,
                           selected_metric_name=selected_metric_name,
                           plot_url=plot_url)


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    app.run(debug=True)
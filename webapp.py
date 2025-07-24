from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import subprocess
from datetime import datetime
from src.analysis import TEST_SUITE_CONFIG, FILE_ROLE_DEFINITIONS

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
DATA_DIR = 'data'

@app.route('/')
def index():
    devices = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    return render_template('index.html', devices=devices)

@app.route('/add_device', methods=['POST'])
def add_device():
    device_name = request.form['device_name']
    # Basic sanitization to prevent directory traversal issues
    device_name = "".join(c for c in device_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if device_name:
        device_path = os.path.join(DATA_DIR, device_name)
        if not os.path.exists(device_path):
            os.makedirs(device_path)
    return redirect(url_for('index'))


@app.route('/device/<device_name>')
def device(device_name):
    device_path = os.path.join(DATA_DIR, device_name)
    if not os.path.exists(device_path):
        return redirect(url_for('index'))
    
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

    return render_template('device.html', device_name=device_name, history=history)

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


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    app.run(debug=True)
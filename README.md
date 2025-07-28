# MapCHECK QA Analysis Suite (Web Application)

This project provides a powerful, user-friendly, and automated web-based system for processing, analyzing, and interpreting quality assurance (QA) data from SunNuclear MapCHECK devices. It replaces manual data handling, minimizes human error, and provides clear, actionable insights through automated reporting and trend analysis.

## Features

- **Automated Analysis**: Process raw `.txt` files from the MapCHECK device to generate detailed PDF and JSON reports.
- **Device Management**: Add, rename, and securely delete MapCHECK devices.
- **Device Metadata**: Automatically extract and display key device information (serial number, model, etc.) from uploaded sample files.
- **Device Comparison**: Compare QA trends across multiple devices with interactive plots.
- **Trend Analysis**: Visualize longitudinal trends for individual MapCHECK devices or groups of devices.

## Prerequisites

- Python 3.x
- The following Python libraries (install via pip): `Flask`, `matplotlib`, `numpy`, `pandas` (Note: `tkinter` is no longer a direct dependency for the web app).

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-repo/MapCHECK_QA.git
    cd MapCHECK_QA
    ```
2.  Install the required Python libraries:
    ```bash
    pip install Flask matplotlib numpy pandas
    ```

## Usage

To run the web application, execute the `webapp.py` script from your terminal:

```bash
python webapp.py
```

This will start the Flask development server. You will see output in your terminal, including a URL like `* Running on http://127.0.0.1:5000`. Open this URL in your web browser to interact with the application.

### Key Features in the Web App:

*   **Adding a New Device**: From the main dashboard, you can add a new device. You will be prompted to upload a sample `.txt` file from the device. This file is used to automatically extract and store device metadata.
*   **Running Analysis**: On each device's page, you can upload new `.txt` files to run analyses, generating PDF reports and JSON data files.
*   **Trend Analysis**: From a device's page, you can navigate to the "Trend Analysis" section to visualize historical data for that specific device.
*   **Compare Data**: From any device's page, you can navigate to the "Compare Data" section to select multiple devices and compare their QA trends.
*   **Renaming a Device**: On a device's page, you can rename the device. This will update the directory name and all associated data.
*   **Deleting a Device**: On a device's page, you can delete the device and all its associated data. **This action requires a password for security.**

### Device Deletion Password:

The password required to delete a device is: `mbpcc1`

This password is case-sensitive.
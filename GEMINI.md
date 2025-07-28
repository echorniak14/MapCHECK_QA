# Gemini Project Reference: MapCHECK QA Suite

## Primary Objective

The project's primary objective is to create a powerful, user-friendly, and automated system for processing, analyzing, and interpreting quality assurance (QA) data from SunNuclear MapCHECK devices. The suite of tools is designed to replace manual data handling, minimize human error, and provide clear, actionable insights through automated reporting and trend analysis.

## Current Architecture

The project has been refactored into a modular, web-based application.

- **Entry Point**: `webapp.py` is the main script that runs the Flask web application.
- **Source Code**: All Python source code is located in the `src/` directory.
- **Modules**: The `src/` directory contains the core logic for various functionalities:
    - `src/analysis.py`: Handles the initial processing of raw `.txt` files from the MapCHECK device, generating detailed PDF and JSON reports.
    - `src/comparison.py`: Compares QA trends across multiple devices using the generated JSON files.
    - `src/correlation.py`: Correlates MapCHECK QA data with external data sources (e.g., CSV files).
    - `src/trending.py`: Analyzes and visualizes longitudinal trends for a single device or a group of devices from the generated JSON files.
- **Common Utilities**: `src/common/utils.py` contains shared functions used across the different modules, such as file parsing, robust date handling, and data loading.

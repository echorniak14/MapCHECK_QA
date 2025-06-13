Of course! This is a fantastic script, and a good README.md file is essential for making it usable for yourself in the future and for anyone else you might share it with.
Based on your code, here is a comprehensive README.md file. It explains the purpose, dependencies, usage, and key features like the automated file-finding convention.
You can copy and paste the text below directly into a new file named README.md in your Git repository.
MapCHECK QA Analysis Suite
This Python script is a comprehensive tool for processing and analyzing data from Sun Nuclear MapCHECK devices. It automates the entire workflow from raw text file conversion to final report generation, significantly speeding up the quality assurance process.
The script extracts the "Dose Interpolated" data from MapCHECK .txt files, performs a series of physics-based analyses, and outputs the results into a clean, multi-page PDF report and a machine-readable JSON file.
Key Features
TXT to CSV Conversion: Automatically converts proprietary MapCHECK .txt files into a clean, usable .csv format, reliably extracting the numerical dose grid.
Interactive GUI: Uses simple dialog boxes to guide the user through file selection and setup, requiring no code modification for day-to-day use.
Automated File Finding: Can automatically find the correct files for each analysis if they follow a standard naming convention, or allows for manual selection if needed.
Comprehensive Analysis: Performs a suite of standard QA tests:
Dose Output Linearity
Dose Rate Dependence
Field Size Dependence
Beam Uniformity / Flatness
Leakage Radiation
Automated Report Generation:
PDF Report: Creates a professional, multi-page PDF containing a summary cover page and detailed plots for each analysis performed.
JSON Output: Saves all calculated numerical results (fit coefficients, standard deviations, etc.) to a .json file for easy post-processing, data logging, or further research.
Data Integrity Checks: Includes a check for device serial number mismatches across files to prevent accidental analysis of data from different devices.
Prerequisites
To run this script, you need to have Python 3 installed, along with a few common data science libraries.
Python 3: If you don't have it, download it from python.org.
Required Libraries: Install the necessary libraries using pip. Open your terminal or command prompt and run:
pip install pandas numpy matplotlib
Use code with caution.
Bash
(Note: tkinter is usually included with standard Python installations on Windows and macOS. On some Linux distributions, you may need to install it separately, e.g., sudo apt-get install python3-tk)
How to Run
Place your MapCHECK .txt files in a folder on your computer.
Open a terminal or PowerShell/Command Prompt.
Navigate to the directory where you saved mapcheck_analysis_v2.py.
Run the script with the following command:
python mapcheck_analysis_v2.py
Use code with caution.
Bash
The script will then guide you through a series of dialog boxes:
Convert TXT to CSV: You'll first be asked if you want to convert new .txt files.
Select Files & Folders: You will be prompted to select your input files and a location to save the final report.
Choose File Selection Method: You can let the script try to find files automatically based on standard names (see below) or select each file manually.
Standard File Naming Convention
For maximum automation, you can name your CSV files according to the script's built-in conventions. If you use these names, the script can run without prompting you to select files for each test.
The script looks for the following filenames in the CSV directory:
Test	Standard Filename(s)
Leakage	Leakage.csv
Dose Rate Dependence	DR_150mupmin.csv, DR_350mupmin.csv, linearity_100MU.csv
Field Size Dependence	FS_small.csv, linearity_100MU.csv, FS_large.csv
Linearity	linearity_10MU.csv, linearity_100MU.csv, linearity_1000MU.csv
Uniformity	FS_large.csv (or your preferred uniformity file)
Note: The script is flexible. If it cannot find these files, it will simply ask you to select them manually.
Input File Format
The script is designed to parse .txt files exported from the MapCHECK software. The key requirement is that the file must contain the Dose Interpolated section, which is where the numerical grid data begins.
Output Files
For a chosen output name like WH_Analysis, the script will generate two files:
WH_Analysis.pdf: A multi-page report.
Cover Page: A summary of all test results, including the device serial number and measurement date.
Analysis Pages: One page for each analysis (Linearity, Dose Rate, etc.), showing a clear plot of the data and the results of the fit.
WH_Analysis_data.json: A structured data file.
Contains all the raw numerical results (e.g., fit slope/intercept, mean values, standard deviations) for easy import into other software, databases, or analysis scripts.
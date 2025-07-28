# Project TODO List

This file tracks planned improvements and known issues for the MapCHECK QA web application.

## Known Issues

- **Device Naming:** The input for creating a new device does not properly sanitize special characters (e.g., '/', '\', ':'). This could lead to errors when creating directories. The input should be validated to only allow filesystem-safe characters.

## Completed Enhancements

- **Edit Device Name:** Added the ability to edit a device's name after it has been created.
- **Device Metadata:** Extract and display detailed device information (e.g., serial number, device ID) on the device hub page. This data can be parsed from the uploaded MapCHECK `.txt` files and stored for easy reference.
- **Comparison Feature Implemented:** The comparison feature is now fully functional in the web application. Users can select multiple devices and a metric to generate a comparison plot.
- **Device Deletion (Password Protected):** Implemented a secure device deletion feature requiring a password for confirmation.

## Future Enhancements


## Current Progress

**July 28, 2025**
- **Device Metadata Feature Implemented:** Users can now upload a sample `.txt` file when creating a new device. The application automatically extracts and displays key metadata (Model, Serial Number, etc.) on the device's page.
- **Trending Feature Implemented:** The trending feature is now fully functional in the web application. Users can select a metric and generate a trend plot for any device. This involved creating a new web-friendly plotting function, adding a new template for the trends page, and updating the web application routes.
- **Comparison Feature Implemented:** The comparison feature is now fully functional in the web application. Users can select multiple devices and a metric to generate a comparison plot.
- **Device Deletion (Password Protected):** Implemented a secure device deletion feature requiring a password for confirmation.

- **NEXT STEPS:** Collect more data from various devices to run and test comparison feature.
- **NEXT STEPS:** Address documentation inconsistency: `app.py` is a CLI, while `README.md` and `GEMINI.md` describe a web application. Decide whether to remove `app.py`, update it, or maintain both versions and clarify documentation.
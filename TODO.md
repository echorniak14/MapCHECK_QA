# Project TODO List

This file tracks planned improvements and known issues for the MapCHECK QA web application.

## Known Issues

- **Device Naming:** The input for creating a new device does not properly sanitize special characters (e.g., '/', '\', ':'). This could lead to errors when creating directories. The input should be validated to only allow filesystem-safe characters.

## Future Enhancements

- **Device Metadata:** Extract and display detailed device information (e.g., serial number, device ID) on the device hub page. This data can be parsed from the uploaded MapCHECK `.txt` files and stored for easy reference.
- **Edit Device Name:** Add the ability to edit a device's name after it has been created. This will require careful implementation to ensure that the corresponding data directory is renamed correctly and that all historical data remains associated with the device. Restrictions on naming will need to be considered.


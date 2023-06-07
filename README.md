# Binary Parameter Fitting for MEAM Potential

## Description
The `binary_fitting.py` script is used to parameterize the binary parameters for the MEAM (Modified Embedded Atom Method) potential using differential evaluation optimization in Scipy.

## Dependencies
- The script requires the unary potential's library and parameter files as they will be merged with the binary parameter files.
- Please make sure you have [Atomsk](https://atomsk.univ-lille.fr/) installed in your system.
- Additionally, the script assumes that you have a LAMMPS executable named `lmp_serial` in the same folder. You can modify the executable name as per your requirements.

## Usage
1. Clone the repository or download the `binary_fitting.py` script.
2. Make sure you have the required unary potential files and parameter files in the same directory.
3. Install any necessary dependencies, including Atomsk.
4. Ensure the `lmp_serial` executable is present in the current folder or update the script with the correct executable name.
5. Run the `binary_fitting.py` script using your preferred Python environment.

## Notes
- Please refer to the documentation for detailed information on the MEAM potential and how to interpret the fitted binary parameters.
- If you encounter any issues or have questions, feel free to open an issue or reach out to the project maintainers.

Happy fitting!

# Parameter Fitting for MEAM Potential


## Description
Welcome to the MEAM Potential Parameter Fitting repository! This project aims to provide a user-friendly and efficient way to parameterize the MEAM potential for various materials. The Modified Embedded Atom Method (MEAM) potential is widely used in materials science for simulating the behavior of atomic systems.

In this repository, you will find two main scripts:

1. `unary_fitting.py`: This script allows you to parameterize the unary parameters of the MEAM potential, enabling the fitting of elastic constants, vacancy, interstitial, free surface, and stacking fault energies for single-phase materials in FCC, BCC, and HCP crystal structures.

2. `binary_fitting.py`: This script focuses on fitting the binary parameters of the MEAM potential. It utilizes differential evaluation optimization in Scipy to provide accurate and reliable parameterization.

To make the process even easier, we have included a sample dataset for a Hafnium fitting as an example of unary fitting, as well as a Titanium-Zirconium fitting as a b2 reference structure. You can also add your own `database.data` file to customize the fitting for your specific material.

## Dependencies
Before using the scripts, please ensure you have the following dependencies:

- [Atomsk](https://atomsk.univ-lille.fr/): Make sure to install Atomsk, a powerful tool for manipulating atomic systems, which is required for the parameterization process.

- LAMMPS Executable: The scripts assume that you have a LAMMPS executable named `lmp_serial` in the same folder. However, you can easily modify the script to use a different executable if necessary.

- lowest_file_search.py: This utility script can assist you in finding the best fitting parameters by multiplying the parameters to narrow down the search bounds. It helps streamline the fitting process and optimize parameter selection.

## Usage
To get started with parameter fitting, follow these steps:

1. Clone the repository or download the `binary_fitting.py` and `unary_fitting.py` scripts.

2. Ensure that the required unary potential files and parameter files are present in the same directory. This includes the `database.data` file for your specific material.

3. Install any necessary dependencies, including Atomsk, by following the provided documentation and instructions.

4. Make sure you have the LAMMPS executable named `lmp_serial` in the current folder. If your executable has a different name, simply update the script accordingly.

5. Run either the `binary_fitting.py` or `unary_fitting.py` script using your preferred Python environment, depending on the type of parameterization you require.


If you encounter any issues, have questions, or would like to contribute to the project, feel free to open an issue or reach out to Mash (mash@lanl.gov)

Happy fitting!


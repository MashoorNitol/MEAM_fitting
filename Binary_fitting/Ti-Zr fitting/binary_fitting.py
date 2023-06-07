#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:24:59 2023

@author: mashroornitol
"""

import numpy as np
import os
import subprocess
import platform
import random
from scipy.optimize import differential_evolution
import glob
import re as regex
from plot_function import*


import warnings
warnings.filterwarnings("ignore", message="color is redundantly defined by the 'color' keyword argument and the fmt string")


# Differential equation parameters
kwords = {"strategy": "best1bin",       # Differential evolution strategy to use
          "maxiter": 500,               # Maximum number of generations over which the population is evolved
          "popsize": 25,                # Multiplier for setting the total population size
          "tol": 0.01,                  # Relative tolerance for convergence
          "mutation": (0.5, 1),         # Differential weight. When (min,max), dithering is used to improve the algorithm by adding a small amount of random noise to the search space
          "recombination": 1.0,         # Progression of mutants into the next generation, from 0 to 1
          "seed": None,                 # Seed for random generator, default is None
          "callback": None,             # Function to follow process of minimization
          "disp": False,                # Prints the evaluated func at every iteration
          "polish": True,               # If true, scipy.optimize.minimize is used to polish the best population  member at the end
          "init": 'sobol',              # Population iniziation type
          "atol": 0,                    # Absolute tolerance for convergence
          "updating": 'immediate',      # With immediate, the best soluction is continuously updated within a single generation
          "workers": 1,                 # CPU cores to be used, use -1 to call all available CPU cores, more than 1 doesn't seem to work because of parallelization issues :)
          "constraints": (),            # Constraints on the solver
           "x0": None,                   # An initial guess to the minimization
          "integrality": None,          # Array of decision variables, gives a boolean indicating whether the decision variable is contrained to integer values
          "vectorized": False}          # If true, func is sent an array and is expected to return an array of the solutions. Alternative to parallelization!


ev_angstrom3 = 160.2176621 # 1 eV/Å3 = 160.2176621 GPa
# MEAM constants #

zbl = 0
nn2 = 1

rc = 6.0
def lammps_run(in_file):
    null_device = '/dev/null' if platform.system() != 'Windows' else 'nul'
    with open(null_device, 'w') as devnull:
        subprocess.call([lammps_executable, '-in', in_file], stdout=devnull, stderr=subprocess.STDOUT)

def minimization():
    return('fix 1 all box/relax aniso 0.0 vmax 0.1\
           \nminimize 1e-16 1e-16 1000 1000\
           \nunfix 1\
           \nrun 0')

def potential_mod(style,potential_file):
    return('pair_style %s\npair_coeff * * %s\
           \nneighbor 2.0 bin\nneigh_modify delay 10 check yes'%(style,potential_file))

def output_mod():
    return('compute eng all pe/atom \ncompute eatoms all reduce sum c_eng\
           \nthermo 1 \nthermo_style    custom step etotal temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol c_eatoms\
               \nthermo_modify norm no\
           \ndump 1 all custom 1 dump_lammps id type x y z fx fy fz c_eng')
           
def lammps_header():
    return('clear\nunits metal\ndimension 3\nboundary p p p \
                \natom_style atomic\natom_modify map array\nbox tilt large\n')
                
def relaxation():
    return('\nminimize 1e-8 1e-8 1000 1000\
           \nrun 0')

def minimized_output(structure):
    return('variable natoms equal count(all)\
           \nvariable teng equal "etotal"\
           \nvariable ecoh equal  v_teng/v_natoms\
           \nvariable length equal "lx"\
           \nvariable lengthZ equal "lz"\
           \nvariable perA_vol equal "vol/v_natoms"\
           \nvariable ca_ratio equal "v_lengthZ/v_length"\
           \nprint "---------------------------------------------------------------------------" append %s\
           \nprint "# minimized structure and energy of %s" append %s\
           \nprint "Cohesive energy (eV) = ${ecoh}" append %s\
           \nprint "Lattice constant (Angstroms) = ${length}" append %s\
           \nprint "C/A ratio  = ${ca_ratio}" append %s\
           \nprint "Volume per atom (Angstrom^3/atom) = ${perA_vol}" append %s\
           \nprint "---------------------------------------------------------------------------" append %s'%(output,structure,output,output,output,output,output,output))


def parse_elasticconstants_file(filename):
    results = {} 
    with open(filename, 'r') as file:
        content = file.read()

    # Extract structure information
    structure_matches = regex.finditer(r'# Elastic constant and bulk modulus of (\w+)', content)
    for match in structure_matches:
        structure = match.group(1)
        result = {}
        lattice_constant_match = regex.search(r'Lattice constant \(Angstroms\) = ([\d.]+)', content[match.end():])
        if lattice_constant_match:
            result[f'{structure}_lattice_constant_a'] = lattice_constant_match.group(1)

        # Extract elastic constants
        elastic_constants = regex.findall(r'c(\d+) = ([\d.]+) GPa', content[match.end():])
        constants_dict = {}
        for constant in elastic_constants:
            constant_name = f'{structure}_ec_c{constant[0]}'
            constant_value = constant[1]
            if not any(abs(float(constant_value) - float(value)) <= 0.1 for value in constants_dict.values()):
                constants_dict[constant_name] = constant_value
        
        result.update(constants_dict)
        results[structure] = result

        result.update(constants_dict)


    return results
                    
def elastic_constant(structure,data_file,output,potential_file):
    properties='elastic_constant_%s'%(structure)
    """
    elastic constant calculaiton  using LAMMPS for any given material
    """
    with open('displace.mod', 'w') as f:
        f.write('if "${dir} == 1" then & \n"variable len0 equal ${lx0}" \nif "${dir} == 2" then &\
\n "variable len0 equal ${ly0}" \nif "${dir} == 3" then & \n   "variable len0 equal ${lz0}"\
\nif "${dir} == 4" then & \n   "variable len0 equal ${lz0}" \nif "${dir} == 5" then &\
\n   "variable len0 equal ${lz0}" \nif "${dir} == 6" then & \n   "variable len0 equal ${ly0}"')
        f.write('\nclear \nbox tilt large\nread_restart restart.equil\n')
        f.write(potential_mod(style,potential_file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write('\nvariable delta equal -${up}*${len0} \nvariable deltaxy equal -${up}*xy \nvariable deltaxz equal -${up}*xz \nvariable deltayz equal -${up}*yz\
\nif "${dir} == 1" then &\
\n   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\
\nif "${dir} == 2" then &\
\n   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\
\nif "${dir} == 3" then &\
\n   "change_box all z delta 0 ${delta} remap units box"\
\nif "${dir} == 4" then &\
\n   "change_box all yz delta ${delta} remap units box"\
\nif "${dir} == 5" then &\
\n   "change_box all xz delta ${delta} remap units box"\
\nif "${dir} == 6" then &\
\n   "change_box all xy delta ${delta} remap units box"\
\nminimize ${etol} ${ftol} ${maxiter} ${maxeval}')
        f.write('\nvariable tmp equal pxx\nvariable pxx1 equal ${tmp} \nvariable tmp equal pyy \nvariable pyy1 equal ${tmp} \nvariable tmp equal pzz\
                \nvariable pzz1 equal ${tmp} \nvariable tmp equal pxy \nvariable pxy1 equal ${tmp} \nvariable tmp equal pxz \nvariable pxz1 equal ${tmp}\
                \nvariable tmp equal pyz \nvariable pyz1 equal ${tmp}')
        f.write('\nclear \nbox tilt large\nread_restart restart.equil\n')
        f.write(potential_mod(style,potential_file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write('variable C1neg equal ${d1}\
                \nvariable C1neg equal ${d1}\
                \nvariable C2neg equal ${d2}\
                    \nvariable C3neg equal ${d3}\
                \nvariable C4neg equal ${d4}\
                \nvariable C5neg equal ${d5}\
                \nvariable C6neg equal ${d6}')
        f.write('\nclear \nbox tilt large\nread_restart restart.equil\n')
        f.write(potential_mod(style,potential_file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write('\
\nvariable delta equal ${up}*${len0}\
\nvariable deltaxy equal ${up}*xy\
\nvariable deltaxz equal ${up}*xz\
\nvariable deltayz equal ${up}*yz\
\nif "${dir} == 1" then &\
\n    "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\
\nif "${dir} == 2" then &\
\n    "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\
\nif "${dir} == 3" then &\
\n    "change_box all z delta 0 ${delta} remap units box"\
\nif "${dir} == 4" then &\
\n    "change_box all yz delta ${delta} remap units box"\
\nif "${dir} == 5" then &\
\n    "change_box all xz delta ${delta} remap units box"\
\nif "${dir} == 6" then &\
\n    "change_box all xy delta ${delta} remap units box"\
\nminimize ${etol} ${ftol} ${maxiter} ${maxeval}\
\nvariable tmp equal pe\
\nvariable e1 equal ${tmp}\
\nvariable tmp equal press\
\nvariable p1 equal ${tmp}\
\nvariable tmp equal pxx\
\nvariable pxx1 equal ${tmp}\
\nvariable tmp equal pyy\
\nvariable pyy1 equal ${tmp}\
\nvariable tmp equal pzz\
\nvariable pzz1 equal ${tmp}\
\nvariable tmp equal pxy\
\nvariable pxy1 equal ${tmp}\
\nvariable tmp equal pxz\
\nvariable pxz1 equal ${tmp}\
\nvariable tmp equal pyz\
\nvariable pyz1 equal ${tmp}\
\nvariable C1pos equal ${d1}\
\nvariable C2pos equal ${d2}\
\nvariable C3pos equal ${d3}\
\nvariable C4pos equal ${d4}\
\nvariable C5pos equal ${d5}\
\nvariable C6pos equal ${d6}\
\nvariable C1${dir} equal 0.5*(${C1neg}+${C1pos})\
\nvariable C2${dir} equal 0.5*(${C2neg}+${C2pos})\
\nvariable C3${dir} equal 0.5*(${C3neg}+${C3pos})\
\nvariable C4${dir} equal 0.5*(${C4neg}+${C4pos})\
\nvariable C5${dir} equal 0.5*(${C5neg}+${C5pos})\
\nvariable C6${dir} equal 0.5*(${C6neg}+${C6pos})')

# # Delete dir to make sure it is not reused

# variable dir delete




# ')
        
    with open('elastic.in', 'w') as f:
        f.write('variable up equal 1.0e-4\nvariable atomjiggle equal 1.0e-5\n')
        f.write(lammps_header() + '\n\n')
        f.write('variable cfac equal 1.0e-4\nvariable cunits string GPa\n')
        f.write('variable etol equal 1.0e-8\nvariable ftol equal 1.0e-8\nvariable maxiter equal 1000\
                \nvariable maxeval equal 1000\
                \nvariable dmax equal 1.0e-2\n')
        f.write('read_data %s'%data_file + '\n\n')
        f.write('change_box all triclinic\n')
        f.write('mass * 1.0e-20\n')
        f.write(potential_mod(style,potential_file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write('\nfix 3 all box/relax  aniso 0.0 vmax 0.1\n')
        f.write('min_style cg\nmin_modify       dmax ${dmax} line quadratic\n')
        f.write(relaxation() + '\n\n')
        f.write('variable tmp equal pxx\nvariable pxx0 equal ${tmp}\nvariable tmp equal pyy\
                \nvariable pyy0 equal ${tmp}\nvariable tmp equal pzz\nvariable pzz0 equal ${tmp}\
                \nvariable tmp equal pyz\nvariable pyz0 equal ${tmp}\nvariable tmp equal pxz\
                \nvariable pxz0 equal ${tmp} \nvariable tmp equal pxy \nvariable pxy0 equal ${tmp}\
                \nvariable tmp equal lx\nvariable lx0 equal ${tmp}\nvariable tmp equal ly\
                \nvariable ly0 equal ${tmp}\nvariable tmp equal lz\nvariable lz0 equal ${tmp}\
                \nvariable d1 equal -(v_pxx1-${pxx0})/(v_delta/v_len0)*${cfac}\
                \nvariable d2 equal -(v_pyy1-${pyy0})/(v_delta/v_len0)*${cfac}\
                \nvariable d3 equal -(v_pzz1-${pzz0})/(v_delta/v_len0)*${cfac}\
                \nvariable d4 equal -(v_pyz1-${pyz0})/(v_delta/v_len0)*${cfac}\
                \nvariable d5 equal -(v_pxz1-${pxz0})/(v_delta/v_len0)*${cfac}\
                \nvariable d6 equal -(v_pxy1-${pxy0})/(v_delta/v_len0)*${cfac}\
                \ndisplace_atoms all random ${atomjiggle} ${atomjiggle} ${atomjiggle} 87287 units box\
                \nunfix 3\nwrite_restart restart.equil\nvariable dir equal 1\
                \ninclude displace.mod\nvariable dir equal 2\ninclude displace.mod\nvariable dir equal 3\
                \ninclude displace.mod\nvariable dir equal 4\ninclude displace.mod  \nvariable dir equal 5\
                \ninclude displace.mod\nvariable dir equal 6\ninclude displace.mod \nvariable C11all equal ${C11} \nvariable C22all equal ${C22}\
                \nvariable C33all equal ${C33} \nvariable C12all equal 0.5*(${C12}+${C21})\
                \nvariable C13all equal 0.5*(${C13}+${C31}) \nvariable C23all equal 0.5*(${C23}+${C32})\
                \nvariable C44all equal ${C44} \nvariable C55all equal ${C55}\
                \nvariable C66all equal ${C66} \nvariable C14all equal 0.5*(${C14}+${C41})\
                \nvariable C15all equal 0.5*(${C15}+${C51}) \nvariable C16all equal 0.5*(${C16}+${C61})\
                \nvariable C24all equal 0.5*(${C24}+${C42}) \nvariable C25all equal 0.5*(${C25}+${C52})\
                \nvariable C26all equal 0.5*(${C26}+${C62}) \nvariable C34all equal 0.5*(${C34}+${C43})\
                \nvariable C35all equal 0.5*(${C35}+${C53}) \nvariable C36all equal 0.5*(${C36}+${C63})\
                \nvariable C45all equal 0.5*(${C45}+${C54}) \nvariable C46all equal 0.5*(${C46}+${C64})\
                \nvariable C56all equal 0.5*(${C56}+${C65}) \nvariable C16all equal 0.5*(${C16}+${C61})\
                \nvariable C24all equal 0.5*(${C24}+${C42}) \nvariable C25all equal 0.5*(${C25}+${C52})\
                \nvariable C26all equal 0.5*(${C26}+${C62}) \nvariable C11cubic equal (${C11all}+${C22all}+${C33all})/3.0\
                \nvariable C12cubic equal (${C12all}+${C13all}+${C23all})/3.0 \nvariable C44cubic equal (${C44all}+${C55all}+${C66all})/3.0\
                \nvariable bulkmodulus equal (${C11cubic}+2*${C12cubic})/3.0 \nvariable shearmodulus1 equal ${C44cubic} \nvariable shearmodulus2 equal (${C11cubic}-${C12cubic})/2.0\
                \nvariable poissonratio equal 1.0/(1.0+${C11cubic}/${C12cubic})')    
        f.write('\nprint "---------------------------------------------------------------------------" append %s\
            \nprint "# Elastic constant and bulk modulus of %s" append %s\
            \nprint "Bulk Modulus = ${bulkmodulus} ${cunits}" append %s\
            \nprint "Shear Modulus 1 = ${shearmodulus1} ${cunits}" append %s\
            \nprint "Shear Modulus 2 = ${shearmodulus2} ${cunits}" append %s\
            \nprint "Poisson Ratio = ${poissonratio}" append %s\
            \nprint "c11 = ${C11all} ${cunits}" append %s\
            \nprint "c12 = ${C12all} ${cunits}" append %s\
            \nprint "c13 = ${C13all} ${cunits}" append %s\
            \nprint "c33 = ${C33all} ${cunits}" append %s\
            \nprint "c44 = ${C44all} ${cunits}" append %s\
            \nprint "c55 = ${C55all} ${cunits}" append %s\
            \nprint "c66 = ${C66all} ${cunits}" append %s\
            \nprint "---------------------------------------------------------------------------" append %s'%(output,structure,output,output,output,output,output,output,output,output,output,output,output,output,output))
        f.write('\nrun 0\n')
        f.write(minimized_output(structure) + '\n\n')
    
    lammps_run('elastic.in')

    
    constants = parse_elasticconstants_file(output)
    if os.path.exists(properties):
        os.system('rm -r %s' % properties)
        
    os.system('mkdir %s'%properties)
    os.system('mv elastic.in restart.equil displace.mod %s %s/'%(data_file,properties))   
    return constants

def gsfe(ref_struct,lattice_constants,elements,output,potential_file):
    ref_struct = element_properties_database['ref_struct']
    if ref_struct == 'b2':
        gsfe_output = 'gsfe_output.txt'
        sorted_elements = reference_info['elements']
        type_atom1, type_atom2 = sorted_elements[:2]
        latparam = float(lattice_constants[ref_struct])
        direction = '111'
        plane = '110'
        properties = 'gsfe_%s_%s_%s'%(ref_struct,plane,direction)
        os.system('atomsk --create bcc %.16f %s %s orient [1-10] [112]  [11-1]  -orthogonal-cell -duplicate 1 1 30 %s_%s.lmp >/dev/null 2>&1'%(latparam,type_atom1, type_atom2,ref_struct,direction))
        with open('gsfe.in', 'w') as f:
            f.write(lammps_header() + '\n\n')
            f.write('\nread_data %s_%s.lmp'%(ref_struct,direction)+ '\n\n')
            f.write(potential_mod(style,potential_file) + '\n\n')
            f.write(output_mod() + '\n\n')
            f.write(minimization() + '\n\n')
            f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
            f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-20"\n')
            f.write('\nvariable updel0 equal "v_z_middle0+20"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
            f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
            f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
            f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
            f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
            f.write('\n\nvariable xmin equal xlo\nvariable xmax equal xhi\nvariable xtot equal v_xmin+v_xmax\n')
            f.write('\nvariable ymin equal ylo\nvariable ymax equal yhi\nvariable ytot equal v_ymin+v_ymax')
            f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                    \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                    \ngroup lower region lower')
            f.write('\nfix 1 lower move linear ${xtot} 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
            f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
            f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-16\n')
            f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
            f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
            f.write('run 0 \nprint "XXX ${GSFE}" append %s'%gsfe_output)
            f.close()

            output_filegsfe = gsfe_output
            if os.path.exists("%s"%output_filegsfe):
                os.remove("%s"%output_filegsfe)

            os.system('cp gsfe.in gsfe_initial.in')
            with open('gsfe_initial.in', 'r') as fileinitial :
                filedatainitial = fileinitial.read()
            filedatainitial = filedatainitial.replace('XXX', '0')
            with open('gsfe_initial.in', 'w') as fileinitial:
                fileinitial.write(filedatainitial)                        
            lammps_run('gsfe_initial.in')
        
            # Read the output.txt file
            with open(output_filegsfe, 'r') as filegsfe:
                linesgsfe = filegsfe.readlines()
        
            # Extract the last value from output.txt
            target_value = float(linesgsfe[-1].split()[1])
            # print(last_value)
            # Initialize the range variables
            start_range = 100
            end_range = start_range
            # Loop until the condition is satisfied
            flag = False
            while not flag:
                # Update the end_range variable
                end_range += 100                
                # Loop through the range
                for i in range(start_range, end_range, 100):
                    # Modify gsfe.in file
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as filegsfe:
                        filedatagsfe = filegsfe.read()                
                    filedatagsfe = filedatagsfe.replace('XXX', str(i))
            
                    with open('gsfe_%d.in' % i, 'w') as filegsfe:
                        filegsfe.write(filedatagsfe)
            
                    # Run the executable using the modified input file
                    lammps_run('gsfe_%d.in' % i)
            
                    # Read the output.txt file
                    with open(output_filegsfe, 'r') as filegsfe:
                        linesgsfe = filegsfe.readlines()
            
                    # Extract the last value from output.txt
                    last_value = float(linesgsfe[-1].split()[1])
                    difference_val = last_value-target_value
                    # Check if the new value satisfies the condition
                    if difference_val < 3.0:
                        flag = True
                        break
            
                # Update the start_range for the next iteration
                start_range = end_range
                if flag:
                    break
                    
        
        data2 = np.genfromtxt(gsfe_output)         
        x_data2 = [i2 for i2 in np.linspace(0,1,len(data2))]
        y_data2 = [(data2[i][1]-data2[0][1])/1 for i in range(len(data2))]
        y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0 <= x2 <= 0.45]
        y_usf2 = max(y_subset2)
        y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0.45 <= x2 <= 0.6]
        y_ssf2 = min(y_subset2)
        fig2, ax2 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
        direction = '110'
        ax2.plot(x_data2,y_data2,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s :  [%s]$<$%s$>$'%(ref_struct,plane,direction))
        ax2.set_xlabel(r'Normalized displacement',fontweight='bold')
        ax2.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
        ax2.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
        fig2.savefig('gsfe_%s_%s_%s.pdf'%(ref_struct,plane,direction))  
        plt.close()
        os.system('mkdir %s'%properties)            
        os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s.pdf %s/'%(gsfe_output,ref_struct,plane,direction,properties))
        filename = output
        new_val = '\n# Stacking fault energies %s plane %s direction\n\
        \nunstable stacking fault for %s  %s plane in %s direction = %0.16f mj/m^2\
            \nstable stacking fault for %s  %s plane in %s direction = %0.16f mj/m^2\
                    \n------------------------------------------------------------\n\
                        '%(plane,direction,ref_struct,plane,direction,y_usf2,ref_struct,plane,direction,y_ssf2)                                   
        if os.path.isfile(filename):
            with open(filename, 'a') as file3:
                file3.write(new_val)
        else:
            with open(filename, 'w') as file3:
                file3.write(new_val)
                
                
def merge_libraryfiles(input_files, output_file):
    with open(output_file, 'w') as output:
        for file_name in input_files:
            with open(file_name, 'r') as file:
                for line in file:
                    output.write(line)
                output.write('\n\n') 


def merge_parameterfiles(file1_path, file2_path, merged_file_path):
    replacement_mappings = {
        "zbl": {"(1,1)": "(2,2)"},
        "nn2": {"(1,1)": "(2,2)"},
        "rho0": {"(1)": "(2)"},
        "Ec": {"(1,1)": "(2,2)"},
        "re": {"(1,1)": "(2,2)"},
        "alpha": {"(1,1)": "(2,2)"},
        "repuls": {"(1,1)": "(2,2)"},
        "attrac": {"(1,1)": "(2,2)"},
        "Cmin": {"(1,1,1)": "(2,2,2)"},
        "Cmax": {"(1,1,1)": "(2,2,2)"}
    }


    file1_name = file1_path.split('.')[0]
    file2_name = file2_path.split('.')[0]


    with open(merged_file_path, 'w') as merged_file:

        with open(file1_path, 'r') as file1:
            file1_contents = file1.readlines()
        file1_zbl_index = next((i for i, line in enumerate(file1_contents) if line.startswith("zbl(1,1)")), None)
        merged_file.writelines(file1_contents[:file1_zbl_index])
        merged_file.write(f"# {file1_name} Parameters\n")
        merged_file.writelines(file1_contents[file1_zbl_index:])

        # Append the relevant lines from the second file to the merged file
        merged_file.write(f"\n\n# {file2_name} Parameters\n")
        with open(file2_path, 'r') as file2:
            append_lines = False
            for line in file2:
                if line.startswith("zbl"):
                    append_lines = True
                if append_lines:
                    for keyword, replacements in replacement_mappings.items():
                        if line.startswith(keyword):
                            for original, replacement in replacements.items():
                                line = line.replace(original, replacement)
                            break
                    merged_file.write(line)





def calculate_binary_parameter(ref_struct,ecoh,bulkmod,lattice_param):  
    unit_vol = 0.0
    if ref_struct == 'b2':
        unit_vol = (lattice_param ** 3) / 2
    elif ref_struct == 'l10':
        unit_vol = (lattice_param ** 3) / 4

    ev_angstrom3 = 160.2176621  

    alpha = np.sqrt(9 * bulkmod * unit_vol / (ecoh * ev_angstrom3))
    
    if ref_struct == 'l10':
        re = (1 / np.sqrt(2.)) * lattice_param
    elif ref_struct == 'b2':
        re = (np.sqrt(3.) / 2) * lattice_param

    return alpha, re



def create_binary_part(cmin_values,cmax_values,attrac_vals,repuls_vals,reference_info, alpha, re):
    with open('%s'%outputparam, 'a') as file:
        file.write('\n\n# Binary parameters of %s reference - %s' % (reference_info["ref_struct"], ', '.join(reference_info['elements'])))
        file.write('\nzbl(1,2) = 0\n')
        file.write('nn2(1,2) = 1\n')
        file.write(f'lattce(1,2) = \'{reference_info["ref_struct"]}\'\n')
        coh_energy = float(reference_info["cohesieve_energy"])
        file.write(f'Ec(1,2) = {coh_energy:.6f}'  + '\n')
        file.write(f're(1,2) = {re:.6f}\n')
        file.write(f'alpha(1,2) = {alpha:.6f}\n')
        file.write(f'\nattrac(1,2) = {attrac_vals:.6f}\n')
        file.write(f'repuls(1,2) = {repuls_vals:.6f}\n\n')
        for values in cmin_values:
            formatted_values = ','.join(str(val) for val in values)
            file.write(f"Cmin({formatted_values}) = {cmin_values[values]:.6f}\n")
        for values in cmax_values:
            formatted_values = ','.join(str(val) for val in values)
            file.write(f"Cmax({formatted_values}) = {cmax_values[values]:.6f}\n")
            



def objective_function(params=(0,)*11):
    if os.path.exists("%s"%output):
        os.remove("%s"%output)
    errors = []
    
    cmin1,cmin2,cmin3,cmin4,cmax1,cmax2,cmax3,cmax4,alpha,repuls,attrac = params
    cmin_values = {
        (1,1,2): cmin1,
        (2,2,1): cmin2,
        (1,2,1): cmin3,
        (1,2,2): cmin4,
        (2,1,2): cmin4,
        (2,1,1): cmin3
    }

    cmax_values = {
        (1,1,2): cmax1,
        (2,2,1): cmax2,
        (1,2,1): cmax3,
        (1,2,2): cmax4,
        (2,1,2): cmax4,
        (2,1,1): cmax3
    }
    attrac_vals = attrac
    repuls_vals = repuls
    

    
    merge_libraryfiles(['%s'%unary_potential_1lib, '%s'%unary_potential_2lib],'%s'%outputlib)
    merge_parameterfiles('%s'%unary_potential_1param, '%s'%unary_potential_2param, '%s'%outputparam)
    create_binary_part(cmin_values, cmax_values,attrac_vals,repuls_vals, reference_info, alpha, re)
    create_structure_files(database_info)

    if os.path.exists("%s"%output):
        os.remove("%s"%output)         
    merged_dict = {}
    elastic_constants = database_info['elastic_constants']

    for structure, constants in lattice_constants.items():
        data_file = f'{structure}.lmp'
        a = elastic_constant(structure, data_file, output, potential_file)
        for key, value in a.items():
            merged_dict.setdefault(key, []).append(value)
    
        if structure in merged_dict:
            struct_data = merged_dict[structure][0]
            dft_constants = elastic_constants.get(structure, {})
            for key, value in struct_data.items():
                if key in dft_constants:
                    valueO = float(value)
                    dft_val = float(dft_constants[key])
                    diff = (valueO - dft_val) ** 2
                    errors.append(diff)
                    
    error = np.sqrt(np.mean(errors))
    serror=np.round(error,decimals=2)
    param_file = outputparam
    print('~~ ERROR:{}'.format(str(serror)))
    if serror < 19.0:
        # print('~~ ERROR:{}'.format(str(serror)))        
        os.system('cp %s %s_%s'%(param_file,param_file,str(serror)))
        os.system('cp %s %s_%s'%(output,output,str(serror)))
        os.system('mv %s_%s %s_%s %s/'%(param_file,str(serror),output,str(serror),pf))
    return error 



def read_database(database_file):

    with open(database_file, "r") as file:
        lines = file.readlines()
    
    lines = [line for line in lines if '"' not in line]
    
    elements_line = next(line for line in lines if line.startswith("# element:"))
    elements = elements_line.split(":")[1].strip().split()

    ref_struct_line = next(line for line in lines if line.startswith("ref_struct"))
    ref_struct = ref_struct_line.split("=")[1].strip()


    refstruct_cohesive_energy_line = next(line for line in lines if line.startswith("ref_struct_cohesieve_energy"))
    refstruct_cohesive_energy = float(refstruct_cohesive_energy_line.split("=")[1].strip())


    refstruct_lattice_constant_a_line = next(line for line in lines if line.startswith("ref_struct_lattice_constant_a"))
    refstruct_lattice_constant_a = float(refstruct_lattice_constant_a_line.split("=")[1].strip())
    
    refstruct_bulkmod_line = next(line for line in lines if line.startswith("ref_struct_bulk_modulus"))
    refstruct_bulkmod = float(refstruct_bulkmod_line.split("=")[1].strip())
    

    reference_info = {
        "elements" : elements,
        "ref_struct": ref_struct,
        "cohesieve_energy": refstruct_cohesive_energy,
        "bulk_modulus": refstruct_bulkmod,        
        "lattice_constant_a": refstruct_lattice_constant_a
    }


    structure_to_consider_line = next(line for line in lines if line.startswith("structure_to_consider"))
    structures_to_consider = structure_to_consider_line.split("=")[1].strip().split(", ")


    lattice_constants = {}
    elastic_constants = {}
    for structure in structures_to_consider:
        if structure == "hcp":
            hcp_lattice_constant_a_line = next(line for line in lines if line.startswith("hcp_lattice_constant_a"))
            hcp_lattice_constant_a = float(hcp_lattice_constant_a_line.split("=")[1].strip())

            hcp_lattice_constant_c_line = next(line for line in lines if line.startswith("hcp_lattice_constant_c"))
            hcp_lattice_constant_c = float(hcp_lattice_constant_c_line.split("=")[1].strip())
            
            hcp_ec_c11_line = next(line for line in lines if line.startswith("hcp_ec_c11"))
            hcp_ec_c11 = float(hcp_ec_c11_line.split("=")[1].strip())
            hcp_ec_c12_line = next(line for line in lines if line.startswith("hcp_ec_c12"))
            hcp_ec_c12 = float(hcp_ec_c12_line.split("=")[1].strip())
            hcp_ec_c13_line = next(line for line in lines if line.startswith("hcp_ec_c13"))
            hcp_ec_c13 = float(hcp_ec_c13_line.split("=")[1].strip())
            hcp_ec_c33_line = next(line for line in lines if line.startswith("hcp_ec_c33"))
            hcp_ec_c33 = float(hcp_ec_c33_line.split("=")[1].strip())
            hcp_ec_c44_line = next(line for line in lines if line.startswith("hcp_ec_c44"))
            hcp_ec_c44 = float(hcp_ec_c44_line.split("=")[1].strip())            

            lattice_constants[structure] = {
                "hcp_lattice_constant_a": hcp_lattice_constant_a,
                "hcp_lattice_constant_c": hcp_lattice_constant_c
            }
            elastic_constants[structure] = {
                "hcp_ec_c11" : hcp_ec_c11,
                "hcp_ec_c12" : hcp_ec_c12,
                "hcp_ec_c13" : hcp_ec_c13,
                "hcp_ec_c33" : hcp_ec_c33,
                "hcp_ec_c44" : hcp_ec_c44               
            }
        elif structure.startswith("dO19"):
            element = structure.split("_")[1]
            dO19_lattice_constant_a_line = next(line for line in lines if line.startswith("dO19_%s_lattice_constant_a" % element))
            dO19_lattice_constant_c_line = next(line for line in lines if line.startswith("dO19_%s_lattice_constant_c" % element))
            dO19_lattice_constant_a = float(dO19_lattice_constant_a_line.split("=")[1].strip())
            dO19_lattice_constant_c = float(dO19_lattice_constant_c_line.split("=")[1].strip())
            
            dO19_ec_c11_line = next(line for line in lines if line.startswith("dO19_%s_ec_c11"%element))
            dO19_ec_c11 = float(dO19_ec_c11_line.split("=")[1].strip())
            dO19_ec_c12_line = next(line for line in lines if line.startswith("dO19_%s_ec_c12"%element))
            dO19_ec_c12 = float(dO19_ec_c12_line.split("=")[1].strip())
            dO19_ec_c13_line = next(line for line in lines if line.startswith("dO19_%s_ec_c13"%element))
            dO19_ec_c13 = float(dO19_ec_c13_line.split("=")[1].strip())
            dO19_ec_c33_line = next(line for line in lines if line.startswith("dO19_%s_ec_c33"%element))
            dO19_ec_c33 = float(dO19_ec_c33_line.split("=")[1].strip())
            dO19_ec_c44_line = next(line for line in lines if line.startswith("dO19_%s_ec_c44"%element))
            dO19_ec_c44 = float(dO19_ec_c44_line.split("=")[1].strip()) 

            lattice_constants[structure] = {
                "dO19_%s_lattice_constant_a" % element: dO19_lattice_constant_a,
                "dO19_%s_lattice_constant_c" % element: dO19_lattice_constant_c
            }
            elastic_constants[structure] = {
                "dO19_%s_ec_c11"%element : dO19_ec_c11,
                "dO19_%s_ec_c12"%element : dO19_ec_c12,
                "dO19_%s_ec_c13"%element : dO19_ec_c13,
                "dO19_%s_ec_c33"%element : dO19_ec_c33,
                "dO19_%s_ec_c44"%element : dO19_ec_c44               
            }            
        elif structure.startswith("b2"):
            b2_lattice_constant_a_line = next(line for line in lines if line.startswith("b2_lattice_constant_a"))
            b2_lattice_constant_a = float(b2_lattice_constant_a_line.split("=")[1].strip())
            b2_ec_c11_line = next(line for line in lines if line.startswith("b2_ec_c11"))
            b2_ec_c11 = float(b2_ec_c11_line.split("=")[1].strip())
            b2_ec_c12_line = next(line for line in lines if line.startswith("b2_ec_c12"))
            b2_ec_c12 = float(b2_ec_c12_line.split("=")[1].strip())
            b2_ec_c44_line = next(line for line in lines if line.startswith("b2_ec_c44"))
            b2_ec_c44 = float(b2_ec_c44_line.split("=")[1].strip()) 

            lattice_constants[structure] = {
                "b2_lattice_constant_a" : b2_lattice_constant_a
            }
            
            elastic_constants[structure] = {
                "b2_ec_c11" : b2_ec_c11,
                "b2_ec_c12" : b2_ec_c12,
                "b2_ec_c44" : b2_ec_c44               
            }
        elif structure.startswith("l10"):
            element = structure.split("_")[1]
            l10_lattice_constant_a_line = next(line for line in lines if line.startswith("l10_%s_lattice_constant_a" % element))
            l10_lattice_constant_c_line = next(line for line in lines if line.startswith("l10_%s_lattice_constant_c" % element))
            l10_lattice_constant_a = float(l10_lattice_constant_a_line.split("=")[1].strip())
            l10_lattice_constant_c = float(l10_lattice_constant_c_line.split("=")[1].strip())
            l10_ec_c11_line = next(line for line in lines if line.startswith("l10_ec_c11"))
            l10_ec_c11 = float(l10_ec_c11_line.split("=")[1].strip())
            l10_ec_c12_line = next(line for line in lines if line.startswith("l10_ec_c12"))
            l10_ec_c12 = float(l10_ec_c12_line.split("=")[1].strip())
            l10_ec_c13_line = next(line for line in lines if line.startswith("l10_ec_c13"))
            l10_ec_c13 = float(l10_ec_c13_line.split("=")[1].strip())
            l10_ec_c33_line = next(line for line in lines if line.startswith("l10_ec_c33"))
            l10_ec_c33 = float(l10_ec_c33_line.split("=")[1].strip())
            l10_ec_c44_line = next(line for line in lines if line.startswith("l10_ec_c44"))
            l10_ec_c44 = float(l10_ec_c44_line.split("=")[1].strip()) 

            lattice_constants[structure] = {
                "l10_%s_lattice_constant_a" % element: l10_lattice_constant_a,
                "l10_%s_lattice_constant_c" % element: l10_lattice_constant_c
            }
            elastic_constants[structure] = {
                "l10_ec_c11" : l10_ec_c11,
                "l10_ec_c12" : l10_ec_c12,
                "l10_ec_c13" : l10_ec_c13,
                "l10_ec_c33" : l10_ec_c33,
                "l10_ec_c44" : l10_ec_c44               
            }
        elif structure.startswith("dO3"):
            element = structure.split("_")[1]
            dO3_lattice_constant_a_line = next(line for line in lines if line.startswith("dO3_%s_lattice_constant_a" % element))
            dO3_lattice_constant_a = float(dO3_lattice_constant_a_line.split("=")[1].strip())

            dO3_ec_c11_line = next(line for line in lines if line.startswith("dO3_%s_ec_c11"%element))
            dO3_ec_c11 = float(dO3_ec_c11_line.split("=")[1].strip())
            dO3_ec_c12_line = next(line for line in lines if line.startswith("dO3_%s_ec_c12"%element))
            dO3_ec_c12 = float(dO3_ec_c12_line.split("=")[1].strip())
            dO3_ec_c13_line = next(line for line in lines if line.startswith("dO3_%s_ec_c13"%element))
            dO3_ec_c13 = float(dO3_ec_c13_line.split("=")[1].strip())
            dO3_ec_c33_line = next(line for line in lines if line.startswith("dO3_%s_ec_c33"%element))
            dO3_ec_c33 = float(dO3_ec_c33_line.split("=")[1].strip())
            dO3_ec_c44_line = next(line for line in lines if line.startswith("dO3_%s_ec_c44"%element))
            dO3_ec_c44 = float(dO3_ec_c44_line.split("=")[1].strip())

            lattice_constants[structure] = {
                "dO3_%s_lattice_constant_a" % element: dO3_lattice_constant_a
            }
            elastic_constants[structure] = {
                "dO3_%s_ec_c11"%element : dO3_ec_c11,
                "dO3_%s_ec_c12"%element : dO3_ec_c12,
                "dO3_%s_ec_c13"%element : dO3_ec_c13,
                "dO3_%s_ec_c33"%element : dO3_ec_c33,
                "dO3_%s_ec_c44"%element : dO3_ec_c44               
            }             

    return {
        "reference_info": reference_info,
        "lattice_constants": lattice_constants,
        "elastic_constants" : elastic_constants
    }


def create_structure_files(database_info):
    for structure, constants in lattice_constants.items():
        sorted_elements = sorted(reference_info['elements'])
        type_atom1, type_atom2 = sorted_elements[:2]
        for constant, value in constants.items():
            # print("  -", constant + ":", value)
            if structure == 'b2':
                os.system('rm POSCAR')
                os.system('rm b2.lmp')
                lx = float(constants['b2_lattice_constant_a'])
                with open('b2.POSCAR', 'w') as file_poscar:
                    file_poscar.write(f'B2 POSCAR file\n'
                                    f'1.0\n'
                                    f'{lx:.16f} 0 0\n'
                                    f'0 {lx:.16f} 0\n'
                                    f'0 0 {lx:.16f}\n'
                                    f'{type_atom1} {type_atom2}\n'
                                    f'1 1\n'
                                    f'Direct\n'
                                    f'0.0 0.0 0.0\n'
                                    f'0.5 0.5 0.5')
        
                os.system('cp b2.POSCAR POSCAR')
                os.system('atomsk POSCAR b2.lmp >/dev/null 2>&1')
            elif structure == 'hcp':
                os.system('rm POSCAR')
                os.system('rm hcp.lmp')
                lx = float(constants['hcp_lattice_constant_a'])
                ly0 = -0.5*lx
                ly1 = (np.sqrt(3.)/2)*lx
                lz = float(constants['hcp_lattice_constant_c'])
                with open('hcp.POSCAR', 'w') as file_poscar:
                    file_poscar.write(f'HCP (lamella) POSCAR file\n'
                                    f'1.0\n'
                                    f'{lx:.16f} 0 0\n'
                                    f'{ly0:.16f} {ly1:.16f} 0\n'
                                    f'0 0 {lz:.16f}\n'
                                    f'{type_atom1} {type_atom2}\n'
                                    f'1 1\n'
                                    f'Direct\n'
                                    f'0.0 0.0 0.0\n'
                                    f'0.33333 0.66667 0.5')
        
                os.system('cp hcp.POSCAR POSCAR')
                os.system('atomsk POSCAR hcp.lmp >/dev/null 2>&1')           
            elif structure == 'dO3_%s'%type_atom1:
                lx = float(constants['dO3_%s_lattice_constant_a'%type_atom1])
                os.system('rm POSCAR')
                os.system('rm dO3_%s.lmp'%type_atom1)
                with open('dO3_%s.POSCAR'%type_atom1, 'w') as file_poscar:
                    file_poscar.write(f'dO3 - {type_atom1} rich POSCAR file\n'
                                    f'1.0\n'
                                    f'{lx:.16f} 0 0\n'
                                    f'0 {lx:.16f} 0\n'
                                    f'0 0 {lx:.16f}\n'
                                    f'{type_atom1} {type_atom2}\n'
                                    f'12 4\n'
                                    f'Direct\n'
                                    f'0.50 0.00 0.00\n'
                                    f'0.25 0.75 0.75 \n'
                                    f'0.25 0.25 0.75 \n'
                                    f'0.50 0.50 0.50 \n'
                                    f'0.25 0.25 0.25 \n'
                                    f'0.25 0.75 0.25 \n'
                                    f'0.00 0.00 0.50 \n'
                                    f'0.75 0.75 0.25 \n'
                                    f'0.75 0.25 0.25 \n'
                                    f'0.00 0.50 0.00 \n'
                                    f'0.75 0.25 0.75 \n'
                                    f'0.75 0.75 0.75 \n'
                                    f'0.00 0.00 0.00 \n'
                                    f'0.00 0.50 0.50 \n'
                                    f'0.50 0.00 0.50 \n'
                                    f'0.50 0.50 0.00 ')                              
                os.system('cp dO3_%s.POSCAR POSCAR'%type_atom1)
                os.system('atomsk POSCAR dO3_%s.lmp >/dev/null 2>&1'%type_atom1)
            elif structure == 'dO3_%s'%type_atom2:
                lx = float(constants['dO3_%s_lattice_constant_a'%type_atom2])
                os.system('rm POSCAR')
                os.system('rm dO3_%s.lmp'%type_atom2)
                with open('dO3_%s.POSCAR'%type_atom2, 'w') as file_poscar:
                    file_poscar.write(f'dO3 - {type_atom2} rich POSCAR file\n'
                                    f'1.0\n'
                                    f'{lx:.16f} 0 0\n'
                                    f'0 {lx:.16f} 0\n'
                                    f'0 0 {lx:.16f}\n'
                                    f'{type_atom1} {type_atom2}\n'
                                    f'4 12\n'
                                    f'Direct\n'
                                    f'0.00 0.00 0.00 \n'
                                    f'0.00 0.50 0.50 \n'
                                    f'0.50 0.00 0.50 \n'
                                    f'0.50 0.50 0.00 \n'                                
                                    f'0.50 0.00 0.00 \n'
                                    f'0.25 0.75 0.75 \n'
                                    f'0.25 0.25 0.75 \n'
                                    f'0.50 0.50 0.50 \n'
                                    f'0.25 0.25 0.25 \n'
                                    f'0.25 0.75 0.25 \n'
                                    f'0.00 0.00 0.50 \n'
                                    f'0.75 0.75 0.25 \n'
                                    f'0.75 0.25 0.25 \n'
                                    f'0.00 0.50 0.00 \n'
                                    f'0.75 0.25 0.75 \n'
                                    f'0.75 0.75 0.75 ')                              
                os.system('cp dO3_%s.POSCAR POSCAR'%type_atom2)
                os.system('atomsk POSCAR dO3_%s.lmp >/dev/null 2>&1'%type_atom2)            
            elif structure == 'dO19_%s'%type_atom1:
                lx = float(constants['dO19_%s_lattice_constant_a'%type_atom1])
                ly0 = -0.5*lx
                ly1 = (np.sqrt(3.)/2)*lx
                lz = float(constants['dO19_%s_lattice_constant_c'%type_atom1]) 
                os.system('rm POSCAR')
                os.system('rm dO19_%s.lmp'%type_atom1)
                with open('dO19_%s.POSCAR'%type_atom1, 'w') as file_poscar:
                    file_poscar.write(f'dO19 - {type_atom1} rich POSCAR file\n'
                                    f'1.0\n'
                                    f'{lx:.16f} 0 0\n'
                                    f'{ly0:.16f} {ly1:.16f} 0\n'
                                    f'0 0 {lz:.16f}\n'
                                    f'{type_atom1} {type_atom2}\n'
                                    f'6 2\n'
                                    f'Direct\n'
                                    f'0.680766 0.840383 0.750000\n'
                                    f'0.840383 0.680766 0.250000\n'
                                    f'0.840383 0.159617 0.250000\n'
                                    f'0.159617 0.840383 0.750000\n'
                                    f'0.159617 0.319234 0.750000\n'
                                    f'0.319234 0.159617 0.250000\n'
                                    f'0.666667 0.333333 0.750000\n'
                                    f'0.333333 0.666667 0.250000 '   
                                    )
                os.system('cp dO19_%s.POSCAR POSCAR'%type_atom1)
                os.system('atomsk POSCAR dO19_%s.lmp >/dev/null 2>&1'%type_atom1)
            elif structure == 'dO19_%s'%type_atom2:
                lx = float(constants['dO19_%s_lattice_constant_a'%type_atom2])
                ly0 = -0.5*lx
                ly1 = (np.sqrt(3.)/2)*lx
                lz = float(constants['dO19_%s_lattice_constant_c'%type_atom2]) 
                os.system('rm POSCAR')
                os.system('rm dO19_%s.lmp'%type_atom2)
                with open('dO19_%s.POSCAR'%type_atom2, 'w') as file_poscar:
                    file_poscar.write(f'dO19 - {type_atom2} rich POSCAR file\n'
                                    f'1.0\n'
                                    f'{lx:.16f} 0 0\n'
                                    f'{ly0:.16f} {ly1:.16f} 0\n'
                                    f'0 0 {lz:.16f}\n'
                                    f'{type_atom1} {type_atom2}\n'
                                    f'2 6\n'
                                    f'Direct\n'
                                    f'0.666667 0.333333 0.750000\n'
                                    f'0.333333 0.666667 0.250000\n'                                
                                    f'0.680766 0.840383 0.750000\n'
                                    f'0.840383 0.680766 0.250000\n'
                                    f'0.840383 0.159617 0.250000\n'
                                    f'0.159617 0.840383 0.750000\n'
                                    f'0.159617 0.319234 0.750000\n'
                                    f'0.319234 0.159617 0.250000')
                os.system('cp dO19_%s.POSCAR POSCAR'%type_atom2)
                os.system('atomsk POSCAR dO19_%s.lmp >/dev/null 2>&1'%type_atom2)
##############################################################################

##############################################################################



lammps_executable = './lmp_serial'
style = 'meam'

unary_potential_1lib = 'Zr.library_27.44'
unary_potential_2lib = 'Ti.library_3.32'
unary_potential_1param = 'Zr.parameter_27.44'
unary_potential_2param = 'Ti.parameter_3.32'
outputlib = 'ZrTi.library'
outputparam = 'ZrTi.parameter'
binary_database = 'database.binary'

database_info = read_database(binary_database)
reference_info = database_info["reference_info"]
lattice_constants = database_info["lattice_constants"]
alpha, re = calculate_binary_parameter(reference_info['ref_struct'],reference_info['cohesieve_energy'],reference_info['bulk_modulus'],reference_info['lattice_constant_a'])
output = f"results_{style}_{sorted(list(reference_info['elements']))[0]}_{sorted(list(reference_info['elements']))[1]}.dat"
potential_file = ' '.join([outputlib] + sorted(list(reference_info['elements']))[:2] + [outputparam] + sorted(list(reference_info['elements']))[:2])
# potential_file = 'ZrTi.library  Zr Ti  ZrTi.parameter_20.51 Zr Ti'


# style = 'eam/alloy'
# potential_file = 'Farkas_Nb-Ti-Al_1996.eam.alloy Nb Ti'
pf = 'best_fit_files'
os.system('mkdir %s'%pf) 

cmin1 = (0.44099992, 0.44543207999999995)
cmin2 = (0.487465425, 0.4923645749999999)
cmin3 = (0.44235908999999995, 0.44680490999999994)
cmin4 = (0.13024749, 0.13155651)
cmax1 = (2.58939198, 2.6154160199999996)
cmax2 = (1.761835555, 1.7795424449999997)
cmax3 = (1.917059535, 1.9363264649999998)
cmax4 = (2.74934022, 2.77697178)
repulsb = (0.0, 0.0)
attracb = (0.0, 0.0)

alpha = 4.462762



bounds = [cmin1,cmin2,cmin3,cmin4,cmax1,cmax2,cmax3,cmax4,(0.995*alpha, 1.005*alpha),repulsb,attracb]


convergence_threshold = 0.005  # Initial convergence threshold

def callback(x, convergence):
    if convergence < convergence_threshold:
        return True
    return False


# result = differential_evolution(objective_function, bounds,**kwords, callback=callback)  
result = differential_evolution(objective_function, bounds, callback=callback)
optimal_params = result.x
optimal_error = result.fun
print("Optimal Parameters:", optimal_params)
print("Optimal Error:", optimal_error)

##############################################################################
# properties

# ref_struct = element_properties_database['ref_struct']
# gsfe(ref_struct,lattice_constants,elements,output,potential_file)



# if os.path.exists("%s"%output):
#     os.remove("%s"%output)
# elem1 =  sorted(list(elements))[0]
# elem2 =  sorted(list(elements))[1]
# # elastic constants            
# merged_dict = {}
# for structure, constants in lattice_constants.items()
#     data_file = f'{structure}.lmp'
#     a = elastic_constant(structure, data_file, output, potential_file)


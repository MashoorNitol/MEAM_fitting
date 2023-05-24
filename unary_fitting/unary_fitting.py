#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 08:49:31 2023

@author: mashroornitol
"""
import numpy as np
import os
from plot_function import*
import subprocess
import platform
import random
from scipy.optimize import differential_evolution
import glob
import re


ev_angstrom3 = 160.2176621 # 1 eV/Å3 = 160.2176621 GPa
# MEAM constants #
rozero = 1.0
ibar = 3
delr = 0.1
augt1 = 0
erose_form = 2
ialloy = 2
ielement = 1
zbl = 0
nn2 = 1
rho0 = 1.0
repuls = 0.0
attrac = 0.0
rc = 6.0
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
"""
MEAM library format:
# elt        lat     z       ielement     atwt
# alpha      b0      b1      b2           b3    alat    esub    asub
# t0         t1              t2           t3            rozero  ibar
"""

def write_library(element_properties_database, element, element_struc, first_near, ielement, atwt,
                       alpha, b0, b1, b2, b3, latparam, cohesieve_energy, asub, t0, t1, t2, t3, rozero, ibar):
    file_name = f"{element_properties_database.get('symbol')}.library"
    with open(file_name, "w") as f:
        f.write(f"'{element}'\t '{element_struc}' \t {first_near} \t {ielement} \t {atwt}\n")
        f.write(f"{alpha:.6f} \t {b0:.6f} \t {b1:.6f} \t {b2:.6f} \t {b3:.6f}\t {latparam:.6f} \t {cohesieve_energy:.6f} \t {asub:.6f}\n")
        f.write(f"{t0} \t {t1:.6f} \t {t2:.6f} \t {t3:.6f} \t {rozero:.6f}\t {ibar:d}")
    
    return file_name
        
def write_parameter(element_properties_database, rc, delr, augt1, erose_form, ialloy, zbl, nn2, rho0,
                         cohesieve_energy, re, alpha, repuls, attrac, Cmin, Cmax):
    iter_over = 1
    file_name = f"{element_properties_database.get('symbol')}.parameter"
    with open(file_name, "w") as f:
        f.write(f"rc = {rc:.6f}\n")
        f.write(f"delr = {delr:.6f}\n")
        f.write(f"augt1 = {augt1:d}\n")
        f.write(f"erose_form = {erose_form:d}\n")
        f.write(f"ialloy = {ialloy:d}\n\n")
        f.write(f"zbl({iter_over},{iter_over}) = {zbl:d}\n")
        f.write(f"nn2({iter_over},{iter_over}) = {nn2:d}\n")
        f.write(f"rho0({iter_over}) = {rho0:.6f}\n")
        f.write(f"Ec({iter_over},{iter_over}) = {cohesieve_energy:.6f}\n")
        f.write(f"re({iter_over},{iter_over}) = {re:.6f}\n")
        f.write(f"alpha({iter_over},{iter_over}) = {alpha:.6f}\n")
        f.write(f"repuls({iter_over},{iter_over}) = {repuls:.6f}\n")
        f.write(f"attrac({iter_over},{iter_over}) = {attrac:.6f}\n")
        f.write(f"Cmin({iter_over},{iter_over},{iter_over}) = {Cmin:.6f}\n")
        f.write(f"Cmax({iter_over},{iter_over},{iter_over}) = {Cmax:.6f}")
    
    return file_name


def find_file_with_lowest_float(file_list):
    lowest_float = float('inf')
    lowest_file = None

    for file_name in file_list:
        match = re.search(r'(\d+\.\d+)$', file_name)
        if match:
            float_num = float(match.group(1))
            if float_num < lowest_float:
                lowest_float = float_num
                lowest_file = file_name

    return lowest_file
        
def meam_prop_calc(element,element_properties_database):
    latparam = float(element_properties_database.get("lattice_constant_a"))
    if element_properties_database.get("ground_state") == 'hcp':
        latparam_z = float(element_properties_database.get("ca_ratio"))*latparam
    cohesieve_energy = float(element_properties_database.get("cohesieve_energy"))
    element_struc = element_properties_database.get("ground_state")
    if element_properties_database.get("ground_state") == 'hcp' or element_properties_database.get("ground_state") == 'fcc':
        first_near = int(12)
    elif element_properties_database.get("ground_state") == 'bcc':
        first_near = int(8)
    atwt = element_properties_database.get("atomic_weight")
    if element_properties_database.get("ground_state") == 'bcc':
        unit_vol = (latparam**3)/2
    elif element_properties_database.get("ground_state") == 'fcc':
        unit_vol = (latparam**3)/4
    elif element_properties_database.get("ground_state") == 'hcp':
        #unit_vol = (3*np.sqrt(2.)*(latparam**3))/6
        unit_vol = (latparam*((np.sqrt(3.)/2)*latparam)*latparam_z)/2
    bulk_modulus = float(element_properties_database.get("bulk_modulus"))
    shear_modulus = float(element_properties_database.get("shear_modulus"))
    alpha = np.sqrt(9*bulk_modulus*unit_vol/(cohesieve_energy*ev_angstrom3))
    if element_properties_database.get("ground_state") == 'fcc':
        re = (1/np.sqrt(2.))*latparam
    elif element_properties_database.get("ground_state") == 'bcc':
        re = (np.sqrt(3.)/2)*latparam
    elif element_properties_database.get("ground_state") == 'hcp':
        re = latparam
    return (latparam,cohesieve_energy,element_struc,first_near,atwt,alpha,re)

def extract_element_properties(filename, element):
    properties = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("# element:") and line.endswith(element):
            while i < len(lines) - 1:
                i += 1
                line = lines[i].strip()
                if line == "":
                    break
                if "=" in line:
                    key, value = line.split("=")
                    properties[key.strip()] = value.strip()

    return properties
################# LAMMPS functions #######################
# def lammps_run(in_file):
#     return('%s -in %s'%(lammps_executable,in_file))

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

def minimized_output():
    return('variable natoms equal count(all)\
           \nvariable teng equal "etotal"\
           \nvariable ecoh equal  v_teng/v_natoms\
           \nvariable length equal "lx"\
           \nvariable lengthZ equal "lz"\
           \nvariable perA_vol equal "vol/v_natoms"\
           \nvariable ca_ratio equal "v_lengthZ/v_length"\
           \nprint "---------------------------------------------------------------------------" append %s\
           \nprint "# minimized structure and energy" append %s\
           \nprint "Cohesive energy (eV) = ${ecoh}" append %s\
           \nprint "Lattice constant (Angstroms) = ${length}" append %s\
           \nprint "C/A ratio  = ${ca_ratio}" append %s\
           \nprint "Volume per atom (Angstrom^3/atom) = ${perA_vol}" append %s\
           \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output,output,output,output))

def cold_curve(cryst,latparam,type_atom,output,potential_file):
    properties='energy_volume_%s_%s'%(cryst,type_atom)
    os.system('rm ev.pdf')
    """
    energy-volume curve calculaiton  using LAMMPS for any given material
    """
    with open('ev.in', 'w') as f:
        f.write(lammps_header() + '\n\n')
        f.write(struct_mod(cryst,element_properties_database) + '\n\n')
        f.write(potential_mod(style,potential_file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write(minimization() + '\n\n')
        f.write(minimized_output() + '\n\n')
        f.write('\nvariable etot equal etotal\
                \nvariable lenx equal lx\
                \nvariable Volume equal vol\
                \nvariable peratom_vol equal v_Volume/v_natoms\
                \nvariable peratom_energy equal v_etot/v_natoms\
                \n\nchange_box all x scale 0.75 y scale 0.75 z scale 0.75 remap\
                \nfix 1 all deform 1 x scale 2.0 y scale 2.0 z scale 2.0\
                \nfix EV all print 10 "${peratom_vol} ${peratom_energy}" file energy_vol.dat\
                \nrun 500\
                \nunfix 1')
                
    lammps_run('ev.in')
    ev_data = np.genfromtxt('energy_vol.dat',skip_header=1)
    ax.plot(ev_data[:,0],ev_data[:,1],'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'EV curve of %s %s'%(cryst.upper(),type_atom))
    ax.set_xlabel(r'Volume per atom (\AA$^3$)',fontweight='bold')
    ax.set_ylabel(r'Energy per atom (eV/atom)',fontweight='bold')
    ax.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
    fig.savefig('ev.pdf')
    os.system('mkdir %s'%properties)
    os.system('mv dump_lammps ev.pdf log.lammps ev.in energy_vol.dat %s/'%properties)
                
def struct_mod(cryst,element_properties_database):
    if cryst=='fcc':
        return('variable a equal %.6f\nlattice fcc $a\
               \nregion box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0\
                    \ncreate_box 1 box\
                    \nlattice 	fcc $a  orient x 1 0 0 orient y 0 1 0 orient z 0 0 1\
                    \ncreate_atoms 1 box'%latparam)
    elif cryst=='bcc':
        return('variable a equal %.6f\nlattice bcc $a\
               \nregion box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0\
                    \ncreate_box 1 box\ncreate_atoms 1 box'%latparam)
    elif cryst=='hcp':
        if element_properties_database.get("symbol") == 'Ti' or element_properties_database.get("symbol") == 'Zr' or element_properties_database.get("symbol") == 'Hf':
            return('variable a equal %.6f\nlattice hcp $a\
                \nregion box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0\
                    \ncreate_box 1 box\ncreate_atoms 1 box\
                        \nchange_box all z scale 0.9712 remap\n'%latparam)
        else:
            return('variable a equal %.6f\nlattice hcp $a\
                   \nregion box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0\
                   \ncreate_box 1 box\ncreate_atoms 1 box'%latparam)
    # elif cryst=='hcp':
    #     return('variable a equal %.6f\nlattice hcp $a\
    #             \nregion box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0\
    #                 \ncreate_box 1 box\ncreate_atoms 1 box'%latparam)

def extract_constants(crystal,output):
    # Read the output.txt file
    with open("%s"%output, "r") as f:
        lines = f.readlines()

    if crystal == "hcp":
        # Extract C11, C12, C13, C33, C44 values for hcp crystal
        constants = {}
        for line in lines:
            if "c11 =" in line:
                constants["c11"] = float(line.split("=")[1].strip().split()[0])
            elif "c12 =" in line:
                constants["c12"] = float(line.split("=")[1].strip().split()[0])
            elif "c13 =" in line:
                constants["c13"] = float(line.split("=")[1].strip().split()[0])
            elif "c33 =" in line:
                constants["c33"] = float(line.split("=")[1].strip().split()[0])
            elif "c44 =" in line:
                constants["c44"] = float(line.split("=")[1].strip().split()[0])
        
        return constants

    elif crystal == "bcc" or crystal == "fcc":
        # Extract C11, C12, C44 values for FCC or BCC crystal
        constants = {}
        for line in lines:
            if "c11 =" in line:
                constants["c11"] = float(line.split("=")[1].strip().split()[0])
            elif "c12 =" in line:
                constants["c12"] = float(line.split("=")[1].strip().split()[0])
            elif "c44 =" in line:
                constants["c44"] = float(line.split("=")[1].strip().split()[0])
        
        return constants

    else:
        return None  # Return None if crystal type is not recognized

                    
def elastic_constant(cryst,latparam,type_atom,output,potential_file):
    properties='elastic_constant_%s_%s'%(cryst,type_atom)
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
        f.write('variable up equal 1.0e-4\nvariable atomjiggle equal 1.0e-6\n')
        f.write(lammps_header() + '\n\n')
        f.write('variable cfac equal 1.0e-4\nvariable cunits string GPa\n')
        f.write('variable etol equal 1.0e-8\nvariable ftol equal 1.0e-8\n\
                variable maxiter equal 1000\nvariable maxeval equal 1000\
                \nvariable dmax equal 1.0e-2\n')
        f.write(struct_mod(cryst,element_properties_database) + '\n\n')
        f.write('mass * 1.0e-20\n')
        f.write(potential_mod(style,potential_file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write('\nfix 3 all box/relax  aniso 0.0 vmax 0.001\n')
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
            \nprint "# Elastic constant and bulk modulus" append %s\
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
            \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output,output,output,output,output,output,output,output,output,output,output))
        f.write('\nrun 0\n')
        f.write(minimized_output() + '\n\n')
    
    lammps_run('elastic.in')

    
    constants = extract_constants(cryst, output)
    # if any(value < 0 for value in constants.values()):
    #     print("Found negative elastic constants. Rerunning...")
    #     lammps_run('elastic.in')
    #     constants = extract_constants(cryst, output)
    if os.path.exists(properties):
        os.system('rm -r %s' % properties)
        
    os.system('mkdir %s'%properties)
    os.system('mv elastic.in restart.equil displace.mod %s/'%properties)        
    return constants



def extract_vacancy_formation_energy(type_sim,output):
    if type_sim == 'vacformation':
        vacancy_energy_key = '# vacancy formation energy'
        with open(output, 'r') as file:
            for line in file:
                if line.startswith(vacancy_energy_key):
                    # Read the next line and extract the energy value
                    next_line = next(file)
                    energy_parts = next_line.split('=')
                    if len(energy_parts) > 1:
                        energy = energy_parts[1].strip()
                        return {'e_vac': energy}
        # If the line is not found or the file is empty
        return None
    
def vacancy_formation(cryst,latparam,type_atom,output,potential_file):
    properties='vacancy_formation_%s_%s'%(cryst,latparam)
    """
    vacancy formation calculaiton  using LAMMPS for any given material
    """
    with open('vac.in', 'w') as f:
        f.write(lammps_header() + '\n\n')
        f.write(struct_mod(cryst,element_properties_database) + '\n\n')
        f.write(potential_mod(style,potential_file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write(minimization() + '\n\n')
        f.write('\nreplicate 4 4 4\
                \nrun 0\
                \nvariable N equal count(all)\
                \nvariable No equal $N\
                \nvariable E equal "c_eatoms"\
                \nvariable Ei equal $E\
                \nvariable r2 equal sqrt(${a}^2+${a}^2)/4\
                \nregion select sphere 0 0 0 ${r2} units box\
                \ndelete_atoms region select compress yes\
                \nmin_style cg\
                \nminimize 1e-16 1e-16 5000 5000\
                \nvariable Ef equal "c_eatoms"\
                \nvariable Ev equal (${Ef}-((${No}-1)/${No})*${Ei})\
                \nprint "---------------------------------------------------------------------------" append %s\
                \nprint "# vacancy formation energy" append %s\
                \nprint "Vacancy formation energy (eV) = ${Ev}" append %s\
                \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
                
    lammps_run('vac.in')
    type_sim = 'vacformation' 
    vacinfo = extract_vacancy_formation_energy(type_sim, output)
    if os.path.exists(properties):
        os.system('rm -r %s' % properties)        
    os.system('mkdir %s'%properties)
    os.system('mv dump_lammps log.lammps vac.in %s/'%properties)
    return vacinfo
    
def extract_octainterstetial_formation_energy(cryst, defect, output):
    if defect == 'octahedral':
        octa_energy_key = '# %s Octahedral interstetial energy' % cryst.upper()
        with open(output, 'r') as file:
            for line in file:
                if line.startswith(octa_energy_key):
                    # Read the next line and extract the energy value
                    next_line = next(file)
                    energy_parts = next_line.split('=')
                    if len(energy_parts) > 1:
                        energy = energy_parts[1].strip()
                        return {'e_octa': energy}
        # If the line is not found or the file is empty
        return None

            
def extract_tetrainterstetial_formation_energy(cryst,defect,output):
    if defect == 'tetrahedral':
        tetra_energy_key = '# %s tetrahedral interstetial energy' % cryst.upper()
        with open(output, 'r') as file:
            for line in file:
                if line.startswith(tetra_energy_key):
                    # Read the next line and extract the energy value
                    next_line = next(file)
                    energy_parts = next_line.split('=')
                    if len(energy_parts) > 1:
                        energy = energy_parts[1].strip()
                        return {'e_tetra': energy}
        # If the line is not found or the file is empty
        return None     

def interstetial_octa_fcc(cryst,latparam,type_atom,output,defect,potential_file):
        # Remove POSCAR file
    if os.path.exists("POSCAR"):
        os.remove("POSCAR")
    
    # Remove all *.lmp files
    for filename in os.listdir("."):
        if filename.endswith(".lmp"):
            os.remove(filename)
    properties = 'interstetial_octa_%s_%s'%(cryst,type_atom)
    """
    find the interstetial position in certiasian coordinate
    and insert in lammps script acoording to replication
    """
    interstetial_prop = ['octahedral']
    replication = 6
    if cryst == 'fcc':
        for idx, defect in enumerate(interstetial_prop):
            if defect == 'octahedral':
                coord = [0.5,0.5,0.5]
                with open('inter.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write(struct_mod(cryst,element_properties_database) + '\n\n')
                    f.write(potential_mod(style,potential_file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            lammps_run('inter.in')
            os.system('atomsk bulk_simbox.data option -frac -wrap POSCAR >/dev/null 2>&1')
            N = 1
            with open('POSCAR', 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if "Direct" in line: 
                        lineno = int(i)
                        j = i-N if i>N else 0
                        for k in range(j,i):
                            natoms=int(lines[k])
            new_natoms=int(natoms)+1
            if platform.system() == 'Darwin':
                subprocess.call(["sed -i '' \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            else:
                subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new = [i/replication for i in coord]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new[0],coord_new[1],coord_new[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp >/dev/null 2>&1')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# FCC Octahedral interstetial energy" append %s\
                    \nprint "FCC Octahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            
            lammps_run('inter2.in')
            defect = 'octahedral' 
            interinfoocta = extract_octainterstetial_formation_energy(cryst,defect, output)
            if os.path.exists(properties):
                os.system('rm -r %s' % properties)        
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)
            return interinfoocta


def interstetial_tetra_fcc(cryst,latparam,type_atom,output,defect,potential_file):
        # Remove POSCAR file
    if os.path.exists("POSCAR"):
        os.remove("POSCAR")
    
    # Remove all *.lmp files
    for filename in os.listdir("."):
        if filename.endswith(".lmp"):
            os.remove(filename)
    properties = 'interstetial_tetra_%s_%s'%(cryst,type_atom)
    """
    find the interstetial position in certiasian coordinate
    and insert in lammps script acoording to replication
    """
    interstetial_prop = ['tetrahedral']
    replication = 6
    if cryst == 'fcc':
        for idx, defect in enumerate(interstetial_prop):
            if defect == 'tetrahedral':
                coord_tet = [3/4.0,3/4.0,0.0]
                with open('inter.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write(struct_mod(cryst,element_properties_database) + '\n\n')
                    f.write(potential_mod(style,potential_file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            lammps_run('inter.in')
            os.system('atomsk bulk_simbox.data option -frac -wrap POSCAR')
            N = 1
            with open('POSCAR', 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if "Direct" in line: 
                        lineno = int(i)
                        j = i-N if i>N else 0
                        for k in range(j,i):
                            natoms=int(lines[k])
            new_natoms=int(natoms)+1
            if platform.system() == 'Darwin':
                subprocess.call(["sed -i '' \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            else:
                subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new_tet = [i/replication for i in coord_tet]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new_tet[0],coord_new_tet[1],coord_new_tet[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp >/dev/null 2>&1')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# FCC tetrahedral interstetial energy" append %s\
                    \nprint "FCC tetrahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "--------------------------------------------------------------------------- append %s'%(output,output,output,output))
            lammps_run('inter2.in')
            defect = 'tetrahedral' 
            interinfotetra = extract_tetrainterstetial_formation_energy(cryst, defect, output)
            if os.path.exists(properties):
                os.system('rm -r %s' % properties)        
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)
            return interinfotetra
                
        
def interstetial_octa_bcc(cryst,latparam,type_atom,output,defect,potential_file):
        # Remove POSCAR file
    if os.path.exists("POSCAR"):
        os.remove("POSCAR")
    
    # Remove all *.lmp files
    for filename in os.listdir("."):
        if filename.endswith(".lmp"):
            os.remove(filename)
    properties = 'interstetial_octa_%s_%s'%(cryst,type_atom)
    """
    find the interstetial position in certiasian coordinate
    and insert in lammps script acoording to replication
    """
    interstetial_prop = ['octahedral']
    replication = 6
    if cryst == 'bcc':
        for idx, defect in enumerate(interstetial_prop):
            if defect == 'octahedral':
                coord = [0.5,0.5,0.0]
                with open('inter.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write(struct_mod(cryst,element_properties_database) + '\n\n')
                    f.write(potential_mod(style,potential_file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            lammps_run('inter.in')
            os.system('atomsk bulk_simbox.data option -frac -wrap POSCAR')
            N = 1
            with open('POSCAR', 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if "Direct" in line: 
                        lineno = int(i)
                        j = i-N if i>N else 0
                        for k in range(j,i):
                            natoms=int(lines[k])
            new_natoms=int(natoms)+1 
            if platform.system() == 'Darwin':
                subprocess.call(["sed -i '' \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            else:
                subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new = [i/replication for i in coord]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new[0],coord_new[1],coord_new[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp >/dev/null 2>&1')
            with open('inter3.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# BCC Octahedral interstetial energy" append %s\
                    \nprint "BCC Octahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            
            lammps_run('inter3.in')
            defect = 'octahedral' 
            interinfoocta = extract_octainterstetial_formation_energy(cryst,defect, output)
            if os.path.exists(properties):
                os.system('rm -r %s' % properties)        
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter3.in %s/'%properties)
            return interinfoocta                

def interstetial_tetra_bcc(cryst,latparam,type_atom,output,defect,potential_file):
        # Remove POSCAR file
    if os.path.exists("POSCAR"):
        os.remove("POSCAR")
    
    # Remove all *.lmp files
    for filename in os.listdir("."):
        if filename.endswith(".lmp"):
            os.remove(filename)
    properties = 'interstetial_tetra_%s_%s'%(cryst,type_atom)
    """
    find the interstetial position in certiasian coordinate
    and insert in lammps script acoording to replication
    """
    interstetial_prop = ['tetrahedral']
    replication = 6
    if cryst == 'bcc':
        for idx, defect in enumerate(interstetial_prop):
            if defect == 'tetrahedral':
                coord_tet = [1/4.0,1/4.0,1/4.0]
                with open('inter.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write(struct_mod(cryst,element_properties_database) + '\n\n')
                    f.write(potential_mod(style,potential_file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            lammps_run('inter.in')
            os.system('atomsk bulk_simbox.data option -frac -wrap POSCAR')
            N = 1
            with open('POSCAR', 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if "Direct" in line: 
                        lineno = int(i)
                        j = i-N if i>N else 0
                        for k in range(j,i):
                            natoms=int(lines[k])
            new_natoms=int(natoms)+1 
            if platform.system() == 'Darwin':
                subprocess.call(["sed -i '' \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            else:
                subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new_tet = [i/replication for i in coord_tet]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new_tet[0],coord_new_tet[1],coord_new_tet[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp >/dev/null 2>&1')
            with open('inter25.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# BCC tetrahedral interstetial energy" append %s\
                    \nprint "BCC tetrahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            lammps_run('inter25.in')
            defect = 'tetrahedral' 
            interinfotetra = extract_tetrainterstetial_formation_energy(cryst,defect, output)
            if os.path.exists(properties):
                os.system('rm -r %s' % properties)        
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter25.in %s/'%properties)
            return interinfotetra

        
def interstetial_octa_hcp(cryst,latparam,type_atom,output,defect,potential_file):
        # Remove POSCAR file
    if os.path.exists("POSCAR"):
        os.remove("POSCAR")
    
    # Remove all *.lmp files
    for filename in os.listdir("."):
        if filename.endswith(".lmp"):
            os.remove(filename)
    properties = 'interstetial_octa_%s_%s'%(cryst,type_atom)
    """
    find the interstetial position in certiasian coordinate
    and insert in lammps script acoording to replication
    """
    interstetial_prop = ['octahedral']
    replication = 10
    if cryst == 'hcp':
        lz = np.sqrt(8/3.0)*latparam
        os.system('atomsk --create hcp %.16f %.16f %s -duplicate 1 1 1 unitcell.lmp'%(latparam,lz,type_atom))
        for idx, defect in enumerate(interstetial_prop):
            if defect == 'octahedral':
                coord = [0.5,0.5,0.25]
                with open('inter.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data unitcell.lmp'+ '\n\n')
                    #f.write(struct_mod(cryst,element_properties_database) + '\n\n')
                    f.write(potential_mod(style,potential_file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            lammps_run('inter.in')
            os.system('atomsk bulk_simbox.data option -frac -wrap POSCAR >/dev/null 2>&1')
            N = 1
            with open('POSCAR', 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if "Direct" in line: 
                        lineno = int(i)
                        j = i-N if i>N else 0
                        for k in range(j,i):
                            natoms=int(lines[k])
            new_natoms=int(natoms)+1 
            if platform.system() == 'Darwin':
                subprocess.call(["sed -i '' \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            else:
                subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new = [i/replication for i in coord]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new[0],coord_new[1],coord_new[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp >/dev/null 2>&1')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# HCP Octahedral interstetial energy" append %s\
                    \nprint "HCP Octahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            lammps_run('inter2.in')
            defect = 'octahedral' 
            interinfoocta = extract_octainterstetial_formation_energy(cryst,defect, output)
            if os.path.exists(properties):
                os.system('rm -r %s' % properties)        
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)
            return interinfoocta

def interstetial_tetra_hcp(cryst,latparam,type_atom,output,defect,potential_file):
        # Remove POSCAR file
    if os.path.exists("POSCAR"):
        os.remove("POSCAR")
    
    # Remove all *.lmp files
    for filename in os.listdir("."):
        if filename.endswith(".lmp"):
            os.remove(filename)
    properties = 'interstetial_tetra_%s_%s'%(cryst,type_atom)
    """
    find the interstetial position in certiasian coordinate
    and insert in lammps script acoording to replication
    """
    interstetial_prop = ['tetrahedral']
    replication = 10
    if cryst == 'hcp':
        lz = np.sqrt(8/3.0)*latparam
        os.system('atomsk --create hcp %.16f %.16f %s -duplicate 1 1 1 unitcell.lmp >/dev/null 2>&1'%(latparam,lz,type_atom))
        for idx, defect in enumerate(interstetial_prop):
            if defect == 'tetrahedral':
                coord_tet = [0,0,5/8.]
                with open('inter.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data unitcell.lmp'+ '\n\n')
                    #f.write(struct_mod(cryst,element_properties_database) + '\n\n')
                    f.write(potential_mod(style,potential_file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            lammps_run('inter.in')
            os.system('atomsk bulk_simbox.data option -frac -wrap POSCAR >/dev/null 2>&1')
            N = 1
            with open('POSCAR', 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if "Direct" in line: 
                        lineno = int(i)
                        j = i-N if i>N else 0
                        for k in range(j,i):
                            natoms=int(lines[k])
            new_natoms=int(natoms)+1 
            if platform.system() == 'Darwin':
                subprocess.call(["sed -i '' \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            else:
                subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new_tet = [i/replication for i in coord_tet]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new_tet[0],coord_new_tet[1],coord_new_tet[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp >/dev/null 2>&1')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# HCP tetrahedral interstetial energy" append %s\
                    \nprint "HCP tetrahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            lammps_run('inter2.in')
            defect = 'tetrahedral' 
            interinfotetra = extract_tetrainterstetial_formation_energy(cryst,defect, output)
            if os.path.exists(properties):
                os.system('rm -r %s' % properties)        
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)
            return interinfotetra






def extract_surface_formation_energy(cryst, surface, output):
    surface_energy_key = '# Free surface energy'
    target_line = 'Surface energy %s =' % surface
    with open(output, 'r') as file:
        for line in file:
            if line.startswith(target_line):
                energy_parts = line.split('=')
                if len(energy_parts) > 1:
                    energy_with_unit = energy_parts[1].strip()
                    if 'j/m^2' in energy_with_unit:
                        energy = energy_with_unit.split(' ')[0]
                        return {'fs_%s' % surface: energy}
    # If the line is not found or the file is empty
    return None
    
    

def freesurfaceenergy(cryst,latparam,type_atom,output,surface,potential_file):
        # Remove POSCAR file
    if os.path.exists("POSCAR"):
        os.remove("POSCAR")
    
    # Remove all *.lmp files
    for filename in os.listdir("."):
        if filename.endswith(".lmp"):
            os.remove(filename)
    if cryst == 'fcc':      
        properties = 'surface_%s_%s_%s'%(cryst,surface,type_atom)
        if surface == '100':
            os.system('atomsk --create %s %.16f %s orient [100] [010] [001] -orthogonal-cell -duplicate 2 2 8 fcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "-----------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '110':
            os.system('atomsk --create %s %.16f %s orient [112] [11-1] [1-10] -orthogonal-cell -duplicate 2 2 8 fcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)                    
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '111':
            os.system('atomsk --create %s %.16f %s orient [112] [1-10] [11-1] -orthogonal-cell -duplicate 2 2 8 fcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))                
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output) 
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '332':
            os.system('atomsk --create %s %.16f %s orient [-2-26] [1-10] [332] -orthogonal-cell -duplicate 2 2 4  fcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))  
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
                
    if cryst == 'bcc':
        properties = 'surface_%s_%s_%s'%(cryst,surface,type_atom)
        if surface == '100':
            os.system('atomsk --create %s %.16f %s orient [100] [010] [001] -orthogonal-cell -duplicate 2 2 8 bcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '110':
            os.system('atomsk --create %s %.16f %s orient [111] [-1-12] [1-10] -orthogonal-cell -duplicate 2 2 8 bcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '111':
            os.system('atomsk --create %s %.16f %s orient [-1-12] [1-10] [111] -orthogonal-cell -duplicate 2 2 8 bcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))                
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '112':
            os.system('atomsk --create %s %.16f %s orient [111] [1-10] [-1-12] -orthogonal-cell -duplicate 2 2 8  bcc_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,surface))  
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)                
            return surface_info
        
    if cryst == 'hcp':
        lz_latparam = np.sqrt(8/3.)*latparam
        if type_atom == 'Zn' or type_atom == 'Cd':
            lz_latparam = np.sqrt(8/3.)*latparam*1.1365632406513948 
        # surfaces = ['0001','10m10','1011','1122']        
        properties = 'surface_%s_%s_%s'%(cryst,surface,type_atom)
        if surface == '0001':
            os.system('atomsk --create %s %.16f %.16f %s orient [1-210] [1-100] [0001] -orthogonal-cell -duplicate 2 2 8 hcp_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,surface))
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '1010':
            os.system('atomsk --create %s %.16f %.16f %s orient [1-210] [0001] [10-10] -orthogonal-cell -duplicate 2 2 8 hcp_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,surface))
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '1101':
            os.system('atomsk --create %s %.16f %.16f %s orient [-12-10] [10-12] [-1011] -orthogonal-cell -duplicate 2 1 1 hcp_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,surface))                
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')    
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info
        elif surface == '1122':
            os.system('atomsk --create %s %.16f %.16f %s orient [-1100] [11-23] [11-22] -orthogonal-cell -duplicate 1 1 10  hcp_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,surface))  
            with open('surf.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nchange_box all z scale 2.0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable Ef equal "c_eatoms" \nvariable LX equal "lx"\
                        \nvariable LY equal "ly" \nvariable Esurf equal (${Ef}-${Ei})/(2*${LX}*${LY})\
                        \nvariable Esurf_erg_cm2 equal ${Esurf}*16021.765650000003\
                        \nvariable Esurf_j_m2 equal ${Esurf}*16.021765650000003')
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
                f.write('\nprint "# Free surface energy" append %s'%output)
                f.write('\nprint "Surface energy %s = ${Esurf} eV/angstrom^2" append %s\
\nprint "Surface energy %s = ${Esurf_j_m2} j/m^2" append %s \
\nprint "Surface energy %s = ${Esurf_erg_cm2} erg/cm^2 or mJ/m^2" append %s'%(surface,output,surface,output,surface,output))
                f.write('\nprint "---------------------------------------------------------------------------" append %s'%output)
            lammps_run('surf.in')
            surface_info = extract_surface_formation_energy(cryst,surface,output)
            os.system('mkdir %s'%properties)
            os.system('mv surf.in *.lmp  dump_lammps log.lammps  %s/'%properties)
            return surface_info 
            
def extract_energy_difference(cryst, output):
    energy_diff = {}
    energy_key_map = {
        'fcc': {
            'deltaE_bcc': 'The energy difference between fcc and bcc',
            'deltaE_hcp': 'The energy difference between fcc and hcp'
        },
        'bcc': {
            'deltaE_fcc': 'The energy difference between bcc and fcc',
            'deltaE_hcp': 'The energy difference between bcc and hcp'
        },
        'hcp': {
            'deltaE_fcc': 'The energy difference between hcp and fcc',
            'deltaE_bcc': 'The energy difference between hcp and bcc'
        }
    }

    if cryst in energy_key_map:
        key_map = energy_key_map[cryst]
        with open(output, 'r') as file:
            for line in file:
                for key, energy_key in key_map.items():
                    if energy_key in line:
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_string = energy_parts[1].split(' ')
                            energy = energy_string[1]
                            energy_diff[key] = energy
                           

    return energy_diff



def phase_energy_difference(cryst,latparam,type_atom,output,potential_file):
    properties='phase_energy_differences_%s_%s'%(cryst,type_atom)
    formation_out = 'formation_output.txt'
    if os.path.exists("%s"%formation_out):
        os.remove("%s"%formation_out)
    file_patterns = ["*.POSCAR", "*.lmp", "POSCAR"]
    files_to_remove = []
    for pattern in file_patterns:
        files = [file for file in os.listdir() if file.endswith(pattern[1:])]
        files_to_remove.extend(files)
    if files_to_remove:
        os.system("rm " + " ".join(files_to_remove))
    # os.system('rm *.POSCAR *.lmp POSCAR')
    with open('fcc.POSCAR', 'w') as file_fcc:
        lx = latparam
        file_fcc.write('FCC POSCAR file\n1.0\n%.16f 0 0\n0 %.16f 0\n0 0 %.16f\
                       \n%s\n4\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.0\n0.5 0.0 0.5\n0.0 0.5 0.5'\
                           %(lx,lx,lx,type_atom))
    os.system('cp fcc.POSCAR POSCAR')
    os.system('atomsk POSCAR fcc.lmp >/dev/null 2>&1')
    file_fcc.close()
    with open('bcc.POSCAR', 'w') as file_bcc:
        lx = latparam
        file_bcc.write('BCC POSCAR file\n1.0\n%.16f 0 0\n0 %.16f 0\n0 0 %.16f\
                        \n%s\n2\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.5'\
                            %(lx,lx,lx,type_atom))        
    os.system('cp bcc.POSCAR POSCAR')
    os.system('atomsk POSCAR bcc.lmp >/dev/null 2>&1') 
    file_bcc.close()
    with open('hcp.POSCAR', 'w') as file_hcp:
        lx = latparam
        xy =-0.5*lx
        ly = (np.sqrt(3.)/2.)*lx
        lz = (np.sqrt(8./3.))*lx
        file_hcp.write('HCP POSCAR file\n1.0\n%.16f 0 0\n%.16f %.16f 0\n0 0 %.16f\
                        \n%s\n2\nDirect\n0.0 0.0 0.0\n0.66666667  0.33333333  0.50'\
                            %(lx,xy,ly,lz,type_atom))
    os.system('cp hcp.POSCAR POSCAR')
    os.system('atomsk POSCAR hcp.lmp >/dev/null 2>&1')
    file_hcp.close()        
    with open('sc.POSCAR', 'w') as file_sc:
        lx=latparam
        file_sc.write('simple cubic POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n1\nDirect\n0.0 0.0 0.0'%(lx,lx,lx,type_atom))
    os.system('cp sc.POSCAR POSCAR')
    os.system('atomsk POSCAR sc.lmp >/dev/null 2>&1')  
    file_sc.close()         
    with open('dc.POSCAR', 'w') as file_dc: #damond cubic
        lx=latparam
        file_dc.write('diamond cubic POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n8\nDirect\n0.25 0.25 0.25\n0.50 0.00 0.50\
                        \n0.25 0.75 0.75\n0.50 0.50 0.00\n0.75 0.25 0.75\n0.00 0.00 0.00\
                        \n0.75 0.75 0.25\n0.00 0.50 0.50'%(lx,lx,lx,type_atom))
    os.system('cp dc.POSCAR POSCAR')
    os.system('atomsk POSCAR dc.lmp >/dev/null 2>&1')  
    file_dc.close()                       
    with open('a15.POSCAR', 'w') as file_a15:
        lx=latparam
        file_a15.write('a15 Sn POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n8\nDirect\n0.25 0.0 0.5\n0.75 0.0 0.5\
                        \n0.0 0.5 0.25\n0.0 0.5 0.75\n0.5 0.75 0.0\n0.5 0.25 0.0\
                        \n0.0 0.0 0.0\n0.5 0.5 0.5'%(lx,lx,lx,type_atom))
    os.system('cp a15.POSCAR POSCAR')
    os.system('atomsk POSCAR a15.lmp >/dev/null 2>&1')  
    file_a15.close()
    with open('beta_sn.POSCAR', 'w') as file_betasn: ## beta-Sn
        lx=latparam
        ly=lx
        lz=lx*0.55
        file_betasn.write('beta Sn POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n4\nDirect\n0.0 0.0 0.0\n0.5 0.0 0.25\
                        \n0.5 0.5 0.5\n0.0 0.5 0.75'%(lx,ly,lz,type_atom))
    os.system('cp beta_sn.POSCAR POSCAR')
    os.system('atomsk POSCAR beta_sn.lmp >/dev/null 2>&1')  
    file_betasn.close()
    with open('omega.POSCAR', 'w') as file_omega: #C6
        lx=latparam*2.0
        xy=-0.5*lx 
        ly=(np.sqrt(3.)/2.)*lx
        lz=lx*0.62
        file_omega.write('omega POSCAR file\n1.0\n%.16f 0 0\n%.16f %.16f 0\n0 0 %.16f\
                        \n%s\n3\nDirect\n0.0 0.0 0.0\n0.666667 0.333333 0.5\
                        \n0.333333 0.666667 0.5'%(lx,xy,ly,lz,type_atom))
    os.system('cp omega.POSCAR POSCAR')
    os.system('atomsk POSCAR omega.lmp >/dev/null 2>&1')
    file_omega.close()    
    with open('a7.POSCAR', 'w') as file_a7: # alpha-As # R3m
        lx=latparam
        xy=-0.5*lx
        ly=(np.sqrt(3.)/2.)*lx
        lz=3.0*lx
        file_a7.write('a7 POSCAR file\n1.0\n%.16f 0 0\n%.16f %.16f 0\n0 0 %.16f\
                        \n%s\n6\nDirect\n0.0 0.0 0.227320\n0.666667 0.333333 0.106013\
                        \n0.666667 0.333333 0.560653\n0.333333 0.666667 0.439347\
                        \n0.333333 0.666667 0.893987\n0.0 0.0 0.772680'%(lx,xy,ly,lz,type_atom))
    os.system('cp a7.POSCAR POSCAR')
    os.system('atomsk POSCAR a7.lmp >/dev/null 2>&1')
    file_a7.close()  
    with open('a11.POSCAR', 'w') as file_a11: # alpha-Ga #Cmce
        lx=latparam
        ly=1.7108347466942362*lx
        lz=1.0137728745950843*lx
        file_a11.write('a11 POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n8\nDirect\n0.5 0.155818 0.581037\n0.5 0.844182 0.418963\
                        \n0.5 0.344182 0.081037\n0.5 0.655818 0.918963\
                        \n0.0 0.655818 0.581037\n0.0 0.344182 0.418963\
                        \n0.0 0.844182 0.081037\n0.0 0.155818 0.918963'%(lx,ly,lz,type_atom))
    os.system('cp a11.POSCAR POSCAR')
    os.system('atomsk POSCAR a11.lmp >/dev/null 2>&1')
    file_a11.close()
    with open('a20.POSCAR', 'w') as file_a20: # alpha-U # Cmcm
        lx=latparam
        ly=2.0641907721492485*lx
        lz=1.7480584826470433*lx
        file_a20.write('a20 POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n4\nDirect\n0.0 0.099017 0.75\n0.5 0.400983 0.25\
                        \n0.5 0.599017 0.75\n0.0 0.900983 0.25'%(lx,ly,lz,type_atom))
    os.system('cp a20.POSCAR POSCAR')
    os.system('atomsk POSCAR a20.lmp >/dev/null 2>&1')
    file_a20.close()
    
    files = os.listdir('./')
    lmp_files = [f for f in files if f.endswith('.lmp')]
    for lmp in lmp_files:
        with open('formation.in', 'w') as f:
            f.write(lammps_header() + '\n\n')
            f.write('\nread_data %s'%lmp+ '\n\n')
            f.write(potential_mod(style,potential_file) + '\n\n')
            f.write(output_mod() + '\n\n')
            # f.write(relaxation() + '\n\n')
            f.write(minimization() + '\n\n')
            f.write(minimization() + '\n\n')
            f.write('variable natoms equal count(all)\
                       \nvariable teng equal "etotal"\
                       \nvariable ecoh equal  v_teng/v_natoms')
            f.write('\nprint "Per Atom energy for structure : %s =  ${ecoh} eV/atom" append %s'%(lmp,formation_out))
        lammps_run('formation.in')

    with open('formation_output.txt', 'r') as f:
        lines = f.readlines()
    energy_dict = {}
    for line in lines:
        if 'Per Atom energy for structure :' in line:
            lmp_name, energy = line.split('=')[0].split(':')[-1].strip(), float(line.split('=')[-1].split()[0])
            energy_dict[lmp_name] = energy
    
  
    type_structure = '%s.lmp'%cryst
    filename = output
    if os.path.isfile(filename):
        with open(filename, 'a') as file_check:
            file_check.write("---------------------------------------------------------------------------\n")
    else:
        with open(filename, 'w') as file_check:
            file_check.write("---------------------------------------------------------------------------\n")
    if type_structure in energy_dict:
        energy_type_structure = energy_dict[type_structure]
        for lmp_name, energy in energy_dict.items():
            if lmp_name != type_structure:
                energy_diff = energy - energy_type_structure
                filename = output
                if os.path.isfile(filename):
                    with open(filename, 'a') as file_check:
                        file_check.write("The energy difference between %s and %s  = %.16f eV/atom\n"% (type_structure.split('.')[0], lmp_name.split('.')[0], energy_diff))
                else:
                    with open(filename, 'w') as file_check:
                        file_check.write("The energy difference between %s and %s  = %.16f eV/atom\n"% (type_structure.split('.')[0], lmp_name.split('.')[0], energy_diff))                    
    if os.path.isfile(filename):
        with open(filename, 'a') as file_check:
            file_check.write("---------------------------------------------------------------------------\n")
    else:
        with open(filename, 'w') as file_check:
            file_check.write("---------------------------------------------------------------------------\n")
            
    energy_diff = extract_energy_difference(cryst, output)
    os.system('mkdir %s'%properties)
    os.system('mv dump_lammps log.lammps formation.in formation_output.txt *.POSCAR *.lmp POSCAR %s/'%properties)
    return energy_diff



def extract_gsfe(cryst,type_atom, output,plane,direction):
    result_dict = {}
    if cryst == 'fcc':
        if plane == '111' and direction == '110':
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
        if plane == '111' and direction == '110':
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction) : energy})
        if plane == '111' and direction == '112':
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction) : energy})
        if plane == '111' and direction == '112':
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction) : energy})
        if plane == '111' and direction == '112':
            target_line = 'twin boundary for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_tb' % (plane, direction) : energy})
   
    if cryst == 'bcc':
        if plane == '110' and direction == '111':
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
        if plane == '112' and direction == '111':
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction) : energy})
        if plane == '123' and direction == '112':
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,plane,direction)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction) : energy})
    
    if cryst == 'hcp':
        if plane == '0001' and direction == '1120':
            pl = '0001'
            dt = '1-210'
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl,dt)
            # print(target_line)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl,dt)
            print(target_line)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction): energy})
        if plane == '0001' and direction == '1010':
            pl2 = '0001'
            dt2 = '10-10'
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl2,dt2)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl2,dt2)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction): energy})
            target_line = 'twin energy %s %s %s plane in %s direction' % (type_atom,cryst,pl2,dt2)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_tb' % (plane, direction): energy})
        if plane == '1010' and direction == '1120':
            pl3 = '10-10'
            dt3 = '1-210'
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl3,dt3)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl3,dt3)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction): energy})  
        if plane == '1101' and direction == '1120':
            pl4 = '10-11'
            dt4 = '1-210'
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl,dt4)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl4,dt4)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane4, direction4): energy}) 
        if plane == '1101' and direction == '1123':
            pl5 = '10-11'
            dt5 = '-1-123'
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl5,dt5)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl5,dt5)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction): energy})    
            target_line = 'twin energy %s %s %s plane in %s direction' % (type_atom,cryst,pl5,dt5)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_tb' % (plane, direction): energy})
        if plane == '1122' and direction == '1123':
            pl6 = '11-22'
            dt6 = '11-23'
            target_line = 'unstable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl6,dt6)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_us' % (plane, direction): energy})
            target_line = 'stable stacking fault for %s %s %s plane in %s direction' % (type_atom,cryst,pl6,dt6)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_ss' % (plane, direction): energy})    
            target_line = 'twin energy %s %s %s plane in %s direction' % (type_atom,cryst,pl6,dt6)
            with open(output, 'r') as file:
                for line in file:
                    if line.startswith(target_line):
                        energy_parts = line.split('=')
                        if len(energy_parts) > 1:
                            energy_with_unit = energy_parts[1].strip()
                            if 'j/m^2' in energy_with_unit:
                                energy = energy_with_unit.split(' ')[0]
                                result_dict.update({'sf_%s_%s_tb' % (plane, direction): energy})                                 
    return result_dict 



  
def gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file):
    # os.system('rm -rf *.lmp *.in  gsfe*')
    if cryst == 'fcc':      
        if plane == '111' and direction == '112':
            gsfe_output = 'gsfe_output.txt'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %s orient  [112] [1-10] [11-1] -orthogonal-cell -duplicate 2 2 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,direction))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%direction+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data = np.genfromtxt(gsfe_output)          
            x_data = [i for i in np.linspace(0,1,len(data))]
            y_data = [(data[i][1])/1000 - (data[0][1])/1000 for i in range(len(data))]
            y_subset = [y for x, y in zip(x_data, y_data) if 0 <= x <= 0.3]
            y_usf = max(y_subset)
            y_subset = [y for x, y in zip(x_data, y_data) if 0.3 <= x <= 0.6]
            y_ssf = min(y_subset)
            y_subset = [y for x, y in zip(x_data, y_data) if 0.6 <= x < 1.0]
            y_twin = max(y_subset)
            fig0, ax0 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)                
            ax0.plot(x_data,y_data,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax0.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax0.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax0.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig0.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                    \ntwin boundary for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usf,type_atom,cryst,plane,direction,y_ssf,type_atom,cryst,plane,direction,y_twin)                   
            
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)                         

            
        if plane == '111' and direction == '110':
            gsfe_output = 'gsfe_output.txt'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)             
            os.system('atomsk --create %s %.16f %s orient  [1-10] [112]  [11-1] -orthogonal-cell -duplicate 2 2 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,direction))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%direction+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data2 = np.genfromtxt(gsfe_output)         
            x_data2 = [i2 for i2 in np.linspace(0,1,len(data2))]
            y_data2 = [(data2[i][1])/1000 - (data2[0][1])/1000 for i in range(len(data2))]
            y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0 <= x2 <= 0.45]
            y_usf2 = max(y_subset2)
            y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0.45 <= x2 <= 0.6]
            y_ssf2 = min(y_subset2)
            fig2, ax2 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            direction = '110'
            ax2.plot(x_data2,y_data2,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax2.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax2.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax2.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig2.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))  
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usf2,type_atom,cryst,plane,direction,y_ssf2)                                   
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)
                    
            
    if cryst == 'bcc':        
        if plane == '110' and direction == '111':
            gsfe_output = 'gsfe_output.txt'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %s orient [111] [-1-12] [1-10] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')

                f.write('\n\nvariable xmin equal xlo\nvariable xmax equal xhi\nvariable xtot equal v_xmin+v_xmax\n')
                f.write('\nvariable ymin equal ylo\nvariable ymax equal yhi\nvariable ytot equal v_ymin+v_ymax')
                f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                        \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                        \ngroup lower region lower')
                f.write('\nfix 1 lower move linear  ${xtot} 0.0 0.0\nfix 2 all setforce 0.0 0.0 NULL\n')


                # f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                #         \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                #         \ngroup lower region lower')
                # f.write('\nfix 1 lower move linear 20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            dataB = np.genfromtxt(gsfe_output)     
            x_dataB = [iB for iB in np.linspace(0,1,len(dataB))]
            y_dataB = [(dataB[iB][1])/1000 - (dataB[0][1])/1000 for iB in range(len(dataB))]
            y_subsetB = [yB for xB, yB in zip(x_dataB, y_dataB) if 0.3 <= xB <= 0.6]
            y_usfB = max(y_subsetB)           
            direction = '111'
            fig2b, ax2b = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            ax2b.plot(x_dataB,y_dataB,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax2b.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax2b.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax2b.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig2b.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usfB)                   
            
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)                         

        if plane == '112' and direction == '111':
            gsfe_output = 'gsfe_output.txt'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %s orient [111]  [1-10] [-1-12] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data2 = np.genfromtxt(gsfe_output)    
            x_data2 = [i2 for i2 in np.linspace(0,1,len(data2))]
            y_data2 = [(data2[i][1])/1000 - (data2[0][1])/1000 for i in range(len(data2))]
            y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0.3 <= x2 <= 0.6]
            y_usf2 = max(y_subset2)
            fig2, ax2 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            direction = '111'
            ax2.plot(x_data2,y_data2,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax2.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax2.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax2.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig2.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))  
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usf2)                   
            
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)            

        if plane == '123' and direction == '111':
            gsfe_output = 'gsfe_output.txt'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %s orient [11-1] [-54-1] [123] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data3 = np.genfromtxt(gsfe_output)    
            x_data3 = [i3 for i3 in np.linspace(0,1,len(data3))]
            y_data3 = [(data3[i][1])/1000 - (data3[0][1])/1000 for i in range(len(data3))]
            y_subset3 = [y3 for x3, y3 in zip(x_data3, y_data3) if 0.3 <= x3 <= 0.6]
            y_usf3 = max(y_subset3)
            fig3, ax3 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            direction = '111'
            ax3.plot(x_data3,y_data3,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax3.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax3.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax3.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig3.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))  
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usf3)                   
            
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)
                    

    if cryst == 'hcp':
        lz_latparam = np.sqrt(8/3.)*latparam
        if type_atom == 'Zn' or type_atom == 'Cd':
            lz_latparam = np.sqrt(8/3.)*latparam*1.1365632406513948        
        if plane == '0001' and direction == '1120' :
            gsfe_output = 'gsfe_output.txt'
            plane = '0001'
            direction = '1-210'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %.16f %s orient [1-210] [10-10] [0001] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            datahcp = np.genfromtxt(gsfe_output)    
            x_datahcp = [ihcp for ihcp in np.linspace(0,1,len(datahcp))]
            y_datahcp = [(datahcp[ihcp][1])/1000 - (datahcp[0][1])/1000 for ihcp in range(len(datahcp))]
            y_subsethcp = [yhcp for xhcp, yhcp in zip(x_datahcp, y_datahcp) if 0.2 <= xhcp <= 0.3]
            y_usfhcp = max(y_subsethcp)
            y_subsethcp = [yhcp for xhcp, yhcp in zip(x_datahcp, y_datahcp) if 0.35 <= xhcp <= 0.55]
            y_ssfhcp = min(y_subsethcp) 
            fighcp, axhcp = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            axhcp.plot(x_datahcp,y_datahcp,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            axhcp.set_xlabel(r'Normalized displacement',fontweight='bold')
            axhcp.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            axhcp.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fighcp.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usfhcp, type_atom,cryst,plane,direction,y_ssfhcp)                   
                            
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)
                            
        if plane == '0001' and direction == '1010':
            gsfe_output = 'gsfe_output.txt'
            plane = '0001'
            direction = '10-10'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %.16f %s orient  [10-10] [1-210] [0001] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\n\nvariable xmin equal xlo\nvariable xmax equal xhi\nvariable xtot equal v_xmin+v_xmax\n')
                f.write('\nvariable ymin equal ylo\nvariable ymax equal yhi\nvariable ytot equal v_ymin+v_ymax')
                f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                        \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                        \ngroup lower region lower')
                f.write('\nfix 1 lower move linear -${xtot} 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data2 = np.genfromtxt(gsfe_output)     
            x_data2 = [i2 for i2 in np.linspace(0,1,len(data2))]
            y_data2 = [(data2[i2][1])/1000 - (data2[0][1])/1000 for i2 in range(len(data2))]
            y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0.0 <= x2 <= 0.25]
            y_usf2 = max(y_subset2)
            y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0.25 <= x2 <= 0.5]
            y_ssf2 = min(y_subset2)
            y_subset2 = [y2 for x2, y2 in zip(x_data2, y_data2) if 0.5 <= x2 <= 0.8]
            y_twin2 = max(y_subset2)
            fig2, ax2 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            ax2.plot(x_data2,y_data2,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax2.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax2.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax2.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig2.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
            \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
            \ntwin energy %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usf2,type_atom,cryst,plane,direction,y_ssf2,type_atom,cryst,plane,direction,y_twin2)                   
                            
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)
                    
        if plane == '1010' and direction == '1120':
            gsfe_output = 'gsfe_output.txt'
            plane = '10-10'
            direction = '1-210'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %.16f %s orient  [1-210] [0001]  [10-10] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\n\nvariable xmin equal xlo\nvariable xmax equal xhi\nvariable xtot equal v_xmin+v_xmax\n')
                f.write('\nvariable ymin equal ylo\nvariable ymax equal yhi\nvariable ytot equal v_ymin+v_ymax')
                f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                        \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                        \ngroup lower region lower')
                f.write('\nfix 1 lower move linear -${xtot} 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data3 = np.genfromtxt(gsfe_output)     
            x_data3 = [i3 for i3 in np.linspace(0,1,len(data3))]
            y_data3 = [(data3[i3][1]) - (data3[0][1])/1000 for i3 in range(len(data3))]
            y_subset3 = [y3 for x3, y3 in zip(x_data3, y_data3) if 0.0 <= x3 <= 0.45]
            y_usf3 = max(y_subset3)
            y_subset3 = [y3 for x3, y3 in zip(x_data3, y_data3) if 0.45 <= x3 <= 0.65]
            y_ssf3 = min(y_subset3)
            fig3, ax3 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            ax3.plot(x_data3,y_data3,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax3.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax3.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax3.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig3.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
            \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usf3,type_atom,cryst,plane,direction,y_ssf3)                   
                            
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)  
                       
        if plane == '1101' and direction == '1120':
            gsfe_output = 'gsfe_output.txt'
            plane = '-1011'
            direction = '-12-10'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %.16f %s orient  [-12-10] [10-12] [-1011] -orthogonal-cell -duplicate 2 1 1 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\n\nvariable xmin equal xlo\nvariable xmax equal xhi\nvariable xtot equal v_xmin+v_xmax\n')
                f.write('\nvariable ymin equal ylo\nvariable ymax equal yhi\nvariable ytot equal v_ymin+v_ymax')
                f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                        \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                        \ngroup lower region lower')
                f.write('\nfix 1 lower move linear -${xtot} 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data4 = np.genfromtxt(gsfe_output)     
            x_data4 = [i4 for i4 in np.linspace(0,1,len(data4))]
            y_data4 = [(data4[i4][1])/1000 - (data4[0][1])/1000 for i4 in range(len(data4))]
            y_subset4 = [y4 for x4, y4 in zip(x_data4, y_data4) if 0.0 <= x4 <= 0.45]
            y_usf4 = max(y_subset4)
            y_subset4 = [y4 for x4, y4 in zip(x_data4, y_data4) if 0.45 <= x4 <= 0.65]
            y_ssf4 = min(y_subset4)
            fig4, ax4 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            ax4.plot(x_data4,y_data4,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax4.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax4.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax4.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig4.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
            \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                        \n---------------------------------------------------------------------------\n\
                            '%(plane,direction,type_atom,cryst,plane,direction,y_usf4,type_atom,cryst,plane,direction,y_ssf4)   
                            
        if plane == '1101' and direction == '1123':
            gsfe_output = 'gsfe_output.txt'
            plane = '10-11'
            direction = '-1-123'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %.16f %s orient  [-12-10] [10-12] [-1011] -orthogonal-cell -duplicate 1 1 1 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\n\nvariable xmin equal xlo\nvariable xmax equal xhi\nvariable xtot equal v_xmin+v_xmax\n')
                f.write('\nvariable ymin equal ylo\nvariable ymax equal yhi\nvariable ytot equal v_ymin+v_ymax')
                f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                        \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                        \ngroup lower region lower')
                f.write('\nfix 1 lower move linear  ${xtot} ${ytot} 0.0\nfix 2 all setforce 0.0 0.0 NULL\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data5 = np.genfromtxt(gsfe_output)    
            x_data5 = [i5 for i5 in np.linspace(0,1,len(data5))]
            y_data5 = [(data5[i5][1])/1000 - (data5[0][1])/1000 for i5 in range(len(data5))]
            y_subset5 = [y5 for x5, y5 in zip(x_data5, y_data5) if 0.2 <= x5 <= 0.35]
            y_usf5 = max(y_subset5)
            y_subset5 = [y5 for x5, y5 in zip(x_data5, y_data5) if 0.35 <= x5 <= 0.55]
            y_ssf5 = min(y_subset5)
            y_subset5 = [y5 for x5, y5 in zip(x_data5, y_data5) if 0.6 <= x5 <= 0.8]
            y_twin5 = max(y_subset5)                
            fig5, ax5 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            ax5.plot(x_data5,y_data5,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax5.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax5.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax5.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig5.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
            \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                \ntwin energy  for %s %s %s plane in %s direction = %0.16f j/m^2\
                \n---------------------------------------------------------------------------\n\
                '%(plane,direction,type_atom,cryst,plane,direction,y_usf5,type_atom,cryst,plane,direction,y_ssf5,type_atom,cryst,plane,direction,y_twin5)                                
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val) 
                    
        if plane == '1122' and direction == '1123':
            gsfe_output = 'gsfe_output.txt'
            plane = '11-22'
            direction = '11-23'
            properties = 'gsfe_%s_%s_%s_%s'%(cryst,plane,direction,type_atom)
            os.system('atomsk --create %s %.16f %.16f %s orient [-1100] [11-23] [11-22] -orthogonal-cell -duplicate 1 1 10 struct_%s.lmp >/dev/null 2>&1'%(cryst,latparam,lz_latparam,type_atom,plane))
            with open('gsfe.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                f.write(potential_mod(style,potential_file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                f.write('\nvariable ymin equal ylo\nvariable ymax equal yhi\nvariable ytot equal v_ymin+v_ymax')
                f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                        \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                        \ngroup lower region lower')
                f.write('\nfix 1 lower move linear  0.0 -${ytot} 0.0\nfix 2 all setforce 0.0 0.0 NULL\n')
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
                start_range = 25
                end_range = start_range
                # Loop until the condition is satisfied
                flag = False
                while not flag:
                    # Update the end_range variable
                    end_range += 25                
                    # Loop through the range
                    for i in range(start_range, end_range, 25):
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
                        if difference_val < 0.5:
                            flag = True
                            break
                
                    # Update the start_range for the next iteration
                    start_range = end_range
                    if flag:
                        break
                        
            
            data6 = np.genfromtxt(gsfe_output)     
            x_data6 = [i6 for i6 in np.linspace(0,1,len(data6))]
            y_data6 = [(data6[i6][1])/1000 - (data6[0][1])/1000 for i6 in range(len(data6))]
            y_subset6 = [y6 for x6, y6 in zip(x_data6, y_data6) if 0.2 <= x6 <= 0.35]
            y_usf6 = max(y_subset6)
            y_subset6 = [y6 for x6, y6 in zip(x_data6, y_data6) if 0.35 <= x6 <= 0.55]
            y_ssf6 = min(y_subset6)
            y_subset6 = [y6 for x6, y6 in zip(x_data6, y_data6) if 0.6 <= x6 <= 0.8]
            y_twin6 = max(y_subset6)                
            fig6, ax6 = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
            ax6.plot(x_data6,y_data6,'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'%s : %s [%s]$<$%s$>$'%(type_atom,cryst.upper(),plane,direction))
            ax6.set_xlabel(r'Normalized displacement',fontweight='bold')
            ax6.set_ylabel(r'GSFE, E$_{f}^{sf}$ (mJ/m$^2$)',fontweight='bold')
            ax6.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
            fig6.savefig('gsfe_%s_%s_%s_%s.pdf'%(type_atom,cryst,plane,direction))
            plt.close()
            os.system('mkdir %s'%properties)            
            os.system('mv *.in %s dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(gsfe_output,type_atom,cryst,plane,direction,properties))
            filename = output
            new_val = '\n# Stacking fault energies %s plane %s direction\n\
            \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
            \nstable stacking fault for %s %s %s plane in %s direction = %0.16f j/m^2\
                \ntwin energy  for %s %s %s plane in %s direction = %0.16f j/m^2\
                \n---------------------------------------------------------------------------\n\
                '%(plane,direction,type_atom,cryst,plane,direction,y_usf6,type_atom,cryst,plane,direction,y_ssf6,type_atom,cryst,plane,direction,y_twin6)                                
            if os.path.isfile(filename):
                with open(filename, 'a') as file3:
                    file3.write(new_val)
            else:
                with open(filename, 'w') as file3:
                    file3.write(new_val)   
################ Optimization functions
def objective_function(params=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)):
    if os.path.exists("%s"%output):
        os.remove("%s"%output)
    b0, b1, b2, b3, t0, t1, t2, t3, Cmin, Cmax, asub, alpha = params
    # Save potential and parameter files
    lib_file = write_library(element_properties_database, element, element_struc, first_near, ielement, atwt,
                            alpha, b0, b1, b2, b3, latparam, cohesieve_energy, asub, t0, t1, t2, t3, rozero, ibar)
    param_file = write_parameter(element_properties_database, rc, delr, augt1, erose_form, ialloy, zbl, nn2, rho0,
                              cohesieve_energy, re, alpha, repuls, attrac, Cmin, Cmax)

    potential_file = lib_file + ' ' + element + ' ' + param_file + ' ' + element
# whatever functios you want to calculate
# update the errors list
# update the weights
    constants=elastic_constant(cryst,latparam,type_atom,output,potential_file)
    # cold_curve(cryst,latparam,type_atom,output,potential_file)
    if 'e_vac' in element_properties_database:
        vac_form = vacancy_formation(cryst,latparam,type_atom,output,potential_file)
    if 'e_octa' in element_properties_database and cryst == 'bcc':
        defect = 'octahedral'
        interinfo_octra = interstetial_octa_bcc(cryst, latparam, type_atom, output, defect, potential_file)
    if 'e_tetra' in element_properties_database and cryst == 'bcc':
        defect = 'tetrahedral'
        interinfo_tetra = interstetial_tetra_bcc(cryst, latparam, type_atom, output, defect, potential_file)        
    if 'e_octa' in element_properties_database and cryst == 'fcc':
        defect = 'octahedral'
        interinfo_octra = interstetial_octa_fcc(cryst, latparam, type_atom, output, defect, potential_file)
    if 'e_tetra' in element_properties_database and cryst == 'fcc':
        defect = 'tetrahedral'
        interinfo_tetra = interstetial_tetra_fcc(cryst, latparam, type_atom, output, defect, potential_file)
    if 'e_octa' in element_properties_database and cryst == 'hcp':
        defect = 'octahedral'
        interinfo_octra = interstetial_octa_hcp(cryst, latparam, type_atom, output, defect, potential_file)
    if 'e_tetra' in element_properties_database and cryst == 'hcp':
        defect = 'tetrahedral'
        interinfo_tetra = interstetial_tetra_hcp(cryst, latparam, type_atom, output, defect, potential_file) 
    if 'fs_100' in element_properties_database and cryst == 'fcc':
        surface = '100'
        surface_info100fcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)
    if 'fs_110' in element_properties_database and cryst == 'fcc':
        surface = '110'
        surface_info110fcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)
    if 'fs_111' in element_properties_database and cryst == 'fcc':
        surface = '111'
        surface_info111fcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)  
    if 'fs_332' in element_properties_database and cryst == 'fcc':
        surface = '332'
        surface_info332fcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)        
    if 'fs_100' in element_properties_database and cryst == 'bcc':
        surface = '100'
        surface_info100bcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)    
    if 'fs_110' in element_properties_database and cryst == 'bcc':
        surface = '110'
        surface_info110bcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)    
    if 'fs_111' in element_properties_database and cryst == 'bcc':
        surface = '111'
        surface_info111bcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)
    if 'fs_112' in element_properties_database and cryst == 'bcc':
        surface = '112'
        surface_info112bcc = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)        
    if 'fs_0001' in element_properties_database and cryst == 'hcp':
        surface = '0001'
        surface_info0001hcp = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)    
    if 'fs_1010' in element_properties_database and cryst == 'hcp':
        surface = '1010'
        surface_info1010hcp = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)    
    if 'fs_1101' in element_properties_database and cryst == 'hcp':
        surface = '1101'
        surface_info1101hcp = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)  
    if 'fs_1122' in element_properties_database and cryst == 'hcp':
        surface = '1122'
        surface_info1122hcp = freesurfaceenergy(cryst,latparam,type_atom,output,surface, potential_file)
    if 'deltaE_' in  element_properties_database:  
        pd_diff = phase_energy_difference(cryst,latparam,type_atom,output,potential_file)
    
    if cryst == 'fcc' and 'sf_111_110_us' in element_properties_database:
        plane = '111'
        direction = '110'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe1 = extract_gsfe(cryst,type_atom, output,plane,direction)
    if cryst == 'fcc' and 'sf_111_112_us' in element_properties_database:
        plane = '111'
        direction = '112'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe2 = extract_gsfe(cryst,type_atom, output,plane,direction)
    if cryst == 'bcc' and 'sf_110_111_us' in element_properties_database:
        plane = '110'
        direction = '111'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe3 = extract_gsfe(cryst,type_atom, output,plane,direction)  
    if cryst == 'bcc' and 'sf_112_111_us' in element_properties_database:
        plane = '112'
        direction = '111'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe4 = extract_gsfe(cryst,type_atom, output,plane,direction)
    if cryst == 'bcc' and 'sf_123_111_us' in element_properties_database:
        plane = '123'
        direction = '111'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe5 = extract_gsfe(cryst,type_atom, output,plane,direction)        
    if cryst == 'hcp' and 'sf_0001_1120_us' in element_properties_database:
        plane = '0001'
        direction = '1120'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe6 = extract_gsfe(cryst,type_atom, output,plane,direction)
    if cryst == 'hcp' and 'sf_0001_1010_us' in element_properties_database:
        plane = '0001'
        direction = '1010'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe7 = extract_gsfe(cryst,type_atom, output,plane,direction)
    if cryst == 'hcp' and 'sf_1010_1120_us' in element_properties_database:
        plane = '1010'
        direction = '1120'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe8 = extract_gsfe(cryst,type_atom, output,plane,direction)  
    if cryst == 'hcp' and 'sf_1101_1120_us' in element_properties_database:
        plane = '1101'
        direction = '1120'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe9 = extract_gsfe(cryst,type_atom, output,plane,direction)         
    if cryst == 'hcp' and 'sf_1101_1123_us' in element_properties_database:
        plane = '1101'
        direction = '1123'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe10 = extract_gsfe(cryst,type_atom, output,plane,direction)   
    if cryst == 'hcp' and 'sf_1122_1123_us' in element_properties_database:
        plane = '1122'
        direction = '1123'
        gsfe(cryst,latparam,type_atom,output,plane,direction,potential_file)
        result_gsfe11 = extract_gsfe(cryst,type_atom, output,plane,direction)         
        
        
    # Compute the weighted discrepancy between the predicted and experimental values
    errors = []
    for key, value in element_properties_database.items():
        if key.startswith('ec_'):
            param_name = key
            if param_name[3:] in constants and param_name in element_properties_database:
                valueO = constants[param_name[3:]]
                # diff =  weights[param_name[3:]]*(valueO - float(element_properties_database[param_name]))**2
                diff =  (valueO - float(element_properties_database[param_name]))**2
                errors.append(diff)
        if key == 'e_vac':
            param_name2 = key
            if param_name2 in vac_form and param_name2 in element_properties_database:
                valueO2 = vac_form[param_name2]
                diff2 =  (float(valueO2) - float(element_properties_database[param_name2]))**2
                errors.append(diff2)
        if key == 'e_octa':
            param_name2 = key
            if param_name2 in interinfo_octra and param_name2 in element_properties_database:
                valueO2 = interinfo_octra[param_name2]
                diff3 =  (float(valueO2) - float(element_properties_database[param_name2]))**2
                errors.append(diff3)
        if key == 'e_tetra':
            param_name2 = key
            if param_name2 in interinfo_tetra and param_name2 in element_properties_database:
                valueO2 = interinfo_tetra[param_name2]
                diff4 =  (float(valueO2) - float(element_properties_database[param_name2]))**2
                errors.append(diff4)
        if key == 'fs_100'and cryst == 'bcc':
            param_name100bcc = key 
            if param_name100bcc in surface_info100bcc and 'fs_100' in element_properties_database :
                value100bcc = surface_info100bcc[param_name100bcc]
                diff100bcc =  (float(value100bcc) - float(element_properties_database[param_name100bcc]))**2
                errors.append(diff100bcc)
        if key == 'fs_110'and cryst == 'bcc':
            param_name110bcc = key 
            if param_name110bcc in surface_info110bcc and 'fs_110' in element_properties_database :
                value110bcc = surface_info110bcc[param_name110bcc]
                diff110bcc =  (float(value110bcc) - float(element_properties_database[param_name110bcc]))**2
                errors.append(diff110bcc)
        if key == 'fs_111'and cryst == 'bcc':
            param_name111bcc = key 
            if param_name111bcc in surface_info111bcc and 'fs_111' in element_properties_database :
                value111bcc = surface_info111bcc[param_name111bcc]
                diff111bcc =  (float(value111bcc) - float(element_properties_database[param_name111bcc]))**2
                errors.append(diff111bcc)
        if key == 'fs_112'and cryst == 'bcc':
            param_name112bcc = key 
            if param_name112bcc in surface_info112bcc and 'fs_112' in element_properties_database :
                value112bcc = surface_info112bcc[param_name112bcc]
                diff112bcc =  (float(value112bcc) - float(element_properties_database[param_name112bcc]))**2
                errors.append(diff112bcc)
        if cryst == 'fcc' and  key == 'fs_100':
            param_name100fcc = key 
            if param_name100fcc in surface_info100fcc and 'fs_100' in element_properties_database :
                value100fcc = surface_info100fcc[param_name100fcc]
                diff100fcc =  (float(value100fcc) - float(element_properties_database[param_name100fcc]))**2
                errors.append(diff100fcc)
        if cryst == 'fcc' and key == 'fs_110':
            param_name110fcc = key 
            if param_name110fcc in surface_info110fcc and 'fs_110' in element_properties_database :
                value110fcc = surface_info110fcc[param_name110fcc]
                diff110fcc =  (float(value110fcc) - float(element_properties_database[param_name110fcc]))**2
                errors.append(diff110fcc)
        if cryst == 'fcc' and key == 'fs_111':
            param_name111fcc = key 
            if param_name111fcc in surface_info111fcc and 'fs_111' in element_properties_database :
                value111fcc = surface_info111fcc[param_name111fcc]
                diff111fcc =  (float(value111fcc) - float(element_properties_database[param_name111fcc]))**2
                errors.append(diff111fcc)
        if cryst == 'fcc' and  key == 'fs_332':
            param_name332fcc = key 
            if param_name332fcc in surface_info332fcc and 'fs_332' in element_properties_database :
                value332fcc = surface_info332fcc[param_name332fcc]
                diff332fcc =  (float(value332fcc) - float(element_properties_database[param_name332fcc]))**2
                errors.append(diff332fcc)
        if cryst == 'hcp' and key == 'fs_0001':
            param_name0001hcp = key 
            if param_name0001hcp in surface_info0001hcp and 'fs_0001' in element_properties_database :
                value0001hcp = surface_info0001hcp[param_name0001hcp]
                diff0001hcp =  (float(value0001hcp) - float(element_properties_database[param_name0001hcp]))**2
                errors.append(diff0001hcp)
        if cryst == 'hcp' and key == 'fs_1010':
            param_name1010hcp = key 
            if param_name1010hcp in surface_info1010hcp and 'fs_1010' in element_properties_database :
                value1010hcp = surface_info1010hcp[param_name1010hcp]
                diff1010hcp =  (float(value1010hcp) - float(element_properties_database[param_name1010hcp]))**2
                errors.append(diff1010hcp)
        if cryst == 'hcp' and key == 'fs_1101':
            param_name1101hcp = key 
            if param_name1101hcp in surface_info1101hcp and 'fs_1101' in element_properties_database :
                value1101hcp = surface_info1101hcp[param_name1101hcp]
                diff1101hcp =  (float(value1101hcp) - float(element_properties_database[param_name1101hcp]))**2
                errors.append(diff1101hcp)
        if cryst == 'hcp' and key == 'fs_1122':
            param_name1122hcp = key 
            if param_name1122hcp in surface_info1122hcp and 'fs_1122' in element_properties_database :
                value1122hcp = surface_info1122hcp[param_name1122hcp]
                diff1122hcp =  (float(value1122hcp) - float(element_properties_database[param_name1122hcp]))**2
                errors.append(diff1122hcp)
        if cryst == 'fcc' and key == 'deltaE_hcp':
            param_namediffhcp = key
            if param_namediffhcp in pd_diff and param_namediffhcp in element_properties_database:
                valuefhcp = pd_diff[param_namediffhcp]
                diffvaluefhcp = (float(valuefhcp) - float(element_properties_database[param_namediffhcp]))**2
                errors.append(diffvaluefhcp)
        if cryst == 'fcc' and key == 'deltaE_bcc':
            param_namediffhcp = key
            if param_namediffhcp in pd_diff and param_namediffhcp in element_properties_database:
                valuefhcp = pd_diff[param_namediffhcp]
                diffvaluefhcp = (float(valuefhcp) - float(element_properties_database[param_namediffhcp]))**2
                errors.append(diffvaluefhcp)                  
        if cryst == 'bcc' and key == 'deltaE_fcc':
            param_namediffhcp = key
            if param_namediffhcp in pd_diff and param_namediffhcp in element_properties_database:
                valuefhcp = pd_diff[param_namediffhcp]
                diffvaluefhcp = (float(valuefhcp) - float(element_properties_database[param_namediffhcp]))**2
                errors.append(diffvaluefhcp)  
        if cryst == 'bcc' and key == 'deltaE_hcp':
            param_namediffhcp = key
            if param_namediffhcp in pd_diff and param_namediffhcp in element_properties_database:
                valuefhcp = pd_diff[param_namediffhcp]
                print(valuefhcp)
                diffvaluefhcp = (float(valuefhcp) - float(element_properties_database[param_namediffhcp]))**2
                errors.append(diffvaluefhcp)                  
        if cryst == 'hcp' and key == 'deltaE_fcc':
            param_namediffhcp = key
            if param_namediffhcp in pd_diff and param_namediffhcp in element_properties_database:
                valuefhcp = pd_diff[param_namediffhcp]
                diffvaluefhcp = (float(valuefhcp) - float(element_properties_database[param_namediffhcp]))**2
                errors.append(diffvaluefhcp)  
        if cryst == 'hcp' and key == 'deltaE_bcc':
            param_namediffhcp = key
            if param_namediffhcp in pd_diff and param_namediffhcp in element_properties_database:
                valuefhcp = pd_diff[param_namediffhcp]
                diffvaluefhcp = (float(valuefhcp) - float(element_properties_database[param_namediffhcp]))**2
                errors.append(diffvaluefhcp)
        if cryst == 'fcc' and key == 'sf_111_110_us':
            param_name111_110us = key
            if param_name111_110us in result_gsfe1 and param_name111_110us in element_properties_database:
                value111_110us = result_gsfe1[param_name111_110us]
                diffvalue111_110us = (float(value111_110us) - float(element_properties_database[param_name111_110us]))**2
                errors.append(diffvalue111_110us)  
        if cryst == 'fcc' and key == 'sf_111_110_ss':
            param_name111_110ss = key
            if param_name111_110ss in result_gsfe1 and param_name111_110ss in element_properties_database:
                value111_110ss = result_gsfe1[param_name111_110ss]
                diffvalue111_110ss = (float(value111_110ss) - float(element_properties_database[param_name111_110ss]))**2
                errors.append(diffvalue111_110ss)
        if cryst == 'fcc' and key == 'sf_111_112_us':
            param_name111_112us = key
            if param_name111_112us in result_gsfe2 and param_name111_112us in element_properties_database:
                value111_112us = result_gsfe2[param_name111_112us]
                diffvalue111_112us = (float(value111_112us) - float(element_properties_database[param_name111_112us]))**2
                errors.append(diffvalue111_112us) 
        if cryst == 'fcc' and key == 'sf_111_112_ss':
            param_name111_112ss = key
            if param_name111_112ss in result_gsfe2 and param_name111_112ss in element_properties_database:
                value111_112ss = result_gsfe2[param_name111_112ss]
                diffvalue111_112ss = (float(value111_112ss) - float(element_properties_database[param_name111_112ss]))**2
                errors.append(diffvalue111_112ss) 
        if cryst == 'fcc' and key == 'sf_111_112_tb':
            param_name111_112tb = key
            if param_name111_112tb in result_gsfe2 and param_name111_112tb in element_properties_database:
                value111_112tb = result_gsfe2[param_name111_112tb]
                diffvalue111_112tb = (float(value111_112tb) - float(element_properties_database[param_name111_112tb]))**2
                errors.append(diffvalue111_112tb)
        if cryst == 'bcc' and key == 'sf_110_111_us':
            param_name110_111us = key
            if param_name110_111us in result_gsfe3 and param_name110_111us in element_properties_database:
                value110_111us = result_gsfe3[param_name110_111us]
                diffvalue110_111us = (float(value110_111us) - float(element_properties_database[param_name110_111us]))**2
                errors.append(diffvalue110_111us)
        if cryst == 'bcc' and key == 'sf_112_111_us':
            param_name112_111us = key
            if param_name112_111us in result_gsfe4 and param_name112_111us in element_properties_database:
                value110_111ss = result_gsfe4[param_name112_111us]
                diffvalue111_112tb = (float(value110_111ss) - float(element_properties_database[param_name112_111us]))**2
                errors.append(diffvalue111_112tb)
        if cryst == 'bcc' and key == 'sf_123_111_us':
            param_name123_111us = key
            if param_name123_111us in result_gsfe5 and param_name123_111us in element_properties_database:
                value123_111us = result_gsfe5[param_name123_111us]
                diffvalue123_111us = (float(value123_111us) - float(element_properties_database[param_name123_111us]))**2
                errors.append(diffvalue123_111us)
        if cryst == 'hcp' and key == 'sf_0001_1210_us':
            param_name0001_1210_us = key
            if param_name0001_1210_us in result_gsfe6 and param_name0001_1210_us in element_properties_database:
                value0001_1210_us = result_gsfe6[param_name0001_1210_us]
                diffvalue0001_1210_us = (float(value0001_1210_us) - float(element_properties_database[param_name0001_1210_us]))**2
                errors.append(diffvalue0001_1210_us) 
        if cryst == 'hcp' and key == 'sf_0001_1210_ss':
            param_name0001_1210_ss = key
            if param_name0001_1210_ss in result_gsfe6 and param_name0001_1210_ss in element_properties_database:
                value0001_1210_ss = result_gsfe6[param_name0001_1210_ss]
                diffvalue0001_1210_ss = (float(value0001_1210_ss) - float(element_properties_database[param_name0001_1210_ss]))**2
                errors.append(diffvalue0001_1210_ss)                  
        if cryst == 'hcp' and key == 'sf_0001_1010_us':
            param_name0001_1010_us = key
            if param_name0001_1010_us in result_gsfe7 and param_name0001_1010_us in element_properties_database:
                value0001_1010_us = result_gsfe7[param_name0001_1010_us]
                diffvalue0001_1010_us = (float(value0001_1010_us) - float(element_properties_database[param_name0001_1010_us]))**2
                errors.append(diffvalue0001_1010_us)
        if cryst == 'hcp' and key == 'sf_0001_1010_ss':
            param_name0001_1010_ss = key
            if param_name0001_1010_ss in result_gsfe7 and param_name0001_1010_ss in element_properties_database:
                value0001_1010_ss = result_gsfe7[param_name0001_1010_ss]
                diffvalue0001_1010_ss = (float(value0001_1010_ss) - float(element_properties_database[param_name0001_1010_ss]))**2
                errors.append(diffvalue0001_1010_ss) 
        if cryst == 'hcp' and key == 'sf_0001_1010_tb':
            param_name0001_1010_tb = key
            if param_name0001_1010_tb in result_gsfe7 and param_name0001_1010_tb in element_properties_database:
                value0001_1010_tb = result_gsfe7[param_name0001_1010_tb]
                diffvalue0001_1010_tb = (float(value0001_1010_tb) - float(element_properties_database[param_name0001_1010_tb]))**2
                errors.append(diffvalue0001_1010_tb)
        if cryst == 'hcp' and key == 'sf_1010_1210_us':
            param_name1010_1210_us = key
            if param_name1010_1210_us in result_gsfe8 and param_name1010_1210_us in element_properties_database:
                value1010_1210_us = result_gsfe8[param_name1010_1210_us]
                diffvalue1010_1210_us = (float(value1010_1210_us) - float(element_properties_database[param_name1010_1210_us]))**2
                errors.append(diffvalue1010_1210_us)
        if cryst == 'hcp' and key == 'sf_1010_1210_ss':
            param_name1010_1210_ss = key
            if param_name1010_1210_ss in result_gsfe8 and param_name1010_1210_ss in element_properties_database:
                value1010_1210_ss = result_gsfe8[param_name1010_1210_ss]
                diffvalue1010_1210_ss = (float(value1010_1210_ss) - float(element_properties_database[param_name1010_1210_ss]))**2
                errors.append(diffvalue1010_1210_ss)
        if cryst == 'hcp' and key == 'sf_1101_1210_us':
            param_name1101_1210_us = key
            if param_name1101_1210_us in result_gsfe9 and param_name1101_1210_us in element_properties_database:
                value1101_1210_us = result_gsfe9[param_name1101_1210_us]
                diffvalue1101_1210_us = (float(value1101_1210_us) - float(element_properties_database[param_name1101_1210_us]))**2
                errors.append(diffvalue1101_1210_us) 
        if cryst == 'hcp' and key == 'sf_1101_1210_ss':
            param_name1101_1210_ss = key
            if param_name1101_1210_ss in result_gsfe9 and param_name1101_1210_ss in element_properties_database:
                value1101_1210_ss = result_gsfe9[param_name1101_1210_ss]
                diffvalue1101_1210_ss = (float(value1101_1210_ss) - float(element_properties_database[param_name1101_1210_ss]))**2
                errors.append(diffvalue1101_1210_ss)  
        if cryst == 'hcp' and key == 'sf_1101_1123_us':
            param_name1101_1123_us = key
            if param_name1101_1123_us in result_gsfe10 and param_name1101_1123_us in element_properties_database:
                value1101_1123_us = result_gsfe10[param_name1101_1123_us]
                diffvalue1101_1123_us = (float(value1101_1123_us) - float(element_properties_database[param_name1101_1123_us]))**2
                errors.append(diffvalue1101_1123_us) 
        if cryst == 'hcp' and key == 'sf_1101_1123_ss':
            param_name1101_1123_ss = key
            if param_name1101_1123_ss in result_gsfe10 and param_name1101_1123_ss in element_properties_database:
                value1101_1123_ss = result_gsfe10[param_name1101_1123_ss]
                diffvalue1101_1123_ss = (float(value1101_1123_ss) - float(element_properties_database[param_name1101_1123_ss]))**2
                errors.append(diffvalue1101_1123_ss)  
        if cryst == 'hcp' and key == 'sf_1101_1123_tb':
            param_name1101_1123_tb = key
            if param_name1101_1123_tb in result_gsfe10 and param_name1101_1123_tb in element_properties_database:
                value1101_1123_tb = result_gsfe10[param_name1101_1123_tb]
                diffvalue1101_1123_tb = (float(value1101_1123_tb) - float(element_properties_database[param_name1101_1123_tb]))**2
                errors.append(diffvalue1101_1123_tb)
        if cryst == 'hcp' and key == 'sf_1122_1123_us':
            param_name1122_1123_us = key
            if param_name1122_1123_us in result_gsfe11 and param_name1122_1123_us in element_properties_database:
                value1122_1123_us = result_gsfe11[param_name1122_1123_us]
                diffvalue1122_1123_us = (float(value1122_1123_us) - float(element_properties_database[param_name1122_1123_us]))**2
                errors.append(diffvalue1122_1123_us)
        if cryst == 'hcp' and key == 'sf_1122_1123_ss':
            param_name1122_1123_ss = key
            if param_name1122_1123_ss in result_gsfe11 and param_name1122_1123_ss in element_properties_database:
                value1122_1123_ss = result_gsfe11[param_name1122_1123_ss]
                diffvalue1122_1123_ss = (float(value1122_1123_ss) - float(element_properties_database[param_name1122_1123_ss]))**2
                errors.append(diffvalue1122_1123_ss)
        if cryst == 'hcp' and key == 'sf_1122_1123_tb':
            param_name1122_1123_tb = key
            if param_name1122_1123_tb in result_gsfe11 and param_name1122_1123_tb in element_properties_database:
                value1122_1123_tb = result_gsfe11[param_name1122_1123_tb]
                diffvalue1122_1123_tb = (float(value1122_1123_tb) - float(element_properties_database[param_name1122_1123_tb]))**2
                errors.append(diffvalue1122_1123_tb)                 
    error = np.sqrt(np.mean(errors))
    serror=np.round(error,decimals=2)
    # ccp=np.round(constants['c11'],decimals=2)
    # elp=np.round(float(element_properties_database['ec_c11']),decimals=2)
    # ccp1=np.round(constants['c12'],decimals=2)
    # elp1=np.round(float(element_properties_database['ec_c12']),decimals=2)    
    # ccp2=np.round(constants['c44'],decimals=2)
    # elp2=np.round(float(element_properties_database['ec_c44']),decimals=2)    
    # ccp3v=np.round(float(vac_form['e_vac']),decimals=2)
    # elp3v=np.round(float(element_properties_database['e_vac']),decimals=2)    
    # ccp3=np.round(float(interinfo_octra['e_octa']),decimals=2)
    # elp3=np.round(float(element_properties_database['e_octa']),decimals=2)
    # ccp4=np.round(float(interinfo_tetra['e_tetra']),decimals=2)
    # elp4=np.round(float(element_properties_database['e_tetra']),decimals=2)
    # ccp5=np.round(float(surface_info110bcc['fs_110']),decimals=2)
    # elp5=np.round(float(element_properties_database['fs_110']),decimals=2)
    # if serror < 4.0:
        # print('~~ ERROR:{} {} {} {} {} {} {} {} {}'.format(str(serror),str(ccp2),str(elp2),str(elp3v),str(ccp3),str(elp3),str(ccp4),str(elp4)))
    print('~~ ERROR:{}'.format(str(serror)))
    if serror < 60.0:
        # print('~~ ERROR:{}'.format(str(serror)))        
        os.system('cp %s %s_%s'%(lib_file,lib_file,str(serror)))
        os.system('cp %s %s_%s'%(param_file,param_file,str(serror)))
        os.system('mv %s_%s %s_%s %s/'%(lib_file,str(serror),param_file,str(serror),pf))
    return error 
 

###############################################################################
########################## Read data from User#################################
###############################################################################

filename = "database.data"
element = "Ti"
style = 'meam'
lammps_executable = './lmp_serial'
element_properties_database = extract_element_properties(filename, element)
element_keys = element_properties_database.keys()
# print(element_keys)
# for key, value in element_properties_database.items():
#     if key.startswith('ec_'):
#         print(key[3:],value)

##  write a MEAM potential first from the constant's and a starting point
latparam,cohesieve_energy, element_struc, first_near, atwt, alpha, re = meam_prop_calc(element,element_properties_database)
# elastic constant functions input
cryst = element_properties_database.get("ground_state")
latparam = float(element_properties_database.get("lattice_constant_a"))
type_atom = element_properties_database.get("symbol")
output = 'output_lammps_sims.data'
if os.path.exists("%s"%output):
    os.remove("%s"%output)

pf = 'potential_files_%s'%element
os.system('mkdir %s'%pf)               
#### Bound values, very important!!!!!!!!!!!!!!!!
# weight from 0 to 1
# weights = {'c11': 1.0, 'c12': 1.0, 'c13':0.0, 'c33':0.0, 'c44':1.0}  # adjust these weights as needed

b0b=(2.0, 3.0)
b1b=(2.0, 3.0)
b2b=(4.0, 5.0)
b3b=(2.0, 3.0)
t0b=(1.0, 1.0) # must be 1
t1b=(0.1, 1)
t2b=(4, 5 )
t3b=(-14, -12)
Cminb= (0.38,0.49)
Cmaxb=(2.8, 2.8)
asubb=(0.6, 0.7)
# alpha=4.719331
bounds = [b0b,b1b,b2b,b3b,t0b,t1b,t2b,t3b,Cminb,Cmaxb,asubb,(1.0*alpha, 1.0*alpha)]

convergence_threshold = 0.005  # Initial convergence threshold

def callback(x, convergence):
    if convergence < convergence_threshold:
        return True
    return False

# def callback(x, convergence):
#     if objective_function(x) < 0.5:
#         return True
#     return False

# result = differential_evolution(objective_function, bounds,**kwords, callback=callback)  
result = differential_evolution(objective_function, bounds, callback=callback)
optimal_params = result.x
optimal_error = result.fun
print("Optimal Parameters:", optimal_params)
print("Optimal Error:", optimal_error)









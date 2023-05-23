#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:08:16 2023

@author: mashroornitol
"""

import numpy as np
import os
from plot_function import*
import subprocess

def lammps_run(in_file):
    return('%s -in %s'%(lammps_executable,in_file))

def lammps_header():
    return('clear\nunits metal\ndimension 3\nboundary p p p \
                \natom_style atomic\natom_modify map array\nbox tilt large\n')
def struct_mod(cryst):
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
        return('variable a equal %.6f\nlattice hcp $a\
                \nregion box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0\
                    \ncreate_box 1 box\ncreate_atoms 1 box'%latparam)                           
def potential_mod(style,file):
    return('pair_style %s\npair_coeff * * %s\
           \nneighbor 2.0 bin\nneigh_modify delay 10 check yes'%(style,file))

def output_mod():
    return('compute eng all pe/atom \ncompute eatoms all reduce sum c_eng\
           \nthermo 1 \nthermo_style    custom step etotal temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol c_eatoms\
               \nthermo_modify norm no\
           \ndump 1 all custom 1 dump_lammps id type x y z fx fy fz c_eng')
 
def minimization():
    return('fix 1 all box/relax aniso 0.0 vmax 0.1\
           \nminimize 1e-16 1e-16 1000 1000\
           \nunfix 1\
           \nrun 0')
           
def relaxation():
    return('\nminimize 1e-16 1e-16 1000 1000\
           \nrun 0')
           
def minimized_output():
    return('variable natoms equal count(all)\
           \nvariable teng equal "etotal"\
           \nvariable ecoh equal  v_teng/v_natoms\
           \nvariable length equal "lx"\
           \nvariable perA_vol equal "vol/v_natoms"\
           \nprint "---------------------------------------------------------------------------" append %s\
           \nprint "# minimized structure and energy" append %s\
           \nprint "Cohesive energy (eV) = ${ecoh}" append %s\
           \nprint "Lattice constant (Angstoms) = ${length}" append %s\
           \nprint "Volume per atom (Angstom^3/atom) = ${perA_vol}" append %s\
           \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output,output,output))
           


def cold_curve(cryst,latparam,type_atom,output):
    properties='energy_volume_%s'%cryst
    """
    energy-volume curve calculaiton  using LAMMPS for any given material
    """
    with open('ev.in', 'w') as f:
        f.write(lammps_header() + '\n\n')
        f.write(struct_mod(cryst) + '\n\n')
        f.write(potential_mod(style,file) + '\n\n')
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
                
    os.system(lammps_run('ev.in'))
    ev_data = np.genfromtxt('energy_vol.dat',skip_header=1)
    ax.plot(ev_data[:,0],ev_data[:,1],'ro-',color=adjust_lightness('red',0.7),linewidth=2,markersize=8,fillstyle='full',label=r'EV curve of %s %s'%(cryst.upper(),type_atom))
    ax.set_xlabel(r'Volume per atom (\AA$^3$)',fontweight='bold')
    ax.set_ylabel(r'Energy per atom (eV/atom)',fontweight='bold')
    ax.legend(loc='best',fontsize=16,frameon=True,fancybox=False,edgecolor='k')
    fig.savefig('ev.pdf')
    os.system('mkdir %s'%properties)
    os.system('mv dump_lammps ev.pdf log.lammps ev.in energy_vol.dat %s/'%properties)
    
def vacancy_formation(cryst,latparam,type_atom,output):
    properties='vacancy_formation_%s'%cryst
    """
    vacancy formation calculaiton  using LAMMPS for any given material
    """
    with open('vac.in', 'w') as f:
        f.write(lammps_header() + '\n\n')
        f.write(struct_mod(cryst) + '\n\n')
        f.write(potential_mod(style,file) + '\n\n')
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
                
    os.system(lammps_run('vac.in'))
    os.system('mkdir %s'%properties)
    os.system('mv dump_lammps log.lammps vac.in %s/'%properties)
        
def elastic_constant(cryst,latparam,type_atom,output):
    properties='elastic_constant_%s'%cryst
    """
    elastic constant calculaiton  using LAMMPS for any given material
    """
    with open('displace.mod', 'w') as f:
        f.write('if "${dir} == 1" then & \n"variable len0 equal ${lx0}" \nif "${dir} == 2" then &\
\n "variable len0 equal ${ly0}" \nif "${dir} == 3" then & \n   "variable len0 equal ${lz0}"\
\nif "${dir} == 4" then & \n   "variable len0 equal ${lz0}" \nif "${dir} == 5" then &\
\n   "variable len0 equal ${lz0}" \nif "${dir} == 6" then & \n   "variable len0 equal ${ly0}"')
        f.write('\nclear \nbox tilt large\nread_restart restart.equil\n')
        f.write(potential_mod(style,file) + '\n\n')
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
        f.write(potential_mod(style,file) + '\n\n')
        f.write(output_mod() + '\n\n')
        f.write('variable C1neg equal ${d1}\
                \nvariable C1neg equal ${d1}\
                \nvariable C2neg equal ${d2}\
                    \nvariable C3neg equal ${d3}\
                \nvariable C4neg equal ${d4}\
                \nvariable C5neg equal ${d5}\
                \nvariable C6neg equal ${d6}')
        f.write('\nclear \nbox tilt large\nread_restart restart.equil\n')
        f.write(potential_mod(style,file) + '\n\n')
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
        f.write('variable etol equal 1.0e-32\nvariable ftol equal 1.0e-32\n\
                variable maxiter equal 100000\nvariable maxeval equal 100000\
                \nvariable dmax equal 1.0e-2\n')
        f.write(struct_mod(cryst) + '\n\n')
        f.write('mass * 1.0e-20\n')
        f.write(potential_mod(style,file) + '\n\n')
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
            \nprint "# Elastic constant and bulk modulus" append %s\
            \nprint "Bulk Modulus = ${bulkmodulus} ${cunits}" append %s\
            \nprint "Shear Modulus 1 = ${shearmodulus1} ${cunits}" append %s\
            \nprint "Shear Modulus 2 = ${shearmodulus2} ${cunits}" append %s\
            \nprint "Poisson Ratio = ${poissonratio}" append %s\
            \nprint "C11 = ${C11all} ${cunits}" append %s\
            \nprint "C12 = ${C12all} ${cunits}" append %s\
            \nprint "C13 = ${C13all} ${cunits}" append %s\
            \nprint "C33 = ${C33all} ${cunits}" append %s\
            \nprint "C44 = ${C44all} ${cunits}" append %s\
            \nprint "C55 = ${C55all} ${cunits}" append %s\
            \nprint "C66 = ${C66all} ${cunits}" append %s\
            \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output,output,output,output,output,output,output,output,output,output,output))
        f.write('\nrun 0\n')
    os.system(lammps_run('elastic.in'))
    os.system('mkdir %s'%properties)
    os.system('mv elastic.in restart.equil displace.mod %s/'%properties)

    
def interstetial_octa_fcc(cryst,latparam,type_atom,output,defect):
    properties = 'interstetial_octa_%s'%cryst
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
                    f.write(struct_mod(cryst) + '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            os.system(lammps_run('inter.in'))
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
            subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new = [i/replication for i in coord]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new[0],coord_new[1],coord_new[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# FCC Octahedral interstetial energy" append %s\
                    \nprint "FCC Octahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            os.system(lammps_run('inter2.in'))  
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)

def interstetial_tetra_fcc(cryst,latparam,type_atom,output,defect):
    properties = 'interstetial_tetra_%s'%cryst
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
                    f.write(struct_mod(cryst) + '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            os.system(lammps_run('inter.in'))
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
            subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new_tet = [i/replication for i in coord_tet]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new_tet[0],coord_new_tet[1],coord_new_tet[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# FCC tetrahedral interstetial energy" append %s\
                    \nprint "FCC tetrahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "--------------------------------------------------------------------------- append %s'%(output,output,output,output))
            os.system(lammps_run('inter2.in'))
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)
                
        
def interstetial_octa_bcc(cryst,latparam,type_atom,output,defect):
    properties = 'interstetial_octa_%s'%cryst
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
                    f.write(struct_mod(cryst) + '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            os.system(lammps_run('inter.in'))
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
            subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new = [i/replication for i in coord]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new[0],coord_new[1],coord_new[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# BCC Octahedral interstetial energy" append %s\
                    \nprint "BCC Octahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            os.system(lammps_run('inter2.in'))  
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)

def interstetial_tetra_bcc(cryst,latparam,type_atom,output,defect):
    properties = 'interstetial_tetra_%s'%cryst
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
                    f.write(struct_mod(cryst) + '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            os.system(lammps_run('inter.in'))
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
            subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new_tet = [i/replication for i in coord_tet]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new_tet[0],coord_new_tet[1],coord_new_tet[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# BCC tetrahedral interstetial energy" append %s\
                    \nprint "BCC tetrahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            os.system(lammps_run('inter2.in'))
            os.system('mkdir %s'%properties)        
            os.system('mv bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)

        
def interstetial_octa_hcp(cryst,latparam,type_atom,output,defect):
    properties = 'interstetial_octa_%s'%cryst
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
                    #f.write(struct_mod(cryst) + '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            os.system(lammps_run('inter.in'))
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
            subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new = [i/replication for i in coord]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new[0],coord_new[1],coord_new[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# HCP Octahedral interstetial energy" append %s\
                    \nprint "HCP Octahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            os.system(lammps_run('inter2.in'))  
            os.system('mkdir %s'%properties)        
            os.system('mv unitcell.lmp bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)

def interstetial_tetra_hcp(cryst,latparam,type_atom,output,defect):
    properties = 'interstetial_tetra_%s'%cryst
    """
    find the interstetial position in certiasian coordinate
    and insert in lammps script acoording to replication
    """
    interstetial_prop = ['tetrahedral']
    replication = 10
    if cryst == 'hcp':
        lz = np.sqrt(8/3.0)*latparam
        os.system('atomsk --create hcp %.16f %.16f %s -duplicate 1 1 1 unitcell.lmp'%(latparam,lz,type_atom))
        for idx, defect in enumerate(interstetial_prop):
            if defect == 'tetrahedral':
                coord_tet = [0,0,5/8.]
                with open('inter.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data unitcell.lmp'+ '\n\n')
                    #f.write(struct_mod(cryst) + '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('\nreplicate %d %d %d'%(replication,replication,replication))
                    f.write('\nwrite_data bulk_simbox.data')   
            os.system(lammps_run('inter.in'))
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
            subprocess.call(["sed -i  \"%ds/%d/%d/g\" POSCAR"%(lineno,natoms,new_natoms)], shell=True)
            coord_new_tet = [i/replication for i in coord_tet]
            subprocess.call(["echo '%s %s %s' >> POSCAR"%(coord_new_tet[0],coord_new_tet[1],coord_new_tet[2])], shell=True)
            os.system('atomsk POSCAR bulk_simbox.lmp')
            with open('inter2.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.data'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')                
                f.write(relaxation() + '\n\n')
                f.write('\nvariable N equal count(all)\
                        \nvariable No equal $N\
                        \nvariable E equal "c_eatoms"\
                        \nvariable Ei equal $E'+ '\n')
                #f.write('\nclear' + '\n')       
                f.write(lammps_header() + '\n\n')
                f.write('\nread_data bulk_simbox.lmp'+ '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(relaxation() + '\n\n')
                f.write('\nvariable Eint equal "c_eatoms"\
                    \nvariable Eint2 equal (${Eint}-((${No}+1)/${No})*${Ei})\
                    \n#variable Eint2 equal (${Eint}-(${Ei}/(${No}))*(${No}+1))\
                    \nprint "---------------------------------------------------------------------------" append %s\
                    \nprint "# HCP tetrahedral interstetial energy" append %s\
                    \nprint "HCP tetrahedral interstetial energy (eV) = ${Eint2}" append %s\
                    \nprint "---------------------------------------------------------------------------" append %s'%(output,output,output,output))
            os.system(lammps_run('inter2.in'))
            os.system('mkdir %s'%properties)        
            os.system('mv unitcell.lmp bulk_simbox.data bulk_simbox.lmp POSCAR dump_lammps log.lammps inter.in inter2.in %s/'%properties)


def freesurfaceenergy(cryst,latparam,type_atom,output):
    if cryst == 'fcc':
        surfaces = ['100','110','111','332']
        for idx, surface in enumerate(surfaces):        
            properties = 'surface_%s_%s'%(cryst,surface)
            if surface == '100':
                os.system('atomsk --create %s %.16f %s orient [100] [010] [001] -orthogonal-cell -duplicate 2 2 8 fcc_%s.lmp'%(cryst,latparam,type_atom,surface))
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '110':
                os.system('atomsk --create %s %.16f %s orient [112] [11-1] [1-10] -orthogonal-cell -duplicate 2 2 8 fcc_%s.lmp'%(cryst,latparam,type_atom,surface))
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '111':
                os.system('atomsk --create %s %.16f %s orient [112] [1-10] [11-1] -orthogonal-cell -duplicate 2 2 8 fcc_%s.lmp'%(cryst,latparam,type_atom,surface))                
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '332':
                os.system('atomsk --create %s %.16f %s orient [-2-26] [1-10] [332] -orthogonal-cell -duplicate 2 2 4  fcc_%s.lmp'%(cryst,latparam,type_atom,surface))  
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data fcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
                
    if cryst == 'bcc':
        surfaces = ['100','110','111','112']
        for idx, surface in enumerate(surfaces):        
            properties = 'surface_%s_%s'%(cryst,surface)
            if surface == '100':
                os.system('atomsk --create %s %.16f %s orient [100] [010] [001] -orthogonal-cell -duplicate 2 2 8 bcc_%s.lmp'%(cryst,latparam,type_atom,surface))
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '110':
                os.system('atomsk --create %s %.16f %s orient [111] [-1-12] [1-10] -orthogonal-cell -duplicate 2 2 8 bcc_%s.lmp'%(cryst,latparam,type_atom,surface))
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '111':
                os.system('atomsk --create %s %.16f %s orient [-1-12] [1-10] [111] -orthogonal-cell -duplicate 2 2 8 bcc_%s.lmp'%(cryst,latparam,type_atom,surface))                
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '112':
                os.system('atomsk --create %s %.16f %s orient [111] [1-10] [-1-12] -orthogonal-cell -duplicate 2 2 8  bcc_%s.lmp'%(cryst,latparam,type_atom,surface))  
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data bcc_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)                

    if cryst == 'hcp':
        lz_latparam = np.sqrt(8/3.)*latparam
        surfaces = ['0001','10m10','1011','1122']
        # surfaces = ['1011']
        for idx, surface in enumerate(surfaces):        
            properties = 'surface_%s_%s'%(cryst,surface)
            if surface == '0001':
                os.system('atomsk --create %s %.16f %.16f %s orient [1-210] [1-100] [0001] -orthogonal-cell -duplicate 2 2 8 hcp_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,surface))
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '10m10':
                os.system('atomsk --create %s %.16f %.16f %s orient [1-210] [0001] [10-10] -orthogonal-cell -duplicate 2 2 8 hcp_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,surface))
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '1011':
                os.system('atomsk --create %s %.16f %.16f %s orient [-12-10] [10-12] [-1011] -orthogonal-cell -duplicate 2 1 1 hcp_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,surface))                
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)
            elif surface == '1122':
                os.system('atomsk --create %s %.16f %.16f %s orient [-1100] [11-23] [11-22] -orthogonal-cell -duplicate 1 1 10  hcp_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,surface))  
                with open('surf.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data hcp_%s.lmp'%surface+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                os.system(lammps_run('surf.in'))
                os.system('mkdir %s'%properties)
                os.system('mv surf.in *.lmp POSCAR dump_lammps log.lammps  %s/'%properties)

def gsfe(cryst,latparam,type_atom,output):
    # os.system('rm -rf *.lmp *.in  gsfe*')
    if cryst == 'fcc':
        plane = '111'
        directions = ['112','110']
        for idx, direction in enumerate(directions):        
            if direction == '112':
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %s orient  [112] [1-10] [11-1] -orthogonal-cell -duplicate 2 2 16 struct_%s.lmp'%(cryst,latparam,type_atom,direction))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%direction+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear 20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_1.txt')
                    f.close()
                for i in range(0,260,10):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                data = np.genfromtxt('gsfe_1.txt')     
                x_data = [i for i in np.linspace(0,1,len(data))]
                y_data = [data[i][1] - data[0][1] for i in range(len(data))]
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
                os.system('mv *.in gsfe_1.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                    \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                        \ntwin boundary for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usf,type_atom,cryst,plane,direction,y_ssf,type_atom,cryst,plane,direction,y_twin)                   
                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)                         
            
            elif direction == '110':
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)             
                os.system('atomsk --create %s %.16f %s orient  [1-10] [112]  [11-1] -orthogonal-cell -duplicate 2 2 16 struct_%s.lmp'%(cryst,latparam,type_atom,direction))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%direction+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear 20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_2fcc.txt')
                    f.close()
                for i in range(0,145,5):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d'%i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                
                data2 = np.genfromtxt('gsfe_2fcc.txt')     
                x_data2 = [i2 for i2 in np.linspace(0,1,len(data2))]
                y_data2 = [data2[i][1] - data2[0][1] for i in range(len(data2))]
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
                os.system('mv *.in gsfe_2fcc.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                    \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usf2,type_atom,cryst,plane,direction,y_ssf2)                                   
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)
                        
    if cryst == 'bcc':
        
            if plane == '110' and direction == '111':
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %s orient [111] [-1-12] [1-10] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp'%(cryst,latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_1bcc.txt')
                    f.close()
                # for i in range(0,260,10):
                for i in range(0,135,5):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                dataB = np.genfromtxt('gsfe_1bcc.txt')     
                x_dataB = [iB for iB in np.linspace(0,1,len(dataB))]
                y_dataB = [dataB[iB][1] - dataB[0][1] for iB in range(len(dataB))]
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
                os.system('mv *.in gsfe_1bcc.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usfB)                   
                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)                         

            if plane == '112' and direction == '111':
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %s orient [111]  [1-10] [-1-12] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp'%(cryst,latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear 20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_2bcc.txt')
                    f.close()
                # for i in range(0,260,10):
                for i in range(0,135,5):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                data2 = np.genfromtxt('gsfe_2bcc.txt')     
                x_data2 = [i2 for i2 in np.linspace(0,1,len(data2))]
                y_data2 = [data2[i][1] - data2[0][1] for i in range(len(data2))]
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
                os.system('mv *.in gsfe_2bcc.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usf2)                   
                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)            

            if plane == '123' and direction == '111':
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %s orient [11-1] [-54-1] [123] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp'%(cryst,latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear 20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_3bcc.txt')
                    f.close()
                # for i in range(0,260,10):
                for i in range(0,135,5):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                data3 = np.genfromtxt('gsfe_3bcc.txt')     
                x_data3 = [i3 for i3 in np.linspace(0,1,len(data3))]
                y_data3 = [data3[i][1] - data3[0][1] for i in range(len(data3))]
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
                os.system('mv *.in gsfe_3bcc.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usf3)                   
                
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
        planes = ['0001_1120','0001_10-10','10-10-1120','1101-1120','1101-1123','1122-1123']
        # direction = ['1120','10-10','1123']
        for idx, plane in enumerate(planes):        
            if plane == '0001_1120':
                plane = '0001'
                direction = '1-210'
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %.16f %s orient [1-210] [10-10] [0001] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear 20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_1hcp.txt')
                    f.close()
                for i in range(0,165,5):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                datahcp = np.genfromtxt('gsfe_1hcp.txt')     
                x_datahcp = [ihcp for ihcp in np.linspace(0,1,len(datahcp))]
                y_datahcp = [datahcp[ihcp][1] - datahcp[0][1] for ihcp in range(len(datahcp))]
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
                os.system('mv *.in gsfe_1hcp.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                    \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usfhcp, type_atom,cryst,plane,direction,y_ssfhcp)                   
                                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)                            
            elif plane == '0001_10-10':
                plane = '0001'
                direction = '10-10'
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %.16f %s orient  [10-10] [1-210] [0001] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear -20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_2hcp.txt')
                    f.close()
                for i in range(0,290,10):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                data2 = np.genfromtxt('gsfe_2hcp.txt')     
                x_data2 = [i2 for i2 in np.linspace(0,1,len(data2))]
                y_data2 = [data2[i2][1] - data2[0][1] for i2 in range(len(data2))]
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
                os.system('mv *.in gsfe_2hcp.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                \ntwin energy %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usf2,type_atom,cryst,plane,direction,y_ssf2,type_atom,cryst,plane,direction,y_twin2)                   
                                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)   
            elif plane == '10-10-1120':
                plane = '10-10'
                direction = '1-210'
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %.16f %s orient  [1-210] [0001]  [10-10] -orthogonal-cell -duplicate 1 1 16 struct_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear -20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_3hcp.txt')
                    f.close()
                for i in range(0,165,5):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                data3 = np.genfromtxt('gsfe_3hcp.txt')     
                x_data3 = [i3 for i3 in np.linspace(0,1,len(data3))]
                y_data3 = [data3[i3][1] - data3[0][1] for i3 in range(len(data3))]
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
                os.system('mv *.in gsfe_3hcp.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usf3,type_atom,cryst,plane,direction,y_ssf3)                   
                                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)                         
            elif plane == '1101-1120':
                plane = '-1011'
                direction = '-12-10'
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %.16f %s orient  [-12-10] [10-12] [-1011] -orthogonal-cell -duplicate 2 1 1 struct_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
                    f.write(output_mod() + '\n\n')
                    f.write(minimization() + '\n\n')
                    f.write('variable E equal "c_eatoms"\nvariable Ei equal $E\n')
                    f.write('\nvariable z_middle0 equal (zlo+zhi)/2\nvariable lowdel0 equal "v_z_middle0-22"\n')
                    f.write('\nvariable updel0 equal "v_z_middle0+22"\nregion topatoms0 block INF INF INF INF ${updel0} INF units lattice\n')
                    f.write('\ngroup topatoms0 region topatoms0\ndelete_atoms group topatoms0\n')
                    f.write('\nregion bottomatoms0 block INF INF INF INF -1000000 ${lowdel0} units lattice\n')
                    f.write('\ngroup bottomatoms0 region bottomatoms0\ndelete_atoms group bottomatoms0\n')
                    f.write('\nfix 3 all box/relax x 0.0 y 0.0\nmin_style cg\nminimize 1e-16 1e-16 5000 5000\nunfix 3')
                    f.write('\nvariable z_middle equal ((zlo+zhi)/2)+2\
                            \nregion lower prism INF INF INF INF ${z_middle} INF 0 0 0 units box\
                            \ngroup lower region lower')
                    f.write('\nfix 1 lower move linear -20.0 0.0 0.0\nfix 2 all setforce 0.0 NULL NULL\n')
                    f.write('\nrun XXX \nminimize 1e-8 1e-8 10000 10000 \nunfix 2 \nunfix 1\n')
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_4hcp.txt')
                    f.close()
                for i in range(0,165,5):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                data4 = np.genfromtxt('gsfe_4hcp.txt')     
                x_data4 = [i4 for i4 in np.linspace(0,1,len(data4))]
                y_data4 = [data4[i4][1] - data4[0][1] for i4 in range(len(data4))]
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
                os.system('mv *.in gsfe_4hcp.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                            \n---------------------------------------------------------------------------\n\
                                '%(type_atom,cryst,plane,direction,y_usf4,type_atom,cryst,plane,direction,y_ssf4)                   
            elif plane == '1101-1123':
                plane = '10-11'
                direction = '-1-123'
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %.16f %s orient  [-12-10] [10-12] [-1011] -orthogonal-cell -duplicate 1 1 1 struct_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_5hcp.txt')
                    f.close()
                for i in range(0,510,10):    
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                
                data5 = np.genfromtxt('gsfe_5hcp.txt')     
                x_data5 = [i5 for i5 in np.linspace(0,1,len(data5))]
                y_data5 = [data5[i5][1] - data5[0][1] for i5 in range(len(data5))]
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
                os.system('mv *.in gsfe_5hcp.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                    \ntwin energy  for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                    \n---------------------------------------------------------------------------\n\
                    '%(type_atom,cryst,plane,direction,y_usf5,type_atom,cryst,plane,direction,y_ssf5,type_atom,cryst,plane,direction,y_twin5)                                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val) 
            elif plane == '1122-1123':
                plane = '11-22'
                direction = '11-23'
                properties = 'gsfe_%s_%s_%s'%(cryst,plane,direction)
                os.system('atomsk --create %s %.16f %.16f %s orient [-1100] [11-23] [11-22] -orthogonal-cell -duplicate 1 1 10 struct_%s.lmp'%(cryst,latparam,lz_latparam,type_atom,plane))
                with open('gsfe.in', 'w') as f:
                    f.write(lammps_header() + '\n\n')
                    f.write('\nread_data struct_%s.lmp'%plane+ '\n\n')
                    f.write(potential_mod(style,file) + '\n\n')
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
                    f.write('\nvariable sys_energy equal pe\nvariable sys_energy_J equal v_sys_energy*1.60217733e-19\n')
                    f.write('\nvariable LXO equal "lx*1e-10"\nvariable LYO equal "ly*1e-10"\n')
                    f.write('\nvariable GSFE equal "v_sys_energy_J/(v_LXO*v_LYO)"\n')
                    f.write('run 0 \n print "XXX ${GSFE}" append gsfe_6hcp.txt')
                    f.close()
                    
                for i in range(0,1025,25):
                    os.system('cp gsfe.in gsfe_%d.in'%i)
                    with open('gsfe_%d.in'%i, 'r') as file2 :
                        filedata = file2.read()
                    filedata = filedata.replace('XXX', '%d' %i)
                    with open('gsfe_%d.in'%i, 'w') as file2:
                        file2.write(filedata)
                    os.system(lammps_run('gsfe_%d.in'%i))
                    
                
                data6 = np.genfromtxt('gsfe_6hcp.txt')     
                x_data6 = [i6 for i6 in np.linspace(0,1,len(data6))]
                y_data6 = [data6[i6][1] - data6[0][1] for i6 in range(len(data6))]
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
                os.system('mv *.in gsfe_6hcp.txt dump_lammps *.lmp log.lammps gsfe_%s_%s_%s_%s.pdf %s/'%(type_atom,cryst,plane,direction,properties))
                filename = output
                new_val = '\n----------------Stacking fault energies ---------------------\n\
                \nunstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                \nstable stacking fault for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                    \ntwin energy  for %s %s %s plane in %s direction = %0.16f mJ/m^2\
                    \n---------------------------------------------------------------------------\n\
                    '%(type_atom,cryst,plane,direction,y_usf6,type_atom,cryst,plane,direction,y_ssf6,type_atom,cryst,plane,direction,y_twin6)                                
                if os.path.isfile(filename):
                    with open(filename, 'a') as file3:
                        file3.write(new_val)
                else:
                    with open(filename, 'w') as file3:
                        file3.write(new_val)   


def phase_energy_difference(cryst,latparam,type_atom,output):
    properties='phase_energy_differences_%s'%cryst
    formation_out = 'formation_output.txt'
    if os.path.exists("%s"%formation_out):
        os.remove("%s"%formation_out)
    os.system('rm *.POSCAR *.lmp POSCAR')
    with open('fcc.POSCAR', 'w') as file_fcc:
        lx = latparam
        file_fcc.write('FCC POSCAR file\n1.0\n%.16f 0 0\n0 %.16f 0\n0 0 %.16f\
                       \n%s\n4\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.0\n0.5 0.0 0.5\n0.0 0.5 0.5'\
                           %(lx,lx,lx,type_atom))
    os.system('cp fcc.POSCAR POSCAR')
    os.system('atomsk POSCAR fcc.lmp')
    file_fcc.close()
    with open('bcc.POSCAR', 'w') as file_bcc:
        lx = latparam
        file_bcc.write('BCC POSCAR file\n1.0\n%.16f 0 0\n0 %.16f 0\n0 0 %.16f\
                        \n%s\n2\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.5'\
                            %(lx,lx,lx,type_atom))        
    os.system('cp bcc.POSCAR POSCAR')
    os.system('atomsk POSCAR bcc.lmp') 
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
    os.system('atomsk POSCAR hcp.lmp')
    file_hcp.close()        
    with open('sc.POSCAR', 'w') as file_sc:
        lx=latparam
        file_sc.write('simple cubic POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n1\nDirect\n0.0 0.0 0.0'%(lx,lx,lx,type_atom))
    os.system('cp sc.POSCAR POSCAR')
    os.system('atomsk POSCAR sc.lmp')  
    file_sc.close()         
    with open('dc.POSCAR', 'w') as file_dc: #damond cubic
        lx=latparam
        file_dc.write('diamond cubic POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n8\nDirect\n0.25 0.25 0.25\n0.50 0.00 0.50\
                        \n0.25 0.75 0.75\n0.50 0.50 0.00\n0.75 0.25 0.75\n0.00 0.00 0.00\
                        \n0.75 0.75 0.25\n0.00 0.50 0.50'%(lx,lx,lx,type_atom))
    os.system('cp dc.POSCAR POSCAR')
    os.system('atomsk POSCAR dc.lmp')  
    file_dc.close()                       
    with open('a15.POSCAR', 'w') as file_a15:
        lx=latparam
        file_a15.write('a15 Sn POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n8\nDirect\n0.25 0.0 0.5\n0.75 0.0 0.5\
                        \n0.0 0.5 0.25\n0.0 0.5 0.75\n0.5 0.75 0.0\n0.5 0.25 0.0\
                        \n0.0 0.0 0.0\n0.5 0.5 0.5'%(lx,lx,lx,type_atom))
    os.system('cp a15.POSCAR POSCAR')
    os.system('atomsk POSCAR a15.lmp')  
    file_a15.close()
    with open('beta_sn.POSCAR', 'w') as file_betasn: ## beta-Sn
        lx=latparam
        ly=lx
        lz=lx*0.55
        file_betasn.write('beta Sn POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n4\nDirect\n0.0 0.0 0.0\n0.5 0.0 0.25\
                        \n0.5 0.5 0.5\n0.0 0.5 0.75'%(lx,ly,lz,type_atom))
    os.system('cp beta_sn.POSCAR POSCAR')
    os.system('atomsk POSCAR beta_sn.lmp')  
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
    os.system('atomsk POSCAR omega.lmp')
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
    os.system('atomsk POSCAR a7.lmp')
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
    os.system('atomsk POSCAR a11.lmp')
    file_a11.close()
    with open('a20.POSCAR', 'w') as file_a20: # alpha-U # Cmcm
        lx=latparam
        ly=2.0641907721492485*lx
        lz=1.7480584826470433*lx
        file_a20.write('a20 POSCAR file\n1.0\n%.16f 0 0\n0.0 %.16f 0\n0 0 %.16f\
                        \n%s\n4\nDirect\n0.0 0.099017 0.75\n0.5 0.400983 0.25\
                        \n0.5 0.599017 0.75\n0.0 0.900983 0.25'%(lx,ly,lz,type_atom))
    os.system('cp a20.POSCAR POSCAR')
    os.system('atomsk POSCAR a20.lmp')
    file_a20.close()
    
    files = os.listdir('./')
    lmp_files = [f for f in files if f.endswith('.lmp')]
    for lmp in lmp_files:
        with open('formation.in', 'w') as f:
            f.write(lammps_header() + '\n\n')
            f.write('\nread_data %s'%lmp+ '\n\n')
            f.write(potential_mod(style,file) + '\n\n')
            f.write(output_mod() + '\n\n')
            # f.write(relaxation() + '\n\n')
            f.write(minimization() + '\n\n')
            f.write(minimization() + '\n\n')
            f.write('variable natoms equal count(all)\
                       \nvariable teng equal "etotal"\
                       \nvariable ecoh equal  v_teng/v_natoms')
            f.write('\nprint "Per Atom energy for structure : %s =  ${ecoh} eV/atom" append %s'%(lmp,formation_out))
        os.system(lammps_run('formation.in'))

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
            
    
    os.system('mkdir %s'%properties)
    os.system('mv dump_lammps log.lammps formation.in formation_output.txt *.POSCAR *.lmp POSCAR %s/'%properties)  



def thermodynamic_props(cryst,latparam,type_atom,output,pressure,melting_point,total_run):
    half_run = int(total_run/2)
    properties='thermodynamics_%s'%cryst
    for press in pressure:
        press_bar = 10000*press
        for temp in range(50,int(melting_point)+800,150):
            with open('equil.in', 'w') as f:
                f.write(lammps_header() + '\n\n')
                f.write(struct_mod(cryst) + '\n\n')
                f.write(potential_mod(style,file) + '\n\n')
                f.write(output_mod() + '\n\n')
                f.write(minimization() + '\n\n')
                if cryst == 'bcc':
                    f.write('replicate 18 18 18')
                elif cryst == 'fcc' or cryst == 'hcp':
                    f.write('replicate 15 15 15\n')
                f.write(minimization() + '\n\n')
                output_variables = ('variable n_atoms equal "count(all)"'
                        '\nvariable temperature equal "temp" '
                        '\nvariable sys_energy equal "etotal" '
                        '\nvariable sys_enthalpy equal "enthalpy"'
                        '\nvariable pressure equal "press"'
                        '\nvariable perAtomE equal v_sys_energy/v_n_atoms'
                        '\nvariable perAtomH equal v_sys_enthalpy/v_n_atoms'
                        '\nvariable volumeEQ equal "vol"'
                        '\nvariable perAtom_volumeEQ equal v_volumeEQ/v_n_atoms'
                        '\nvariable LX_length equal "lx"'
                        '\nvariable LY_length equal "ly"'
                        '\nvariable LZ_length equal "lz"'
                        '\nvariable CA_ratio equal v_LZ_length/v_LX_length'
                        '\nvariable BA_ratio equal v_LY_length/v_LX_length'
                        '\nfix npt_eq all print 1 "${n_atoms} ${temperature} '
                        '${sys_energy} ${sys_enthalpy} ${pressure} ${perAtomE} ${perAtomH} '
                        '${volumeEQ} ${perAtom_volumeEQ} ${LX_length} '
                        '${LY_length} ${LZ_length} ${CA_ratio} ${BA_ratio}"'
                        ' file equil_npt_TEMP%d_PRESS%d.txt'%(temp,int(press_bar)))
                
                f.write(output_variables)
                velocity = 2*temp
                seed = int.from_bytes(os.urandom(2), byteorder="big") % 100000
                f.write('\nvelocity all create %d %d mom yes rot no'%(velocity,seed))
                f.write('\nfix 1 all npt temp %d %d 1 aniso %d %d 1 drag 1'%(int(temp),int(temp),int(press_bar),int(press_bar)))
                f.write('\nrun %d\nunfix 1\n'%half_run)
                seed = int.from_bytes(os.urandom(2), byteorder="big") % 100000
                f.write('\nvelocity all create %d %d mom yes rot no'%(velocity,seed))
                f.write('\nfix 1 all npt temp %d %d 1 aniso %d %d 1 drag 1'%(int(temp),int(temp),int(press_bar),int(press_bar)))
                f.write('\nrun %d\nunfix 1\n'%half_run)
            os.system(lammps_run('equil.in'))
            os.system('cp equil.in equil_temp%d_press%d.in'%(temp,press))
            os.system('cp log.lammps log.lammps_temp%d_press%d'%(temp,press))
            os.system('cp dump_lammps dump_lammps_temp%d_press%d'%(temp,press))
            
    os.system('mkdir %s'%properties)
    os.system('mv equil_npt_TEMP* dump_* log.* equil_temp* f %s/'%properties)            
                

def find_phase_ranges(temp_enthalpy_pairs):
    phase_ranges = []
    incr_list = []
    prev_enthalpy = temp_enthalpy_pairs[0][1]
    start_temp = temp_enthalpy_pairs[0][0]
    end_temp = None
    for temp, enthalpy in temp_enthalpy_pairs[1:]:
        incr_list.append(abs(enthalpy - prev_enthalpy))
        prev_enthalpy = enthalpy
    preceding_value = None
    for i in range(1, len(incr_list)):
        diff = incr_list[i-1]/incr_list[i]
        if diff > 1.2:
            preceding_value = incr_list[i]
    increament = preceding_value
    for temp, enthalpy in temp_enthalpy_pairs[1:]:
        if abs(enthalpy - prev_enthalpy) > increament:
            end_temp = temp
            phase_ranges.append((start_temp, end_temp))
            start_temp = end_temp
        prev_enthalpy = enthalpy
    phase_ranges.append((start_temp, temp))
    return phase_ranges

def thermodynamic_prop_analysis(total_run):
    if total_run < 5000:
        print("I will not analyze this unconverged files")
        return
    else:
        directory = 'thermodynamics_fcc/'
        keyword = 'npt'
        converge_line = total_run-1000
        output_thermo = []
        for filename in os.listdir(directory):
            if keyword in filename:
                with open(os.path.join(directory, filename), 'r') as f:
                    next(f)
                    pressure = []
                    temp = []
                    e = []
                    h = []
                    vol = []
                    for i, line in enumerate(f):
                        if i >= converge_line:
                            data = line.split()
                            pressure.append(float(data[4]))
                            temp.append(float(data[1]))
                            e.append(float(data[5]))
                            h.append(float(data[6]))
                            vol.append(float(data[8]))
                    index = filename.find("PRESS")
                    if index != -1:
                        p_value = float(filename[index + len("PRESS"):].split(".")[0])
                    output_thermo.append((p_value, np.mean(temp), np.mean(e), np.mean(h), np.mean(vol)))
        
        output_thermo.sort()
        with open('thermo_output_avg.txt', 'w') as f:
            f.write('# Pressure Temperature Per_Atom_E Per_Atom_H Per_Atom_Volume_EQ\n')
            for row in output_thermo:
                f.write('{:.2f} {:.2f} {:.4f} {:.4f} {:.4f}\n'.format(row[0], row[1], row[2], row[3], row[4]))
                
        df = pd.read_csv("thermo_output_avg.txt", comment="#", delim_whitespace=True, header=None, names=["Pressure", "Temperature", "Per_Atom_E", "Per_Atom_H", "Per_Atom_Volume_EQ"])
        pressures = df['Pressure'].unique()

        figenth, axenth = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
        for p in pressures:
            sub_df = df[df['Pressure'] == p]
            axenth.plot(sub_df['Temperature'], sub_df['Per_Atom_H'], 'o-', label=f'{p} MPa')
            
        axenth.set_xlabel(r'Temperature (K)',fontweight='bold')
        axenth.set_ylabel(r'Per atom H (eV/atom)',fontweight='bold')
        legend = axenth.legend(loc='upper center', fontsize=16, frameon=True, fancybox=False, edgecolor='k', bbox_to_anchor=(0.5, 1.2), ncol=4)
        legend.set_title('Pressure', prop={'size': 16, 'weight': 'bold'})
        figenth.subplots_adjust(top=0.85)
        figure_width = 10.0
        figure_height = 10.0
        figenth.set_size_inches(figure_width, figure_height)
        # plt.show()
        figenth.savefig('temp_enthalpy.pdf',bbox_inches='tight')
        
        data = {}
        with open("thermo_output_avg.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue
                pressure, temp,per_atom_e, per_atom_h, per_atom_vol= line.split()
                if pressure not in data:
                    data[pressure] = []
                data[pressure].append((float(temp), float(per_atom_h)))

        with open('thermo_output_extract.txt', 'w') as f:
            for pressure, temp_enthalpy_pairs in data.items():    
                f.write(f"Pressure: {pressure}"+"\n")
                phase_ranges = find_phase_ranges(temp_enthalpy_pairs)
                prev_end_temp = 0
                for i, (start_temp, end_temp) in enumerate(phase_ranges):
                    f.write(f"Phase change {i+1}:"+"\n")
                    if i == 0:
                        f.write(f"Temperature range before phase change {i+1}: {prev_end_temp}K to {start_temp}K"+"\n")
                        range_data = [(t, h) for t, h in temp_enthalpy_pairs if prev_end_temp < t < start_temp]
                    else:
                        f.write(f"Temperature range between phase change {i} and {i+1}: {prev_end_temp}K to {start_temp}K"+"\n")
                        range_data = [(t, h) for t, h in temp_enthalpy_pairs if prev_end_temp <= t < start_temp]
                    if range_data:
                        temps, enthalpies = zip(*range_data)
                        f.write(f"Temperature values: {temps}"+"\n")
                        f.write(f"Enthalpy values: {enthalpies}"+"\n")
                    f.write(f"Temperature range after phase change {i}: {start_temp}K to {end_temp}K"+"\n")
                    range_data = [(t, h) for t, h in temp_enthalpy_pairs if start_temp <= t <= end_temp]
                    temps, enthalpies = zip(*range_data)
                    f.write(f"Temperature values: {temps}"+"\n")
                    f.write(f"Enthalpy values: {enthalpies}"+"\n")
                    prev_end_temp = end_temp
             
     
            
"""
User input
A few notes to user:
    
    # installing atomsk and add excutable path in the ~/.bashrc
    Download atomsk https://atomsk.univ-lille.fr/dl.php depends on system
    
    # install appropriate python packages : numpy, matplotlib
    
    # LAMMPS excutable
    To change LAMMPS executable change "lammps_executable" variable.
    By default the code assumes it has "lmp_serial" in the same directory as the code.
    
    # LAMMPS potential
    The code assumes the potential file is the same directoy as the code.
    If you want to use EAM/MEAM/ML potential just change the how LAMMPS reads that potential in "file" variable below
    
    # Atom type, crystal structure and starting lattice parameter
    You have to input the which element you are interested in, the code will extract crystal structure and corresponding lattice parameter from its database text file.
    If you want to use your own custom lattice paramter and crystal structure, just comment "cryst" and "latparam=" variable's line and add your own string and floating number.
"""  


      
lammps_executable = './lmp_serial'


target_element = 'Nb'
#potential_type = 'eam'
potential_type = 'meam'
style='meam'
#style = 'eam/alloy'
potential_file = 'Nb.library_1.64 Nb Nb.parameter_1.64 Nb'
# potential_file = 'al-cu-set.eam.alloy Al'
# potential_file = 'library.meam V V.meam V'
###############################################################################

# with open('element_properties.data', 'r') as f:
#     for line in f:
#         if line.startswith(target_element):
#             properties = line.split()
#             ground_state = properties[1].lower()
#             lattice_parameter = float(properties[5])
#             melting_point = float(properties[4])
#             # print(f"Ground state: {ground_state}, Lattice parameter: {lattice_parameter}, Melting point: {melting_point} K")
#             break  # exit loop once the element is found
# 


latparam = 3.3
cryst = 'bcc'
type_atom = 'Nb'

   
output = 'results_%s_%s_%s.dat'%(potential_type,cryst,type_atom)
if os.path.exists("%s"%output):
    os.remove("%s"%output)    
    
file = potential_file



## Molecular static calculations:
elastic_constant(cryst,latparam,type_atom,output) 
cold_curve(cryst,latparam,type_atom,output)
vacancy_formation(cryst,type_atom,latparam,output)
phase_energy_difference(cryst,latparam,type_atom,output) 
defect='octahedral'
interstetial_octa_fcc(cryst,latparam,type_atom,output,defect)
defect='tetrahedral'
interstetial_tetra_fcc(cryst,latparam,type_atom,output,defect)
freesurfaceenergy(cryst,latparam,type_atom,output)
gsfe(cryst,latparam,type_atom,output)


## Molecular dyamic calculations:
# run at least 5000 run in LAMMPS please!    
#total_run=10000
#pressure = list(np.arange(0, 15.5, 0.5)) # for variable pressure
# pressure = list(np.arange(0, 0.5, 0.5)) # for single pressure
# thermodynamic_props(cryst,latparam,type_atom,output,pressure,melting_point,total_run)
#thermodynamic_prop_analysis(total_run)


# # Calculate properties of a FCC strucuture #
# latparam=4.045
# cryst='fcc'
# style='meam'
# type_atom='Al'

# file = 'MgAlZn.library.meam Mg Al Zn MgAlZn.parameter.meam Al'

# output = 'results_%s_%s_%s.dat'%(style,cryst,type_atom)
# if os.path.exists("%s"%output):
#     os.remove("%s"%output)

# # Calculate properties of a BCC strucuture #
# latparam=3.01
# cryst='bcc'
# style='meam'
# type_atom='V'

# file = 'library.meam V V.meam V'

# output = 'results_%s_%s_%s.dat'%(style,cryst,type_atom)
# if os.path.exists("%s"%output):
#     os.remove("%s"%output)

# # Calculate properties of a HCP strucuture #
# latparam=3.2
# cryst='hcp'
# style='meam'
# type_atom='Mg'

# file = 'MgAlZn.library.meam Mg Al Zn MgAlZn.parameter.meam Mg'

# print(file)
# output = 'results_%s_%s_%s.dat'%(style,cryst,type_atom)
# if os.path.exists("%s"%output):
#     os.remove("%s"%output)


# elastic_constant(cryst,latparam,type_atom,output) 
# cold_curve(cryst,latparam,type_atom,output)
# vacancy_formation(cryst,type_atom,latparam,output)
# defect='octahedral'
# interstetial_octa_hcp(cryst,latparam,type_atom,output,defect)
# defect='tetrahedral'
# interstetial_tetra_hcp(cryst,latparam,type_atom,output,defect)
# freesurfaceenergy(cryst,latparam,type_atom,output)
# gsfe(cryst,latparam,type_atom,output)
# phase_energy_difference(cryst,latparam,type_atom,output)



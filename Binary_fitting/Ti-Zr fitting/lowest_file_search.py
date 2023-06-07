#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:36:13 2023

@author: mashroornitol
"""

import re
import glob

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

def process_file(filename, lower_bound, upper_bound):
    # Step 1: Read the lines only after the "Binary" keyword
    binary_flag = False
    lines_after_binary = []

    with open(filename, 'r') as file:
        for line in file:
            if binary_flag:
                lines_after_binary.append(line)
            elif "Binary" in line:
                binary_flag = True

    # Step 2: Read the lines that contain 'attrac', 'repuls', 'Cmin', or 'Cmax' strings
    target_lines = []
    keywords = ['attrac', 'repuls', 'Cmin', 'Cmax']

    for line in lines_after_binary:
        if any(keyword in line for keyword in keywords):
            target_lines.append(line)

    # Step 3: Read the values after the '=' sign
    values = {}

    for line in target_lines:
        keyword, value = line.split('=')
        keyword = keyword.strip()
        value = float(value.strip())
        values[keyword] = value

    # Step 4: Multiply the values within the specified range
    modified_values = {
        'repuls': (lower_bound * values['repuls(1,2)'], upper_bound * values['repuls(1,2)']),
        'attrac': (lower_bound * values['attrac(1,2)'], upper_bound * values['attrac(1,2)']),
        'Cmin1': (lower_bound * values['Cmin(1,1,2)'], upper_bound * values['Cmin(1,1,2)']),
        'Cmin2': (lower_bound * values['Cmin(2,2,1)'], upper_bound * values['Cmin(2,2,1)']),
        'Cmin3': (lower_bound * values['Cmin(1,2,1)'], upper_bound * values['Cmin(1,2,1)']),
        'Cmin4': (lower_bound * values['Cmin(1,2,2)'], upper_bound * values['Cmin(1,2,2)']),
        'Cmin5': (lower_bound * values['Cmin(2,1,2)'], upper_bound * values['Cmin(2,1,2)']),
        'Cmin6': (lower_bound * values['Cmin(2,1,1)'], upper_bound * values['Cmin(2,1,1)']),
        'Cmax1': (lower_bound * values['Cmax(1,1,2)'], upper_bound * values['Cmax(1,1,2)']),
        'Cmax2': (lower_bound * values['Cmax(2,2,1)'], upper_bound * values['Cmax(2,2,1)']),
        'Cmax3': (lower_bound * values['Cmax(1,2,1)'], upper_bound * values['Cmax(1,2,1)']),
        'Cmax4': (lower_bound * values['Cmax(1,2,2)'], upper_bound * values['Cmax(1,2,2)']),
        'Cmax5': (lower_bound * values['Cmax(2,1,2)'], upper_bound * values['Cmax(2,1,2)']),
        'Cmax6': (lower_bound * values['Cmax(2,1,1)'], upper_bound * values['Cmax(2,1,1)'])
    }

    # Step 5: Print the values with lower and upper bounds
    print(f"cmin1 = ({modified_values['Cmin1'][0]}, {modified_values['Cmin1'][1]})")
    print(f"cmin2 = ({modified_values['Cmin2'][0]}, {modified_values['Cmin2'][1]})")
    print(f"cmin3 = ({modified_values['Cmin3'][0]}, {modified_values['Cmin3'][1]})")
    print(f"cmin4 = ({modified_values['Cmin4'][0]}, {modified_values['Cmin4'][1]})")
#    print(f"cmin5 = ({modified_values['Cmin5'][0]}, {modified_values['Cmin5'][1]})")
#    print(f"cmin6 = ({modified_values['Cmin6'][0]}, {modified_values['Cmin6'][1]})")
    print(f"cmax1 = ({modified_values['Cmax1'][0]}, {modified_values['Cmax1'][1]})")
    print(f"cmax2 = ({modified_values['Cmax2'][0]}, {modified_values['Cmax2'][1]})")
    print(f"cmax3 = ({modified_values['Cmax3'][0]}, {modified_values['Cmax3'][1]})")
    print(f"cmax4 = ({modified_values['Cmax4'][0]}, {modified_values['Cmax4'][1]})")
#    print(f"cmax5 = ({modified_values['Cmax5'][0]}, {modified_values['Cmax5'][1]})")
#    print(f"cmax6 = ({modified_values['Cmax6'][0]}, {modified_values['Cmax6'][1]})")
    print(f"repulsb = ({modified_values['repuls'][0]}, {modified_values['repuls'][1]})")
    print(f"attracb = ({modified_values['attrac'][0]}, {modified_values['attrac'][1]})")





ele1 = 'Zr'
ele2 = 'Ti'
pf = 'best_fit_files'
folder_path = '%s/' % pf  # Replace './' with the actual folder path if different
file_pattern2 = "%s%s%s.parameter_*" % (folder_path,ele1,ele2)
file_list2 = glob.glob(file_pattern2)
lowest_file2 = find_file_with_lowest_float(file_list2)
print(lowest_file2)
lower_bound = 0.995
upper_bound = 1.005
process_file(lowest_file2, lower_bound, upper_bound)

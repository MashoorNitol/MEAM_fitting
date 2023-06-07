#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:23:43 2023

@author: mashroornitol
"""

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
from matplotlib.pyplot import figure
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
###########################################
plt.rcParams["font.family"] = "Times New Roman"
rc('text', usetex=True)
font = {'family':'Times New Roman','size':20,'weight' : 'normal'}  # desired use
plt.rc('font',**font)
fig, ax = plt.subplots(facecolor='w',edgecolor='k',tight_layout=True)
plt.gca()
ax.tick_params(direction='in',length=7.0,width=1.5)
right_side = ax.spines["right"]
right_side.set_visible(True)
top_side = ax.spines["top"]
top_side.set_visible(True)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='minor',direction='in', length=4)
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

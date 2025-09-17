# -*- coding: utf-8 -*-
"""
Modified on 05012025. Remove the 2nd S population. Check the other

Created on Fri Nov  8 10:49:25 2024
Loading the saved file by using the pickle
@author: yawnt
"""

import pickle
import os
import sys

# The path is to noHomeo learned with 1000 s. 
# This is for testing with Homeo
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\runs\\1109_1047'
FigPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate'

'''
# Fig. 3. This is for testing with Homeo
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\runs\\1109_1047'
FigPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate'


# Fig. 3. Supp . 1 This is for testing noHomeo
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\runs\\1108_1001'
FigPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate_noHomeo'

# This is for testing stimulus selectivity
# TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\selectivitytest\\runs\\1111_1502'
FigPath= 'C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\selectivitytest\\figs\\plasticity_saturate_misAlign'


'''


# Specify the file path
file_path = TempPath + '\Output.pkl'

# Open the file and load the objects
with open(file_path, 'rb') as f:
    Output_dic = pickle.load(f)
    para_save = pickle.load(f)
    
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
lib_dir = os.path.join(parent_dir, 'lib')
sys.path.append(lib_dir)

import basic_functions as bf
import parameters_tuningcurve as params
import data_structure
import update_functions
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle
from numpy.random import default_rng
from scipy.stats import truncnorm
import AnalysisTool as AT

n_frames=para_save['n_frames']

PltTool=AT.Tools(Output_dic, para_save['ndt'],para_save['n_frames'], para_save['dt'], para_save, 
                         Plot_flag=True, save_dir=FigPath+'\\')  # Can add a output directory at the end.



PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,3],Output_dic['E_pop_rate'][:,n_frames-2],
                          start_color='grey',end_color='#31a354', labels=('Pre','Post'),
                          name='Stim',bin_width=0.5)
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,4],Output_dic['E_pop_rate'][:,n_frames-1],
                          start_color='grey',end_color='#3182bd',labels=('Pre','Post'),
                          name='Pred',bin_width=0.5)
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,2],Output_dic['E_pop_rate'][:,n_frames-3],
                          start_color='grey',end_color='#e34a33',labels=('Pre','Post'),
                          name='Stim+Pred',bin_width=0.5,left=-0.5)
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,5],Output_dic['E_pop_rate'][:,n_frames-4],
                          start_color='grey',end_color='#ffff99',labels=('Pre','Post'),
                          name='Trained Stim',bin_width=0.5)
init_trace=Output_dic['error']
init_trace = init_trace[~np.isnan(init_trace)]

PltTool.Plot_input_hist(Output_dic['error'],
                          start_color='grey',labels=None,
                          name='Input error',bin_width=0.1)

rep=['Base', 'Stim', 'Pred','Stim+Pred']
colors=['grey','#31a354','#3182bd','#e34a33']
rates=[np.mean(Output_dic['E_pop_rate'][:,1]),
       np.mean(Output_dic['E_pop_rate'][:,3]),
       np.mean(Output_dic['E_pop_rate'][:,4]),
       np.mean(Output_dic['E_pop_rate'][:,2])
       ]

errors=[np.std(Output_dic['E_pop_rate'][:,1]),    
       np.std(Output_dic['E_pop_rate'][:,3]),
       np.std(Output_dic['E_pop_rate'][:,4]),
       np.std(Output_dic['E_pop_rate'][:,2])
       ]
PltTool.MyBarPlot(rep,rates,errors=errors,colors=colors,tag='Init',top=10)




rates=[np.mean(Output_dic['E_pop_rate'][:,1]),
       np.mean(Output_dic['E_pop_rate'][:,n_frames-2]),
       np.mean(Output_dic['E_pop_rate'][:,n_frames-1]),
       np.mean(Output_dic['E_pop_rate'][:,n_frames-3])
       ]

errors=[np.std(Output_dic['E_pop_rate'][:,1]),
       np.std(Output_dic['E_pop_rate'][:,n_frames-2]),
       np.std(Output_dic['E_pop_rate'][:,n_frames-1]),
       np.std(Output_dic['E_pop_rate'][:,n_frames-3])
       ]
PltTool.MyBarPlot(rep,rates,errors=errors,colors=colors,tag='Final',top=10)


rep2=['Base', 'Pred', 'Stim','Stim+Pred']
colors2=['grey','#3182bd','#31a354','#e34a33']

rates2=[np.mean(Output_dic['E_pop_rate'][:,1]),
       np.mean(Output_dic['E_pop_rate'][:,n_frames-1]),
       np.mean(Output_dic['E_pop_rate'][:,n_frames-2]),
       np.mean(Output_dic['E_pop_rate'][:,n_frames-3])
       ]

errors2=[np.std(Output_dic['E_pop_rate'][:,1]),
       np.std(Output_dic['E_pop_rate'][:,n_frames-1]),
       np.std(Output_dic['E_pop_rate'][:,n_frames-2]),
       np.std(Output_dic['E_pop_rate'][:,n_frames-3])
       ]
PltTool.MyBarPlot(rep2,rates2,errors=errors2,colors=colors2,tag='FinalSecond',top=10)


PltTool.Plot_weight_diagram()





PltTool.sdir







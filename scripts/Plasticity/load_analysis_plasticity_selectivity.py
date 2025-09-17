# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:49:25 2024
Loading the saved file by using the pickle
@author: yawnt
"""

import pickle
import os
import sys

# The path is to noHomeo learned with 1000 s. 

# This is for testing stimulus selectivity N_neu=200
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\selectivitytest\\runs\\1114_0945'
FigPath= 'C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate_misAlign_n200'



'''
# This is for testing with Homeo
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\runs\\1115_1035'
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\runs\\1109_1047'
FigPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate'

# This is for testing noHomeo
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\runs\\1115_1122'
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\runs\\1108_1001'
FigPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate_noHomeo'
# This is for testing stimulus selectivity N_neu=40
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\selectivitytest\\runs\\1111_1502'
FigPath= 'C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate_misAlign'

# Figure 4: This is for testing stimulus selectivity N_neu=200
TempPath='C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\selectivitytest\\runs\\1114_0945'
FigPath= 'C:\\Users\\yawnt\\Dropbox\\NYU\\ResearchNYU\\Project_predictivecoding\\aaa10212024EverythingHere\\figs\\plasticity_saturate_misAlign_n200'


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

# In the future, this parameter should be saved in the parameter output file
N_eachC=200
np.random.seed(5)
shuffle_input_indices = np.random.permutation(N_eachC)
recovered_pred_indices = np.argsort(shuffle_input_indices)

PltTool.Plot_FixframeInt(Output_dic['W_EdS'][recovered_pred_indices,:],50,name='W_Eds_frames')
PltTool.Plot_FixframeInt(Output_dic['W_EdS2'][recovered_pred_indices,:],50,name='W_Eds2_frames')

PltTool.Plot_FixframeInt(Output_dic['W_EP'],50,name='W_EP_frames_raw')

recovered_stim2_indices=np.argsort(para_save['shuffle_stim_indices'])
PltTool.Plot_FixframeInt(Output_dic['W_EP'][recovered_stim2_indices,:],50,name='W_EP_frames_stim2')




PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[3,4,2,1]],
                           labels=['Stim','Pred','Expected','Baseline',],
                           colors=['#31a354','#3182bd','#e34a33','grey'],
                           name='InitalSnapshots')
PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-2,n_frames-1,n_frames-3,n_frames-6]],
                           labels=['Stim','Pred','Expected','Baseline',],
                           colors=['#31a354','#3182bd','#e34a33','grey'],
                           name='LearnedSnapshots')




PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,3],Output_dic['E_pop_rate'][:,n_frames-2],
                          start_color='grey',end_color='#31a354', labels=('Pre','Post'),
                          name='Stim',bin_width=0.5)
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,4],Output_dic['E_pop_rate'][:,n_frames-1],
                          start_color='grey',end_color='#3182bd',labels=('Pre','Post'),
                          name='Pred',bin_width=0.5)
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,2],Output_dic['E_pop_rate'][:,n_frames-3],
                          start_color='grey',end_color='#e34a33',labels=('Pre','Post'),
                          name='Stim+Pred',bin_width=0.5)


PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,2],Output_dic['E_pop_rate'][:,n_frames-3],
                          start_color='grey',end_color='#e34a33',labels=('Pre','Post'),
                          name='Stim2+Pred')
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,5],Output_dic['E_pop_rate'][:,n_frames-4],
                          start_color='grey',end_color='#ffff99',labels=('Pre','Post'),
                          name='Trained Stim')
'''
PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[3,4,2,5,6,1]],
                           labels=['Stim','Pred','Expected','Trained Stim','Trained Expect','Baseline',],
                           linestyles=['g','b','k','r','grey','--k'],
                           name='InitalSnapshots')
PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-2,n_frames-1,n_frames-3,n_frames-4,n_frames-5,n_frames-2]],
                           labels=['Stim','Pred','Expected','Trained Stim','Trained Expect','Baseline',],
                           linestyles=['g','b','k','r','grey','--k'],
                           name='LearnedSnapshots')
'''
PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-4,n_frames-5,n_frames-2,n_frames-3,n_frames-6]],
                           labels=['Expected Stim','Expected Stim + Pred','Probed Stim', 'Probed Stim + Pred','Baseline'],
                           linestyles=['-','-','-','-','--'],
                           colors=['#31a354','#e34a33','grey','grey','grey'],
                           name='SnapshotsExpectedvsProbedstim')

PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-4,n_frames-5,n_frames-6]],
                           labels=['Expected Stim','Expected Stim + Pred','Baseline'],
                           linestyles=['-','-','--'],
                           colors=['#31a354','#e34a33','grey'],
                           name='SnapshotsExpectedStim',
                           ymax=50,   linewidth=0.7)


PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-2,n_frames-3,n_frames-6]],
                           labels=['Probed Stim','Expected Stim + Pred','Baseline'],
                           linestyles=['-','-','--'],
                           colors=['#feb24c','#bd0026','grey'],
                           name='SnapshotsProbedStim',
                           ymax=50,   linewidth=0.7)

PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-4,n_frames-2]],
                           labels=['Expected Stim','Probed Stim'],
                           linestyles=['-','-'],
                           colors=['#feb24c','#31a354'],
                           name='SnapshotsStimlus+ProbedStim',
                           ymax=50,   linewidth=0.7,flag_scatter=True)

PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-3,n_frames-5]],
                           labels=['',''],
                           linestyles=['-','-'],
                           colors=['#feb24c','#31a354'],
                           name='SnapshotsPwithStimlus+ProbedStim',
                           ymax=50,   linewidth=0.7,flag_scatter=True)

PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-4,n_frames-2]],
                           labels=['Expected Stim','Probed Stim'],
                           linestyles=['-','-'],
                           colors=['#feb24c','#31a354'],
                           name='SnapshotsDownSampledStimlus+ProbedStim',
                           ymax=50,   linewidth=1,downsample=5)

PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[n_frames-3,n_frames-5]],
                           labels=['Probed Stim','Expected Stim'],
                           linestyles=['-','-'],
                           colors=['#feb24c','#31a354'],
                           name='SnapshotsDownSampledPwithStimlus+ProbedStim',
                           ymax=50,   linewidth=1,downsample=5)





MIdata=Output_dic['E_pop_rate'][:,[n_frames-5, n_frames-3, n_frames-4,n_frames-2, n_frames-6]  ]

PltTool.ModulationIndex(MIdata,labels=('Expected','Probed'),
                        start_color='#31a354',end_color='#feb24c')











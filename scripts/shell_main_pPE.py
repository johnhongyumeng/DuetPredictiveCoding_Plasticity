# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:39:04 2024

@author: yawnt
"""
import numpy as np

from class_main_nPE_perturbe import pPE_model 

case=2 
# ATTENTION! Need to change 1st, which one to loop,

#case=0
SST_pert_array=np.arange(-1,1.01,1)
#case=1
PV_pert_array=np.arange(-2,2.01,0.1)
#case=2
NMDA_pert_array=np.arange(0,1.01,0.5)

if case==0:
    fig_name='DifOverSSTpPEtest'
elif case==1:
    fig_name='DifOverPVpPEtest'
elif case==2:
    fig_name='DifOverNMDApPEtest'




DD_list=[]
nPE_neus_list=[]
pPE_neus_list=[]
#for ite_pert in SST_pert_array:
#for ite_pert in PV_pert_array:
for ite_pert in NMDA_pert_array:
    if case==0:
        DD_temp,Output_dic,PltTool=pPE_model.main(ratio_SST_perturb=ite_pert,Plot_flag=True)
    elif case==1:
        DD_temp,Output_dic,PltTool=pPE_model.main(ratio_PV_perturb=ite_pert,Plot_flag=True)
    elif case==2:
        DD_temp,Output_dic,PltTool=pPE_model.main(ratio_NMDA_perturb=ite_pert,Plot_flag=True)
    DD_list.append(DD_temp)
    pPE_neus_temp=Output_dic['E_pop_rate'][-1,29]- Output_dic['E_pop_rate'][-1,-1]
    nPE_neus_temp=Output_dic['E_pop_rate'][0,29]- Output_dic['E_pop_rate'][0,-1]
    
    nPE_neus_list.append(nPE_neus_temp)
    pPE_neus_list.append(pPE_neus_temp)

'''
import matplotlib.pyplot as plt  # Import matplotlib for plotting
DD_array=np.array(DD_list, dtype=float)
nPE_neus_array=np.array(nPE_neus_list, dtype=float)
pPE_neus_array=np.array(pPE_neus_list, dtype=float)

fig, axes=plt.subplots(1,1,figsize=(3,2.25))
if case==0:
    axes.plot(SST_pert_array, DD_array, marker='o', linestyle='-', color='k')
    axes.set_xlabel('SST perturb')
elif case==1:
    axes.plot(PV_pert_array, DD_array, marker='o', linestyle='-', color='k')
    axes.set_xlabel('PV perturb')
elif case==2:
    axes.plot(NMDA_pert_array, DD_array, marker='o', linestyle='-', color='k')
    axes.set_xlabel('NMDA perturb')

axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)

axes.set_ylabel('DD value')
fig.savefig('../figs/looped/pPE/'+fig_name+ 'pop.jpg',bbox_inches='tight')
fig.savefig('../figs/looped/pPE/'+fig_name+ 'pop.svg',bbox_inches='tight')

fig, axes=plt.subplots(1,1,figsize=(3,2.25))

#            labels=['Stim','Pred','Expected','Baseline',],
#            colors=['#31a354','#3182bd','#e34a33','grey'],
if case==0:
    axes.plot(SST_pert_array, nPE_neus_array, marker='o', color='k',linestyle='-',
              markerfacecolor='#3182bd',markeredgecolor=None,label='nPE')
    axes.plot(SST_pert_array, pPE_neus_array, marker='o', color='k',linestyle='-',
              markerfacecolor='#31a354',markeredgecolor=None,label='pPE')
    axes.plot(SST_pert_array, 0*SST_pert_array, color='grey', linestyle='--')
    axes.legend()
    axes.set_xlabel('SST perturb')
elif case==1:
    axes.plot(PV_pert_array, nPE_neus_array, marker='o', color='k',linestyle='-',
              markerfacecolor='#3182bd',markeredgecolor=None,label='nPE')
    axes.plot(PV_pert_array, pPE_neus_array, marker='o', color='k',linestyle='-',
              markerfacecolor='#31a354',markeredgecolor=None,label='pPE')
    axes.plot(PV_pert_array, 0*PV_pert_array, color='grey', linestyle='--')
    axes.set_xlabel('PV perturb')
elif case==2:
    axes.plot(NMDA_pert_array, nPE_neus_array, marker='o', color='k',linestyle='-',
              markerfacecolor='#3182bd',markeredgecolor=None,label='nPE')
    axes.plot(NMDA_pert_array, pPE_neus_array, marker='o', color='k',linestyle='-',
              markerfacecolor='#31a354',markeredgecolor=None,label='pPE')
    axes.plot(NMDA_pert_array, 0*NMDA_pert_array, color='grey', linestyle='--')
    axes.legend()
    axes.set_xlabel('NMDA perturb')

axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)
fig.savefig('../figs/looped/pPE/'+fig_name+ 'IndNeu.svg',bbox_inches='tight')
axes.set_ylabel('nPE and pPE neuron dR')
fig.savefig('../figs/looped/pPE/'+fig_name+ 'IndNeu.jpg',bbox_inches='tight')
'''

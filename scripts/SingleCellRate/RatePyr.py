# -*- coding: utf-8 -*-
"""
Created on 03072023 15:33:15 2023
This is not a model Kevin used, but something rewrite from scratch. 
Just use the concepts from Kevin's early exploration.
Tested on11012024. Retested for the manuscript. Need to get a Layer23 and Layer 5 neuron data out. 
Maybe I have to rerun everything. But this is how science should be done.
@author: John Meng
"""
 # This value close to Larkum L5 neuron.

####################
# Same trivial setup for input and output function

import os
import sys
import importlib
from datetime import date,datetime

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

lib_dir = os.path.join(grandparent_dir, 'lib')
sys.path.append(lib_dir)

sav_dir = os.path.join(grandparent_dir, 'runs')

now=datetime.now()
today=date.today()
current_time=now.strftime("%H%M")
current_day=today.strftime("%m%d_")
timestamp=current_day+current_time
'''
TempPath=sav_dir+'\\'+timestamp

if os.path.isdir(TempPath)==False:
    os.makedirs(TempPath)

filename = os.path.basename(__file__)

from shutil import copyfile      # Save a copy of the code to the saving folder.
copyfile(filename,TempPath+'\\'+filename)
'''


import basic_functions as bf
import parameters_tuningcurve as params
import data_structure
import update_functions
import numpy as np
from collections import OrderedDict



N_neu=1
PARAS=params.PARAMS_ALL.copy()

params_stim=PARAS['Stimulus'].copy()  # Remained here just for consistency.

params_stim['num_perG'] = N_neu   # Number of cells in each group. 
params_stim['Input_dend']= Injd_ite  # Unit pA
params_stim['Input_soma']= Injs_ite  # Unit pA
params_stim['Syns_dendE']= Condd_ite  # Unit nS  20 for leaking  
params_stim['Syns_dendI']= ConddI_ite  # Unit nS
params_stim['Syns_soma']= CondSE_ite  # Unit nS   50 mV difference, so 20 nS would be same as 1000 pA



params_sim=PARAS['Simulation'].copy()
    
# The following par values are from the 1st parameter set in script7.    
delay=1000*0.001   # The unit should be second
params_stim['Tinter'] = 1
params_stim['nbr_rep_std']=1
params_stim['Tstim'] = 1

# Generate a stimulus. This may change into a class.

n_stim_rep=1
params_sim['t_total'] = params_stim['Tresting'] + n_stim_rep*(params_stim['Tinter']+params_stim['Tstim'])   
ndt=round(params_sim['t_total']/params_sim['dt'])

#### Now setting up the single cells. Using the structure set up in data_structure.py.
# Notice the STP related features are saved heretoo
E_pop= data_structure.Pyramidal(N_neu)  # Initial pyramidal cells data strcuture. Including rate(some), vdend

### Initiate the connectivity matrix. Using the create_ring_connectivity() from Basic_functions.
# This should be constrained by the Allen's data Compagnota et al. 2022. In the sense of relative width. 
# Potential modulation of existing parameters here. 
params_E = PARAS['PARAMS_Pyr'].copy()
params_E['g_ds']= coupling_conductance
'''
G_back = dict(                   # Conductance.
    Es=params_E['Ibg_soma'],
)
'''

### Initialize a dataframe to save desired output. 

Output_dic= dict(
    E_pop_rate= np.full((N_neu,ndt),np.nan) ,
    E_pop_Vdend= np.full((N_neu,ndt),np.nan) ,
    E_pop_Isd= np.full((N_neu,ndt),np.nan) ,
    )


### Initiate the updating functions, using update_functions.py
#RHS_update= update_functions.RHS_update(W_dict, I_back)

LHS_update_E= update_functions.LHS_update.Pyr(params_E,params_sim['dt'])



### main loop. Noticing currently, I don't include NMDA, STP, LTP, Adaptation.
for i_time in range(ndt):
    # In each timestep, rate, conductance, and current are updated.
    # tau_I=0, so it's calculated as RHS. 
    # conductance (pre-synaptic) depends on the firing rate of the same cell. Update second
    # rate, doesn't depend on conductance. So update third.
    
    time=i_time*params_sim['dt']
    if time> params_stim['Tresting']:
        sti_d= params_stim['Input_dend']
        sti_s= params_stim['Input_soma']
        sys_dE= params_stim['Syns_dendE']
        sys_dI= params_stim['Syns_dendI']
        sys_sE= params_stim['Syns_soma']
    else:
        sti_d =0
        sti_s =0   
        sys_dE= 0
        sys_sE= params_stim['Syns_soma']  #Testing
        sys_dI =0
        
    # RHS update
    E_pop.input_soma= sti_s
    E_pop.input_dend= sti_d
    E_pop.gE_soma =sys_sE
    E_pop.gE_dend =sys_dE
    E_pop.gI_dend =sys_dI

# No synaptic input here    
#    E_pop.I_soma= RHS_update.I_pyr_soma(E_pop.S, P_pop.S)
#    E_pop.Iexc_to_dend= RHS_update.I_pyr_dendE(Int_pop.S)
#    E_pop.Iinh_to_dend= RHS_update.I_pyr_dendI(S_pop.S)
    
 
    # LHS update all the gating.
    E_pop.h= LHS_update_E.h(E_pop.h,E_pop.rate)

    # LHS update all the firing rates.
    E_pop.rate, E_pop.Vdend, Isd, Itotal = LHS_update_E.rate(E_pop )
    
    # saving. 
    Output_dic['E_pop_rate'][:,i_time]=E_pop.rate
    Output_dic['E_pop_Vdend'][:,i_time]=E_pop.Vdend
    Output_dic['E_pop_Isd'][:,i_time] = Isd
    
    
### output, maybe later analyze in a separate code. 
'''
import matplotlib.pyplot as plt
t_axis=np.arange(ndt)*params_sim['dt']

fig, axes=plt.subplots(2,1,figsize=(4,3))
axes[0].plot(t_axis,Output_dic['E_pop_rate'][0,:])
axes[0].set_title('Ending rate'+ f"{Output_dic['E_pop_rate'][0,-1]:.4f}")
axes[1].plot(t_axis,Output_dic['E_pop_Vdend'][0,:])
axes[1].set_title('Ending DendV'+ f"{Output_dic['E_pop_Vdend'][0,-1]:.4f}")

fig.subplots_adjust(hspace=0.5)
plt.show()
plt.close(fig)   # Otherwise it consumes the memory
'''

Rate_end=Output_dic['E_pop_rate'][0,-1]
Vdend_end=Output_dic['E_pop_Vdend'][0,-1]
Isd_end=Output_dic['E_pop_Isd'][0,-1]

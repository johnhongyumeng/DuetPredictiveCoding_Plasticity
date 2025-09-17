# -*- coding: utf-8 -*-
"""
Rewrite on 11182024. Testing different strength of the pred and stim, 
generate a pearson correlation before and after the learning
@author: John Meng
"""
import numpy as np
# Mod 
flag_homeo=0
Fig_folder='\\figs\\Selectivity'

#Trained_Pathway='\\net\\Output.pkl'
N_eachC=200
Trained_Pathway='\\net\\Output_noHomeoN200.pkl'
#FrameUsed=0   # -1 frame for trained network, 1 for untrained network.
#tag='\\Initial'
#measured_angle= (90-9)/180*np.pi
FrameUsed=-1
tag='\\Learned'
#measured_angle= (90+41)/180*np.pi
Fig_folder=Fig_folder+tag
alpha = 0.1  # 0.3 at 0508 Learning rate for matching case
eta_S=1  # 500 before 06102024. Learning rate on WEd_S bias

gamma_saturate=2.5   # How much deviant will stop learning. Testing
r_saturate=3
totalTime = 20.01 # Start with 200 data points. Should be enough.
jitter=0.0

WsInhFlag = 1
WpInhFlag = 1 

Input_S_ratio=1

sigma_W=0.1  # Default 0.01


kappa=0.3             # default 0.5. Requires a fine tuning.
Int_baseline=5      # 
E_baseline= 2        # baseline 2 at 0508 for both, I should collect the 1st time frame rate as baseline firing rate.
E_anti_baseline=2     # baseline 2  0508


####################
# Same trivial setup for input and output function

import os
import sys
import importlib
from datetime import date,datetime
from tqdm import tqdm

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

lib_dir = os.path.join(parent_dir, 'lib')
sys.path.append(lib_dir)


sav_dir = os.path.join(parent_dir, 'runs')

now=datetime.now()
today=date.today()
current_time=now.strftime("%H%M")
current_day=today.strftime("%m%d_")
timestamp=current_day+current_time

TempPath=os.path.join(sav_dir+'\\'+timestamp)

if os.path.isdir(TempPath)==False:
    os.makedirs(TempPath)

FigPath=os.path.join(parent_dir+Fig_folder)
if os.path.isdir(FigPath)==False:
    os.makedirs(FigPath)
    
    
filename = os.path.basename(__file__)

from shutil import copyfile      # Save a copy of the code to the saving folder.
copyfile(filename,TempPath+'\\'+filename)
para_loc='..\..\lib\\'


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

with open(parent_dir+Trained_Pathway, 'rb') as f:
    my_dict_loaded = pickle.load(f)  
    my_para_loaded= pickle.load(f)
        
Trained_W_EdS=my_dict_loaded['W_EdS'][:,FrameUsed];
Trained_W_EP=my_dict_loaded['W_EP'][:, FrameUsed];

# Set a global seed for reproducibility
np.random.seed(5)

# All right, I really hate this. The work is not that transparaent. But it is one way
PARAS=params.PARAMS_ALL.copy()
PARAS['PARAMS_Pyr']['kappa_NMDA']=0
PARAS['PARAMS_INT']['kappa_NMDA']=0
params_E = PARAS['PARAMS_Pyr'].copy()
params_Int = PARAS['PARAMS_INT'].copy()
params_P = PARAS['PARAMS_PV'].copy()
params_S = PARAS['PARAMS_SST'].copy()

# list_param= np.array([-0.01,2,0.5,8,50,1,1,1,1,0.1,2])   

# adaptation, top-down feedback, Tinter, dev_iud, delay, 
# ndf_plasticity, int_plasticity. bool_sigma_ndf, bool_sigma_int, adaptation_timeconstant
# nbr_rep. This description is based on the dict_param in Script7

params_stim=PARAS['Stimulus'].copy()
params_stim['prob_std'] = 0.8


params_sim=PARAS['Simulation'].copy()
    
# The following par values are from the 1st parameter set in script7.    
delay=50*0.001   # The unit should be second
params_stim['Tinter'] = 0.5
params_stim['nbr_rep_std']=6

# Generate a stimulus. This may change into a class.

#nbr_stim_std = 0
#nbr_stim_dev = 0   # default is 2. number of repetited deviation
n_stim_rep=params_stim['nbr_rep_std']+params_stim['nbr_rep_dev']+2
params_sim['t_total'] = params_stim['Tresting'] + n_stim_rep*(params_stim['Tinter']+params_stim['Tstim'])   # Why is adding 4? 
params_sim['t_total'] = totalTime
params_stim['t_total'] = totalTime

ndt=round(params_sim['t_total']/params_sim['dt'])
ndt_per_trial=round((params_stim['Tinter']+params_stim['Tstim'])/params_sim['dt'])
ndt_rest= round(params_stim['Tresting']/params_sim['dt'])

N_column=1
if flag_homeo:
    shuffle_input_indices = np.arange(N_eachC)
else:
    shuffle_input_indices = np.random.permutation(N_eachC)


N_neu=N_column*N_eachC
params_stim['num_perG'] = N_neu   # Number of cells in each group. 

N_column=int(round(N_column))
N_eachC=int(round(N_eachC))
N_neu=int(round(N_neu))


params_stim['N_eachC'] = N_eachC   # Number of cells in each group. 
params_stim['N_column'] = N_column   # Number of cells in each group. 
params_stim['N_neu'] = N_neu   # Number of cells in each group. 

Ind_vec_rest=np.zeros(ndt_rest)
Ind_vec_per_trial= np.arange(0,ndt_per_trial)
Ind_vec_control= np.zeros(ndt_per_trial)

Ind_vec_std_temp= np.concatenate((Ind_vec_rest,
                               np.tile(Ind_vec_per_trial,params_stim['nbr_rep_std']),
                               np.tile(Ind_vec_control,params_stim['nbr_rep_dev']),
                               np.tile(Ind_vec_per_trial,2)))
Ind_vec_dev_temp= np.concatenate((Ind_vec_rest,
                               np.tile(Ind_vec_control,params_stim['nbr_rep_std']),
                               np.tile(Ind_vec_per_trial,params_stim['nbr_rep_dev']),
                               np.tile(Ind_vec_control,2)))

fig, ax=plt.subplots(dpi=250,figsize=(2,1.7))
ax.plot(Ind_vec_std_temp)
plt.show()

Ind_vec_std=Ind_vec_std_temp>(params_stim['Tinter']/params_sim['dt'])
Ind_vec_dev=Ind_vec_dev_temp>(params_stim['Tinter']/params_sim['dt'])


#index_1d = j * N_col + i   for stim_2d[i,j]
# Here stim2d for debugging.

stimulus_std_inc,stim2d =bf.generate_ringblock_stim(params_stim['std_id'], N_column, N_eachC,
                                            params_stim['str_g']*params_stim['ratio_min'],
                                            params_stim['str_g']*params_stim['ratio_max'],
                                            params_stim['sigma'])
fig, ax=plt.subplots(dpi=250,figsize=(2,1.7))
ax.plot(stimulus_std_inc)
plt.show()

temp_vec=  np.linspace(1.0,1.1, N_eachC)
background_inc_ratio=np.tile(temp_vec, N_column)

stimuli_inc_learn = np.linspace(
    params_stim['str_g']*params_stim['ratio_min'], params_stim['str_g']*params_stim['ratio_max'], N_eachC)

stimuli_inc_back = np.linspace(
    params_stim['str_g_back']*params_stim['ratio_min'], params_stim['str_g_back']*params_stim['ratio_max'], N_eachC)

stimuli_inc_back=stimuli_inc_back[shuffle_input_indices]

jitter_vec=np.random.uniform(1 - jitter, 1 + jitter, size=N_eachC)

#stimuli_ave_learn = np.ones(N_eachC)*params_stim['str_g']*params_stim['SST_ratio']*jitter_vec
stimuli_ave_learn = np.ones(N_eachC)*params_stim['str_g']*params_stim['SST_ratio']

'''
stimuli_dec_learn = np.linspace(params_stim['str_g']*params_stim['ratio_max'],
                                params_stim['str_g']*params_stim['ratio_min'], N_eachC)*Ratio_topstimuli
'''





#### Now setting up the single cells. Using the structure set up in data_structure.py.
# Notice the STP related features are saved heretoo
E_pop= data_structure.Pyramidal(params_stim['N_neu'])  # Initial pyramidal cells data strcuture. Including rate(some), vdend
Int_pop = data_structure.Integrator(params_stim['N_column'])
Int_pop.rate= Int_baseline
P_pop= data_structure.Interneuron(params_stim['N_neu'])
S_pop= data_structure.Interneuron(params_stim['N_neu'])



### Initiate the connectivity matrix. Using the create_ring_connectivity() from Basic_functions.
# This should be constrained by the Allen's data Compagnota et al. 2022. In the sense of relative width. 
# Potential modulation of existing parameters here. 



W_dict = OrderedDict()   # 10052023 Now here would be the major changes. 

W_dict['EE']= 0 # Hope this work. Double check if error returns.
W_dict['PE']= 0  # Maybe later test the results with all realistic connections. 


W_dict['SE']= 0

#W_dict['EE']= bf.create_ring_connectivity(N_neu,N_neu,params_E['weight_to_soma']*Wscale*WeeScale,params_E['sigma_to_pc'])  # Jmax, sigma=43.2, Jmin=0
#W_dict['PE']= bf.create_ring_connectivity(N_neu,N_neu,params_E['weight_to_pv']*Wscale,params_E['sigma_to_pv'])    # sigma=43.2
#W_dict['SE']= bf.create_ring_connectivity(N_neu,N_neu,params_E['weight_to_sst']*Wscale,params_E['sigma_to_sst'])  # Uniform excitation. I 
# I need a bulk connectivity that do not specify the source of Top area.
W_dict['IntE']=bf.create_ringbulk_connectivity(
    params_E['weight_to_integrator'],params_E['weight_to_integrator'],
    N_col=N_column, N_eachc=N_eachC, type='BulktoInt')


#W_dict['IntE']= bf.create_ring_connectivity(N_neu,N_neu,params_E['weight_to_integrator']*Wscale*WbottopupScale,params_E['sigma_to_integrator'])  #sigma=1
'''
fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
im=ax.imshow(W_dict['IntE'])    
cbar = ax.figure.colorbar(im, ax=ax)
plt.show()

'''
rng1 = default_rng(1)
rng2 = default_rng(2)
# Randomizing here
'''
W_dict['EP'] = bf.create_random_delta_connectivity(
    params_P['weight_to_soma'], N_neu, sigma_r=sigma_W, rng=rng1)
mean_EP_value = params_P['weight_to_soma']*100 #N_ref=100
'''
W_dict['EP'] =  bf.load_ringbulk_connectivity(
    Trained_W_EP,N_column,type='BulktoBulk_diag')

W_dict['PP'] =0 
W_dict['SP'] =0

#W_dict['EP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_soma']*Wscale*WPVInhScale,params_P['sigma_to_pc'])  # Sigma=20. Need to study later. Diagonal in Kevin code
#W_dict['PP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_pv']*Wscale,params_P['sigma_to_pv'])  #  Sigma=20
#W_dict['SP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_sst']*Wscale,params_P['sigma_to_sst'])

'''
W_dict['EdS'] = bf.create_random_delta_connectivity(
    params_S['weight_to_dend'], N_neu, sigma_r=sigma_W, rng=rng2)
mean_EdS_value = params_S['weight_to_dend']*100 #N_ref=100
'''

W_dict['EdS']=bf.load_ringbulk_connectivity(Trained_W_EdS,N_column,type='BulktoBulk_diag')

#W_dict['EdS']= bf.create_delta_connectivity(params_S['weight_to_dend'],N_neu)*flag_StoE
#W_dict['EdS']= bf.create_ring_connectivity(N_neu,N_neu,params_S['weight_to_dend']*WSSTInhScale,params_S['sigma_to_pc'])

W_dict['PS']= 0
#W_dict['PS']= bf.create_ring_connectivity(N_neu,N_neu,params_S['weight_to_pv'],params_S['sigma_to_pv'])

# Here I should do more systematically, so far just directly change it.
'''
W_dict['EdInt']= bf.create_ringbulk_connectivity(
    params_Int['weight_to_dend']*params_Int['scaleMax'],params_Int['weight_to_dend']*params_Int['scaleMin'],
    N_col=N_column, N_eachc=N_eachC, type='InttoBulk')
'''


# Here I should do more systematically, shuffle make PS misalign
W_dict['EdInt']= bf.create_ringbulk_connectivity(
    params_Int['weight_to_dend']*params_Int['scaleMax'],params_Int['weight_to_dend']*params_Int['scaleMin'],
    N_col=N_column, N_eachc=N_eachC, type='InttoBulk',shuffle_inds=shuffle_input_indices)


# Testing the randomize the top-down Pred to dendrite.
W_dict['SInt']= np.zeros((N_neu,N_column))

#W_dict['SInt']= bf.load_ringbulk_connectivity(Trained_W_Edst,N_column)*params_S['ToptoSST']
# Using the trained connectivity to mimick the learned results. 

#W_dict['SInt']= bf.create_ring_connectivity(N_neu,N_neu,params_Int['weight_to_sst'],params_Int['sigma_to_sst'])   # Wasn't existed.

W_dict['PInt']=bf.create_ringbulk_connectivity(
    params_Int['weight_to_pv'],params_Int['weight_to_pv'],
    N_col=N_column, N_eachc=N_eachC, type='InttoBulk',jitter=jitter)

#W_dict['PInt']= bf.create_ring_connectivity(N_neu,N_neu,params_Int['weight_to_pv']*WsTopscale,params_Int['sigma_to_pv'])   # Wasn't existed.

W_dict['IntInt']= bf.create_ring_connectivity(
    N_column,N_column,params_Int['Jmax'],params_Int['sigma_to_int'], 
    N_column= N_column, Jmin=params_Int['Jmin'])

fig, ax =  plt.subplots(dpi=250,figsize=(4,3))
#im=ax.imshow(W_dict['PInt'],aspect='auto') 
im=ax.imshow(W_dict['EdInt'],aspect='auto') 

cbar = ax.figure.colorbar(im, ax=ax)
#plt.title('Connectivity Int to P')
plt.title('Connectivity Int to Ed')
plt.show()


'''
fig, ax =  plt.subplots(dpi=200,figsize=(2,1.7))
im=ax.imshow(W_dict['SInt'])   
cbar = ax.figure.colorbar(im, ax=ax)
ax.set_title('From Int to Int')
plt.show()
'''
g_back = dict(
    Es=params_E['gbg_soma'],
    Ed=params_E['gbg_dend'],
    Int=params_Int['gbg'],

    P=params_P['gbg'],

    S=params_S['gbg'],
    
#    Ekappa=params_E['kappa_NMDA'],  # How much excitation can get from NMDA.
#    Intkappa=params_Int['kappa_NMDA'],  # default 0.5 for both cases

    Ekappa=kappa,  # How much excitation can get from NMDA.
    Intkappa=kappa,  # default 0.5 for both cases
    IntIntkappa=1, # at top circuit, only NMDA works. 

)


### Initialize a dataframe to save desired output. 
#n_frames=2000   # Number of time frames I save. 
#save_interval = ndt // n_frames  # Calculate interval at which to save frames

save_interval = 0.5 / params_sim['dt']
save_interval = round(save_interval)
n_frames = ndt/save_interval
n_frames = round(n_frames)+1

i_frames=0
learning_protocol = np.nan

Output_dic= dict(
    E_pop_rate= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    E_pop_Vdend= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    E_pop_Isd= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    E_pop_Isum= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    P_pop_rate= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,

    P_pop_sumI= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    
    Int_pop_rate= np.full((N_column,n_frames),np.nan, dtype=np.float32) ,
    Int_pop_sumI= np.full((N_column,n_frames),np.nan, dtype=np.float32) ,
    S_pop_rate= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    S_pop_sumI= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    
    g_E_sE=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_E_sI=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_E_dE=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_E_dI=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,

    g_E_sEN=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_E_dEN=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,

    g_Int=np.full((N_column,n_frames),np.nan, dtype=np.float32) ,
    g_Int_N=np.full((N_column,n_frames),np.nan, dtype=np.float32) ,

    g_P_E=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_P_EN=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_P_I=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,

    g_S_E=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_S_EN=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    g_S_I=np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    
    h_E= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    h_EN= np.full((N_neu,n_frames),np.nan, dtype=np.float32) ,
    h_Int= np.full((N_column,n_frames),np.nan, dtype=np.float32) ,
    h_IntN= np.full((N_column,n_frames),np.nan, dtype=np.float32) ,

    W_EdS=np.full((N_neu, n_frames), np.nan, dtype=np.float32),
    W_EP=np.full((N_neu, n_frames), np.nan, dtype=np.float32),
    dWEds=np.full((N_neu, n_frames), np.nan, dtype=np.float32),
    dWEP=np.full((N_neu, n_frames), np.nan, dtype=np.float32),
    error=np.full(n_frames, np.nan, dtype=np.float32),
    error_shuffle=np.full(n_frames, np.nan, dtype=np.float32),
    rho_pred=np.full(n_frames, np.nan, dtype=np.float32),
    rho_stim=np.full(n_frames, np.nan, dtype=np.float32),
    tested_stimulus=np.full((N_neu,n_frames),np.nan,dtype=np.float32),   # Test different stimulus.
    shuffle_stim_indices=np.full((N_neu,n_frames),np.nan,dtype=np.float32),
    
    )
# Create a truncnorm distribution dist = truncnorm(a, b, loc=mu, scale=sigma)
#a, b = (lower - mu) / sigma, (upper - mu) / sigma, 
temp_dist = truncnorm(-2, 2, loc=0, scale=0.5)  
Output_dic['error']=   temp_dist.rvs(n_frames)
shuffle_error_indices = np.random.permutation(n_frames)
Output_dic['error_shuffle']=Output_dic['error'][shuffle_error_indices]

Output_dic['error'][1:7]=np.array([0,0,1,-1,1,0])   # Baseline, expected2, stim2, pred, stim, expected         
Output_dic['error'][n_frames-6:n_frames]=np.array([0, 0,1,0,1,-1])     # baseline, expected, stim, expeted2, stim2, pred

Output_dic['error_shuffle'][1:7]=np.array([0,0,1,-1,1,0])            
Output_dic['error_shuffle'][n_frames-6:n_frames]=np.array([0,0,1,0,1,-1])                    

pattern=[0,1]
Output_dic['rho_pred'] = np.tile(pattern, n_frames // 2+1)[:n_frames]
Output_dic['rho_stim'] = np.random.rand(n_frames)

# Including the intial. 
Output_dic['tested_stimulus'][:,0]=stimuli_inc_learn
Output_dic['tested_stimulus'][:,1]=stimuli_inc_learn
Output_dic['shuffle_stim_indices'][:,0]=np.arange(N_eachC)
shuffle_stim_indices=np.arange(N_eachC)

stimuli_test=stimuli_inc_learn

### Initiate the updating functions, using update_functions.py
RHS_update= update_functions.RHS_update(W_dict, g_back)
RHS_update.g_back['Es'] +=stimuli_inc_back
LHS_update_E= update_functions.LHS_update.Pyr(params_E,params_sim['dt'])
LHS_update_Int= update_functions.LHS_update.Int(params_Int,params_sim['dt'])

LHS_update_P= update_functions.LHS_update.P(params_P,params_sim['dt'])

LHS_update_S= update_functions.LHS_update.S(params_S,params_sim['dt'])


### main loop. Noticing currently, I don't include NMDA, STP, LTP, Adaptation.
#t_PVadvance=int(0.05//params_sim['dt']) 

# for i_time in range(ndt):
for i_time in tqdm(range(ndt)):
#for i_time in tqdm(range(int(3/params_sim['dt']))):    
    

    # In each timestep, rate, conductance, and current are updated.
    # tau_I=0, so it's calculated as RHS. 
    # conductance (pre-synaptic) depends on the firing rate of the same cell. Update second
    # rate, doesn't depend on conductance. So update third.

    if (i_time-1) % save_interval == 0 and i_frames < n_frames and i_time > save_interval:
        # For every save_interval, change the tested interval and learning rule.
        if i_frames % 2 == 0 and i_frames > 3:  # Redraw the activity
            shuffle_stim_indices = np.random.permutation(N_eachC)
            stimuli_test= stimuli_inc_learn[shuffle_stim_indices]
            Output_dic['rho_stim'][i_frames]=1 
            # print(f'shuffling the stim at i_time{i_time} i_frame{i_frames}\n' )

        # RHS_update.g_back['Es'] = stimuli_inc_learn*Output_dic['rho_stim'][i_frames]+g_back['Es']
        # RHS_update.g_back['S'] = stimuli_ave_learn*Input_S_ratio*Output_dic['rho_stim'][i_frames]+g_back['S']
        # Int_pop.rate=5.0+  Output_dic['rho_pred'][i_frames]  * 5.0
        
        rho_pred_temp=Output_dic['rho_pred'][i_frames]
        # print(f'Updating the prediction{rho_pred_temp} at i_time{i_time} i_frame{i_frames} \n' )
        
        RHS_update.g_back['Es'] = stimuli_test*1.0+g_back['Es']
        RHS_update.g_back['S'] = stimuli_ave_learn*Input_S_ratio*1.0+g_back['S']
        Int_pop.rate=5.0+  Output_dic['rho_pred'][i_frames]  * 5.0       
         
        
        RHS_update.g_back['Es'] +=stimuli_inc_back   # So that the baseline are similar


    
    
    # RHS update. Calculate the post-synaptic conductance based on connectivity.
    E_pop.gE_soma= RHS_update.g_pyr_somaE(E_pop.h)
    E_pop.gEN_soma= RHS_update.g_pyr_somaEN(E_pop.h)
    E_pop.gI_soma= RHS_update.g_pyr_somaI(P_pop.h*WpInhFlag)


    E_pop.gE_dend= RHS_update.g_pyr_dendE(Int_pop.h)
    E_pop.gEN_dend= RHS_update.g_pyr_dendEN(Int_pop.hN)
    E_pop.gI_dend= RHS_update.g_pyr_dendI(S_pop.h*WsInhFlag)    
    
    #Int_pop.g= RHS_update.g_Int(E_pop.h, Int_pop.h)
    #Int_pop.gN= RHS_update.g_IntN(E_pop.hN, Int_pop.hN)
    #Int_pop.input=stimuli[:,i_time]

    P_pop.gE= RHS_update.g_pv_E(E_pop.h, Int_pop.h)    
    P_pop.gEN= RHS_update.g_pv_EN(E_pop.hN, Int_pop.hN)    
    P_pop.gI= RHS_update.g_pv_I(P_pop.h, S_pop.h) 


    S_pop.gE= RHS_update.g_sst_E(E_pop.h, Int_pop.h)   
    S_pop.gEN= RHS_update.g_sst_EN(E_pop.hN, Int_pop.hN)    
    S_pop.gI= RHS_update.g_sst_I(P_pop.h) 

    # LHS update all the conductance.
    E_pop.h= LHS_update_E.h(E_pop.h,E_pop.rate)
    E_pop.hN= LHS_update_E.hNMDA(E_pop.h,E_pop.rate)
    
    Int_pop.h= LHS_update_Int.h(Int_pop.h,Int_pop.rate)
    Int_pop.hN= LHS_update_Int.hNMDA(Int_pop.hN,Int_pop.rate)

    P_pop.h= LHS_update_P.h(P_pop.h,P_pop.rate)
    
    
    S_pop.h= LHS_update_S.h(S_pop.h,S_pop.rate)

    # LHS update all the firing rates.
    E_pop.rate, E_pop.Vdend, Isd, Isum= LHS_update_E.rate(E_pop )
    P_pop.rate, p_total_current=LHS_update_P.rate(P_pop )


    #Int_pop.rate, Int_total_current=LHS_update_Int.rate(Int_pop )
    S_pop.rate, S_total_current=LHS_update_S.rate(S_pop )

    
    # saving. 
    if i_time % save_interval == 0 and i_frames < n_frames:
        # print(f'saving at i_time{i_time} and i_frames{i_frames}\n' )

        Output_dic['W_EdS'][:, i_frames] = np.diag(W_dict['EdS']).copy()
        Output_dic['W_EP'][:, i_frames] = np.diag(W_dict['EP']).copy()

     
        Output_dic['E_pop_rate'][:,i_frames]=E_pop.rate
        Output_dic['E_pop_Vdend'][:,i_frames]=E_pop.Vdend
        Output_dic['E_pop_Isd'][:,i_frames] = Isd
        Output_dic['E_pop_Isum'][:,i_frames] = Isum
    
        Output_dic['P_pop_rate'][:,i_frames]=P_pop.rate
        Output_dic['P_pop_sumI'][:,i_frames]=p_total_current

        
        Output_dic['Int_pop_rate'][:,i_frames]=Int_pop.rate
        #Output_dic['Int_pop_sumI'][:,i_frames]=Int_total_current
        
        Output_dic['S_pop_rate'][:,i_frames]=S_pop.rate
        Output_dic['S_pop_sumI'][:,i_frames]=S_total_current
    
        Output_dic['g_E_sE'][:,i_frames]=E_pop.gE_soma
        Output_dic['g_E_sI'][:,i_frames]=E_pop.gI_soma
        Output_dic['g_E_dE'][:,i_frames]=E_pop.gE_dend
        Output_dic['g_E_dI'][:,i_frames]=E_pop.gI_dend
    
        Output_dic['g_E_sEN'][:,i_frames]=E_pop.gEN_soma
        Output_dic['g_E_dEN'][:,i_frames]=E_pop.gEN_dend
    
        Output_dic['g_Int'][:,i_frames]=Int_pop.g
        Output_dic['g_Int_N'][:,i_frames]=Int_pop.gN

    
        Output_dic['g_P_E'][:,i_frames]=P_pop.gE
        Output_dic['g_P_EN'][:,i_frames]=P_pop.gEN
        Output_dic['g_P_I'][:,i_frames]=P_pop.gI
        Output_dic['g_S_E'][:,i_frames]=S_pop.gE
        Output_dic['g_S_EN'][:,i_frames]=S_pop.gEN
        Output_dic['g_S_I'][:,i_frames]=S_pop.gI
        
        Output_dic['h_E'][:,i_frames]=E_pop.h
        Output_dic['h_EN'][:,i_frames]=E_pop.hN
        Output_dic['h_Int'][:,i_frames]=Int_pop.h
        Output_dic['h_IntN'][:,i_frames]=Int_pop.hN

        Output_dic['error'][i_frames]=learning_protocol
        Output_dic['tested_stimulus'][:,i_frames]=stimuli_test
        Output_dic['shuffle_stim_indices'][:,i_frames]=shuffle_stim_indices
        i_frames+=1
    

### output, maybe later analyze in a separate code. 
checkind_0=20                         
t_axis=np.arange(ndt)*params_sim['dt']

# Assume Output_dic['tested_stimulus'] has shape (N_stimuli, N_frames)


import AnalysisTool as AT
import matplotlib.pyplot as plt
from scipy.stats import sem  # standard error of the mean

PltTool=AT.Tools(Output_dic, ndt,n_frames, params_sim['dt'], params_stim, 
                         Plot_flag=True, save_dir=FigPath+'\\')  # Can add a output directory at the end.
'''
PltTool=AT.Tools(Output_dic, ndt,n_frames, params_sim['dt'], params_stim, True, '../figs/SaturatePlasticity/')  # Can add a output directory at the end.
'''

corr_stim = PltTool.column_vs_matrix_corr(stimuli_inc_learn, Output_dic['tested_stimulus'])


corr_resp = PltTool.column_vs_matrix_corr(Output_dic['E_pop_rate'][:,2], Output_dic['E_pop_rate'][:,2::2])
corr_resp_mod = PltTool.column_vs_matrix_corr(Output_dic['E_pop_rate'][:,1], Output_dic['E_pop_rate'][:,3::2])  # with prediction

corr_resp_ctrl = PltTool.column_vs_matrix_corr(Output_dic['E_pop_rate'][:,4], Output_dic['E_pop_rate'][:,2::2])
corr_resp_mod_ctrl = PltTool.column_vs_matrix_corr(Output_dic['E_pop_rate'][:,3], Output_dic['E_pop_rate'][:,1::2])  # with prediction

dprime_resp = PltTool.column_vs_matrix_fisher(Output_dic['E_pop_rate'][:,2], Output_dic['E_pop_rate'][:,4::2])
dprime_resp_mod = PltTool.column_vs_matrix_fisher(Output_dic['E_pop_rate'][:,3], Output_dic['E_pop_rate'][:,5::2])  # with prediction

dprime_resp_ctrl = PltTool.column_vs_matrix_fisher(Output_dic['E_pop_rate'][:,4], Output_dic['E_pop_rate'][:,2::2])
dprime_resp_mod_ctrl = PltTool.column_vs_matrix_fisher(Output_dic['E_pop_rate'][:,5], Output_dic['E_pop_rate'][:,3::2])  # with prediction

all_dprime_ctrl = []
all_dprime_mod_ctrl = []

# 2,3 trained, 4,5; 6,7 ... tested.
for i_ctrl in range(6, Output_dic['E_pop_rate'].shape[1], 2):  # Replace with your actual iterable
    temp_cols=list( range(6, Output_dic['E_pop_rate'].shape[1], 2))
    temp_cols.remove(i_ctrl)
    even_cols=np.array(temp_cols)
    dprime_resp_ctrl = PltTool.column_vs_matrix_fisher(
        Output_dic['E_pop_rate'][:, i_ctrl],
        Output_dic['E_pop_rate'][:, even_cols]
    )
    
    temp_cols= list( range(5, Output_dic['E_pop_rate'].shape[1], 2))
    temp_cols.remove(i_ctrl-1)
    odd_cols=np.array(temp_cols)
    dprime_resp_mod_ctrl = PltTool.column_vs_matrix_fisher(
        Output_dic['E_pop_rate'][:, i_ctrl-1],
        Output_dic['E_pop_rate'][:, odd_cols]
    )
    all_dprime_ctrl.append(dprime_resp_ctrl)
    all_dprime_mod_ctrl.append(dprime_resp_mod_ctrl)

# Flatten into single vectors
all_dprime_ctrl = np.concatenate(all_dprime_ctrl)
all_dprime_mod_ctrl = np.concatenate(all_dprime_mod_ctrl)

# Collect data in order
groups = [dprime_resp, all_dprime_ctrl, dprime_resp_mod, dprime_resp_mod_ctrl]
group_labels = ['trained Stim', 'ctrl', 'mod_trained_stim', 'mod_ctrl_stim']

# Compute means and SEMs
means = [np.mean(g) for g in groups]
errors = [sem(g) for g in groups]

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(group_labels, means, yerr=errors, capsize=5)

# Aesthetics
plt.ylabel('d-prime')
plt.title('Comparison of d-prime across conditions')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()






PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[4,6,2]],
                           labels=['4','6','2'],
                           colors=['#31a354','#3182bd','#e34a33'],
                           name='InitalSnapshots')
PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[3,5,1]],
                           labels=['3','5','1'],
                           colors=['#31a354','#3182bd','#e34a33'],
                           name='InitalSnapshots')


PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[3,4]],
                           labels=['3','4',],
                           colors=['#31a354','#3182bd'],
                           name='InitalSnapshots')

PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[0,2,3]],
                           labels=['0','2','3'],
                           colors=['#31a354','#3182bd','#e34a33'],
                           name='InitalSnapshots')
PltTool.Plot_2Snapshots(Output_dic['W_EdS'][:,1],Output_dic['W_EdS'][:,n_frames-1],'W_EdSLearning')
PltTool.Plot_2Snapshots(Output_dic['W_EP'][:,1],Output_dic['W_EP'][:,n_frames-1],'W_EPLearning')



#'''
# saving the trained network.
import pickle

para_save=params_stim.copy()
para_save['ndt']=ndt
para_save['n_frames']=n_frames
para_save['dt']=params_sim['dt']
para_save['stimuli_ave_learn']=stimuli_ave_learn
para_save['shuffle_pred_indices']=shuffle_input_indices



'''
# Open a file for writing. The 'wb' parameter denotes 'write binary'
with open(TempPath+'\Output.pkl', 'wb') as f:
    pickle.dump(Output_dic, f)
    pickle.dump(para_save, f)

# copying the trained network in the following run
with open(parent_dir+'\\net\\nPEcorr\\Output_trained.pkl', 'wb') as f:
with open(parent_dir+'\\net\\nPEcorr\\Output_initial.pkl', 'wb') as f:
    pickle.dump(Output_dic, f)
    pickle.dump(para_save, f)


# copying the trained network in the following run
with open(parent_dir+'\\net\\Output.pkl', 'wb') as f:
with open(parent_dir+'\\net\\Output_noHomeoN200.pkl', 'wb') as f:
    pickle.dump(Output_dic, f)
    pickle.dump(para_save, f)
'''


'''
# Random control test.
with open(parent_dir+'\\net\\Output_rand\\Output0.pkl', 'rb') as f:
    dict_rand = pickle.load(f)  
    para_rand= pickle.load(f)

PltTool.PlotPyrTimes_randomcontrol(dict_rand)
PltTool.PlotPyrTimesDif_randomcontrol(dict_rand)

PltTool.Plot_Dict(dict_rand['Int_pop_rate'][6,:],name='IntRandom')

n_frame2=my_para_loaded['n_frames']
PltTool.Plot_Snapshots(my_dict_loaded['E_pop_rate'][:,n_frame2-3],my_dict_loaded['E_pop_rate'][:,n_frame2-2],my_dict_loaded['E_pop_rate'][:,n_frame2-1],'After Learning')
PltTool.Plot_Snapshots(my_dict_loaded['E_pop_rate'][:,2],my_dict_loaded['E_pop_rate'][:,3],my_dict_loaded['E_pop_rate'][:,4],'Before Learning')

'''
recovered_pred_indices = np.argsort(shuffle_input_indices)
PltTool.Plot_FixframeInt(my_dict_loaded['W_EdS'][recovered_pred_indices,:],200,name='W_Eds_frames')
PltTool.Plot_FixframeInt(my_dict_loaded['W_EP'],200,name='W_EP_frames_raw')


recovered_SP_indices=np.argsort(Output_dic['shuffle_stim_indices'][:,4])
PltTool.Plot_FixframeInt(my_dict_loaded['W_EP'][recovered_SP_indices,:],200,name='W_EP_frames_SP')

#2,3 trained   #4,5 prob   #2,4 stim, #3, 5 modulated
PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[2,4]],
                           labels=['Expected Stim','Probed Stim'],
                           linestyles=['-','-'],
                           colors=['#31a354','#feb24c'],
                           name='SnapshotsStimli',
                           ymax=25,   linewidth=0.7,flag_scatter=True)

PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[3,5]],
                           labels=['',''],
                           linestyles=['-','-'],
                           colors=['#31a354','#feb24c'],
                           name='SnapshotsPwithStimli',
                           ymax=25,   linewidth=0.7,flag_scatter=True)

# Need to include baseline, indeed.
MIdata=Output_dic['E_pop_rate'][:,[3, 5, 2,4, 1]  ]

PltTool.ModulationIndex(MIdata,labels=('Expected','Probed'),
                        start_color='#31a354',end_color='#feb24c')

'''
Here are the plots that need regeneratation
Output_dic_learned=Output_dic
PltTool_learned=PltTool

# comparing expected stim in the intial or trianed network
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,4],Output_dic_learned['E_pop_rate'][:,4],
                          start_color='grey',end_color='#feb24c',labels=('Pre','Post'),
                          name='Probed Stim',bin_width=0.5)
Probe_stim_index=5
PltTool.Plot_overlay_hist(Output_dic['E_pop_rate'][:,Probe_stim_index],Output_dic_learned['E_pop_rate'][:,Probe_stim_index],
                          start_color='grey',end_color='#feb24c',labels=('Pre','Post'),
                          name='Probed Stim+ mod',bin_width=0.5)

# Now the dprime and change

dprime_resp_learned = PltTool.column_vs_matrix_fisher(Output_dic_learned['E_pop_rate'][:,2], Output_dic_learned['E_pop_rate'][:,4::2])

all_dprime_ctrl_learned = []
# 2,3 trained, 4,5; 6,7 ... tested.
for i_ctrl in range(6, Output_dic_learned['E_pop_rate'].shape[1], 2):  # Replace with your actual iterable
    temp_cols=list( range(6, Output_dic_learned['E_pop_rate'].shape[1], 2))
    temp_cols.remove(i_ctrl)
    even_cols=np.array(temp_cols)
    dprime_resp_ctrl = PltTool.column_vs_matrix_fisher(
        Output_dic_learned['E_pop_rate'][:, i_ctrl],
        Output_dic_learned['E_pop_rate'][:, even_cols]
    )
    
    all_dprime_ctrl_learned.append(dprime_resp_ctrl)

# Flatten into single vectors
# all_dprime_ctrl_learned = np.concatenate(all_dprime_ctrl_learned)

# So here are the four learned catogory. Looking for pairwise discriminality.
dprime_resp, all_dprime_ctrl, dprime_resp_learned, all_dprime_ctrl_learned
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from itertools import combinations

group_names = ['Resp', 'Ctrl', 'Resp_Learned', 'Ctrl_Learned']
data_groups = [dprime_resp, all_dprime_ctrl, dprime_resp_learned, all_dprime_ctrl_learned]

# Compute means and SEMs
means = [np.mean(g) for g in data_groups]
sems = [np.std(g, ddof=1) / np.sqrt(len(g)) for g in data_groups]
colors = ['#31a354', '#feb24c', '#31a354', '#feb24c']

# Plot
fig, ax = plt.subplots(figsize=(2.4, 2.4))
bars = ax.bar(range(len(means)), means, yerr=sems, capsize=5, alpha=0.9, color=colors)
ax.set_xticks(range(len(means)))
ax.set_xticklabels(group_names)
ax.set_ylabel("d-prime")
#ax.set_title("Discriminability (d′) across conditions")

# # Annotate p-values for each pair
# y_max = max(means) + max(sems) + 0.1
# offset = 100
# for i, j in combinations(range(len(data_groups)), 2):
#     # T-test
#     stat, p = ttest_ind(data_groups[i], data_groups[j], equal_var=False)
#     # Annotation
#     x = (i + j) / 2
#     height = y_max + offset * abs(i - j)+offset/5*i
#     ax.plot([i, i, j, j], [height, height + 0.02, height + 0.02, height], lw=1.5, c='k')
#     ax.text(x, height + 0.025, f"p = {p:.3g}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
fig.savefig(PltTool_learned.sdir+'dprimeAcrossConds'+'.jpg',bbox_inches='tight')
fig.savefig(PltTool_learned.sdir+'dprimeAcrossConds'+'.svg',bbox_inches='tight')

'''




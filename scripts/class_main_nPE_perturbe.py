# -*- coding: utf-8 -*-
"""
Modified on 09192024 to do the perturbation on nPE test to have a curve. 
Now I have include the pPE test by removing the  stimuli. Test again.
@author: John Meng
"""
# Certain control parameters. Should set up better later.
ratio_SST_perturb=0
ratio_PV_perturb=0
ratio_NMDA_perturb= 0

class nPE_model:
    def main(ratio_SST_perturb=0.0,ratio_PV_perturb=0.0,ratio_NMDA_perturb= 0.0,
             Trained_Pathway='\\net\\Output.pkl', FrameUsed=-1,Plot_flag=True,
             flag_homeo=True,tag=''):
        
        
        alpha = 0.1  # 0.3 at 0508 Learning rate for matching case
        eta_S=1  # Learning rate on WEd_S bias
        beta = 500   # Leaking term for the Hebbian learning. About right.
        gamma = 0.04  # 0.05 at 0508 Learning rate for anti-Hebbian learning
        gamma2 =0.04  # 0.2 at 0508
        r_saturate=5
        totalTime = 5.01 # Put it here for easy modification
        jitter=0.00
        
        WsInhFlag = 1 
        WpInhFlag = 1
        sigma_W=0.1  
        
        Input_S_ratio=1
        
        kappa=0.3             # default 0.5. Requires a fine tuning.
        Int_baseline=5      # I should collect the 1st time frame rate as baseline firing rate.
        E_baseline= 2
        E_anti_baseline=2
        
        
        ####################
        # Same trivial setup for input and output function
        
        import os
        import sys
        import importlib
        from datetime import date,datetime
        from tqdm import tqdm
        
        current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        
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
        
        
        # Format the ratios to 1 decimal place
        formatted_SST = f"{ratio_SST_perturb:.1f}"
        formatted_PV = f"{ratio_PV_perturb:.1f}"
        formatted_NMDA = f"{ratio_NMDA_perturb:.1f}"
        
        FigPath = os.path.join(parent_dir, f'figs\\nPEtests\\nPEtest_SST{formatted_SST}_PV{formatted_PV}_NMDA{formatted_NMDA}'+tag)
        
        if os.path.isdir(FigPath)==False:
            os.makedirs(FigPath)
            
            
        filename = os.path.basename(__file__)
        
        from shutil import copyfile      # Save a copy of the code to the saving folder.
        copyfile(filename,TempPath+'\\'+filename)
        para_loc='..\lib\\'
        
        #copyfile(para_loc+'parameters_tuningcurve.py',TempPath+'\parameters_tuningcurve.py')
        #copyfile('main_nPEtest.py',TempPath+'\main_nPEtest.py')
        
        import basic_functions as bf
        import parameters_tuningcurve as params
        import data_structure
        import update_functions
        import numpy as np
        from collections import OrderedDict
        import matplotlib.pyplot as plt
        import pickle
        from numpy.random import default_rng
        
        with open(parent_dir+Trained_Pathway, 'rb') as f:
            my_dict_loaded = pickle.load(f)  
            my_para_loaded= pickle.load(f)
        
        
        
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
        
        ndt=round(params_sim['t_total']/params_sim['dt'])
        ndt_per_trial=round((params_stim['Tinter']+params_stim['Tstim'])/params_sim['dt'])
        ndt_rest= round(params_stim['Tresting']/params_sim['dt'])
        params_stim['t_total']=params_sim['t_total']
        
        N_column=1
        N_eachC=my_para_loaded['N_neu']
        N_neu=N_column*N_eachC
        params_stim['num_perG'] = N_neu   # Number of cells in each group. 
        
        N_column=int(round(N_column))
        N_eachC=int(round(N_eachC))
        N_neu=int(round(N_neu))
        
        #my_para_loaded['n_frames']-1
        Trained_W_EdS=my_dict_loaded['W_EdS'][:,FrameUsed];
        Trained_W_EP=my_dict_loaded['W_EP'][:, FrameUsed];
        stimuli_inc_base_eachC = np.linspace(
            params_stim['str_g_back']*params_stim['ratio_min'], params_stim['str_g_back']*params_stim['ratio_max'], N_eachC)
        stimuli_inc_base=np.tile(stimuli_inc_base_eachC,N_column)
        
        
        
        
        
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
        if not flag_homeo:
            np.random.seed(5)
            shuffle_input_indices = np.random.permutation(N_eachC)
            stimuli_inc_back=stimuli_inc_back[shuffle_input_indices]
        
        
        jitter_vec=np.random.uniform(1 - jitter, 1 + jitter, size=N_eachC)
        stimuli_ave_learn = np.ones(N_eachC)*params_stim['str_g']*params_stim['SST_ratio']*jitter_vec
        
        '''
        stimuli_dec_learn = np.linspace(params_stim['str_g']*params_stim['ratio_max'],
                                        params_stim['str_g']*params_stim['ratio_min'], N_eachC)*Ratio_topstimuli
        '''
        
        
        
        
        
        #### Now setting up the single cells. Using the structure set up in data_structure.py.
        # Notice the STP related features are saved heretoo
        E_pop= data_structure.Pyramidal(params_stim['N_neu'])  # Initial pyramidal cells data strcuture. Including rate(some), vdend
        Int_pop = data_structure.Integrator(params_stim['N_column'])
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
        fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
        im=ax.imshow(W_dict['PE'])    
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()
        '''
        rng1 = default_rng(1)
        rng2 = default_rng(2)
        # Randomizing here
        #W_dict['EP'] =  bf.load_ringbulk_connectivity(
        #    Trained_W_EP,N_column,type='BulktoBulk_diag')
        W_dict['EP'] =  np.diag(Trained_W_EP)   

        W_dict['PP'] =0 
        W_dict['SP'] =0
        
        #W_dict['EP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_soma']*Wscale*WPVInhScale,params_P['sigma_to_pc'])  # Sigma=20. Need to study later. Diagonal in Kevin code
        #W_dict['PP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_pv']*Wscale,params_P['sigma_to_pv'])  #  Sigma=20
        #W_dict['SP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_sst']*Wscale,params_P['sigma_to_sst'])
        
        '''
        fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
        im=ax.imshow(W_dict['EP'])   
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_title('From P to E')   
        plt.show()
        '''
        '''
        fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
        im=ax.imshow(W_dict['PP'])  
        cbar = ax.figure.colorbar(im, ax=ax)  
        plt.show()
        '''
        #W_dict['EdS']=bf.load_ringbulk_connectivity(Trained_W_EdS,N_column,type='BulktoBulk_diag')
        W_dict['EdS'] =  np.diag(Trained_W_EdS)   
        mean_EdS_value = params_S['weight_to_dend']*100 #N_ref=100
        
        
        #W_dict['EdS']= bf.create_delta_connectivity(params_S['weight_to_dend'],N_neu)*flag_StoE
        #W_dict['EdS']= bf.create_ring_connectivity(N_neu,N_neu,params_S['weight_to_dend']*WSSTInhScale,params_S['sigma_to_pc'])
        
        W_dict['PS']= 0
        #W_dict['PS']= bf.create_ring_connectivity(N_neu,N_neu,params_S['weight_to_pv'],params_S['sigma_to_pv'])
        
        # Here I should do more systematically, so far just directly change it.
        W_dict['EdInt']= bf.create_ringbulk_connectivity(
            params_Int['weight_to_dend']*params_Int['scaleMax'],params_Int['weight_to_dend']*params_Int['scaleMin'],
            N_col=N_column, N_eachc=N_eachC, type='InttoBulk')
        if not flag_homeo:
            W_dict['EdInt']= bf.create_ringbulk_connectivity(
            params_Int['weight_to_dend']*params_Int['scaleMax'],params_Int['weight_to_dend']*params_Int['scaleMin'],
            N_col=N_column, N_eachc=N_eachC, type='InttoBulk',shuffle_inds=shuffle_input_indices)
                
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
        im=ax.imshow(W_dict['EdS'],aspect='auto') 
        
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.title('Connectivity S to Ed')
        
        plt.show()
        fig, ax =  plt.subplots(dpi=250,figsize=(4,3))
        #im=ax.imshow(W_dict['PInt'],aspect='auto') 
        ax.plot(Trained_W_EP,'b')
        ax.plot(Trained_W_EdS,'g')
        ax.plot(W_dict['EdInt'],'r')
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.title('Connectivity P,S to E')
        
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
        
        save_interval = 0.1 / params_sim['dt']
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
            )
        
        
        ### Initiate the updating functions, using update_functions.py
        RHS_update= update_functions.RHS_update(W_dict, g_back)
        RHS_update.g_back['Es'] +=stimuli_inc_back
        RHS_update.g_back['S'] = np.maximum(0,RHS_update.g_back['S'] +(ratio_SST_perturb)*max(stimuli_ave_learn))    
        RHS_update.g_back['P'] = np.maximum(0,g_back['P'] +  ratio_PV_perturb*max(stimuli_ave_learn))  
        Int_pop.rate=5.0 

        LHS_update_E= update_functions.LHS_update.Pyr(params_E,params_sim['dt'])
        LHS_update_Int= update_functions.LHS_update.Int(params_Int,params_sim['dt'])
        
        LHS_update_P= update_functions.LHS_update.P(params_P,params_sim['dt'])
        
        LHS_update_S= update_functions.LHS_update.S(params_S,params_sim['dt'])
        
        
        ### main loop. Noticing currently, I don't include NMDA, STP, LTP, Adaptation.
        #t_PVadvance=int(0.05//params_sim['dt']) 
        
        for i_time in tqdm(range(ndt)):
        #for i_time in tqdm(range(int(3/params_sim['dt']))):    
            # tau_I=0, so it's calculated as RHS. 
            # conductance (pre-synaptic) depends on the firing rate of the same cell. Update second
            # rate, doesn't depend on conductance. So update third.
            if (i_time-1) % save_interval == 0 and i_frames < n_frames and i_time > 10*save_interval:
                # For every save_interval, change the tested interval and learning rule.
        
                #learning_protocol= random.randint(0,2)
                learning_protocol = 0   # Just testing on the matching case.
                        
                if i_frames>=20 and i_frames<30 :
                    learning_protocol = 2
        
                if learning_protocol == 0:   # Matching learning. Needs to improve more
                    RHS_update.g_back['Es'] = stimuli_inc_learn+g_back['Es']
                    RHS_update.g_back['S'] = stimuli_ave_learn*Input_S_ratio+g_back['S']
                    Int_pop.rate=10.0
                elif learning_protocol == 1:  # Stimuli only
                    RHS_update.g_back['Es'] = stimuli_inc_learn+g_back['Es']
                    RHS_update.g_back['S'] = stimuli_ave_learn*Input_S_ratio+g_back['S']
                    Int_pop.rate=Int_baseline
                elif learning_protocol == 2:  # Prediction only
                    RHS_update.g_back['Es'] = g_back['Es']   
                    RHS_update.g_back['S'] = g_back['S'] 
                    Int_pop.rate=10.0
                RHS_update.g_back['Es'] +=stimuli_inc_back   # So that the baseline are similar
                RHS_update.g_back['S'] = np.maximum(0,RHS_update.g_back['S'] +(ratio_SST_perturb)*max(stimuli_ave_learn))    
                RHS_update.g_back['P'] = np.maximum(0,g_back['P'] +  ratio_PV_perturb*max(stimuli_ave_learn))  
    
            '''
            # Input
            RHS_update.g_back['Es'] = stimuli_inc[:,i_time]+g_back['Es']
            RHS_update.g_back['Ed'] = g_back['Ed']*background_inc_ratio
            RHS_update.g_back['S'] = stimuli_dec_SST[:,i_time]+g_back['S']
            '''
            
        
            
            # RHS update. Calculate the post-synaptic conductance based on connectivity.
            E_pop.gE_soma= RHS_update.g_pyr_somaE(E_pop.h)
            E_pop.gEN_soma= RHS_update.g_pyr_somaEN(E_pop.h)*(1-ratio_NMDA_perturb)
            #E_pop.gI_soma= RHS_update.g_pyr_somaI(P_pop.h*WpInhFlag,P0_pop.h*Wp0InhFlag)
            E_pop.gI_soma= RHS_update.g_pyr_somaI(P_pop.h*WpInhFlag)
        
            E_pop.gE_dend= RHS_update.g_pyr_dendE(Int_pop.h)
            E_pop.gEN_dend= RHS_update.g_pyr_dendEN(Int_pop.hN)*(1-ratio_NMDA_perturb)
            E_pop.gI_dend= RHS_update.g_pyr_dendI(S_pop.h*WsInhFlag)    
            
            #Int_pop.g= RHS_update.g_Int(E_pop.h, Int_pop.h)
            #Int_pop.gN= RHS_update.g_IntN(E_pop.hN, Int_pop.hN)
            #Int_pop.input=stimuli[:,i_time]
        
            P_pop.gE= RHS_update.g_pv_E(E_pop.h, Int_pop.h)    
            P_pop.gEN= RHS_update.g_pv_EN(E_pop.hN, Int_pop.hN)*(1-ratio_NMDA_perturb)    
            P_pop.gI= RHS_update.g_pv_I(P_pop.h, S_pop.h) 
        
        
            S_pop.gE= RHS_update.g_sst_E(E_pop.h, Int_pop.h)   
            S_pop.gEN= RHS_update.g_sst_EN(E_pop.hN, Int_pop.hN)*(1-ratio_NMDA_perturb)    
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
                W_EP = np.diag(W_dict['EP']).copy()
                W_EdS = np.diag(W_dict['EdS']).copy()
                dWEP = 0
                dWEds = 0
                #if i_frames==1:   # update the mean value. 
                    #E_baseline= np.mean(E_pop.rate)
                    #E_anti_baseline= 0
                '''
                if not(i_time == 0):  # learning rate, varying the connectivity matrix.
                    print(f"\n i_time = {i_time}, updating at {learning_protocol}")
                    if learning_protocol == 0:   # Matching learning. Needs to improve more
                        print('learning matching')
                        # Learning is propotional to the existing weight.
                        dWEds = (E_pop.rate-E_baseline)*S_pop.rate + (-W_EdS+mean_EdS_value/2) / beta
                        dWEP =  (E_pop.rate-E_baseline)*P_pop.rate + (-W_EP+mean_EP_value/8) / beta
                        W_EdS += alpha*dWEds*eta_S
                        W_EP  += alpha*dWEP
                        W_dict['EdS'] = np.diag(bf.relu(W_EdS))
                        W_dict['EP'] = np.diag(bf.relu(W_EP))
                    elif learning_protocol == 1:  # Stimuli only
                        # beta is a leaking term for satuation
                        dWEds = -(E_pop.rate-E_anti_baseline)*S_pop.rate + (-W_EdS+mean_EdS_value/2) / beta
                        dWEP = - (E_pop.rate-E_anti_baseline)*P_pop.rate + (-W_EP+mean_EP_value/8) / beta
                        W_EdS += gamma*dWEds*eta_S
                        W_EP  += gamma*dWEP
                        W_dict['EdS'] = np.diag(bf.relu(W_EdS))
                        W_dict['EP'] = np.diag(bf.relu(W_EP))
                    elif learning_protocol == 2:  # Prediction only
                        dWEds = -(E_pop.rate-E_anti_baseline)*S_pop.rate + (-W_EdS+mean_EdS_value/2) / beta
                        dWEP = -(E_pop.rate-E_anti_baseline)*P_pop.rate + (-W_EP+mean_EP_value/8) / beta
                        W_EdS += gamma2*dWEds*eta_S
                        W_EP  += gamma2*dWEP
                        W_dict['EdS'] = np.diag(bf.relu(W_EdS))
                        W_dict['EP'] = np.diag(bf.relu(W_EP))
                '''        
                RHS_update.W_dict = W_dict    # Update the update RHS_update
        
                Output_dic['W_EdS'][:, i_frames] = np.diag(W_dict['EdS']).copy()
                Output_dic['W_EP'][:, i_frames] = np.diag(W_dict['EP']).copy()
                Output_dic['dWEds'][:, i_frames] = dWEds
                Output_dic['dWEP'][:, i_frames] = dWEP
             
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
        
                i_frames+=1
            

        import AnalysisTool as AT
        #PltTool=AT.Tools(Output_dic, ndt,n_frames, params_sim['dt'], params_stim, Plot_flag=True, save_dir='../figs/nPEtest/')  # Can add a output directory at the end.
        PltTool=AT.Tools(Output_dic, ndt,n_frames, params_sim['dt'], params_stim, 
                         Plot_flag=Plot_flag, save_dir=FigPath+'\\')  # Can add a output directory at the end.
        PltTool.PlotPyrPop_Keller(start_color='#62BC43',end_color='#F7A541')
        PltTool.PlotP_current(shift_id=-N_eachC//2)
        PltTool.PlotS_current(shift_id=-N_eachC//4)
        PltTool.PlotPyr_current(shift_id=N_eachC//2-1)
        #PltTool.PlotPyr_conductance(shift_id=N_eachC//2-1,Rmax=24,DEmax=4, SImin=-5,DImin=-10,DImax=2)
        PltTool.PlotPyr_current(shift_id=-N_eachC//2)
        #PltTool.PlotPyr_conductance(shift_id=-N_eachC//2,Rmax=24,DEmax=4, SImin=-5,DImin=-10,DImax=2)        
        trace=Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,n_frames-1]
        PltTool.Plot_Snapshot_nPE(Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,n_frames-1] ,'E_pop nPE')
        PltTool.Plot_3Snapshots(Output_dic['E_pop_rate'][:,9],
                        Output_dic['E_pop_rate'][:,n_frames-1],
                        Output_dic['E_pop_rate'][:,29],'CompareEInd',
                        colors=['grey','#e34a33','#3182bd'],
                        labels=['Baseline','Stim + Pred','Pred'])
        PltTool.Plot_nPE_barplot(['Base','Stim + Pred', 'Pred'],
                         [np.mean( Output_dic['E_pop_rate'][:,9]), 
                          np.mean( Output_dic['E_pop_rate'][:,n_frames-1]   ), 
                          np.mean( Output_dic['E_pop_rate'][:,29])  ] ,name='nPEbarplot',
                        colors=['grey','#e34a33','#3182bd'],top=10)
        
        return np.mean(trace), Output_dic,PltTool
    
class pPE_model:
    def main(ratio_SST_perturb=0.0,ratio_PV_perturb=0.0,ratio_NMDA_perturb= 0.0,
             Trained_Pathway='\\net\\Output.pkl',Plot_flag= True, 
             FrameUsed=-1,flag_homeo=True,tag=''):
        
        
        alpha = 0.1  # 0.3 at 0508 Learning rate for matching case
        eta_S=1  # Learning rate on WEd_S bias
        beta = 500   # Leaking term for the Hebbian learning. About right.
        gamma = 0.04  # 0.05 at 0508 Learning rate for anti-Hebbian learning
        gamma2 =0.04  # 0.2 at 0508
        r_saturate=5
        totalTime = 5.01 # Put it here for easy modification
        jitter=0.00
        
        WsInhFlag = 1 
        WpInhFlag = 1
        sigma_W=0.1  # Default 0.3
        
        Input_S_ratio=1
        
        kappa=0.3             # default 0.5. Requires a fine tuning.
        Int_baseline=5      # I should collect the 1st time frame rate as baseline firing rate.
        E_baseline= 2
        E_anti_baseline=2
        
        
        ####################
        # Same trivial setup for input and output function
        
        import os
        import sys
        import importlib
        from datetime import date,datetime
        from tqdm import tqdm
        
        current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        
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
        
        
        # Format the ratios to 1 decimal place
        formatted_SST = f"{ratio_SST_perturb:.1f}"
        formatted_PV = f"{ratio_PV_perturb:.1f}"
        formatted_NMDA = f"{ratio_NMDA_perturb:.1f}"
        
        FigPath = os.path.join(parent_dir, f'figs\\pPEtests\\pPEtest_SST{formatted_SST}_PV{formatted_PV}_NMDA{formatted_NMDA}'+tag)
        
        if os.path.isdir(FigPath)==False:
            os.makedirs(FigPath)
            
            
        filename = os.path.basename(__file__)
        
        from shutil import copyfile      # Save a copy of the code to the saving folder.
        copyfile(filename,TempPath+'\\'+filename)
        para_loc='..\lib\\'
        
       
        import basic_functions as bf
        import parameters_tuningcurve as params
        import data_structure
        import update_functions
        import numpy as np
        from collections import OrderedDict
        import matplotlib.pyplot as plt
        import pickle
        from numpy.random import default_rng
        
        with open(parent_dir+Trained_Pathway, 'rb') as f:
            my_dict_loaded = pickle.load(f)  
            my_para_loaded= pickle.load(f)
        
        
        
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
        
        ndt=round(params_sim['t_total']/params_sim['dt'])
        ndt_per_trial=round((params_stim['Tinter']+params_stim['Tstim'])/params_sim['dt'])
        ndt_rest= round(params_stim['Tresting']/params_sim['dt'])
        params_stim['t_total']=params_sim['t_total']
        
        N_column=1
        N_eachC=my_para_loaded['N_neu']
        N_neu=N_column*N_eachC
        params_stim['num_perG'] = N_neu   # Number of cells in each group. 
        
        N_column=int(round(N_column))
        N_eachC=int(round(N_eachC))
        N_neu=int(round(N_neu))
        
        
        Trained_W_EdS=my_dict_loaded['W_EdS'][:, FrameUsed];
        Trained_W_EP=my_dict_loaded['W_EP'][:, FrameUsed];
        stimuli_inc_base_eachC = np.linspace(
            params_stim['str_g_back']*params_stim['ratio_min'], params_stim['str_g_back']*params_stim['ratio_max'], N_eachC)
        stimuli_inc_base=np.tile(stimuli_inc_base_eachC,N_column)
        
        
        
        
        
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
        if not flag_homeo:
            np.random.seed(5)
            shuffle_input_indices = np.random.permutation(N_eachC)
            stimuli_inc_back=stimuli_inc_back[shuffle_input_indices]
            
        
        jitter_vec=np.random.uniform(1 - jitter, 1 + jitter, size=N_eachC)
        stimuli_ave_learn = np.ones(N_eachC)*params_stim['str_g']*params_stim['SST_ratio']*jitter_vec
        
        '''
        stimuli_dec_learn = np.linspace(params_stim['str_g']*params_stim['ratio_max'],
                                        params_stim['str_g']*params_stim['ratio_min'], N_eachC)*Ratio_topstimuli
        '''
        
        
        
        
        
        #### Now setting up the single cells. Using the structure set up in data_structure.py.
        # Notice the STP related features are saved heretoo
        E_pop= data_structure.Pyramidal(params_stim['N_neu'])  # Initial pyramidal cells data strcuture. Including rate(some), vdend
        Int_pop = data_structure.Integrator(params_stim['N_column'])
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
        fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
        im=ax.imshow(W_dict['PE'])    
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()
        '''
        rng1 = default_rng(1)
        rng2 = default_rng(2)
        # Randomizing here
        W_dict['EP'] =  bf.load_ringbulk_connectivity(
            Trained_W_EP,N_column,type='BulktoBulk_diag')
        
        
        
        W_dict['PP'] =0 
        W_dict['SP'] =0
        
        #W_dict['EP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_soma']*Wscale*WPVInhScale,params_P['sigma_to_pc'])  # Sigma=20. Need to study later. Diagonal in Kevin code
        #W_dict['PP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_pv']*Wscale,params_P['sigma_to_pv'])  #  Sigma=20
        #W_dict['SP'] = bf.create_ring_connectivity(N_neu,N_neu, params_P['weight_to_sst']*Wscale,params_P['sigma_to_sst'])
        
        '''
        fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
        im=ax.imshow(W_dict['EP'])   
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_title('From P to E')   
        plt.show()
        '''
        '''
        fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
        im=ax.imshow(W_dict['PP'])  
        cbar = ax.figure.colorbar(im, ax=ax)  
        plt.show()
        '''
        W_dict['EdS']=bf.load_ringbulk_connectivity(Trained_W_EdS,N_column,type='BulktoBulk_diag')
        
        mean_EdS_value = params_S['weight_to_dend']*100 #N_ref=100
        
        
        #W_dict['EdS']= bf.create_delta_connectivity(params_S['weight_to_dend'],N_neu)*flag_StoE
        #W_dict['EdS']= bf.create_ring_connectivity(N_neu,N_neu,params_S['weight_to_dend']*WSSTInhScale,params_S['sigma_to_pc'])
        
        W_dict['PS']= 0
        #W_dict['PS']= bf.create_ring_connectivity(N_neu,N_neu,params_S['weight_to_pv'],params_S['sigma_to_pv'])
        
        # Here I should do more systematically, so far just directly change it.
        W_dict['EdInt']= bf.create_ringbulk_connectivity(
            params_Int['weight_to_dend']*params_Int['scaleMax'],params_Int['weight_to_dend']*params_Int['scaleMin'],
            N_col=N_column, N_eachc=N_eachC, type='InttoBulk')
        if not flag_homeo:
            W_dict['EdInt']= bf.create_ringbulk_connectivity(
            params_Int['weight_to_dend']*params_Int['scaleMax'],params_Int['weight_to_dend']*params_Int['scaleMin'],
            N_col=N_column, N_eachc=N_eachC, type='InttoBulk',shuffle_inds=shuffle_input_indices)
        
        
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
        
        fig, ax =  plt.subplots(dpi=200,figsize=(2,1.7))
        im=ax.imshow(W_dict['EdS'])   
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_title('From S to Ed')
        plt.show()

        fig, ax =  plt.subplots(dpi=200,figsize=(2,1.7))
        im=ax.imshow(W_dict['EP'])   
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_title('From P to E')
        plt.show()

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
        
        save_interval = 0.1 / params_sim['dt']
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
            )
        
        
        ### Initiate the updating functions, using update_functions.py
        RHS_update= update_functions.RHS_update(W_dict, g_back)
        RHS_update.g_back['Es'] +=stimuli_inc_back
        RHS_update.g_back['S'] = np.maximum(0,RHS_update.g_back['S'] +(ratio_SST_perturb)*max(stimuli_ave_learn))    
        RHS_update.g_back['P'] = np.maximum(0,g_back['P'] +  ratio_PV_perturb*max(stimuli_ave_learn))  
        Int_pop.rate=5.0 

        LHS_update_E= update_functions.LHS_update.Pyr(params_E,params_sim['dt'])
        LHS_update_Int= update_functions.LHS_update.Int(params_Int,params_sim['dt'])
        
        LHS_update_P= update_functions.LHS_update.P(params_P,params_sim['dt'])
        
        LHS_update_S= update_functions.LHS_update.S(params_S,params_sim['dt'])
        
        
        ### main loop. Noticing currently, I don't include NMDA, STP, LTP, Adaptation.
        #t_PVadvance=int(0.05//params_sim['dt']) 
        
        for i_time in tqdm(range(ndt)):
        #for i_time in tqdm(range(int(3/params_sim['dt']))):    
        
            # In each timestep, rate, conductance, and current are updated.
            # tau_I=0, so it's calculated as RHS. 
            # conductance (pre-synaptic) depends on the firing rate of the same cell. Update second
            # rate, doesn't depend on conductance. So update third.
            if (i_time-1) % save_interval == 0 and i_frames < n_frames and i_time > 10*save_interval:
                # For every save_interval, change the tested interval and learning rule.
        
                #learning_protocol= random.randint(0,2)
                learning_protocol = 0   # Just testing on the matching case.
                        
                if i_frames>=20 and i_frames<30 :
                    learning_protocol = 1
        
                if learning_protocol == 0:   # Matching learning. Needs to improve more
                    RHS_update.g_back['Es'] = stimuli_inc_learn+g_back['Es']
                    RHS_update.g_back['S'] = stimuli_ave_learn*Input_S_ratio+g_back['S']
                    Int_pop.rate=10.0
                elif learning_protocol == 1:  # Stimuli only
                    RHS_update.g_back['Es'] = stimuli_inc_learn+g_back['Es']
                    RHS_update.g_back['S'] = stimuli_ave_learn*Input_S_ratio+g_back['S']
                    Int_pop.rate=Int_baseline
                elif learning_protocol == 2:  # Prediction only
                    RHS_update.g_back['Es'] = g_back['Es']   
                    RHS_update.g_back['S'] = g_back['S'] 
                    Int_pop.rate=10.0
                RHS_update.g_back['Es'] +=stimuli_inc_back   # So that the baseline are similar
            
                RHS_update.g_back['S'] = np.maximum(0,RHS_update.g_back['S'] +(ratio_SST_perturb)*max(stimuli_ave_learn))    
                RHS_update.g_back['P'] = np.maximum(0,g_back['P'] +  ratio_PV_perturb*max(stimuli_ave_learn))  
        
            '''
            # Input
            RHS_update.g_back['Es'] = stimuli_inc[:,i_time]+g_back['Es']
            RHS_update.g_back['Ed'] = g_back['Ed']*background_inc_ratio
            RHS_update.g_back['S'] = stimuli_dec_SST[:,i_time]+g_back['S']
            '''
            
        
            
            # RHS update. Calculate the post-synaptic conductance based on connectivity.
            E_pop.gE_soma= RHS_update.g_pyr_somaE(E_pop.h)
            E_pop.gEN_soma= RHS_update.g_pyr_somaEN(E_pop.h)*(1-ratio_NMDA_perturb)
            #E_pop.gI_soma= RHS_update.g_pyr_somaI(P_pop.h*WpInhFlag,P0_pop.h*Wp0InhFlag)
            E_pop.gI_soma= RHS_update.g_pyr_somaI(P_pop.h*WpInhFlag)
        
            E_pop.gE_dend= RHS_update.g_pyr_dendE(Int_pop.h)
            E_pop.gEN_dend= RHS_update.g_pyr_dendEN(Int_pop.hN)*(1-ratio_NMDA_perturb)
            E_pop.gI_dend= RHS_update.g_pyr_dendI(S_pop.h*WsInhFlag)    
            
            #Int_pop.g= RHS_update.g_Int(E_pop.h, Int_pop.h)
            #Int_pop.gN= RHS_update.g_IntN(E_pop.hN, Int_pop.hN)
            #Int_pop.input=stimuli[:,i_time]
        
            P_pop.gE= RHS_update.g_pv_E(E_pop.h, Int_pop.h)    
            P_pop.gEN= RHS_update.g_pv_EN(E_pop.hN, Int_pop.hN)*(1-ratio_NMDA_perturb)    
            P_pop.gI= RHS_update.g_pv_I(P_pop.h, S_pop.h) 

        
            S_pop.gE= RHS_update.g_sst_E(E_pop.h, Int_pop.h)   
            S_pop.gEN= RHS_update.g_sst_EN(E_pop.hN, Int_pop.hN)*(1-ratio_NMDA_perturb)    
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
                W_EP = np.diag(W_dict['EP']).copy()
                W_EdS = np.diag(W_dict['EdS']).copy()
                dWEP = 0
                dWEds = 0
                #if i_frames==1:   # update the mean value. 
                    #E_baseline= np.mean(E_pop.rate)
                    #E_anti_baseline= 0
                '''
                if not(i_time == 0):  # learning rate, varying the connectivity matrix.
                    print(f"\n i_time = {i_time}, updating at {learning_protocol}")
                    if learning_protocol == 0:   # Matching learning. Needs to improve more
                        print('learning matching')
                        # Learning is propotional to the existing weight.
                        dWEds = (E_pop.rate-E_baseline)*S_pop.rate + (-W_EdS+mean_EdS_value/2) / beta
                        dWEP =  (E_pop.rate-E_baseline)*P_pop.rate + (-W_EP+mean_EP_value/8) / beta
                        W_EdS += alpha*dWEds*eta_S
                        W_EP  += alpha*dWEP
                        W_dict['EdS'] = np.diag(bf.relu(W_EdS))
                        W_dict['EP'] = np.diag(bf.relu(W_EP))
                    elif learning_protocol == 1:  # Stimuli only
                        # beta is a leaking term for satuation
                        dWEds = -(E_pop.rate-E_anti_baseline)*S_pop.rate + (-W_EdS+mean_EdS_value/2) / beta
                        dWEP = - (E_pop.rate-E_anti_baseline)*P_pop.rate + (-W_EP+mean_EP_value/8) / beta
                        W_EdS += gamma*dWEds*eta_S
                        W_EP  += gamma*dWEP
                        W_dict['EdS'] = np.diag(bf.relu(W_EdS))
                        W_dict['EP'] = np.diag(bf.relu(W_EP))
                    elif learning_protocol == 2:  # Prediction only
                        dWEds = -(E_pop.rate-E_anti_baseline)*S_pop.rate + (-W_EdS+mean_EdS_value/2) / beta
                        dWEP = -(E_pop.rate-E_anti_baseline)*P_pop.rate + (-W_EP+mean_EP_value/8) / beta
                        W_EdS += gamma2*dWEds*eta_S
                        W_EP  += gamma2*dWEP
                        W_dict['EdS'] = np.diag(bf.relu(W_EdS))
                        W_dict['EP'] = np.diag(bf.relu(W_EP))
                '''        
                RHS_update.W_dict = W_dict    # Update the update RHS_update
        
                Output_dic['W_EdS'][:, i_frames] = np.diag(W_dict['EdS']).copy()
                Output_dic['W_EP'][:, i_frames] = np.diag(W_dict['EP']).copy()
                Output_dic['dWEds'][:, i_frames] = dWEds
                Output_dic['dWEP'][:, i_frames] = dWEP
             
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
        
                i_frames+=1
            

        import AnalysisTool as AT
        #PltTool=AT.Tools(Output_dic, ndt,n_frames, params_sim['dt'], params_stim, Plot_flag=True, save_dir='../figs/nPEtest/')  # Can add a output directory at the end.
        PltTool=AT.Tools(Output_dic, ndt,n_frames, params_sim['dt'], params_stim, 
                         Plot_flag=Plot_flag, save_dir=FigPath+'\\')  # Can add a output directory at the end.
        PltTool.PlotPyrPop_Keller(start_color='#62BC43',end_color='#F7A541')
        PltTool.PlotP_current(shift_id=-N_eachC//2)

        PltTool.PlotS_current(shift_id=-N_eachC//4)
        PltTool.PlotPyr_current(shift_id=N_eachC//2-1)
        #PltTool.PlotPyr_conductance(shift_id=N_eachC//2-1,Rmax=24,DEmax=4, SImin=-5,DImin=-10,DImax=2)
        PltTool.PlotPyr_current(shift_id=-N_eachC//2)
        #PltTool.PlotPyr_conductance(shift_id=-N_eachC//2,Rmax=24,DEmax=4, SImin=-5,DImin=-10,DImax=2)        
        
        trace=Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,n_frames-1]
        
        PltTool.Plot_Snapshot_nPE(Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,n_frames-1] ,'E_pop nPE')
        PltTool.Plot_3Snapshots(Output_dic['E_pop_rate'][:,9],
                        Output_dic['E_pop_rate'][:,n_frames-1],
                        Output_dic['E_pop_rate'][:,29],'CompareEInd',
                        colors=['grey','#e34a33','#31a354'],
                        labels=['Baseline','Stim + Pred','Stim'])
        PltTool.Plot_nPE_barplot(['Base','Stim + Pred', 'Stim'],
                         [np.mean( Output_dic['E_pop_rate'][:,9]), 
                          np.mean( Output_dic['E_pop_rate'][:,n_frames-1]   ), 
                          np.mean( Output_dic['E_pop_rate'][:,29])  ] ,name='pPEbarplot',
                        colors=['grey','#e34a33','#31a354'],top=10)
        
        return np.mean(trace), Output_dic,PltTool
if __name__=="__main__":
    # some parameters can be changed here
    import matplotlib.pyplot as plt
    import numpy as np
    
    #DD, Output_dic,PltTool=nPE_model.main(ratio_SST_perturb=0,
    #                                      Trained_Pathway='\\net\\Output_noHomeoN200.pkl',
    #                                      flag_homeo=False,tag='noHomeoN200')    # can be tested by setting SST_pertube=-0.5
    DD, Output_dic,PltTool=nPE_model.main(ratio_SST_perturb=0)    # can be tested by setting SST_pertube=-0.5
    #DD, Output_dic,PltTool=nPE_model.main(ratio_SST_perturb=0,FrameUsed=0,tag='Unlearned')    # can be tested by setting SST_pertube=-0.5

    Output_dic['E_pop_rate'].shape
    #fig, axes=plt.subplots(1,1,figsize=(6,6))
    #axes.plot(Output_dic['E_pop_rate'][0,:],'k',label='First')
    
    nPE_response=Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,-1]

    #PltTool.PlotPyrPop_Keller(start_color='green')
    PltTool.PlotPyrPop_Keller(start_color='green',cmax=17)  # 17 is value for learned

    PltTool.Plot_Dict(Output_dic['E_pop_rate'][0,:],start_frame=5,name='id0',top=20)
    PltTool.Plot_Dict(Output_dic['E_pop_rate'][-1,:],start_frame=5,name='id-1',top=6)
    PltTool.Plot_Snapshot_nPE(Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,-1] ,'E_pop nPE')
    PltTool.Plot_nPE_barplot(['Base','Stim + Pred', 'Pred'],
                     [np.mean( Output_dic['E_pop_rate'][:,9]), 
                      np.mean( Output_dic['E_pop_rate'][:,-1]   ), 
                      np.mean( Output_dic['E_pop_rate'][:,29])  ] ,name='nPEbarplot',
                    colors=['grey','#e34a33','#3182bd'])
    '''
    DD, Output_dic,PltTool=pPE_model.main(ratio_SST_perturb=0)    # can be tested by setting SST_pertube=-0.5
    

    PltTool.PlotPyrPop_Keller(start_color='green')
    PltTool.Plot_Dict(Output_dic['E_pop_rate'][0,:],start_frame=5,name='id0')
    PltTool.Plot_Dict(Output_dic['E_pop_rate'][-1,:],start_frame=5,name='id-1')
    
    PltTool.Plot_Snapshot_nPE(Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,-1] ,'E_pop nPE')
    
    PltTool.Plot_nPE_barplot(['Base','Stim + Pred', 'Pred'],
                     [np.mean( Output_dic['E_pop_rate'][:,9]), 
                      np.mean( Output_dic['E_pop_rate'][:,-1]   ), 
                      np.mean( Output_dic['E_pop_rate'][:,29])  ] ,name='pPEbarplot',
                    colors=['grey','#e34a33','#31a354'],top=5)    
    
    pPE_response=Output_dic['E_pop_rate'][:,29]- Output_dic['E_pop_rate'][:,-1]
    '''
    
'''
# copying the trained network in the following run
with open(parent_dir+'\\net\\Output.pkl', 'wb') as f:
cdata=nPE_response
with open(parent_dir+'\\net\\Output_nPErespnseNoHomeoN200.pkl', 'wb') as f:
    pickle.dump(cdata, f)
'''

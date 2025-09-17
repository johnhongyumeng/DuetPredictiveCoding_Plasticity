#This file will define the parameters of 
# the simulations

import numpy as np



PARAMS_Pyr= {
    # single cell
    'fI_tau': 0.015, # second, get from Oswald, Reyes 2011
    'fI_gl': 10, # nS

    'Vdend_rest': -65, # mv
    'gL_dend': 20,    # nS.  Default 20. In Larkum 2004. it is 1/43 Mohm, while 1/50Mohm=20 nS. 
    'C_dend': 0.12,     # nfarad. It is the same as in Larkum 2004. Leading tau_d= 5 ms
    'g_ds':  80,        # 60 before 0521. Increase because SST cannot inihibit the pyramidal cells along. 65Mohm for L5 Larkum 2004. About 20    
    'tau_dend': 0.005, #seconds
    'tau_soma': 0.002*50 , #seconds, but to be fair, this is the timescale of the rate model. Modify to suppress the spur behavior
    # background synaptc input
    'gbg_soma': 7, #nS   10 is good without PV. 10.5 include PV
    'gbg_dend': 3.3, #nS  


    # synapses related:
    'tau_AMPA': 0.002,
    'tau_NMDA': 60 * 0.001,
    
    # connectivity. gmax.  
    'weight_to_integrator': 6, # 2 before 11202024 0.15
    #'sigma_to_sst': 30.0,   #360.0,#43.2, Based on the Dipoppa et al. May need to check the robustness later
    #'sigma_to_pv': 30.0/2,
    #sigma_to_integrator': 1.0/2,#43.2,

    
    'gamma_NMDA': 0.641 ,
    'kappa_NMDA': 0.5,       # How many excitatory synapses are NMDA. 1-kappa would be AMPA.

    # dummy parameters used in the rate vector. Include here for completeness.
    'tau_AP': 0.002,     # AP time ms 
    'v_bAP': 10,         # bAP volt       
}

PARAMS_Integrator = {
    # single cell
    #'a': 0.135,    
    #'b': 54.0,
    #'d': 0.308, #seconds
    'fI_tau': 0.015, # second, get from Scala L23
    'fI_gl': 15, # nS
    'gbg': 53, #nS 
    'tau':0.4, #seconds    # 0.3 before 11202024
    #'alpha': 0.6,  # What's this?

    # synapses related:
    'tau_AMPA': 0.002,
    'tau_NMDA': 60 * 0.001,
    'gamma_NMDA': 0.641 ,
    'kappa_NMDA': 0.5,       # How many excitatory synapses are NMDA. 1-kappa would be AMPA.

    # connectivity. gmax.     # Maybe match this to Andreas Tolias data.
    'weight_to_dend':  22,   #  10 before 0508. testing 0.1. Defalut 2.42. For testing, change it to a small number for now.
    'scaleMax':2,
    'scaleMin':0.,
    #'weight_to_sst': 1.0,    #Tuned to 1 so baseline is low. Need to be constrained by Shen et al. 2022 By John
    'weight_to_pv': 20.0,    #   Copied from weight to SST on 02192024.   Previous 2.97. Need to be constrained by Shen et al. 2022 By John
    'weight_to_int': 0,    # Seems lead to unstablity. Need to be constrained by Shen et al. 2022 By John
    'weight_to_int_min': 0,  #0.5   This needs to be investigated

    #'weight_globalI_int': 4000,  # The global inhibtion to every unit. This term introduce competition, 
                              # and prediction part in MMN/DD.  

    #'sigma_to_dend' :  5.0,  
    #'sigma_to_pv' :  6.0,  
    #'sigma_to_sst' :  5.0,  
    'sigma_to_int' :  45,     # Default 43.2. This doesn't impact the results

    # Ring connectivity parameters.
    #'Jmax':907,   #2200 pA *256/90 / (50mV*0.138)=, in nA. 2.2 nA comes from Engel 2011. BUmp attactor 
    #'Jmin':-206,  #-0.5*256/90, in nA. Same as abpve 
    # Tested parameters are 6, 3
    # Recurrent excitation is too strong. Reducing it. So it's possible the units are pA, but not nA.
    'Jmax':5,   #default 10,2.2*256/90, in pA. 2.2 nA comes from Engel 2011. BUmp attactor 
    'Jmin':-15,  #default -8 -0.5*256/90, in pA. Same as abpve
}


PARAMS_PV= {
    # single cell
    'a': 0.135,
    'b': 54.0,
    'd': 0.308, #seconds
    'fI_tau': 0.007, # second, get from Gouwen L23 data
    'fI_gl': 12.5, # nS

    'tau':0.002, #seconds
    'gbg': 5.0, #nS Background synaptic input
    #'Input_ratio' : 0.05,   # share of input comparing to the pyramidal cells.
    # synapse related
    'tau_GABA': 0.005,



    # Connectivity. gmax  #nS per unit
    'weight_to_soma': 0.2,   # 0.2 before 0508
    'weight_to_soma0': 0.00,
    'sigma_to_pc' :  1.0,  
    'weight_to_pv': 17.25,
    'sigma_to_pv' :  1.0,  # This selection does not have too good reason.
    'weight_to_sst': 0.8,  # based on Allen's data in V1. Check again.
    'sigma_to_sst' :  15.0,  

}


PARAMS_SST= {
    # single cell
    'a': 0.135,
    'b': 54.0,
    'd': 0.308, #seconds
    'fI_tau': 0.020, # second, get from Gouwen L23 data
    'fI_gl': 5.0, # nS

    'tau':0.002, #seconds
    'gbg': 2.5, # 2.5nS when only bottom simulation. Background synaptic input
    'tau_GABA': 0.005,        # Maybe I want to have a slower time constant. 

    # Connectivity. gmax  #nS per unit
    'weight_to_dend': 0.2 ,    # 0.2 before 0508, same as PV to soma, appropriate should between 0.5 to 1
    'weight_to_pv': 10.65,
    'sigma_to_pc' :  46.0,      # so far, the tuning curve is identical in I cells.
    'sigma_to_pv' :  37.0,      # so far, the tuning curve is identical in I cells.

    # Include the parameters from trained connectivity to input. Original is trained from 10Hz to 5Hz
    #'StimtoSST':  0.15,    # 0.15 when bottom can get a good one. Assuming prestimuli is 10Hz. 
    'ToptoSST': 2,        # 1 is OK. Maybe need to fine tune later.
}


PARAMS_Simulation = {
    'dt': 0.0001,   # second, or 0.1 ms
    't_start': 0.0,
    't_total': 0.05,  
}

PARAMS_Stimulus = {
    'type': 'deterministic_MMN',
    'str_std': 100,   #pA Can be changed into the conductance input
    'str_dev': 100,
    'str_g': 0.7,  # 0.5 before 05212024 in nS. 
    'SST_ratio': 1,   # 1 befoer 05082024, The average input SST can get. Tuned such that on average SST can get 10 Hz firing rate.
    'sigma': 43.2,
    'sigma_std': 20,   # Origin: 43.2 John: WHY? Seems coming from a traditonal model. 
    'sigma_dev': 20,    # before 0920, 40
    'Tinter': 0.5,
    'Tstim': 0.5,
    'nbr_rep_std': 6,
    'nbr_rep_dev': 2,
    'std_id': 90, # the idx of the neuron receiving the stimulus
    'dev_id': 270, # Later may scale back to 0 to 180 degree.
    'Tresting': 2.0, #initial resting time in seconds
    'ratio_min': 0.,  # Here these two values better avarged to 1. 
    'ratio_max': 2.0 ,
    'str_g_back': 0.8,  # 0.095 before 0508 in nS.   
}



PARAMS_ALL = {
    'Simulation': PARAMS_Simulation,
    'PARAMS_INT': PARAMS_Integrator,
    'PARAMS_Pyr': PARAMS_Pyr,
    'PARAMS_PV': PARAMS_PV,
    'PARAMS_SST': PARAMS_SST,
    'Stimulus': PARAMS_Stimulus,
}

#  -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:45:42 2023
Here is to store all the functions required.
Now just listed all the functions you want to include. Now let's get what you
need. So maybe I just want to include the essentials. Or just go beyond what
Kevin had. 
@author:John Meng
"""
import numpy as np

import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

lib_dir = os.path.join(parent_dir, 'lib')
sys.path.append(lib_dir)
import basic_functions as bf




class RHS_update:  # include the current. However, the current here has the unit of voltage. in
    def __init__(self, W_dict,g_back):  # saving the fixed connectivity matrix here.    
        self.W_dict=W_dict  # Shouldn't be changed if without LTP.
        self.g_back=g_back.copy()  # Conductance based. The background input for integrator is not included here.
# Updating the current for pyramidal cells.
    def g_pyr_somaE(self, h_E):    # Conductance depended updating. Assuming the voltage is always around pre-threshold. So the current is propotionate to the conductance.
        return (np.dot(self.W_dict['EE'],h_E) + self.g_back['Es'])*(1-self.g_back['Ekappa'])
    def g_pyr_somaEN(self, h_EN):    # Conductance depended updating. Assuming the voltage is always around pre-threshold. So the current is propotionate to the conductance.
        return (np.dot(self.W_dict['EE'],h_EN) + self.g_back['Es'])*self.g_back['Ekappa']
    def g_pyr_somaI(self, h_P):    # Conductance depended updating. Assuming the voltage is always around pre-threshold. So the current is propotionate to the conductance.
        return np.dot(self.W_dict['EP'],h_P)

    def g_pyr_dendE(self, h_Int):   # Noticing these needs to be modified by the voltage.
        return (np.dot(self.W_dict['EdInt'],h_Int)+ self.g_back['Ed'])*(1-self.g_back['Intkappa'])
    def g_pyr_dendEN(self, h_IntN):   # Noticing these needs to be modified by the voltage.
        return (np.dot(self.W_dict['EdInt'],h_IntN)+ self.g_back['Ed'])*self.g_back['Intkappa']
    def g_pyr_dendI(self,h_S):    
        return np.dot(self.W_dict['EdS'],h_S)
    def g_pyr_dendI2(self,h_S,h_S2):    
        return np.dot(self.W_dict['EdS'],h_S)+np.dot(self.W_dict['EdS2'],h_S2)
    
# Updating the current for Integrators.
    def g_Int(self,h_E, h_Int):
        return (np.dot(self.W_dict['IntE'],h_E)*(1-self.g_back['Ekappa'])+ 
                (np.dot(self.W_dict['IntInt'],h_Int) + 
                self.g_back['Int'])*(1-self.g_back['IntIntkappa']))
    def g_IntN(self,h_EN, h_IntN):
        return (np.dot(self.W_dict['IntE'],h_EN)*self.g_back['Ekappa']+ 
                (np.dot(self.W_dict['IntInt'],h_IntN) + 
                self.g_back['Int'])*self.g_back['IntIntkappa']) 
    def g_Int_Iglobal(self, h_Int):   # This term is about the top-down competition. A with not-A.
        return self.W_dict['Int_Iglobal']*sum(h_Int)
    
# Updating the current for PV and SSTs.
    def g_pv_E(self,h_E,h_int):  
        return ( (np.dot(self.W_dict['PE'],h_E)+ self.g_back['P']  )*(1-self.g_back['Ekappa']) + 
                np.dot(self.W_dict['PInt'],h_int)* (1-self.g_back['Intkappa'])   )
    def g_pv_EN(self,h_EN,h_intN):    
        return ( (np.dot(self.W_dict['PE'],h_EN)+ self.g_back['P']  )*self.g_back['Ekappa'] + 
                np.dot(self.W_dict['PInt'],h_intN)* self.g_back['Intkappa']   )
# Update a normalization pv0 population
    def g_pv0_E(self,h_E):  
        return (np.dot(self.W_dict['P0E'],h_E)+ self.g_back['P0']  )*(1-self.g_back['Ekappa']) 
    def g_pv0_EN(self,h_EN):    
        return (np.dot(self.W_dict['P0E'],h_EN)+ self.g_back['P0']  )*self.g_back['Ekappa'] 


    def g_pv_I(self, h_P, h_S): 
        return np.dot(self.W_dict['PP'],h_P) + np.dot(self.W_dict['PS'],h_S)
    def g_pv_twoI(self, h_P, h_St,h_Sb): 
        return np.dot(self.W_dict['PP'],h_P) + np.dot(self.W_dict['PSt'],h_St)+np.dot(self.W_dict['PSb'],h_Sb)

    def g_sst_E(self,h_E, h_int): # !!! Notice here I removed the S pop
        return ( (np.dot(self.W_dict['SE'],h_E)+ self.g_back['S'])*(1-self.g_back['Ekappa'])+
                  np.dot(self.W_dict['SInt'],h_int) * (1-self.g_back['Intkappa'])  )
    def g_sst_EN(self,h_EN, h_intN): # !!! Notice here I removed the S pop
        return ( (np.dot(self.W_dict['SE'],h_EN)+ self.g_back['S'])*self.g_back['Ekappa']+
                  np.dot(self.W_dict['SInt'],h_intN) * self.g_back['Intkappa']  )
    def g_sst2_E(self,h_E, h_int): 
        return ( (np.dot(self.W_dict['S2E'],h_E)+ self.g_back['S2'])*(1-self.g_back['Ekappa'])+
                  np.dot(self.W_dict['S2Int'],h_int) * (1-self.g_back['Intkappa'])  )
    def g_sst2_EN(self,h_EN, h_intN): 
        return ( (np.dot(self.W_dict['S2E'],h_EN)+ self.g_back['S2'])*self.g_back['Ekappa']+
                  np.dot(self.W_dict['S2Int'],h_intN) * self.g_back['Intkappa']  )

    def g_sst_I(self,h_P):
        return np.dot(self.W_dict['SP'],h_P)
    def I_IntCompte(self,Int_pop, PARAMS_compte):   # Here pass the GEE, GEI, GIE, GII in. P
        IE=  PARAMS_compte['GEE'] * self.W_dict['IntInt'] @ Int_pop.rE + (- PARAMS_compte['GIE'] * np.mean(Int_pop.rI)) 
        II = PARAMS_compte['GEI'] * np.mean(Int_pop.rE) - PARAMS_compte['GII'] * np.mean(Int_pop.rI) 
        
        return IE, II
#    def dLTP(W,preRate,postRate,paras):

class LHS_update:  # Update the values of left hand side. Including update the conductance and the rate:
    class Pyr:
        def __init__(self, Paras,dt):
            self.Paras=Paras
            self.dt=dt
        def h(self,h,rate):  # Here is the AMPA conductance. If NMDA is needed, include another conductance parameter. 
            dh = (-h/ self.Paras['tau_AMPA']+rate)* self.dt
            return h+dh
        def hNMDA(self,hN,rate):  # Here is the NMDA current
            dh = (-hN/ self.Paras['tau_NMDA']+  
                  (1-hN)* rate*self.Paras['gamma_NMDA'])  * self.dt
            return hN+dh
        def rate(self, E_pop):    # Calculate the new rate and the dend voltage. Key of dendritic calculation.
            rate= E_pop.rate
            # potentially, this could include a term about the bAP. 
            # Need to test whether include or not AP can change the <V_dend>
            Isd= (E_pop.Vdend + 50) * self.Paras['g_ds']
            dVdend= (- (E_pop.Vdend-self.Paras['Vdend_rest']) *self.Paras['gL_dend'] +
                        E_pop.gE_dend * (0-E_pop.Vdend) +  
                        E_pop.gI_dend * (-65-E_pop.Vdend)  +
                        E_pop.gEN_dend * (0-E_pop.Vdend)* bf.nl_NMDA(E_pop.Vdend) + 
                        E_pop.input_dend - Isd +
                        E_pop.rate*(10-E_pop.Vdend)/500
                        )  /self.Paras['C_dend']    *  self.dt     
            # 1 mV * 1 nS = 1 pA
            # units pA for the current.   1pA/1nF= 1mV/1s

            soma_total_current = (Isd+ 
                        E_pop.gE_soma * 50 + E_pop.gI_soma *(-15) +   # by assuming threshold about 50mV. May want to change later.     
                        + E_pop.gEN_soma * 50*bf.nl_NMDA(-50)+  E_pop.input_soma)    # here include the background input

            # In a simplified version, I can get all the current onto the soma. 
            drate = (-E_pop.rate + bf.abott_fI_curve2(soma_total_current, self.Paras['fI_gl'],  self.Paras['fI_tau'])   
                    ) /self.Paras['tau_soma'] *  self.dt
            
            return bf.relu(E_pop.rate+drate), E_pop.Vdend+dVdend, Isd, soma_total_current
    class Int: # Need to update them all.
        def __init__(self, Paras,dt):
            self.dt=dt
            self.Paras=Paras
        def h(self,h,rate):  # Here is the AMPA conductance. If NMDA is needed, include another conductance parameter. 
            dh = (-h/ self.Paras['tau_AMPA']+rate)* self.dt
            return h+dh
        def hNMDA(self,hN,rate):  # Here is the AMPA conductance. If NMDA is needed, include another conductance parameter. 
            dh = (-hN/ self.Paras['tau_NMDA']+  
                  (1-hN)* rate*self.Paras['gamma_NMDA'])  * self.dt
            return hN+dh 
        def rate(self, Int_pop): 
            total_current= (Int_pop.g * 50 + 
                        Int_pop.gN * 50*bf.nl_NMDA(-50)+
                        Int_pop.g_Iglobal *(-15) +   # by assuming threshold about 50mV. May want to change later.     
                        Int_pop.input)    # here include the background input. Currently, it's 0.
  
            drate = (-Int_pop.rate + bf.abott_fI_curve3(total_current, self.Paras['fI_gl'],  self.Paras['fI_tau'])   
                    ) /self.Paras['tau'] *  self.dt

#            drate = (-Int_pop.rate + bf.abott_fI_curve2(total_current, self.Paras['fI_gl'],  self.Paras['fI_tau'])   
#                    ) /self.Paras['tau'] *  self.dt



            return bf.relu(Int_pop.rate+drate), total_current
    class P:
        def __init__(self, Paras,dt):
            self.Paras=Paras
            self.dt=dt

        def h(self,h,rate):  # Here is the AMPA conductance. If NMDA is needed, include another conductance parameter. 
            dh = (-h/ self.Paras['tau_GABA']+rate)* self.dt
            return h+dh 
        def rate(self, P_pop):
            total_current= (P_pop.gE * 50 + P_pop.gI *(-15) +   # by assuming threshold about 50mV. May want to change later.    bf.nl_NMDA(-50)=0.138 
                    P_pop.gEN * 50*bf.nl_NMDA(-50) +   P_pop.input)    # here include the background input    
            drate = (-P_pop.rate + bf.abott_fI_curve2(total_current, self.Paras['fI_gl'],  self.Paras['fI_tau'])   
                    ) /self.Paras['tau'] *  self.dt
            return bf.relu(P_pop.rate+drate), total_current
    class S:
        def __init__(self, Paras,dt):
            self.Paras=Paras
            self.dt=dt

        def h(self,h,rate):  # Here is the AMPA conductance. If NMDA is needed, include another conductance parameter. 
            dh = (-h/self.Paras['tau_GABA']+rate)* self.dt
            return h+dh         
        
        def rate(self, S_pop):
            total_current= (S_pop.gE * 50 + S_pop.gI *(-15) +   # by assuming threshold about 50mV. May want to change later.     
                    S_pop.gEN * 50*bf.nl_NMDA(-50) +   S_pop.input)    # The background input is 0.                     
            drate = (-S_pop.rate + bf.abott_fI_curve2(total_current, self.Paras['fI_gl'],  self.Paras['fI_tau'])   
                    ) /self.Paras['tau'] *  self.dt
            return bf.relu(S_pop.rate+drate) , total_current

#    def calc_STP(U,D, dU, dD, paras):
#    def calc_W(W, dW, paras): 
        
    class IntCompte: # Only update rE
        def __init__(self, Paras,dt):
            self.dt=dt
            self.Paras=Paras
            self.f= lambda x: x * x * (x > 0) * (x < 1) + np.sqrt(np.maximum(4 * x - 3, 0)) * (x >= 1)
        def rate(self, Int_pop): 
            nu = 2      # an arbitrary scale function to increase the firing rate.
            IEall= Int_pop.IE +  Int_pop.input  + self.Paras['I0E']  # here include the background input. Currently, it's 0.
            IIall= Int_pop.II +  self.Paras['I0I']  # here include the background input. Currently, it's 0.

            drateE = (-Int_pop.rE + nu* self.f(IEall))/self.Paras['tauE'] *  self.dt
            drateI = (-Int_pop.rI + nu* self.f(IIall))/self.Paras['tauI'] *  self.dt
            return bf.relu(Int_pop.rE+drateE), bf.relu(Int_pop.rI+drateI)
  

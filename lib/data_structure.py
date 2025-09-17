# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:48:03 2023
Setup the basics data structure for single cells and synapses. Should
include all the structure to save only the essential data, but not to
pdate states here. The updating functions are in updatefunctions.py
I may add new functions over time.
@author: John Meng
"""
import numpy as np

class Pyramidal:
    def __init__(self, Ncells):
       
        self.rate = 1.0*np.ones(Ncells)         # default 5Hz? Not sure initialization here is good.
        self.Vdend= np.zeros(Ncells) - 65       # set as the reversal potential 
        
        self.input_soma = np.zeros(Ncells)
        self.input_dend = np.zeros(Ncells)                # Maybe I want to move it to current calculation

        self.gE_soma = np.zeros(Ncells)            # For AMPA
        self.gEN_soma = np.zeros(Ncells)            # For NMDA
        self.gI_soma = np.zeros(Ncells)            # 
        self.gE_dend = np.zeros(Ncells)
        self.gEN_dend = np.zeros(Ncells)
        self.gI_dend = np.zeros(Ncells)
        
        self.h = np.zeros(Ncells)                  # Dimensionless integrator of rate that mimicks the opening profile of postsynaptic channels.
        self.hN = np.zeros(Ncells)                  # Dimensionless integrator of rate that mimicks the opening profile of postsynaptic channels.

        #maybe hA is better for AMPA receptors? Later you will need to add NMDA.
        # some parameters for output
        # self.Isd= np.zeros(Ncells)
class Interneuron:
    def __init__(self, Ncells):
        self.rate = np.zeros(Ncells) 
       
        self.input = np.zeros(Ncells)
        self.gE = np.zeros(Ncells)
        self.gEN = np.zeros(Ncells)
        self.gI = np.zeros(Ncells)+20

        self.h = np.zeros(Ncells)

#        self.sA = np.zeros(self.Ncells)
#        self.tau_adaptation = kwargs['tau_adaptation']
#        self.gA = kwargs['gA']    
        
class Integrator:
    def __init__(self, Ncells):    
        self.rate = np.zeros(Ncells)   # Can include baseline here

        self.input = np.zeros(Ncells)
        self.g = np.zeros(Ncells)
        self.gN = np.zeros(Ncells)
        self.h = np.zeros(Ncells)
        self.hN = np.zeros(Ncells)
        self.g_Iglobal = 0.0
#        self.sA = np.zeros(self.Ncells)

class IntCompte:    # Reproducing the bump attractor from Compte. 4D for each cell
    def __init__(self, Ncells):
        self.rE= np.zeros(Ncells)
        self.rI= np.zeros(Ncells)
        self.IE= np.zeros(Ncells)
        self.II= np.zeros(Ncells)
        self.input= np.zeros(Ncells)   # This will integrate input from lower level

# Maybe I will include a connection matrix. Maybe no? Not sure. Seems fine
# with the function create_ring_connectivity        
        
        

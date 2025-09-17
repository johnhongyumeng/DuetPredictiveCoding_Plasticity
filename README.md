# DuetPredictiveCoding_Plasticity

Code for: Global error signal guides local optimization in mismatch calculation. John Meng and Xiao-Jing Wang. 
BioRxiv link: doi: https://doi.org/10.1101/2025.07.07.663505

## Figure 2:
Main_ReluToy.py.  Running time: <10s

## Figure 3:

Training:   ./scripts/Plasticity/main_plasticity_variations               Runing time: ~5 mins
To avoid accidentally overwriting the results, it is required to uncomment the output lines to update the results manually. 

Plotting:  ./scripts/Plasticity/load_analysis_plasticity 
## Figure 4:
Training:   ./scripts/Plasticity/main_plasticity_selectivity           Runing time: ~5 mins

Plotting:  ./scripts/Plasticity/load_analysis_plasticity_selectivity 
## Figures 5:
Run: ./scripts/class_main_nPE_perturbe.py
## Figure 6:
Run: ./scripts/shell_main_nPE.py
## Figure 7:
Run: ./scripts/shell_main_pPE.py
## Figure 8
Model Simulation: run ./scripts/nPEcorrelation/main_selectivity.py

Data analysis:
To run the code, the original data is needed from Attinger et al., (2017), which can be accessed through https://data.fmi.ch/PublicationSupplementRepo/ 

After loading the data, run 

./analyze/vml_Fig2_John.m

Which is a modified version of the original vml_Fig2.m

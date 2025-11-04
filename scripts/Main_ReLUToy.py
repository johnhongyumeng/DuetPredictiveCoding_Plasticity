# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:33:05 2025
This is the toy model to illustrate how the learning rule work on 
ReLU neurons.
@author: GoldenHeart
"""
import os
import sys
import importlib
from datetime import date,datetime
from tqdm import tqdm

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

import pickle
from numpy.random import default_rng
from scipy.stats import truncnorm

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

lib_dir = os.path.join(parent_dir, 'lib')
sys.path.append(lib_dir)

import basic_functions as bf
# import AnalysisTool as AT   Copy the needed AnalysisTool part here. 




np.random.seed(5)    #default 5
temp_dist = truncnorm(-2, 2, loc=0, scale=0.5)  
alpha=0.01   # Learning rate.
E_baseline=0.05  # Baseline rate.
E_saturate=0.3
N_time=100000
N_neu= 40     # 40 in the main code.

# FigPath= '..\\figs\\toymodel\\Homeo'

# flag_saturate= True
# flag_perturbe= False
# mu=0.4
# sigma=0.2



FigPath= '..\\figs\\toymodel\\NoHomeo'
flag_perturbe= True
flag_saturate= True
mu=0.3                     # default 0.3   for w/ homeo 0.4
sigma=0.05                   #default 0.05  for w/ homeo 0.2






sdir=FigPath

if flag_perturbe:
    shuffle_input_indices = np.random.permutation(N_neu)
else:
    shuffle_input_indices = np.arange(N_neu)




#E_cells= np.full(N_neu,np.nan,dtype=np.float32)
#W_1 =  np.full(N_neu,np.nan,dtype=np.float32) 
#W_2 =  np.full(N_neu,np.nan,dtype=np.float32) 
#Input_Stim= np.linspace(0, 1, num=N_neu, endpoint=True)
#Input_Pred= np.linspace(1, 0, num=N_neu, endpoint=True)

Input_Stim= np.linspace(1, 0, num=N_neu, endpoint=True)
Input_Pred= np.linspace(0, 1, num=N_neu, endpoint=True)


Input_Pred=Input_Pred[shuffle_input_indices]

# sigma= sigma_r*mu= 0.1*0.5=0.05


lower_bound_std = (0 - mu) / sigma    #(lower_bound - mu) / sigma
upper_bound_std = (2*mu - mu) / sigma

distribution = truncnorm(lower_bound_std, upper_bound_std, loc=mu, scale=sigma)
W_1 = distribution.rvs(N_neu)
W_2 = distribution.rvs(N_neu)




Output_dic= dict(
    W_1_mat= np.full((N_neu,N_time),np.nan, dtype=np.float32),
    W_2_mat= np.full((N_neu,N_time),np.nan, dtype=np.float32),
    Error_vec= np.full(N_time,np.nan,dtype=np.float32)
    )
Error_vec=   temp_dist.rvs(N_time)
Output_dic['Error_vec']=Error_vec
Output_dic['Input_Stim']=Input_Stim
Output_dic['Input_Pred']=Input_Pred
Output_dic['Init_W1']=W_1.copy()
Output_dic['Init_W2']=W_2.copy()


for i_time in tqdm(range(N_time)):
    
    # The error    
    learning_protocol=Error_vec[i_time]
    learning_error= abs(learning_protocol)
    
    if learning_protocol>=0:  # under predict. Stim> pred
        stim=1
        pred= 1- learning_error
    elif learning_protocol<0: # Pred > stim over pred
        pred=1
        stim= 1- learning_error

    # Updating the E_cells.     
    E_cells= bf.relu(stim* Input_Stim - pred*W_2 + pred*Input_Pred- stim*W_1)
    # Updating synaptic weights.
    dW_1 = -(learning_error-0.5)*((E_cells-E_baseline)*stim )
    dW_2 = -(learning_error-0.5)*((E_cells-E_baseline)*pred )
    if learning_error>0.5 and flag_saturate==True:
        dW_1 = dW_1 *bf.relu(1 - np.mean(E_cells)/E_saturate )    
        dW_2 = dW_2 *bf.relu(1 - np.mean(E_cells)/E_saturate )
        '''
        if np.mean(E_cells)>E_saturate*learning_error:
            dW_1 =0
            dW_2 =0
        '''        



    W_1=bf.relu(W_1+alpha*dW_1)
    W_2=bf.relu(W_2+alpha*dW_2)
    
    
    # Update the Output
    Output_dic['W_1_mat'][:,i_time]=W_1
    Output_dic['W_2_mat'][:,i_time]=W_2


# Testing the output. 



Init_base=0* E_cells
stim=0; pred=1
Init_pred=bf.relu(stim* Input_Stim - pred*Output_dic['Init_W2'] + pred*Input_Pred- stim*Output_dic['Init_W1'])
stim=1; pred=0
Init_stim=bf.relu(stim* Input_Stim - pred*Output_dic['Init_W2'] + pred*Input_Pred- stim*Output_dic['Init_W1'])
stim=1; pred=1
Init_expected=bf.relu(stim* Input_Stim - pred*Output_dic['Init_W2'] + pred*Input_Pred- stim*Output_dic['Init_W1'])

Final_base=0* E_cells
stim=0; pred=1
Final_pred=bf.relu(stim* Input_Stim - pred*W_2 + pred*Input_Pred- stim*W_1)
stim=1; pred=0
Final_stim=bf.relu(stim* Input_Stim - pred*W_2 + pred*Input_Pred- stim*W_1)
stim=1; pred=1
Final_expected=bf.relu(stim* Input_Stim - pred*W_2 + pred*Input_Pred- stim*W_1)

'''
PltTool.Plot_var_Snapshots(Output_dic['E_pop_rate'][:,[3,4,2,1]],
                           labels=['Stim','Pred','Expected','Baseline',],
                           colors=['#31a354','#3182bd','#e34a33','grey'],
                           name='InitalSnapshots')
        if self.pf:
            fig.savefig(self.sdir + name + '.jpg', bbox_inches='tight')
            fig.savefig(self.sdir + name + '.svg', bbox_inches='tight')
        axes.set_title(name)

'''

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.4))
axes.plot(Init_stim, color='#31a354')
axes.plot(Init_pred, color='#3182bd')
axes.plot(Init_expected, color='#e34a33')
#x_vals=np.arange(40, 0, -1) - 1
# axes.scatter(x_vals,Init_stim, color='#31a354', s=5)
# axes.scatter(x_vals,Init_pred, color='#3182bd', s=5)
# axes.scatter(x_vals,Init_expected, color='#e34a33', s=5)
axes.set_yticks([0,0.5, 1])

axes.set_ylim(top=1)
name='InitActicity'

fig.savefig(sdir + name + '.jpg', bbox_inches='tight')
fig.savefig(sdir + name + '.svg', bbox_inches='tight')

axes.set_title(name)

plt.show()


fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.4))
axes.plot(Final_stim, color='#31a354')
axes.plot(Final_pred, color='#3182bd')
axes.plot(Final_expected, color='#e34a33')
axes.set_yticks([0,0.5, 1])

name='FinalActicity'
fig.savefig(sdir + name + '.jpg', bbox_inches='tight')
fig.savefig(sdir + name + '.svg', bbox_inches='tight')
axes.set_title(name)
plt.show()


fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.4))

axes.plot(Output_dic['Init_W1'], color='#31a354')
axes.plot(Output_dic['Init_W2'], color='#3182bd')

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.4))
axes.plot(W_1, color='#31a354')
axes.plot(W_2, color='#3182bd')

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.4))


axes.plot(Output_dic['W_1_mat'][0,:],Output_dic['W_2_mat'][0,:],color='#31a354')
axes.plot(Output_dic['W_1_mat'][-1,:],Output_dic['W_2_mat'][-1,:],color='#3182bd')
axes.plot(Output_dic['W_1_mat'][N_neu//3,:],Output_dic['W_2_mat'][N_neu//3,:],color='black')
axes.plot(Output_dic['W_1_mat'][2*N_neu//3,:],Output_dic['W_2_mat'][2*N_neu//3,:],color='black')
axes.plot(Output_dic['W_1_mat'][9,:],Output_dic['W_2_mat'][9,:],color='black')

plt.show()

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.4))
N_lines=6
for i in range(N_lines):
    index=i*N_neu//N_lines
    axes.plot(Output_dic['W_1_mat'][index,:],Output_dic['W_2_mat'][index,:],color='black')
    Sum_input=Input_Stim[index]+Input_Pred[index]-E_baseline
    axes.plot([Sum_input,0],[0,Sum_input],'--k')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
N_lines = 6
segments_total = 500
base_count = segments_total // N_lines  # Base number of segments per iteration
remainder = segments_total % N_lines  # Extra segments to distribute

all_segments = []  # To store all line segments (as coordinate arrays)
all_colors = []    # To store colors for each segment
dash_segments = [] # To store dashed line segments
dash_colors = []   # Stores colors for dashed lines

# Colormap for unique target colors (e.g., viridis)
cmap = plt.cm.get_cmap('viridis')
end_time = N_time  # N_time = 100000

start_points = []  # Store start points for triangles
end_points = []    # Store end points for circles
marker_colors = []  # Colors for the start and end markers
Neu_indices = np.linspace(0, N_neu - 1, N_lines, dtype=int)  # Guarantees min and max indices are included

for i in range(N_lines):
    # Determine how many segments this iteration gets
    seg_count = base_count + (1 if i < remainder else 0)
    if seg_count < 1:
        continue  # Skip if no segments (shouldn't happen unless N_lines > 100)

    # Select the data index for this iteration
    index = Neu_indices[i]
    x_line = Output_dic['W_1_mat'][index, :end_time]
    y_line = Output_dic['W_2_mat'][index, :end_time]

    # Determine the target color for this iteration
    target_color = np.array(cmap(i / (N_lines+2 - 1)))  # RGBA color
    target_rgb = target_color[:3]
    marker_colors.append(target_rgb)  # Store color for both markers
    ax.plot(x_line, y_line, color=target_rgb, linewidth=0.7)


    # Clustered sampling at start
    t = np.linspace(0.0, 1.0, seg_count + 1) ** 4  # Bias clustering towards start
    positions = t * (len(x_line) - 1)
    x_points = np.interp(positions, np.arange(len(x_line)), x_line)
    y_points = np.interp(positions, np.arange(len(y_line)), y_line)


    # Store start and end points for markers
    start_points.append((x_points[0], y_points[0]))   # First point (triangle)
    end_points.append((x_points[-1], y_points[-1]))   # Last point (circle)

    # Create line segments with gradient coloring
    for j in range(seg_count):
        seg = np.array([[x_points[j], y_points[j]], 
                        [x_points[j+1], y_points[j+1]]])
        all_segments.append(seg)

        # Compute blended color (from black to target)
        frac = j / float(seg_count - 1) if seg_count > 1 else 1.0
        color_rgb = target_rgb * (1-frac)   # Blend from black to target
        all_colors.append((color_rgb[0], color_rgb[1], color_rgb[2], 1.0))

    # Store the dashed line with the final segment's color
    Sum_input = Input_Stim[index] + Input_Pred[index] - E_baseline
    dash_segments.append(np.array([[Sum_input, 0], [0, Sum_input]]))
    dash_colors.append((target_rgb[0], target_rgb[1], target_rgb[2], 1.0))  # Last color from gradient

# Create a LineCollection for all colored segments
line_coll = LineCollection(all_segments, colors=all_colors, linewidths=1)
dash_coll = LineCollection(dash_segments, colors=dash_colors, linestyles='--', linewidths=0.5)

# ax.add_collection(line_coll)
ax.add_collection(dash_coll)

# Convert lists to NumPy arrays for scatter plotting
start_points = np.array(start_points)
end_points = np.array(end_points)
marker_colors = np.array(marker_colors)

# Plot start markers (triangles)
ax.scatter(start_points[:, 0], start_points[:, 1], color=marker_colors, marker='x', s=12, label="Start")

# Plot end markers (circles)
ax.scatter(end_points[:, 0], end_points[:, 1], color=marker_colors, marker='o', s=8, label="End")
ax.set_yticks([0, 1])

ax.autoscale()  # Adjust axes to show all segments
plt.show()
name='WeightDiagram'
fig.savefig(sdir + name + '.jpg', bbox_inches='tight')
fig.savefig(sdir + name + '.svg', bbox_inches='tight')



def Plot_FixframeInt(N_eachC,sdir, mat, int_f, name='',top=60):
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.4))
   
    # Define the start and end colors
    start_color = mcolors.to_rgba("#bae4bc")
    end_color = mcolors.to_rgba("#2b8cbe")
    
    # Create a color map that interpolates between start_color and end_color
    n_lines = len(range(0, mat.shape[1], int_f))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color], N=n_lines)
    colors = [cmap(i / (n_lines - 1)) for i in range(n_lines)]

    #colors = [mcolors.to_rgba(c) for c in cm.get_cmap('Blues', n_lines)(np.linspace(0, 1, n_lines))]

    # Calculate indices for start, middle, and end labels
    label_indices = [0, n_lines // 2, n_lines - 1]

    # Plot each line with the color from the gradient
    for i, (index, color) in enumerate(zip(range(0, mat.shape[1], int_f), colors)):
        label = None
#        axes.plot(np.arange(N_eachC, 0, -1) - 1, mat[:, index], color=color, label=label)
        axes.plot(np.arange(0, N_eachC,) , mat[:, index], color=color, label=label)

    if top is not None:
        axes.set_ylim(top=60)
    axes.set_yticks([0,0.5, 1])

    # Add an inset for the color bar
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # inset_ax = inset_axes(axes, width="30%", height="5%", loc='upper right', borderpad=1)

    # Create a ScalarMappable and add the color bar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=mat.shape[1] - 1))
    # sm.set_array([])
    # cbar = fig.colorbar(sm, cax=inset_ax, orientation='horizontal')
    # cbar.ax.tick_params(labelsize=6)
    # inset_ax.xaxis.set_ticks_position("top")  # Adjust tick position for readability

    axes.legend(fontsize=6)

    fig.savefig(sdir + name + '.jpg', bbox_inches='tight')
    fig.savefig(sdir + name + '.svg', bbox_inches='tight')
    
    axes.set_title(name)

Plot_FixframeInt(N_neu,sdir,Output_dic['W_1_mat'],5000,name='W_1_frames',top=None)
Plot_FixframeInt(N_neu,sdir,Output_dic['W_2_mat'],5000,name='W_2_frames',top=None)

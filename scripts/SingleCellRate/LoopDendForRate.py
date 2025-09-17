# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:56:01 2020
For a loop of InjRate. Change the SynsDendrite and Synaps Soma in the rate code
@author: Hongyu Meng
"""
import numpy as np

import brian2 as b2

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import json, codecs
from tqdm import tqdm


import os
from datetime import date,datetime
import matplotlib.colors as mcolors


now=datetime.now()
today=date.today()
current_time=now.strftime("%H%M")
current_day=today.strftime("%m%d_")
timestamp=current_day+current_time

fig_tag='InjSE8L23DEDI_'   # Seems 5 for L5, 8 for L23 is good.
coupling_conductance=  80     # Coupling 80 for L23, 10 for L5
CondSE_ite = 8


#fig_tag='InjSE6L5DEDI_'
#coupling_conductance=  10     # Coupling 80 for L23, 10 for L5
#CondSE_ite = 6

#TempPath='SavedResearch/'+timestamp
#os.makedirs(TempPath)

Num_Injd=11
g_Injd_min=0   # 0.1 nS for the generic VIP model
g_Injd_max=8 
g_Injd_vec= np.arange(0,Num_Injd-1+0.1,1)/(Num_Injd-1) * (g_Injd_max-g_Injd_min)+g_Injd_min

Num_Injs=11
g_Injs_min=0
g_Injs_max=16
g_Injs_vec= np.arange(0,Num_Injs-1+0.1,1)/(Num_Injs-1) * (g_Injs_max-g_Injs_min)+g_Injs_min




Rate_mat= np.full((Num_Injd,Num_Injs),np.nan)
VoltD_mat= np.full((Num_Injd,Num_Injs),np.nan)
Isd_mat= np.full((Num_Injd,Num_Injs),np.nan)
Itotal_mat=np.full((Num_Injd,Num_Injs),np.nan)

for i_ext in tqdm(range(Num_Injd)):
    for j_ext in range(Num_Injs):
        Condd_ite=g_Injd_vec[i_ext]
        ConddI_ite=g_Injs_vec[j_ext]
        Injd_ite=0
        Injs_ite=0

        exec(open("RatePyr.py").read())   
    
    #    exec(open("SinglePyrTrialL23.py").read())
        Rate_mat[i_ext,j_ext,]=Rate_end
        VoltD_mat[i_ext,j_ext,]=Vdend_end
        Isd_mat[i_ext,j_ext,]=Isd_end
        Itotal_mat[i_ext,j_ext,]=Itotal

# X-axis Inj_vec  Y_axis  weigt_vec  

class mycolormesh:
    def __init__(self, x_vec,y_vec):
        self.xx_mesh, self.yy_mesh= np.meshgrid(x_vec,y_vec)
        self.xv=x_vec
        self.yv=y_vec
        
    def myplot(self,data,title='',s_tag='', vmin=None,contour_value=None):
        fig, ax =  plt.subplots(dpi=250,figsize=(2,1.7))
        if vmin is not None:
            norm = mcolors.Normalize(vmin=vmin, vmax=Rate_mat.max())
            im=ax.pcolormesh(self.xx_mesh,self.yy_mesh,data.transpose(),norm=norm)
        else:
            im=ax.pcolormesh(self.xx_mesh,self.yy_mesh,data.transpose())

        if contour_value is not None:
            contour = ax.contour(self.xx_mesh, self.yy_mesh, data.transpose(),
                             colors='white', linewidths=1, linestyles='dashed',
                             levels=contour_value)
        else:
            contour = ax.contour(self.xx_mesh, self.yy_mesh, data.transpose(),
                             colors='white', linewidths=1, linestyles='dashed')
                
        ax.set_title(title)
        # Adjust axis limits to ensure equal scaling
        xmin=min(self.xv)
        xmax=max(self.xv)
        ymin=min(self.yv)
        ymax=max(self.yv)
        ax.set_xticks([xmin, xmax])
        ax.set_yticks([ymin, ymax])
        
        ax.set_aspect((xmax-xmin)/(ymax-ymin), adjustable='box')
        
        fig.colorbar(im)
        im.set_cmap('plasma')
        plt.show()
        fig.savefig('../../figs/SingleCell/'+s_tag +'_'+ title+'.jpg',bbox_inches='tight')
        fig.savefig('../../figs/SingleCell/'+s_tag +'_'+ title+'.svg',bbox_inches='tight')
       
        plt.close(fig)
                

TempPlt=mycolormesh(g_Injd_vec,g_Injs_vec)
TempPlt.myplot(VoltD_mat,'DendVolt',fig_tag,contour_value=4)
TempPlt.myplot(Isd_mat,'Isd',fig_tag,contour_value=4)
#TempPlt.myplot(Itotal_mat,'Itotal',fig_tag,contour_value=4)
TempPlt.myplot(Rate_mat,'rate',fig_tag,vmin=0,contour_value=3)


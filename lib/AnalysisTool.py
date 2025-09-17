# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:08:58 2023
Plot tools for the rate models. Not work for now
There are so many functions that follow different naming rules. Sorry about that. 
Please look only at the function when called.
@author: John Meng
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import pearsonr
from scipy.optimize import curve_fit


        
class Tools:
    def __init__(self, Output_dic,ndt, n_frames, dt,PARAMS_Stimulus, time_draw_start=1,
                 Plot_flag=False,omission_flag=False, save_dir='../figs/'):  # maybe add more later.
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        self.Paras=PARAMS_Stimulus
        #self.shift=7
        self.ndt=ndt
        self.N_eachC=PARAMS_Stimulus['N_eachC']
        self.N_column=PARAMS_Stimulus['N_column']
        self.Nneu=PARAMS_Stimulus['N_neu']
        self.dtheta= 360.0/self.N_column
        self.n_frames=n_frames
        # Now I have columns. I should think a bit about which neuron to get. 
        # First, index_1d = j * N_eachC + i   for stim_2d[i,j]
        # So column number should be 
        self.stdcn=int(PARAMS_Stimulus['std_id']/self.dtheta)  
        self.devcn=int(PARAMS_Stimulus['dev_id']/self.dtheta)
        # For the median number should be self.N_eachC//2
        self.std= self.stdcn*self.N_eachC+ self.N_eachC//2
        self.dev= self.devcn*self.N_eachC+ self.N_eachC//2
        self.Res_flag=None
        self.std_neus_vec=np.arange(0,self.N_eachC)+self.stdcn*self.N_eachC
        self.dev_neus_vec=np.arange(0,self.N_eachC)+self.devcn*self.N_eachC

        self.t_axis=np.arange(n_frames)*dt*ndt/n_frames
        self.Output_dic=Output_dic

        dtframe= dt*ndt/n_frames
        self.dtframe=dtframe
        self.shift=round(self.Paras['Tstim']// dtframe)
        self.t0 = -(PARAMS_Stimulus['Tinter'] + PARAMS_Stimulus['Tresting'] )  
        self.tn = PARAMS_Stimulus['t_total'] + self.t0

        self.LastT= PARAMS_Stimulus['Tresting']+(PARAMS_Stimulus['Tinter']+
                        PARAMS_Stimulus['Tstim'])*(PARAMS_Stimulus['nbr_rep_std']-1)+PARAMS_Stimulus['Tinter']
        self.LastInd=round(self.LastT/dtframe)   # Should be an average. But this for now. Need to define a function
        self.FirstT= PARAMS_Stimulus['Tresting']+PARAMS_Stimulus['Tinter']
        self.FirstInd=round(self.FirstT/dtframe)   # Should be an average. But this for now.
        self.time_draw_start=time_draw_start
        self.drawstart=round(time_draw_start/dtframe) # Removing the initial when drawing.
        #self.baseInd=self.FirstInd-1
        

        self.OddT= PARAMS_Stimulus['Tresting']+(PARAMS_Stimulus['Tinter']+
                        PARAMS_Stimulus['Tstim'])*(PARAMS_Stimulus['nbr_rep_std'])+PARAMS_Stimulus['Tinter']
        self.OddInd=round(self.OddT/dtframe)
        self.pf=Plot_flag
        self.sdir=save_dir
        self.omission_flag=omission_flag
        
        if omission_flag==False:
            neu_axis=np.arange(self.Nneu)
            if self.dev>=self.std:
                shift_axis=np.append(np.arange(self.dev-self.std,self.Nneu),np.arange(self.dev-self.std))
            else:
                shift_axis=np.append(np.arange(self.Nneu+self.dev-self.std,self.Nneu),np.arange(self.Nneu+self.dev-self.std))
        else:
            neu_axis=np.arange(self.Nneu)
            shift_axis=np.arange(self.Nneu)
        self.shift_axis=shift_axis
        self.neu_axis=neu_axis
        First_rate_vec= np.mean(
            self.Output_dic['E_pop_rate'][neu_axis,self.FirstInd:self.FirstInd+self.shift], 
            axis=1)
        Last_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][neu_axis,self.LastInd:self.LastInd+self.shift], 
            axis=1)
        Odd_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][shift_axis,self.OddInd:self.OddInd+self.shift], 
            axis=1)
        First_pre_rate_vec= np.mean(
            self.Output_dic['E_pop_rate'][neu_axis,self.FirstInd-self.shift:self.FirstInd], 
            axis=1)
        Last_pre_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][neu_axis,self.LastInd-self.shift:self.LastInd], 
            axis=1)
        Odd_pre_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][shift_axis,self.OddInd-self.shift:self.OddInd], 
            axis=1)
        baseline_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][neu_axis,self.drawstart:self.drawstart+self.shift], 
            axis=1)          




        self.First_rate_vec=First_rate_vec
        self.Last_rate_vec=Last_rate_vec
        self.Odd_rate_vec=Odd_rate_vec
        self.First_pre_rate_vec=First_pre_rate_vec
        self.Last_pre_rate_vec=Last_pre_rate_vec
        self.Odd_pre_rate_vec=Odd_pre_rate_vec

        self.baseline_rate_vec=baseline_rate_vec

        if omission_flag==False:
            DD_vec=Odd_rate_vec-First_rate_vec
        else:
            DD_vec=Odd_rate_vec-baseline_rate_vec
        SSA_vec=First_rate_vec-Last_rate_vec
        self.DD_vec=DD_vec
        self.SSA_vec=SSA_vec
        
    def PlotPyr(self,shift_id=0): # Now the code can detect the MMN for any two cells.
        stdid = self.std+shift_id
        devid = self.dev+shift_id
        if stdid>=self.Nneu:
            stdid=stdid-self.Nneu
        if devid>=self.Nneu:
            devid=devid-self.Nneu    

        fig, axes=plt.subplots(1,1,figsize=(4,3))
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['E_pop_rate'][stdid,self.drawstart:],'k',label='Std')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['E_pop_rate'][devid,self.drawstart:],'r',label='dev')
        axes.legend()
        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']

            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            if i != 7 and i != 6:
                axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency

        MMN= (self.Output_dic['E_pop_rate'][stdid,self.LastInd]-
            self.Output_dic['E_pop_rate'][devid,self.OddInd])
        if self.pf:
            fig.savefig(self.sdir+'MMNPyr'+ f"{MMN:.4f}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyr'+ f"{MMN:.4f}"+'.svg',bbox_inches='tight')


        axes.set_title('MMNPyr'+ f"{MMN:.4f}"+'DegreeShift'+ f"{shift_id:.2f}")
        return MMN

    def PlotPyr_pPExnPEy(self,left=50,right=60,labels=['std','dev']): # Now the code can detect the MMN for any two cells.
        pPEx_id=self.std+self.N_eachC//2-1
        nPEy_id=self.dev-self.N_eachC//2

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['E_pop_rate'][pPEx_id,self.drawstart:],'k',label=labels[0])
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['E_pop_rate'][nPEy_id,self.drawstart:],'r',label=labels[1])
        axes.legend()
        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']

            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            if i != 7 and i != 6:
                axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency

        axes.set_xlim(left=left-0.5)  # This changes the view to start from -1 on the x-axis
        if right is not None:
            axes.set_xlim(right=right)
            choices= self.Paras['choices'].copy()    # 1 for std, 2 for dev

            for i in np.arange(right-10,right):
                start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
                end = start + self.Paras['Tstim']
                # Add vertical dashed lines
                axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
                axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)      
                if round(choices[i])==2:    
                    axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           


        if self.pf:
            fig.savefig(self.sdir+'PyrTrace_pPEnPE'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'PyrTrace_pPEnPE'+'.svg',bbox_inches='tight')

        axes.set_title('PyrTrace_pPEnPE')

    def PlotPyr_omiss(self,shift_id=0,left=None): # Now the code can detect the MMN for any two cells.
        stdid = self.std+shift_id
        devid = self.dev+shift_id
        if stdid>=self.Nneu:
            stdid=stdid-self.Nneu
        if devid>=self.Nneu:
            devid=devid-self.Nneu    

        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['E_pop_rate'][stdid,self.drawstart:],'k',label='Std')
        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
            if i != 7 and i != 6:
                axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency
        if left is not None:
            axes.set_xlim(left=left)
        MMN= (self.Output_dic['E_pop_rate'][stdid,self.LastInd]-
            self.Output_dic['E_pop_rate'][devid,self.OddInd])
        if self.pf:
            fig.savefig(self.sdir+'OmissPyrId'+ f"{shift_id}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'OmissPyrId'+ f"{shift_id}"+'.svg',bbox_inches='tight')


        axes.set_title('OmissPyr'+ f"{MMN:.4f}"+'DegreeShift'+ f"{shift_id}")
        return MMN
    





    def PlotPyrTimes(self):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))
        Width=  self.N_eachC//2
        First_rate_vec= np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd:self.FirstInd+self.shift], 
            axis=1)
        Late_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd:self.LastInd+self.shift], 
            axis=1)
        Odd_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd:self.OddInd+self.shift], 
            axis=1)

        axes.plot(np.arange(self.N_eachC,0,-1),First_rate_vec,'k',label='First')
        axes.plot(np.arange(self.N_eachC,0,-1),Late_rate_vec,'b',label='Last')
        axes.plot(np.arange(self.N_eachC,0,-1),Odd_rate_vec,'r',label='Oddball')
        axes.legend()

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.get_xaxis().set_visible(False)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes'+ '.svg',bbox_inches='tight')
        axes.set_title('Firing rate, First, last, dev')
 
    def PlotPyrTimes_all(self):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))

        axes.plot(self.neu_axis,self.First_rate_vec,'k',label='First')
        axes.plot(self.neu_axis,self.Last_rate_vec,'b',label='Last')
        axes.plot(self.neu_axis,self.baseline_rate_vec,'--k',label='Base')
        if self.omission_flag==False:
            axes.plot(self.neu_axis,self.Odd_rate_vec,'r',label='Oddball')
        else:
            axes.plot(self.neu_axis,self.Odd_rate_vec,'r',label='Omission')

#        if self.omission_flag==False:
#            axes.plot(self.neu_axis,self.Odd_rate_vec,'r',label='Oddball')
#        else:
#            axes.plot(self.neu_axis,self.Odd_rate_vec,'r',label='Omission')
        axes.legend()

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        #axes.get_xaxis().set_visible(False)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_all'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_all'+ '.svg',bbox_inches='tight')
        axes.set_title('Firing rate_all, First, last, dev')
    def PlotPyrTimes_col(self,Col_id=2,linewidth=2):
        #color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))
        if Col_id is None:    
            Nneu_vec=np.arange(self.Nneu)
        else:
            Nneu_vec=np.arange(Col_id*self.N_eachC,(Col_id+1)*self.N_eachC)
        x_axis_rev=np.arange(self.N_eachC-1,-0.1,-1)    
        axes.plot(x_axis_rev,self.First_rate_vec[Nneu_vec],'k',label='First',linewidth=linewidth)
        axes.plot(x_axis_rev,self.Last_rate_vec[Nneu_vec],color='#3182BD',label='Last',linewidth=linewidth)
        axes.plot(x_axis_rev,self.baseline_rate_vec[Nneu_vec],'--k',label='Base',linewidth=linewidth)
        if self.omission_flag==False:
            axes.plot(x_axis_rev,self.Odd_rate_vec[Nneu_vec],color='#E34A33',label='Oddball',linewidth=linewidth)
        else:
            axes.plot(x_axis_rev,self.Odd_rate_vec[Nneu_vec],color='#E34A33',label='Omission',linewidth=0.5)

        axes.legend()


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_col'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_col'+ '.svg',bbox_inches='tight')
        axes.set_title('Firing rates_col')
        
        fig, axes=plt.subplots(1,1,figsize=(2,2))
        iOmi= ((self.Odd_rate_vec[Nneu_vec[self.Res_flag]]-self.Last_rate_vec[Nneu_vec[self.Res_flag]])/
              (self.Odd_rate_vec[Nneu_vec[self.Res_flag]]+self.Last_rate_vec[Nneu_vec[self.Res_flag]]
               )  )  #-2*self.baseline_rate_vec[Nneu_vec[self.Res_flag]]
        bin_edges = np.arange(-1, 1.1, 0.1)
        '''
        axes.hist(iOmi, bins=bin_edges, density=True, color='grey', alpha=0.5, 
                  edgecolor='black', linewidth=0.5)        
        axes.axvline(np.mean(iOmi), color='grey', linestyle='--', linewidth=1.5)
        '''       
        
        fig, axes=plt.subplots(1,1,figsize=(2,2))
        # Compute the histogram data
        counts, edges = np.histogram(iOmi, bins=bin_edges, density=True)

        # Plot the histogram manually
        for i in range(len(edges) - 1):
            # Get the bin's start, end, and count
            left = edges[i]-0.05
            right = edges[i + 1]-0.05
            height = counts[i]

            # Set the color based on the bin height
            facecolor = 'red' if left > 0.3 else 'grey'

            # Plot the bar
            axes.bar(left, height, width=right - left, color=facecolor, alpha=0.5, edgecolor='black', linewidth=0.5)
        axes.axvline(np.mean(iOmi), color='grey', linestyle='--', linewidth=1.5)
        if self.pf:
            fig.savefig(self.sdir+'iOmidist'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'iOmidist'+ '.svg',bbox_inches='tight')
        axes.set_title('iOmidist')
        return np.mean(iOmi)
    def PlotPyrTimes_all_parras(self,linewidth=1.5, thr=1):
        #color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red
        
        Res_Parras_flag=(
            (self.First_rate_vec>(1+thr)*self.baseline_rate_vec) |
            (self.Last_rate_vec>(1+thr)*self.baseline_rate_vec) |
            (self.Odd_rate_vec>(1+thr)*self.baseline_rate_vec) 
        ) 
        First_rate_vec=self.First_rate_vec-self.baseline_rate_vec
        Last_rate_vec=self.Last_rate_vec-self.baseline_rate_vec
        Odd_rate_vec=self.Odd_rate_vec-self.baseline_rate_vec

        Norm_vec= np.sqrt(First_rate_vec**2+Last_rate_vec**2+Odd_rate_vec**2)
        Dev_vec= Odd_rate_vec/Norm_vec
        STD_vec= Last_rate_vec/Norm_vec
        CTR_vec= First_rate_vec/Norm_vec
        #iMM_vec= Dev_vec - STD_vec
        #iRS_vec= CTR_vec- STD_vec
        #iPE_vec= Dev_vec -CTR_vec
        iMM_vec= Dev_vec[Res_Parras_flag] - STD_vec[Res_Parras_flag]
        iRS_vec= CTR_vec[Res_Parras_flag]- STD_vec[Res_Parras_flag]
        iPE_vec= Dev_vec[Res_Parras_flag] -CTR_vec[Res_Parras_flag]

        fig,axes=plt.subplots(1,1,figsize=(10,8)) # sanity check
        axes.plot(self.neu_axis,self.First_rate_vec,'k',label='First')
        axes.plot(self.neu_axis,self.Last_rate_vec,'b',label='Last')
        axes.plot(self.neu_axis,self.Odd_rate_vec,'r',label='Odd')
        axes.plot(self.neu_axis,self.baseline_rate_vec,'g',label='base')


        fig,axes=plt.subplots(1,1,figsize=(10,8)) # sanity check
        axes.plot(self.neu_axis,CTR_vec,'k',label='First')
        axes.plot(self.neu_axis,STD_vec,'b',label='Last')
        axes.plot(self.neu_axis,Dev_vec,'r',label='Odd')

        fig,axes=plt.subplots(1,1,figsize=(10,8)) # sanity check
        axes.plot(self.neu_axis[Res_Parras_flag],iMM_vec,'.k',label='First')
        axes.plot(self.neu_axis[Res_Parras_flag],iRS_vec,'.b',label='Last')
        axes.plot(self.neu_axis[Res_Parras_flag],iPE_vec,'.r',label='Odd')



        fig, axes=plt.subplots(1,3,figsize=(5,2))
        bin_edges = np.arange(-1, 1.1, 0.1)
        axes[0].hist(iMM_vec, bins=bin_edges, density=True, color='grey', alpha=0.5, 
                  edgecolor='black', linewidth=0.5)        
        axes[0].axvline(np.mean(iMM_vec), color='grey', linestyle='--', linewidth=1.5)
        
        axes[1].hist(iRS_vec, bins=bin_edges, density=True, color='#3182BD', alpha=0.5, 
                  edgecolor='black', linewidth=0.5)        
        axes[1].axvline(np.mean(iRS_vec), color='#3182BD', linestyle='--', linewidth=1.5)
        
        axes[2].hist(iPE_vec, bins=bin_edges, density=True, color='#E34A33', alpha=0.5, 
                  edgecolor='black', linewidth=0.5)        
        axes[2].axvline(np.mean(iPE_vec), color='#E34A33', linestyle='--', linewidth=1.5)
        
        # Compute the histogram data
        if self.pf:
            fig.savefig(self.sdir+'ParresAnlystDists'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'ParresAnlystDists'+ '.svg',bbox_inches='tight')
        axes[0].set_title('iMM_vec')
        axes[1].set_title('iRS_vec')
        axes[2].set_title('iPE_vec')

        return np.mean(iPE_vec)
    def PlotPyrTimes_col_omiss(self,Col_id=2,linewidth=0.5):
        #color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))
        if Col_id is None:    
            Nneu_vec=np.arange(self.Nneu)
        else:
            Nneu_vec=np.arange(Col_id*self.N_eachC,(Col_id+1)*self.N_eachC)
        x_axis_rev=np.arange(self.N_eachC-1,-0.1,-1)    
        axes.plot(x_axis_rev,self.First_rate_vec[Nneu_vec]-self.First_pre_rate_vec[Nneu_vec],'k',label='First',linewidth=linewidth)
        axes.plot(x_axis_rev,self.Last_rate_vec[Nneu_vec]-self.Last_pre_rate_vec[Nneu_vec],color='#3182BD',label='Last',linewidth=linewidth)
        #axes.plot(x_axis_rev,self.baseline_rate_vec[Nneu_vec]-self.baseline_pre_rate_vec[Nneu_vec],'--k',label='Base',linewidth=linewidth)
        if self.omission_flag==False:
            axes.plot(x_axis_rev,self.Odd_rate_vec[Nneu_vec]-self.Odd_pre_rate_vec[Nneu_vec],color='#E34A33',label='Oddball',linewidth=linewidth)
        else:
            axes.plot(x_axis_rev,self.Odd_rate_vec[Nneu_vec]-self.Odd_pre_rate_vec[Nneu_vec],color='#E34A33',label='Omission',linewidth=linewidth)

        axes.legend()


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_col_omiss'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_col_omiss'+ '.svg',bbox_inches='tight')
        axes.set_title('Firing rates_col')
        
        fig, axes=plt.subplots(1,1,figsize=(2,2))
        iOmi= ((self.Odd_rate_vec[Nneu_vec[self.Res_flag]]-self.Last_rate_vec[Nneu_vec[self.Res_flag]])/
              (self.Odd_rate_vec[Nneu_vec[self.Res_flag]]+self.Last_rate_vec[Nneu_vec[self.Res_flag]]
               )  )  #-2*self.baseline_rate_vec[Nneu_vec[self.Res_flag]]
        bin_edges = np.arange(-1, 1.1, 0.1)
        '''
        axes.hist(iOmi, bins=bin_edges, density=True, color='grey', alpha=0.5, 
                  edgecolor='black', linewidth=0.5)        
        axes.axvline(np.mean(iOmi), color='grey', linestyle='--', linewidth=1.5)
        '''       
        
        fig, axes=plt.subplots(1,1,figsize=(2,2))
        # Compute the histogram data
        counts, edges = np.histogram(iOmi, bins=bin_edges, density=True)

        # Plot the histogram manually
        for i in range(len(edges) - 1):
            # Get the bin's start, end, and count
            left = edges[i]-0.05
            right = edges[i + 1]-0.05
            height = counts[i]

            # Set the color based on the bin height
            facecolor = 'red' if left > 0.3 else 'grey'

            # Plot the bar
            axes.bar(left, height, width=right - left, color=facecolor, alpha=0.5, edgecolor='black', linewidth=0.5)
        axes.axvline(np.mean(iOmi), color='grey', linestyle='--', linewidth=1.5)
        if self.pf:
            fig.savefig(self.sdir+'iOmidist'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'iOmidist'+ '.svg',bbox_inches='tight')
        axes.set_title('iOmidist')
        return np.mean(iOmi)


    def PlotPyrTimes_sample(self,shift_id=0,top=20):
        fig, axes=plt.subplots(1,1,figsize=(1.2,2.4))
        Width=  self.N_eachC//2

        if self.omission_flag==False:
            rep = ['C', 'R', 'D']
        else:
            rep = ['C', 'R', 'O']
        Firstrate= self.First_rate_vec[self.std+shift_id]
        Lastrate= self.Last_rate_vec[self.std+shift_id]
        Oddrate= self.Odd_rate_vec[self.std+shift_id]

        rates=[Firstrate,Lastrate,Oddrate]
        axes.bar(rep, rates,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_ylim(top=top)  # Set the max y-value to 5

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_sample'+ str(self.std+shift_id)+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_sample'+ str(self.std+shift_id)+'.svg',bbox_inches='tight')
        axes.set_title('MMNPyrTimes_sample'+ str(shift_id+Width))

    def PlotPyrTimes_2samples(self,shift_id1=0,shift_id2=0,top=20):
        fig, axes=plt.subplots(1,1,figsize=(1.8,2.4))
        Width=  self.N_eachC//2

        if self.omission_flag==False:
            rep = ['C_pPE', 'D_pPE','C_nPE', 'D_nPE']
        else:
            rep = ['C', 'R', 'O']
        Firstrate1= self.First_rate_vec[self.std+shift_id1]
        Oddrate1= self.Odd_rate_vec[self.std+shift_id1]
        baselinerate1= self.baseline_rate_vec[self.std+shift_id1]

        Firstrate2= self.First_rate_vec[self.std+shift_id2]
        Oddrate2= self.Odd_rate_vec[self.std+shift_id2]
        baselinerate2= self.baseline_rate_vec[self.std+shift_id2]



        rates=[Firstrate1-baselinerate1,Oddrate1-baselinerate1,Firstrate2-baselinerate2,Oddrate2-baselinerate2]
        axes.bar(rep, rates,color=['#808080', '#E34A33','#808080', '#E34A33']) # grey, blue, red

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_ylim(top=top)  # Set the max y-value to 5

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_2samples'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_2samples'+ '.svg',bbox_inches='tight')
        axes.set_title('MMNPyrTimes_pPEnPE')



    def PlotPyrTimes_pop(self):
        fig, axes=plt.subplots(1,1,figsize=(1,2))
        Width=  self.N_eachC//2

        rep = ['C', 'R', 'D']
        Firstrate= np.mean(self.Output_dic['E_pop_rate'][self.std-Width:self.std+Width,self.FirstInd:self.FirstInd+self.shift])
        Lastrate= np.mean(self.Output_dic['E_pop_rate'][self.std-Width:self.std+Width,self.LastInd:self.LastInd+self.shift])
        Oddrate= np.mean(self.Output_dic['E_pop_rate'][self.dev-Width:self.dev+Width,self.OddInd:self.OddInd+self.shift])
        Baselinerate=np.mean(self.Output_dic['E_pop_rate'][self.std-Width:self.std+Width,self.drawstart:self.drawstart+self.shift])
        
        '''
        rates=[
            self.Output_dic['E_pop_rate'][self.std+shift_id,self.FirstInd],
            self.Output_dic['E_pop_rate'][self.std+shift_id,self.LastInd],
            self.Output_dic['E_pop_rate'][self.dev+shift_id,self.OddInd],
        ]
        '''
        rates=[Firstrate,Lastrate,Oddrate]-Baselinerate
        axes.bar(rep, rates,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_ylim(top=3)  # Set the max y-value to 5

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_popmean'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_popmean'+'.svg',bbox_inches='tight')
        axes.set_title('MMNPyrTimes_popmean')
    def PlotPyrTimes_pop_all(self):
        fig, axes=plt.subplots(1,1,figsize=(1,2))
        Width=  self.N_eachC//2

        Baselinerate=np.mean(self.baseline_rate_vec)
        Firstrate=np.mean(self.First_rate_vec)
        Lastrate=np.mean(self.Last_rate_vec)
        Oddrate=np.mean(self.Odd_rate_vec)
        
        '''
        rates=[
            self.Output_dic['E_pop_rate'][self.std+shift_id,self.FirstInd],
            self.Output_dic['E_pop_rate'][self.std+shift_id,self.LastInd],
            self.Output_dic['E_pop_rate'][self.dev+shift_id,self.OddInd],
        ]
        '''
        if self.omission_flag==False:
            rates=[Firstrate,Lastrate,Oddrate]-Baselinerate
            rep = ['C', 'R', 'D']

        else:
            rates=[Firstrate,Lastrate,Oddrate]-Baselinerate
            rep = ['C', 'R', 'O']

        axes.bar(rep, rates,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        #axes.set_ylim(top=auto)  # Set the max y-value to 5

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_pop_all_mean'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_pop_all_mean'+'.svg',bbox_inches='tight')
        axes.set_title('MMNPyrTimes_pop_all_mean')
        if self.omission_flag==False:
            return Oddrate-Firstrate
        else:
            return Oddrate

    def PlotPyrTimes_pop_RespOrNot(self):
        fig, axes=plt.subplots(1,1,figsize=(1,2))
        Width=  self.N_eachC//2
        thr=1
        biase=0.
        Res_flag= self.First_rate_vec> (1+thr)*self.baseline_rate_vec+biase
        self.Res_flag=Res_flag

        Baselinerate_res=np.mean(self.baseline_rate_vec[Res_flag])
        Firstrate_res=np.mean(self.First_rate_vec[Res_flag])
        Lastrate_res=np.mean(self.Last_rate_vec[Res_flag])
        Oddrate_res=np.mean(self.Odd_rate_vec[Res_flag])

        rates=[Firstrate_res,Lastrate_res,Oddrate_res]-Baselinerate_res
        rep = ['C', 'R', 'D']

        axes.bar(rep, rates,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        #axes.set_ylim(top=auto)  # Set the max y-value to 5

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_pop_res_mean'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_pop_res_mean'+'.svg',bbox_inches='tight')
        axes.set_title('MMNPyrTimes_pop_res_mean')

        fig, axes=plt.subplots(1,1,figsize=(1,2))

        Baselinerate_Nores=np.mean(self.baseline_rate_vec[~Res_flag])
        Firstrate_Nores=np.mean(self.First_rate_vec[~Res_flag])
        Lastrate_Nores=np.mean(self.Last_rate_vec[~Res_flag])
        Oddrate_Nores=np.mean(self.Odd_rate_vec[~Res_flag])

        rates=[Firstrate_Nores,Lastrate_Nores,Oddrate_Nores]-Baselinerate_Nores
        rep = ['C', 'R', 'D']

        axes.bar(rep, rates,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        #axes.set_ylim(top=auto)  # Set the max y-value to 5

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_pop_NoRes_mean'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_pop_NoRes_mean'+'.svg',bbox_inches='tight')
        axes.set_title('MMNPyrTimes_pop_NoRes_mean')





    def PlotPyrTimes_pop_all_long(self,left=50,right=60,nRep=None):
        # This one is to detect the overall firing rate changes. 
        # Top aveage firing rate, bottom, expectancy of the vector. Using choices.
        # Include a seperate plot for the average of the whole population
        n_stim= self.Paras['nbr_rep_std']+ self.Paras['nbr_rep_dev']
        # sanity
        #n_rep= round((self.n_frames-self.FirstInd)/n_stim)
        n_trial_frame=round((self.Paras['Tinter']+self.Paras['Tstim'])/self.dtframe)
        
        Pop_average=np.mean(self.Output_dic['E_pop_rate'], axis=0) 

        choices= self.Paras['choices']    # 1 for std, 2 for dev
       
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))  # sanity test on the firing rate
        axes.plot(self.t_axis[self.drawstart:]+self.t0,Pop_average[self.drawstart:],'k')
        for i_stim in np.arange(left,right):
            if (nRep is not None) and i_stim>nRep:
                continue
            start=i_stim
            end= i_stim+self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            if round(choices[i_stim])==1:    
                axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           
            else:
                axes.axvspan(start, end, color='red', alpha=0.3,label='dev')  # Adjust `alpha` for transparency               
        axes.set_xlim([left-0.5,right])
        #axes.set_xlim([-0.5,10])
        if self.pf:
            fig.savefig(self.sdir+'ProbMMN'+'PopAverage'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'ProbMMN'+'PopAverage'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'PopAverage')

        stim_averages = []

        # Iterate over the frames in steps of 5
        for i in range(self.FirstInd, self.n_frames, n_trial_frame):
            if i + self.shift <= self.n_frames:  # Ensure there are at least two frames to average
                avg = np.mean(Pop_average[i:i+self.shift ])  # Average of frames [i, i+1]
                stim_averages.append(avg)

        # Now I plot the expectancy with alpha=0.6
        exp_std_vec=np.full(n_stim,np.nan, dtype=np.float32)
        exp_dev_vec=np.full(n_stim,np.nan, dtype=np.float32)

        decay_rate=0.6
        decay_norm=(decay_rate-decay_rate**11)/(1-decay_rate)
        for i in range (10,n_stim):
            start_idx = max(0, i - 10)
            weights = decay_rate ** (np.arange(i - start_idx, 0, -1))/decay_norm
            exp_std_vec[i] = np.sum(weights * (2-choices[start_idx:i]))
            exp_dev_vec[i] = np.sum(weights * (choices[start_idx:i]-1))

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))  # sanity test on the firing rate
        axes.plot(exp_std_vec,'.b')
        axes.plot(exp_std_vec,'k')
        axes.plot(exp_dev_vec,'.',color='brown')
        axes.plot(exp_dev_vec,'#E34A33')
        for i_stim in np.arange(left,right):
            start=i_stim
            end= i_stim+0.2
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            if round(choices[i_stim])==1:    
                axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           
            else:
                axes.axvspan(start, end, color='red', alpha=0.3,label='dev')  # Adjust `alpha` for transparency               
        if left is not None:
            axes.set_xlim(left=left-0.5)
        if right is not None:
            axes.set_xlim(right=right)
        if self.pf:
            fig.savefig(self.sdir+'ProbMMN'+'Expectancy'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'ProbMMN'+'Expectancy'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'Expectancy')

        fig, axes=plt.subplots(3,1,figsize=(2.4,1.8))  # sanity test on the firing rate
        axes[0].plot(stim_averages,'.')
        axes[0].set_xlim([left-0.5,right])     # left=0
        exp_all=exp_std_vec*(2-choices)+exp_dev_vec*(choices-1)
        axes[1].plot(exp_all)
        axes[1].set_xlim([left-0.5,right])
        axes[2].plot(exp_std_vec,'.-k')
        axes[2].plot(exp_dev_vec,'.-b')
        axes[2].set_xlim([left-0.5,right])
        axes[2].plot(choices,'.')

        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))  # sanity test on the firing rate
        axes.plot(exp_all,stim_averages,'k')
        axes.plot(exp_all,stim_averages,'.b')

    def PlotPyrTimes_pop_nRep(self,nRep,dev_flag=False,left=None,right=None):
        Pop_average=np.mean(self.Output_dic['E_pop_rate'], axis=0) 
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))  # sanity test on the firing rate
        axes.plot(self.t_axis[self.drawstart:]+self.t0,Pop_average[self.drawstart:],'k')
        for i_stim in np.arange(nRep):
            start=i_stim
            end= i_stim+self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           
        if dev_flag is True:
            start=nRep
            end= start+self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
            axes.axvspan(start, end, color='red', alpha=0.3)  # Adjust `alpha` for transparency           
            MMN_pop= np.mean(Pop_average[self.OddInd:self.OddInd+self.shift]-
                Pop_average[self.LastInd:self.LastInd+self.shift])

        axes.set_xlim(left=-1)
        if self.pf:
            fig.savefig(self.sdir+'ProbMMN'+'PopAverage'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'ProbMMN'+'PopAverage'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'PopAverage')

        return MMN_pop

    def PlotPyrTimes_pop_all_prob(self,Output_dic,PltTool,decay_rate=0.6):
        from scipy.stats import linregress
        # This one is to combine all the data together, and generate the scatter plot.
        # Then we consider how to generate some diagram. 
        n_stim= self.Paras['nbr_rep_std']+ self.Paras['nbr_rep_dev']
        # sanity
        #n_rep= round((self.n_frames-self.FirstInd)/n_stim)
        n_rep=round((self.Paras['Tinter']+self.Paras['Tstim'])/self.dtframe)
        
        # Now I need to loop through all the Output_dic
        temp_probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        stim_average_runs=[]
        exp_all_runs=[]
        decay_norm=(decay_rate-decay_rate**11)/(1-decay_rate)
        for i_dic in range(6):
            Pop_average=np.mean(Output_dic[i_dic]['E_pop_rate'], axis=0)        
            stim_averages = []

            # Iterate over the frames in steps of 5
            for i in range(self.FirstInd, self.n_frames, n_rep):
                if i + self.shift <= self.n_frames:  # Ensure there are at least two frames to average
                    avg = np.mean(Pop_average[i:i+self.shift ])  # Average of frames [i, i+1]
                    stim_averages.append(avg)

            # Now I plot the expectancy with alpha=0.6
            exp_std_vec=np.full(n_stim,np.nan, dtype=np.float32)
            exp_dev_vec=np.full(n_stim,np.nan, dtype=np.float32)

            choices= PltTool[i_dic].Paras['choices']    # 1 for std, 2 for dev
            for i in range (10,n_stim):
                start_idx = max(0, i - 10)
                weights = decay_rate ** (np.arange(i - start_idx, 0, -1))/decay_norm
                exp_std_vec[i] = np.sum(weights * (2-choices[start_idx:i]))
                exp_dev_vec[i] = np.sum(weights * (choices[start_idx:i]-1))
            exp_all=exp_std_vec*(2-choices)+exp_dev_vec*(choices-1)    
            stim_average_runs.append(stim_averages)
            exp_all_runs.append(exp_all)
        '''
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))  # sanity test on the firing rate
        axes.plot(exp_all,stim_averages,'k')
        axes.plot(exp_all,stim_averages,'.b')
        '''


        # Now let's generate a plot
        stim_average_norm=stim_averages[-1]
        shapes = ['o', '^','s',  'v', 'D']  # Example markers: circle, square, diamond, triangle up, triangle down
        markersize=20
        # Define shades of blue using a colormap
        #cmap = plt.cm.Blues
        #colors = [cmap(i) for i in np.linspace(0.3, 0.9, 5)]  # Adjust the range for lighter/darker blues

        colors = ['blue', 'red', 'green',  'purple','orange']  # Distinct colors

        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))  # sanity test on the firing rate
        selected_indices = [0, 2, 4]
#        for i_dic in range(5):
        for i_dic in selected_indices:
            x_subset = exp_all_runs[i_dic][::5]
            y_subset = (np.array(stim_average_runs[i_dic]) - stim_average_norm)[::5]

            axes.scatter(x_subset,y_subset,
                    edgecolor='black',     linewidths=0.5,  
                    marker=shapes[i_dic],  facecolor=colors[i_dic],
                    s=markersize, label=f'Prob{temp_probs[i_dic]}',alpha=0.6)
        axes.legend(fontsize=8)

        # Now generate a linear regression.
        x_combined = []
        y_combined = []
        for i_dic in range(5):
            x_subset = exp_all_runs[i_dic]
            y_subset = np.array(stim_average_runs[i_dic]) - stim_average_norm
            # Append to combined arrays
            x_combined.extend(x_subset)
            y_combined.extend(y_subset)            
        
        x_combined = np.array(x_combined)
        y_combined = np.array(y_combined)
        valid_indices = ~np.isnan(x_combined) & ~np.isnan(y_combined)
        x_combined = x_combined[valid_indices]
        y_combined = y_combined[valid_indices]

        slope, intercept, r_value, p_value, std_err = linregress(x_combined, y_combined)

        # Plot the fitted line
        x_fit = np.linspace(min(x_combined), max(x_combined), 100)
        y_fit = slope * x_fit + intercept
        axes.plot(x_fit, y_fit, color='black', linestyle='--')

        if self.pf:
            fig.savefig(self.sdir+'ProbMMN'+f'Decay{decay_rate:.3f}'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'ProbMMN'+f'Decay{decay_rate:.3f}'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+f'Decay{decay_rate:.3f}')


        return r_value**2        

    def PlotPyrTimes_pop_omission(self,Col_id=2,thr=1):
        fig, axes=plt.subplots(1,1,figsize=(1,2))
        
        if Col_id is None:    
            Nneu_vec=np.arange(self.Nneu)
        else:
            Nneu_vec=np.arange(Col_id*self.N_eachC,(Col_id+1)*self.N_eachC)

        Baselinerate=np.mean(self.baseline_rate_vec[Nneu_vec])
        Firstrate=np.mean(self.First_rate_vec[Nneu_vec])
        Lastrate=np.mean(self.Last_rate_vec[Nneu_vec])
        Oddrate=np.mean(self.Odd_rate_vec[Nneu_vec])
        First_prerate=np.mean(self.First_pre_rate_vec[Nneu_vec])
        Last_prerate=np.mean(self.Last_pre_rate_vec[Nneu_vec])        
        Odd_prerate=np.mean(self.Odd_pre_rate_vec[Nneu_vec])

        '''
        rates=[
            self.Output_dic['E_pop_rate'][self.std+shift_id,self.FirstInd],
            self.Output_dic['E_pop_rate'][self.std+shift_id,self.LastInd],
            self.Output_dic['E_pop_rate'][self.dev+shift_id,self.OddInd],
        ]
        '''
        if self.omission_flag==False:
            rates=[Firstrate,Lastrate,Oddrate]-Baselinerate
            rep = ['C', 'R', 'D']

        else:
            rates=[Firstrate,Lastrate,Oddrate]-Baselinerate

            #rates=[Firstrate-First_prerate,Lastrate-Last_prerate,Oddrate-Odd_prerate]
            rep = ['C', 'R', 'O']

        axes.bar(rep, rates,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        #axes.set_ylim(top=auto)  # Set the max y-value to 5

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'OmissionPyrTimes_pop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'OmissionPyrTimes_pop'+'.svg',bbox_inches='tight')
        axes.set_title('OmissionPyrTimes_pop')

        # Plot the omiss response neruons
        #thr=1.   # 1.645 reverse of p=0.05; thr=1.28   # reverse of p=0.1   
        #Res_flag= self.Odd_rate_vec[Nneu_vec]> (1+thr)*self.baseline_rate_vec[Nneu_vec]
        Res_flag= self.Odd_rate_vec[Nneu_vec]> (1+thr)*self.Odd_pre_rate_vec[Nneu_vec]
        
        n_omiss=np.sum(Res_flag)
        self.Res_flag=Res_flag
        Baselinerate_omiss=np.mean(self.baseline_rate_vec[Nneu_vec[Res_flag]])
        Firstrate_omiss=np.mean(self.First_rate_vec[Nneu_vec[Res_flag]])
        Lastrate_omiss=np.mean(self.Last_rate_vec[Nneu_vec[Res_flag]])
        Oddrate_omiss=np.mean(self.Odd_rate_vec[Nneu_vec[Res_flag]])
        First_prerate_omiss=np.mean(self.First_pre_rate_vec[Nneu_vec[Res_flag]])
        Last_prerate_omiss=np.mean(self.Last_pre_rate_vec[Nneu_vec[Res_flag]])        
        Odd_prerate_omiss=np.mean(self.Odd_pre_rate_vec[Nneu_vec[Res_flag]])
        

        rates=[Firstrate_omiss,
               Lastrate_omiss,
               Oddrate_omiss]-Baselinerate_omiss

        '''
        rates=[Firstrate_omiss-First_prerate_omiss,
               Lastrate_omiss-Last_prerate_omiss,
               Oddrate_omiss-Odd_prerate_omiss]

        rates=[Baselinerate_omiss,
               Lastrate_omiss,
               Oddrate_omiss]
        '''       
        rep = ['C', 'R', 'O']
        fig, axes=plt.subplots(1,1,figsize=(1,2))
        axes.bar(rep, rates,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red
        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'OmissionPyrTimes_res'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'OmissionPyrTimes_res'+'.svg',bbox_inches='tight')
        axes.set_title('OmissionPyrTimes_res')


        if self.omission_flag==False:
            return Oddrate-Firstrate, n_omiss
        else:
            return Oddrate-Baselinerate, n_omiss


    def MyBarPlot(self,rep,rates,errors=None,colors=None,  tag='', top=None):
        fig, axes=plt.subplots(1,1,figsize=(2,2))
        if colors is None:
            axes.bar(rep, rates,yerr=errors,capsize=5)
        else:
            axes.bar(rep, rates,yerr=errors,capsize=5,color=colors)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        if top is not None:
            axes.set_ylim(top=top)  # Set the max y-value to 5
        axes.set_xticklabels(rep, rotation=45, ha='center', fontsize=12)  # Tilt the labels 45 degrees, right-aligned
        
        axes.get_xaxis().set_visible(True)


        if self.pf:
            fig.savefig(self.sdir+'MyBarPlot'+tag+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MyBarPlot'+tag+'.svg',bbox_inches='tight')
        axes.set_title('Bar Plot'+tag)

    def PlotInhPopITimes(self):
        fig, axes=plt.subplots(1,2,figsize=(8,3))
        ind_vec=np.arange(self.N_eachC)
        align_shift= int(self.dev-self.std)
        axes[0].plot(ind_vec,self.Output_dic['P_pop_sumI'][self.std_neus_vec,self.FirstInd],label='FirstA')
        axes[0].plot(ind_vec,self.Output_dic['P_pop_sumI'][self.std_neus_vec,self.LastInd],label='LastA')
        axes[0].plot(ind_vec,self.Output_dic['P_pop_sumI'][self.dev_neus_vec,self.OddInd],label='DevB')
        axes[0].legend()
        axes[0].set_title('Flashshot P_Isum, First, last, dev')

        axes[1].plot(ind_vec,self.Output_dic['S_pop_sumI'][self.std_neus_vec,self.FirstInd],label='FirstA')
        axes[1].plot(ind_vec,self.Output_dic['S_pop_sumI'][self.std_neus_vec,self.LastInd],label='LastA')
        axes[1].plot(ind_vec,self.Output_dic['S_pop_sumI'][self.dev_neus_vec,self.OddInd],label='DevB')
        axes[1].legend()
        axes[1].set_title('Flashshot S_Isum, First, last, dev')

        '''
        axes.set_title('Flashshot, First, last, dev')


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes'+ '.jpg',bbox_inches='tight')
        '''

    def PlotPyrDDInd(self):
        fig, axes=plt.subplots(1,1,figsize=(4,3))
        xaxis=np.arange(self.N_eachC,0,-1)
        Width=  self.N_eachC//2
        First_rate_vec= np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd:self.FirstInd+self.shift], 
            axis=1)
        Late_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd:self.LastInd+self.shift], 
            axis=1)
        Odd_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd:self.OddInd+self.shift], 
            axis=1)
        DD_vec=Odd_rate_vec-First_rate_vec
        SSA_vec=First_rate_vec-Late_rate_vec
        '''
        DD_vec= (self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd]-
                  self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd])
        SSA_extra_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd]-
                  self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd])
        '''  
        axes.plot(xaxis,DD_vec,'.',color='#c51b8a')
        axes.plot(xaxis,DD_vec,'r',label='DD')
        axes.plot(xaxis,SSA_vec,'k',label='SSA')
        axes.plot(xaxis,0*np.arange(self.N_eachC),'--',color='#D3D3D3')
        axes.legend()
        axes.plot([0,self.N_eachC],[np.mean(DD_vec),np.mean(DD_vec)],'--r',linewidth=2)
        axes.plot([0,self.N_eachC],[np.mean(SSA_vec),np.mean(SSA_vec)],'--k',linewidth=2)
        axes.plot(xaxis,SSA_vec,'.b')

        #axes.get_xaxis().set_visible(False)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimesDif'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimesDif'+ '.svg',bbox_inches='tight')
        axes.set_title('DD(MMN) over Input Affinity')

    def PlotPyrDDInd_all(self):
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        #xaxis=np.arange(self.N_eachC,0,-1)
        if self.omission_flag==False:
            title_str='DD(MMN) of All neuron over Input Affinity'
        else:
            title_str='omission of All neuron over Input Affinity'
            
        '''
        DD_vec= (self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd]-
                  self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd])
        SSA_extra_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd]-
                  self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd])
        '''
        if self.omission_flag==False:
            axes.plot(self.neu_axis,self.DD_vec[::-1],'r',label='DD')
        else:
            axes.plot(self.neu_axis,self.DD_vec[::-1],'r',label='Omission')



        axes.plot(self.neu_axis,self.SSA_vec[::-1],'k',label='SSA')
        axes.plot(self.neu_axis,0*np.arange(self.Nneu),'--',color='#D3D3D3')
        axes.legend()
        axes.plot([1,self.Nneu],[np.mean(self.DD_vec),np.mean(self.DD_vec)],'--r',linewidth=2)
        axes.plot([1,self.Nneu],[np.mean(self.SSA_vec),np.mean(self.SSA_vec)],'--k',linewidth=2)
        
        for i in range(7):
            start = (i+1) * self.N_eachC
            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)

        #axes.get_xaxis().set_visible(False)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimesDifAll'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimesDifAll'+ '.svg',bbox_inches='tight')
        axes.set_title(title_str)

    def PlotPyrDDOmiss_all(self,omiss_vec,linewidth=2,top=None):
        #color=['#808080', '#3182BD', '#E34A33',#31a354]) # grey, blue, red, green
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        #xaxis=np.arange(self.N_eachC,0,-1)
        if self.omission_flag==False:
            title_str='Activity of All neuron over Input Affinity'
        else:
            title_str='omission of All neuron over Input Affinity'
            
        Odd_rev=      self.Odd_rate_vec[::-1]-        self.baseline_rate_vec[::-1]  
        First_rev=    self.First_rate_vec[::-1]-      self.baseline_rate_vec[::-1]
        Last_rev=     self.Last_rate_vec[::-1]-       self.baseline_rate_vec[::-1]
        axes.plot(self.neu_axis,Odd_rev,color='#E34A33',label='Deviant',linewidth=linewidth)
        axes.plot(self.neu_axis,First_rev,color='black',label='First',linewidth=linewidth)
        axes.plot(self.neu_axis,Last_rev,color='#3182BD',label='Last',linewidth=linewidth)
        
        
        neu_axis_omiss=np.concatenate((np.arange(self.Nneu//2,self.Nneu),np.arange(0,self.Nneu//2)))
        omiss_reverse=omiss_vec[::-1]- self.baseline_rate_vec[::-1]
        omiss_rev=omiss_reverse[neu_axis_omiss]
        axes.plot(self.neu_axis,omiss_rev,color='#31a354',label='Omission',linewidth=linewidth)
        
        axes.plot(self.neu_axis,0*np.arange(self.Nneu),'--',color='#D3D3D3',linewidth=0.5)
        
        for i in range(7):
            start = (i+1) * self.N_eachC
            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrDDOmiss_all'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrDDOmiss_all'+ '.svg',bbox_inches='tight')
        axes.legend()
        axes.set_title(title_str)

        # The next is the difference between corresponding lines and the first.
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        #xaxis=np.arange(self.N_eachC,0,-1)
        if self.omission_flag==False:
            title_str='Compare to First over Input Affinity'
        else:
            title_str='omission Compare to First over Input Affinity'
            

        axes.plot(self.neu_axis,Odd_rev-First_rev,color='#E34A33',label='DD',linewidth=linewidth)
        #axes.plot(self.neu_axis,-Last_rev+ First_rev,color='#3182BD',label='SSA',linewidth=linewidth)
        
        
        neu_axis_omiss=np.concatenate((np.arange(self.Nneu//2,self.Nneu),np.arange(0,self.Nneu//2)))
        omiss_reverse_raw=omiss_vec[::-1]
        
        #axes.plot(self.neu_axis,omiss_rev,color='#31a354',label='Omiss',linewidth=linewidth)
        axes.plot(self.neu_axis,omiss_rev-First_rev,color='#31a354',label='Omiss',linewidth=linewidth)

        axes.plot(self.neu_axis,0*np.arange(self.Nneu),'--',color='#D3D3D3',linewidth=0.5)
        
        for i in range(7):
            start = (i+1) * self.N_eachC
            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrCompareFirst_all'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrCompareFirst_all'+ '.svg',bbox_inches='tight')
        axes.legend()
        axes.set_title(title_str)

        # Nest, two bar charts for the positive part and negative part
        DD_vec=Odd_rev-First_rev
        Omiss_dif=omiss_rev-First_rev
        DD_nPE_flag= (DD_vec>0) & (Omiss_dif>0)   # positive DD and positive Omiss
        DD_pPE_flag= (DD_vec>0) & (Omiss_dif<=0)
        
        categories=['C','R','D','O']
        colors=['#808080', '#3182BD', '#E34A33','#31a354']    # grey, blue, red ,green
        
        nPE_values=[np.mean(First_rev[DD_nPE_flag]),np.mean(Last_rev[DD_nPE_flag]),
                    np.mean(Odd_rev[DD_nPE_flag]),np.mean(omiss_rev[DD_nPE_flag])]
        pPE_values=[np.mean(First_rev[DD_pPE_flag]),np.mean(Last_rev[DD_pPE_flag]),
                    np.mean(Odd_rev[DD_pPE_flag]),np.mean(omiss_rev[DD_pPE_flag])]
        
        fig, axes=plt.subplots(1,1,figsize=(2.5,3))
        axes.bar(categories, nPE_values, color=colors)  # Plot bars with sky blue color
        if top is not None:
            axes.set_ylim(top=top)
        axes.set_xticklabels(categories, rotation=45, ha='center', fontsize=12)  # Tilt the labels 45 degrees, right-aligned
        if self.pf:
            fig.savefig(self.sdir+'DDnPEbar'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'DDnPEbar'+'.svg',bbox_inches='tight')
        axes.set_title('DDnPEbar')
        
        fig, axes=plt.subplots(1,1,figsize=(2.5,3))
        axes.bar(categories, pPE_values, color=colors)  # Plot bars with sky blue color
        if top is not None:
            axes.set_ylim(top=top)
        axes.set_xticklabels(categories, rotation=45, ha='center', fontsize=12)  # Tilt the labels 45 degrees, right-aligned
        if self.pf:
            fig.savefig(self.sdir+'DDpPEbar'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'DDpPEbar'+'.svg',bbox_inches='tight')
        axes.set_title('DDpPEbar')


        fig, axes=plt.subplots(1,1,figsize=(3,3))   # This one is for combined example activity.

        # Reversed index:
        pPEx_idx= self.Nneu- self.std - self.N_eachC//2
        nPEy_idx= self.Nneu- self.dev + self.N_eachC//2-1
        #pPEx_idx=200
        #nPEy_idx=79

        PE_values=[-np.mean(First_rev[DD_nPE_flag])+np.mean(Odd_rev[DD_nPE_flag]),
                   np.mean(Omiss_dif[DD_nPE_flag]),
                   -np.mean(First_rev[DD_pPE_flag])+np.mean(Odd_rev[DD_pPE_flag]),
                   np.mean(Omiss_dif[DD_pPE_flag]),
                   ]

        cat_PE=['nPE DD','nPE Omiss Dif','pPE DD','pPE Omiss Dif']
        colors_PE=[ '#E34A33','#31a354', '#E34A33','#31a354']

        axes.bar(cat_PE, PE_values, color=colors_PE)  # Plot bars with sky blue color
        if top is not None:
            axes.set_ylim(top=top)
        axes.set_xticklabels(cat_PE, rotation=45, ha='center', fontsize=12)  # Tilt the labels 45 degrees, right-aligned
        if self.pf:
            fig.savefig(self.sdir+'DDPEbars'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'DDPEbars'+'.svg',bbox_inches='tight')
        axes.set_title('DDPEbar')



        
        fig, axes=plt.subplots(1,1,figsize=(2.5,3))
        PE_values=[First_rev[pPEx_idx], Odd_rev[pPEx_idx], omiss_rev[pPEx_idx],
                   First_rev[nPEy_idx], Odd_rev[nPEy_idx], omiss_rev[nPEy_idx]
                   ]
        cat_PE=['pPEx F','pPEx D','pPEx O','nPEy F','nPEy D','nPEy O']
        colors_PE=['#808080', '#E34A33','#31a354','#808080', '#E34A33','#31a354']

        axes.bar(cat_PE, PE_values, color=colors_PE)  # Plot bars with sky blue color
        if top is not None:
            axes.set_ylim(top=top)
        axes.set_xticklabels(cat_PE, rotation=45, ha='center', fontsize=12)  # Tilt the labels 45 degrees, right-aligned
        if self.pf:
            fig.savefig(self.sdir+'DDCtxCellActbars'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'DDCtxCellActbars'+'.svg',bbox_inches='tight')
        axes.set_title('DDCtxCellActbar')




        fig, axes=plt.subplots(1,1,figsize=(1.8,1.8)) # scatter
        axes.plot(Omiss_dif,DD_vec,'.',color='grey',markersize=2)
        axes.plot(Omiss_dif[DD_vec>0],DD_vec[DD_vec>0],'.',color='#E34A33',markersize=2)
        if self.pf:
            fig.savefig(self.sdir+'DDOmissScatter'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'DDOmissScatter'+'.svg',bbox_inches='tight')
        axes.set_title('DDOmissScatter')



    def PlotPyrTimesDDSSA(self):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))

        Width=  self.N_eachC//2

        First_rate_vec= np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd:self.FirstInd+self.shift], 
            axis=1)
        Late_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd:self.LastInd+self.shift], 
            axis=1)
        Odd_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd:self.OddInd+self.shift], 
            axis=1)

        DD_vec=Odd_rate_vec-First_rate_vec
        SSA_vec=First_rate_vec-Late_rate_vec

        axes.plot(SSA_vec,DD_vec,'.k',markersize=5)
        axes.plot([min(SSA_vec),max(SSA_vec)],[0,0],'k--')
        #axes.plot([0,0],[min(DD_vec),max(DD_vec)],'k--')
        axes.plot([0,0],[-2,2],'k--')

        axes.set_ylim([-2,2])  # This changes the view to start from -1 on the x-axis
        #axes.legend()
        #axes.get_xaxis().set_visible(False)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimesDDSSA'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimesDDSSA'+ '.svg',bbox_inches='tight')
        axes.set_title('DD over SSA')

    def PlotPyrTimesDDSSA_all(self):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))

        axes.plot(self.SSA_vec,self.DD_vec,'.k',markersize=5)
        axes.plot([min(self.SSA_vec),max(self.SSA_vec)],[0,0],'k--')
        #axes.plot([0,0],[min(DD_vec),max(DD_vec)],'k--')
        axes.plot([0,0],[min(self.DD_vec),max(self.DD_vec)],'k--')

        #axes.set_ylim([-2,4])  # This changes the view to start from -1 on the x-axis
        #axes.legend()
        #axes.get_xaxis().set_visible(False)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimes_all_DDSSA'+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimes_all_DDSSA'+ '.svg',bbox_inches='tight')
        if self.omission_flag==False:
            axes.set_title('DD over SSA, all')
        else:
            axes.set_title('omission over SSA, all')

    def PlotPyrPop(self,start_color='green',end_color='#F7A541',left=None,right=None,color_scale=None,flag_shade=True):
        
        data=self.Output_dic['E_pop_rate'][:,self.drawstart:]

        normalized_data = np.array([row - row[self.FirstInd-1] if row[self.FirstInd-1] != 0 else row for row in data])
        reversed_data = np.flipud(normalized_data)

        # Find the maximum absolute value for normalization
        max_val = np.max(np.abs(normalized_data))
        if color_scale is not None:
            max_val=max_val/color_scale
        # Create a custom colormap with white at the center
        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        # Use custom normalization
        if color_scale is None:
            norm = SymLogNorm(linthresh=3, vmin=-max_val, vmax=max_val)  # Set linthresh for linear behavior near 0
        else:
            norm = SymLogNorm(linthresh=1, vmin=-max_val, vmax=max_val)  # Set linthresh for linear behavior near 0

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        im = axes.imshow(reversed_data, aspect='auto', cmap=cmap, norm=norm, extent=[self.t0+self.time_draw_start, self.tn , self.Nneu - 1, 0],
                    interpolation='none' )  # Removed vmin=-max_val, vmax=max_val
        cbar=fig.colorbar(im)
        if color_scale is None:
            cbar.set_ticks([-10, -1, 0, 1, 10])  # Set specific tick positions
            cbar.set_ticklabels(['-10', '-1', '0', '1', '10'])  # Optional custom labels
        else:
            cbar.set_ticks([-1, -0.5, 0,0.5, 1])  # Set specific tick positions
            cbar.set_ticklabels(['-1','-0.5', '0','0.5', '1'])  # Optional custom labels


        if left is not None:
            axes.set_xlim(left=left-0.5)
        if right is not None:
            axes.set_xlim(right=right)
            choices= self.Paras['choices'].copy()    # 1 for std, 2 for dev

            for i in np.arange(right-10,right):
                start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
                end = start + self.Paras['Tstim']
                # Add vertical dashed lines
                axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
                axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)      
                if round(choices[i])==2 and flag_shade:    
                    axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           
        
        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']

            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

        for i in range(7):
            start = (i+1) * self.N_eachC
            # Add vertical dashed lines
            axes.axhline(start, color='grey', linestyle='--', linewidth=0.5)

        plt.show()
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrPop'+'.svg',bbox_inches='tight')
        axes.set_title('Pyr Pop Activity')

    def PlotPyrPop_debug(self,start_color='green',end_color='#F7A541',left=None):
        
        fig, axes=plt.subplots(1,1,figsize=(4,3))
        #data=self.Output_dic['E_pop_rate'][:,self.drawstart:]
        data=self.Output_dic['E_pop_rate']
        #normalized_data = np.array([row - row[0] if row[0] != 0 else row for row in data])
        normalized_data = np.array([row - row[self.FirstInd-1] if row[self.FirstInd-1] != 0 else row for row in data])
        reversed_data = np.flipud(normalized_data)

        # Find the maximum absolute value for normalization
        max_val = np.max(np.abs(normalized_data))

        # Create a custom colormap with white at the center
        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        # Use custom normalization
        norm = SymLogNorm(linthresh=3, vmin=-max_val, vmax=max_val)  # Set linthresh for linear behavior near 0

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        im = axes.imshow(reversed_data, aspect='auto', cmap=cmap, norm=norm, extent=[self.t0, self.tn-self.dtframe , self.Nneu - 1, 0],
                    interpolation='none' )  # Removed vmin=-max_val, vmax=max_val
        cbar=fig.colorbar(im)
        cbar.set_ticks([-10, -1, 0, 1, 10])  # Set specific tick positions
        cbar.set_ticklabels(['-10', '-1', '0', '1', '10'])  # Optional custom labels
        
        #if left is not None:
            #axes.set_xlim(left=left)

        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']

            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

        for i in range(7):
            start = (i+1) * self.N_eachC
            # Add vertical dashed lines
            axes.axhline(start, color='grey', linestyle='--', linewidth=0.5)

        plt.show()
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrPop'+'.svg',bbox_inches='tight')
        axes.set_title('Pyr Pop Activity')

    def PlotPyrPop_prob(self,start_color='green',end_color='#F7A541',left=None,right=None):
        
        fig, axes=plt.subplots(1,1,figsize=(4,3))
        data=self.Output_dic['E_pop_rate'][:,self.drawstart:]


        # In this parameter set, 1s transient time is not long enough. Normalized at FirstT-1
        normInd= self.FirstInd-self.drawstart-1
        # normInd=0
        normalized_data = np.array([row - row[normInd] if row[normInd] != 0 else row for row in data])
        reversed_data = np.flipud(normalized_data)

        # debug
        n_cols = reversed_data.shape[1]  # Number of columns in the data
        '''
        print(f'n_cols{n_cols}')
        print(f't_axis length{len(self.t_axis)}')
        print(f'drawstart{self.drawstart}')
        print(f't_start{self.t_axis[40]}, t_end{self.t_axis[-1]}')
        print(f't0+1={self.t0+1}, tn{self.tn}, dtframe{self.dtframe}')
        '''
        # Find the maximum absolute value for normalization
        max_val = np.max(np.abs(normalized_data))



        # Create a custom colormap with white at the center
        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        # Use custom normalization
        norm = SymLogNorm(linthresh=3, vmin=-max_val, vmax=max_val)  # Set linthresh for linear behavior near 0

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        im = axes.imshow(reversed_data, aspect='auto', cmap=cmap, norm=norm, extent=[self.t0+self.time_draw_start, self.tn-self.dtframe , self.Nneu - 1, 0],
                         )  # interpolation='none',Removed vmin=-max_val, vmax=max_val
        cbar=fig.colorbar(im)
        cbar.set_ticks([-10, -1, 0, 1, 10])  # Set specific tick positions
        cbar.set_ticklabels(['-10', '-1', '0', '1', '10'])  # Optional custom labels
        
        if left is not None:
            axes.set_xlim(left=left-0.5)
        if right is not None:
            axes.set_xlim(right=right)
        for i in np.arange(10)+left:
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']

            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
        for i in range(7):
            start = (i+1) * self.N_eachC
            # Add vertical dashed lines
            axes.axhline(start, color='grey', linestyle='--', linewidth=0.5)


        plt.show()
        if self.pf:
            fig.savefig(self.sdir+'MMNprobPyrPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNprobPyrPop'+'.svg',bbox_inches='tight')
        axes.set_title('Pyr Pop Activity')


    def PlotPyrPop_Keller(self,start_color='#62BC43',end_color='#F7A541',cmax=None):
        # Define the colors: red, white, and green
        #colors = [(253.0/255, 187.0/255, 132.0/255), (1, 1, 1), (161/255, 217/255, 155/255)]  # RGB values for (253, 187, 132), white, and (161, 217, 155)
    
        data=self.Output_dic['E_pop_rate'][:,5:]


        normalized_data = np.array([row - row[-1] if row[-1] != 0 else row for row in data])
        reversed_data = np.flipud(normalized_data)

        # Find the maximum absolute value for normalization
        max_val = np.max(np.abs(normalized_data))
        if cmax is not None:
            max_val=cmax
        # Create a custom colormap with white at the center
        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        norm = SymLogNorm(linthresh=3, vmin=-max_val, vmax=max_val)  # Set linthresh for linear behavior near 0

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        #im = axes.imshow(reversed_data, aspect='auto', cmap=cmap, extent=[-0.5, 4, self.N_eachC - 1, 0],
        #             vmin=-max_val, vmax=max_val)  # Set symmetric range for the colormap
        im = axes.imshow(reversed_data, aspect='auto', cmap=cmap, norm=norm,
                         extent=[-0.5, 4, self.N_eachC - 1, 0],interpolation='none')
        
        cbar=fig.colorbar(im)

        if self.pf:
            fig.savefig(self.sdir+'MMNPyrPopKeller'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrPopKeller'+'.svg',bbox_inches='tight')
        axes.set_title('Pyr Pop Activity')

    def PlotPyrPop_omission(self,columnid=2,start_color='green',end_color='#F7A541',left=None):
        if columnid is None:    
            data=self.Output_dic['E_pop_rate'][:,self.drawstart:]
        else:
            Nneu_vec=np.arange(columnid*self.N_eachC,(columnid+1)*self.N_eachC)
            data=self.Output_dic['E_pop_rate'][Nneu_vec,self.drawstart:]


        normalized_data = np.array([row - row[0] if row[0] != 0 else row for row in data])
        reversed_data = np.flipud(normalized_data)

        # Find the maximum absolute value for normalization
        max_val = np.max(np.abs(normalized_data))

        # Create a custom colormap with white at the center
        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        # Use custom normalization
        norm = SymLogNorm(linthresh=3, vmin=-max_val, vmax=max_val)  # Set linthresh for linear behavior near 0

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        im = axes.imshow(reversed_data, aspect='auto', cmap=cmap, norm=norm, extent=[self.t0+self.time_draw_start, self.tn , self.N_eachC - 1, 0],
                    interpolation='none' )  # Removed vmin=-max_val, vmax=max_val
        cbar=fig.colorbar(im)
        cbar.set_ticks([-10, -1, 0, 1, 10])  # Set specific tick positions
        cbar.set_ticklabels(['-10', '-1', '0', '1', '10'])  # Optional custom labels
        
        if left is not None:
            axes.set_xlim(left=left)

        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']

            # Add vertical dashed lines
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

        plt.show()
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrPopOmission'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrPopOmission'+'.svg',bbox_inches='tight')
        axes.set_title('Pyr Pop Activity')



        #return MMN
    def PlotPVPop(self):
        fig, axes=plt.subplots(1,1,figsize=(4,3))
        data=self.Output_dic['P_pop_rate']
        im=axes.imshow(data,vmin=0, cmap='inferno',extent=[self.t0,self.tn,self.Nneu-1,0],
                       aspect='auto',interpolation='none')    
        cbar=fig.colorbar(im)
        axes.set_xlim(left=-1)  # This changes the view to start from -1 on the x-axis
        axes.set_title('PV Pop Activity')
        if self.pf:
            fig.savefig(self.sdir+'MMNPVPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPVPop'+'.svg',bbox_inches='tight')

    def PlotPV0Pop(self):
        fig, axes=plt.subplots(1,1,figsize=(4,3))
        data=self.Output_dic['P0_pop_rate']
        im=axes.imshow(data,vmin=0, cmap='inferno',extent=[self.t0,self.tn,self.Nneu-1,0],
                       aspect='auto',interpolation='none')    
        cbar=fig.colorbar(im)
        axes.set_xlim(left=-1)  # This changes the view to start from -1 on the x-axis

        axes.set_title('PV0 Pop Activity')
        if self.pf:
            fig.savefig(self.sdir+'MMNPV0Pop'+'.jpg',bbox_inches='tight')



    def PlotSSTPop(self):
        fig, axes=plt.subplots(1,1,figsize=(4,3))
        
        data=self.Output_dic['S_pop_rate']

        im=axes.imshow(data,vmin=0,extent=[self.t0,self.tn-self.dtframe,self.Nneu-1,0],
                       aspect='auto',interpolation='none')    
        cbar=fig.colorbar(im)
        for i in range(12):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']

            # Add vertical dashed lines
            axes.axvline(start, color='white', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='white', linestyle='--', linewidth=0.5)


        #axes.set_xlim(left=-1)  # This changes the view to start from -1 on the x-axis
        axes.set_title('SST Pop Activity')
        if self.pf:
            fig.savefig(self.sdir+'MMNSSTPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNSSTPop'+'.svg',bbox_inches='tight')


    def PlotIntPop(self,left=-0.5,right=None):
        fig, axes=plt.subplots(1,1,figsize=(4,3))
        data=self.Output_dic['Int_pop_rate']
        im=axes.imshow(data,vmin=0, cmap='inferno',extent=[self.t0,self.tn,self.N_column-1,0],
                       aspect='auto',interpolation='none')    
        cbar=fig.colorbar(im)
        axes.set_xlim(left=left-0.5)  # This changes the view to start from -1 on the x-axis
        if right is not None:
            axes.set_xlim(right=right)  # This changes the view to start from -1 on the x-axis


        if self.pf:
            fig.savefig(self.sdir+'MMNIntPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNIntPop'+'.svg',bbox_inches='tight')
        axes.set_title('Int Pop Activity')




    def PlotPopImshow(self,data,Plot_label=''):
        data_flip=np.flipud(data)

        fig, axes=plt.subplots(1,1,figsize=(4,4))
        im=axes.imshow(data_flip,aspect=data.shape[1]/data.shape[0], cmap='inferno',interpolation='none')    
        cbar=fig.colorbar(im)

        if self.pf:
            fig.savefig(self.sdir+Plot_label+'Pop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+Plot_label+'Pop'+'.svg',bbox_inches='tight')

        axes.set_title(Plot_label+'Pop Activity')



    def PlotIntComptePop(self):
        fig, axes=plt.subplots(1,2,figsize=(6,6))
        data=self.Output_dic['Int_pop_rate']
        im=axes[0].imshow(data,aspect=data.shape[1]/data.shape[0],vmin=0, 
                          cmap='inferno',interpolation='none')    
        cbar = fig.colorbar(im, ax=axes[0])
        axes[0].set_title('Int Pop E Activity')

        data=self.Output_dic['Int_pop_rateI']
        im=axes[1].imshow(data,aspect=data.shape[1]/data.shape[0],vmin=0, 
                          cmap='inferno',interpolation='none')    
        cbar = fig.colorbar(im, ax=axes[1])
        axes[1].set_title('Int Pop i Activity')
        plt.show()

        if self.pf:
            fig.savefig(self.sdir+'MMNIntPop'+'.jpg',bbox_inches='tight')


    def PlotInt(self,left=-1,labels=['Std','Dev'], right=None):
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.stdcn,self.drawstart:],'k',label=labels[0])
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.devcn,self.drawstart:],'r',label=labels[1])
        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
            
            
            if i != 7 and i != 6:
                axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency
            else:
                axes.axvspan(start, end, color='red', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency


        axes.legend()
        axes.set_ylim(bottom=0)  # This changes the view to start from -1 on the x-axis
        axes.set_xlim(left=left-0.5)  # This changes the view to start from -1 on the x-axis
        if right is not None:
            axes.set_xlim(right=right)
            choices= self.Paras['choices'].copy()    # 1 for std, 2 for dev

            for i in np.arange(right-10,right):
                start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
                end = start + self.Paras['Tstim']
                # Add vertical dashed lines
                axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
                axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)      
                if round(choices[i])==2:    
                    axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           

        MMNInt= (self.Output_dic['Int_pop_rate'][self.stdcn,self.LastInd]-
            self.Output_dic['Int_pop_rate'][self.devcn,self.OddInd])

        if self.pf:
            fig.savefig(self.sdir+'MMNInt'+ f"{MMNInt:.4f}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNInt'+ f"{MMNInt:.4f}"+'.svg',bbox_inches='tight')
        axes.set_title('MMNInt'+ f"{MMNInt:.4f}")
    
        return MMNInt

    def PlotInt_nrep(self,nrep=4):
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.stdcn,self.drawstart:],'k',label='Std')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.devcn,self.drawstart:],'r',label='dev')
        for i in np.arange(nrep+1):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
            
            
            if i != nrep:
                axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency
            else:
                axes.axvspan(start, end, color='red', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency


        axes.legend()
        axes.set_ylim(bottom=0)  # This changes the view to start from -1 on the x-axis
        axes.set_xlim(left=-1)  # This changes the view to start from -1 on the x-axis

        MMNInt= (-np.mean(self.Output_dic['Int_pop_rate'][self.stdcn,self.LastInd:self.LastInd+self.shift])+
            np.mean(self.Output_dic['Int_pop_rate'][self.devcn,self.OddInd:self.OddInd+self.shift]))

        if self.pf:
            fig.savefig(self.sdir+'MMNInt'+ f"{MMNInt:.4f}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNInt'+ f"{MMNInt:.4f}"+'.svg',bbox_inches='tight')
        axes.set_title('MMNInt'+ f"{MMNInt:.4f}")
    
        return MMNInt



    def PlotInt_prob(self,left=None,right=None):
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.stdcn,self.drawstart:],'k',label='Std')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.devcn,self.drawstart:],'r',label='dev')
        choices= self.Paras['choices']    # 1 for std, 2 for dev
        for i in np.arange(10)+left:
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
            if round(choices[round(i)])==1:    
                axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           
            else:
                axes.axvspan(start, end, color='red', alpha=0.3)  # Adjust `alpha` for transparency               
       
        axes.legend()
        axes.set_ylim([2,8])  # This changes the view to start from -1 on the x-axis
        if left is not None:
            axes.set_xlim(left=left-0.5)
        if right is not None:
            axes.set_xlim(right=right)

        MMNInt= (self.Output_dic['Int_pop_rate'][self.stdcn,self.LastInd]-
            self.Output_dic['Int_pop_rate'][self.devcn,self.OddInd])

        if self.pf:
            fig.savefig(self.sdir+'MMNprobInt'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNprobInt'+'.svg',bbox_inches='tight')
        axes.set_title('MMNprobInt'+ f"{MMNInt:.4f}")
    



    def PlotInt_omiss(self):
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.stdcn,self.drawstart:],'k',label='Std')
        for i in range(10):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
            if i != 7 and i != 6:
                axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency
        axes.set_ylim(bottom=0)  # This changes the view to start from -1 on the x-axis

        MMNInt= (self.Output_dic['Int_pop_rate'][self.stdcn,self.LastInd]-
            self.Output_dic['Int_pop_rate'][self.devcn,self.OddInd])

        if self.pf:
            fig.savefig(self.sdir+'OmissInt'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'OmissInt'+'.svg',bbox_inches='tight')
        axes.set_title('OmissInt')
    
        return MMNInt    
    
    def PlotInt_omiss_fit(self,Input_blocker=2,Tinter=None, Tstim=None,Tau_int=None):
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        t = self.t_axis[self.drawstart:] + self.t0
        y = self.Output_dic['Int_pop_rate'][self.stdcn, self.drawstart:]
        axes.plot(t, y,'k')
        #axes.plot(self.t_axis[self.drawstart:]+self.t0,self.Output_dic['Int_pop_rate'][self.stdcn,self.drawstart:],'k',label='Std')

        for i in range(Input_blocker):
            start = i * (self.Paras['Tinter'] + self.Paras['Tstim'])
            end = start + self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)
            axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency           axes.axvspan(start, end, color='grey', alpha=0.3)  # Adjust `alpha` for transparency
        axes.set_ylim(bottom=0)  # This changes the view to start from -1 on the x-axis
        axes.set_xlim(left=-0.8)  # This changes the view to start from -1 on the x-axis

        # Fitting part:
        # Define the exponential decay function
        def exp_decay(t, A, k, C):
            return A * np.exp(-t *k) + C
        
        # Select a fitting range
        fit_start = ((Input_blocker-1)*(self.Paras['Tinter'] + self.Paras['Tstim'])+
                     0.5*self.Paras['Tstim'])/self.dtframe+self.drawstart
        fit_end=-1
        fit_start=int(fit_start)
        t_fit = t[fit_start:fit_end]
        y_fit = y[fit_start:fit_end]

        # Perform the fit
        popt, pcov = curve_fit(exp_decay, t_fit, y_fit, p0=(5, 0.1, 5))  # Initial guesses for A, tau, C
        # Extract parameters
        A, k, C = popt
        print(f"Fitted Parameters: A={A}, tau={1.0/k}, C={C}")
        # Plot the fitted curve
        t_full = np.linspace(t_fit[0], t_fit[-1], 500)  # Smooth time values for plotting the fit
        y_fit_curve = exp_decay(t_full, *popt)
        axes.plot(t_full, y_fit_curve, '--', label=f'Fit: tau={1/k:.2f}', color='red')

        if self.pf:
            fig.savefig(self.sdir+'OmissBlockInt_fit'+
                        f'Tinter{Tinter:.2f}'+
                        f'Tstim{Tstim:.2f}'+
                        f'Tau_int{Tau_int:.2f}'+
                        f'Tau_fit{1/k:.2f}'+
                        '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'OmissBlockInt_fit'+
                        f'Tinter{Tinter:.2f}'+
                        f'Tstim{Tstim:.2f}'+
                        f'Tau_int{Tau_int:.2f}'+
                        f'Tau_fit{1/k:.2f}'+
                        '.svg',bbox_inches='tight')
            axes.set_title('OmissBlockInt_fit')

    def PlotSST(self):
        fig, axes=plt.subplots(1,1,figsize=(3,2))
        axes.plot(self.t_axis[self.drawstart:]-self.time_draw_start,self.Output_dic['S_pop_rate'][self.std,self.drawstart:])
        axes.plot(self.t_axis[self.drawstart:]-self.time_draw_start,self.Output_dic['S_pop_rate'][self.dev,self.drawstart:])

        MMNSST= (self.Output_dic['S_pop_rate'][self.std,self.LastInd]-
            self.Output_dic['S_pop_rate'][self.dev,self.OddInd])
        if self.pf:
            fig.savefig(self.sdir+'MMNSST'+ f"{MMNSST:.4f}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNSST'+ f"{MMNSST:.4f}"+'.svg',bbox_inches='tight')

        axes.set_title('MMNSST'+ f"{MMNSST:.4f}")

        return MMNSST
    def PlotPV(self):
        fig, axes=plt.subplots(1,1,figsize=(3,2))
        axes.plot(self.t_axis[self.drawstart:]-self.time_draw_start,self.Output_dic['P_pop_rate'][self.std,self.drawstart:])
        axes.plot(self.t_axis[self.drawstart:]-self.time_draw_start,self.Output_dic['P_pop_rate'][self.dev,self.drawstart:])

        MMNPV= (self.Output_dic['P_pop_rate'][self.std,self.LastInd]-
            self.Output_dic['P_pop_rate'][self.dev,self.OddInd])
        if self.pf:
            fig.savefig(self.sdir+'MMNPV'+ f"{MMNPV:.4f}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPV'+ f"{MMNPV:.4f}"+'.svg',bbox_inches='tight')
        axes.set_title('MMNPV'+ f"{MMNPV:.4f}")

        return MMNPV
    
    def PlotPyr_conductance(self,shift_id=0,plotind=None,
                            Rmax=12,DEmax=0.8, SImin=-2,
                            DImin=-5,DImax=0.1):
        if plotind==None:
            ind=self.std
        else:
            ind=plotind
        testid= ind+shift_id
        if testid>=self.Nneu:
            testid=testid-self.Nneu
        
        fig, axes=plt.subplots(6,1,figsize=(2,5))
        axes[0].plot(self.t_axis[self.drawstart:] + self.t0,self.Output_dic['E_pop_rate'][testid,self.drawstart:]
                     -self.Output_dic['E_pop_rate'][testid,self.drawstart],'k',label='E rate')
        axes[0].set_ylim([-2, Rmax])
        axes[1].plot(self.t_axis[self.drawstart:] + self.t0,self.Output_dic['g_E_sE'][testid,self.drawstart:]
                     -self.Output_dic['g_E_sE'][testid,self.drawstart],color='#F7A541',label='Soma gE')
        axes[1].set_ylim([-0.1, 1.2])
        axes[2].plot(self.t_axis[self.drawstart:] + self.t0,-self.Output_dic['g_E_dI'][testid,self.drawstart:]
                     +self.Output_dic['g_E_dI'][testid,self.drawstart],color='#62BC43',label='dend gI')
        axes[2].set_ylim([DImin, DImax])

        if testid<4*self.N_eachC:
            colid=2
        else:
            colid=6

        axes[3].plot(self.t_axis[self.drawstart:] + self.t0,self.Output_dic['Int_pop_rate'][colid,self.drawstart:]
                     -self.Output_dic['Int_pop_rate'][colid,self.drawstart],color='#3182BD',label='Int rate')
        axes[3].plot(self.t_axis[self.drawstart:],0*self.t_axis[self.drawstart:],'--',color='grey',linewidth=0.5)
        axes[3].set_ylim([-4, 4])

        axes[4].plot(self.t_axis[self.drawstart:] + self.t0,self.Output_dic['g_E_dE'][testid,self.drawstart:]
                     -self.Output_dic['g_E_dE'][testid,self.drawstart],color='#F7A541',label='Dend. gE')
        axes[4].set_ylim([-0.3, DEmax])

        axes[5].plot(self.t_axis[self.drawstart:] + self.t0,-self.Output_dic['g_E_sI'][testid,self.drawstart:]
                     +self.Output_dic['g_E_sI'][testid,self.drawstart],color='#62BC43',label='some gI')
        axes[5].set_ylim([SImin, 0.3])
        for i in range(5):
            axes[i].set_xticklabels([])  # Removes tick labels
            axes[i].set_xlabel('')       # Ensures no x-label text

        if self.pf:
            fig.savefig(self.sdir+'PyrConductance'+'NeuronID'+ f"{ind+shift_id:.2f}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'PyrConductance'+'NeuronID'+ f"{ind+shift_id:.2f}"+'.svg',bbox_inches='tight')
        axes[0].set_title('Neu Id='+f"{ind+shift_id:.2f}")


    def PlotPyr_current(self,shift_id=0,plotind=None,tag=''):
        if plotind==None:
            ind=self.std
        else:
            ind=plotind
        fig, axes=plt.subplots(2,2,figsize=(18,6))
        axes[0,0].plot(self.t_axis + self.t0,self.Output_dic['E_pop_Isum'][ind+shift_id,:],label='Isum')
        axes[0,0].plot(self.t_axis + self.t0,self.Output_dic['E_pop_Isd'][ind+shift_id,:],label='Isd')
        axes[0,0].legend()
        axes[0,0].set_title(tag+'Total Current and Dend-to-soma')
        axes[0,1].plot(self.t_axis,self.Output_dic['E_pop_Vdend'][ind+shift_id,:])
        axes[0,1].set_title('Dend Voltage'+'DegreeShift'+ f"{shift_id:.2f}")
        axes[1,0].plot(self.t_axis + self.t0,self.Output_dic['g_E_sE'][ind+shift_id,:],label='gsE')
        axes[1,0].plot(self.t_axis + self.t0,-self.Output_dic['g_E_sI'][ind+shift_id,:],label='gsI')
        axes[1,0].plot(self.t_axis + self.t0,self.Output_dic['g_E_sEN'][ind+shift_id,:],label='gsEN')
        axes[1,0].legend()
        axes[1,0].set_title('conductance Soma')
        axes[1,1].plot(self.t_axis + self.t0,self.Output_dic['g_E_dE'][ind+shift_id,:],label='gdE')
        axes[1,1].plot(self.t_axis + self.t0,-self.Output_dic['g_E_dI'][ind+shift_id,:],label='gdI')
        axes[1,1].plot(self.t_axis + self.t0,self.Output_dic['g_E_dEN'][ind+shift_id,:],label='gdEN')
        axes[1,1].legend()
        axes[1,1].set_title('conductance dend')

        if self.pf:
            fig.savefig(self.sdir+'PyrCurrent'+'DegreeShift'+ f"{shift_id:.2f}"+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'PyrCurrent'+'DegreeShift'+ f"{shift_id:.2f}"+'.svg',bbox_inches='tight')


    def PlotInt_current(self,plotcolind=None,shift_id=0):
        if plotcolind==None:
            ind=self.stdcn
        else:
            ind=plotcolind
        fig, axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].plot(self.t_axis + self.t0,self.Output_dic['Int_pop_rate'][ind+shift_id,:])
        axes[0].set_title('Integrator Firing rate')
        axes[1].plot(self.t_axis + self.t0,self.Output_dic['Int_pop_sumI'][ind+shift_id,:])
        axes[1].set_title('INtegrator Total Current')
        axes[2].plot(self.t_axis + self.t0,self.Output_dic['g_Int'][ind+shift_id,:],'k')
        axes[2].plot(self.t_axis + self.t0,self.Output_dic['g_Int_N'][ind+shift_id,:],'r')
        axes[2].set_title('Integrator conductance')
        if self.pf:
            fig.savefig(self.sdir+'IntCurrent'+'.jpg',bbox_inches='tight')
    def PlotIntCompte_current(self,plotind=None):
        if plotind==None:
            ind=int(round(self.std/360*self.N_column))
        else:
            ind=int(round(plotind/360*self.N_column))
        fig, axes=plt.subplots(1,2,figsize=(12,4))
        axes[0].plot(self.t_axis + self.t0,self.Output_dic['Int_pop_rate'][ind,:],'k')
        axes[0].plot(self.t_axis + self.t0,self.Output_dic['Int_pop_rateI'][ind,:],'r')
        axes[0].set_title('Integrator rate E and I, ind='+ str(ind))
        axes[1].plot(self.t_axis + self.t0,self.Output_dic['Int_pop_sumIE'][ind,:],'k')
        axes[1].plot(self.t_axis + self.t0,self.Output_dic['Int_pop_sumII'][ind,:],'r')
        axes[1].set_title('Integrator Current E and I')
        if self.pf:
            fig.savefig(self.sdir+'IntCurrent'+'.jpg',bbox_inches='tight')

    def PlotP_current(self,shift_id=0,plotind=None):
        if plotind==None:
            ind=self.std
        else:
            ind=plotind

        fig, axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].plot(self.t_axis + self.t0,self.Output_dic['P_pop_rate'][ind+shift_id,:])
        axes[0].set_title('PV Firing rate')
        axes[1].plot(self.t_axis + self.t0,self.Output_dic['P_pop_sumI'][ind+shift_id,:])
        axes[1].set_title('PV Total Current')
        axes[2].plot(self.t_axis + self.t0,self.Output_dic['g_P_E'][self.std+shift_id,:],label='gE')
        axes[2].plot(self.t_axis + self.t0,self.Output_dic['g_P_EN'][self.std+shift_id,:],label='gEN')
        axes[2].plot(self.t_axis + self.t0,-self.Output_dic['g_P_I'][self.std+shift_id,:],label='gI')
        axes[2].legend()
        axes[2].set_title('PV conductance')
        if self.pf:
            fig.savefig(self.sdir+'PCurrent'+'.jpg',bbox_inches='tight')

    def PlotS_current(self,shift_id=0, plotind= None):
        if plotind==None:
            ind=self.std
        else:
            ind=plotind

        fig, axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].plot(self.t_axis + self.t0,self.Output_dic['S_pop_rate'][ind+shift_id,:])
        axes[0].set_title('SST Firing rate')
        axes[1].plot(self.t_axis + self.t0,self.Output_dic['S_pop_sumI'][ind+shift_id,:])
        axes[1].set_title('SST Total Current')
        axes[2].plot(self.t_axis + self.t0,self.Output_dic['g_S_E'][ind+shift_id,:],label='gE')
        axes[2].plot(self.t_axis + self.t0,self.Output_dic['g_S_EN'][ind+shift_id,:],label='gEN')
        axes[2].plot(self.t_axis + self.t0,-self.Output_dic['g_S_I'][ind+shift_id,:],label='gI')
        axes[2].legend()
        if self.pf:
            fig.savefig(self.sdir+'SCurrent'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'SCurrent'+'.svg',bbox_inches='tight')
        axes[2].set_title('SST conductance')

    def Plot_Dict(self,trace,name='',start_frame=0,x_axis=None,left=None, right=None,top=None):  # This one is to plot anything that from output dictionary.
        if x_axis is None:
            x_axis=self.t_axis-1
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))
        axes.plot(x_axis[start_frame:],trace[start_frame:],'k')
        axes.plot(x_axis[start_frame:],trace[start_frame:],'.')
        #axes.set_xlim([0.5,4])  # This changes the view to start from -1 on the x-axis
        if left is not None:
            axes.set_xlim(left=left)
        if right is not None:
            axes.set_xlim(right=right)
        if top is not None:
            axes.set_ylim(top=top)    
            
        axes.set_ylim(bottom=0)    
        if self.pf:
            fig.savefig(self.sdir+'SingleCellTrace'+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'SingleCellTrace'+name+'.svg',bbox_inches='tight')

        axes.set_title(name)
    
    def Plot_Dict_plot(self,trace,name='',x_axis=None):  # This one is to plot anything that from output dictionary.
        if x_axis is None:
            x_axis=self.t_axis-1
        
        plt.figure(figsize=(4, 4))
        plt.plot(x_axis,trace)
#        axes.plot(x_axis,trace,'.')
        plt.xlim(0,4)  # This changes the view to start from -1 on the x-axis
        plt.show()
        if self.pf:
            plt.savefig(self.sdir+name+'.jpg',bbox_inches='tight')
            plt.savefig(self.sdir+name+'.svg',bbox_inches='tight')
        plt.title(name)

                
    def Plot_Snapshot(self,i_frame,tag=''):  # This one is to plot snapshot at given time.
        fig, axes=plt.subplots(2,1,figsize=(4,4))
        axes[0].plot(self.Output_dic['E_pop_rate'][:,i_frame],'k--',label='E')
        axes[0].plot(self.Output_dic['P_pop_rate'][:,i_frame],'g',label='P')
        axes[0].plot(self.Output_dic['P0_pop_rate'][:,i_frame],'g.-',label='P0')
        axes[0].plot(self.Output_dic['S_pop_rate'][:,i_frame],'b',label='S')
        axes[0].plot(np.arange(20,320,320.0/self.N_column),self.Output_dic['Int_pop_rate'][:,i_frame],'r',label='Int')
        axes[0].legend()
        axes[0].legend(loc='right')
        axes[0].set_ylim(bottom=0)
        axes[0].set_title('Snapshot at frame'+ str(i_frame)+tag)

        axes[1].plot(self.Output_dic['E_pop_Isum'][:,i_frame],'k--',label='E')
        axes[1].plot(self.Output_dic['P_pop_sumI'][:,i_frame],'g',label='P')
        axes[1].plot(self.Output_dic['P0_pop_sumI'][:,i_frame],'g.-',label='P0')
        axes[1].plot(self.Output_dic['S_pop_sumI'][:,i_frame],'b',label='S')
        axes[1].plot(np.arange(20,320,320.0/self.N_column),self.Output_dic['Int_pop_sumI'][:,i_frame],'r',label='Int')
        axes[1].legend(loc='right')
        axes[1].set_title('Current')
        axes[1].set_ylim(bottom=-0.1)

        # Adjust layout to prevent overlap
        fig.tight_layout()
        plt.show()

        if self.pf:
            fig.savefig(self.sdir+tag+'.jpg',bbox_inches='tight')

    def Plot_Snapshot_nPE(self,trace,name=''):  # This one is to plot snapshot at given time.
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        xaxis=np.arange(len(trace),0,-1)
        axes.plot(xaxis,trace,'k')
        axes.plot(xaxis,trace,'.b')
        axes.plot([0,self.Nneu],[0,0],'--',color='#D3D3D3')
        axes.plot([0,self.Nneu],[np.mean(trace),np.mean(trace)],'--k',linewidth=2)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+name+'.svg',bbox_inches='tight')
        axes.set_title(name)


    def Plot_nPE_barplot(self,categories, values, name='',
                         colors= ['#bdbdbd','#7fcdbb','#2c7fb8'],top=None):
        fig, axes=plt.subplots(1,1,figsize=(2.5,3))
        #colors= ['#bdbdbd','#7fcdbb','#2c7fb8']
        axes.bar(categories, values, color=colors)  # Plot bars with sky blue color
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        if top is not None:
            axes.set_ylim(top=top)

        axes.set_xticklabels(categories, rotation=45, ha='center', fontsize=12)  # Tilt the labels 45 degrees, right-aligned


        if self.pf:
            fig.savefig(self.sdir+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+name+'.svg',bbox_inches='tight')
        axes.set_title(name)


    def Plot_Snapshots(self,trace1,trace2, trace3, trace0, name=''):  # This one is to plot snapshot at given time.
        fig, axes=plt.subplots(1,1,figsize=(2.4,2.4))
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace2,'g',label='Stim')
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace3,'b',label='Pred')
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace1,'k',label='Integrate')
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace0,'--k',label='Baseline')
        axes.legend()
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        if self.pf:
            fig.savefig(self.sdir+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+name+'.svg',bbox_inches='tight')
        axes.set_title(name)

    def Plot_var_Snapshots(self,traces, labels=['','','',''],linestyles=['-','-','-','--'],colors=['g','b','pink','grey'],name='',
                           ymax=20,linewidth=1, figheight=2.4,downsample=1,flag_scatter=False):  # This one is to plot snapshot at given time.
        fig, axes=plt.subplots(1,1,figsize=(2.4,figheight))
        
        for i in range(traces.shape[1]):
            x_vals=np.arange(self.N_eachC, 0, -1) - 1
            y_vals=traces[:, i]
            #axes.plot(np.arange(self.N_eachC,0,-1)-1,traces[:, i],linestyles[i% len(linestyles)], color= colors[i% len(linestyles)], label=labels[i],linewidth=linewidth)
            if flag_scatter:
                axes.scatter(x_vals[::downsample], y_vals[::downsample], color= colors[i% len(linestyles)], label=labels[i], s=5)
            else:
                axes.plot(x_vals[::downsample], y_vals[::downsample],linestyles[i% len(linestyles)], color= colors[i% len(linestyles)], label=labels[i],linewidth=linewidth)
 
        axes.legend()
        axes.set_ylim(top=ymax)
        if self.pf:
            fig.savefig(self.sdir+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+name+'.svg',bbox_inches='tight')
        axes.set_title(name)

    def Plot_overlay_hist(self, trace0,trace1, labels=('', ''),start_color='grey',end_color='black',name='', bin_width=2.0, left=None):  
        fig, axes=plt.subplots(1,1,figsize=(2,2))
        init_trace=trace0
        end_trace=trace1
        
        # Calculate the combined data range across both traces
        min_value = min(init_trace.min(), end_trace.min())
        max_value = max(init_trace.max(), end_trace.max())
        
        # Define bin edges based on the combined range and specified bin width
        bin_edges = np.arange(min_value, max_value + bin_width, bin_width)

        
        axes.hist(init_trace, bins=bin_edges, density=True, color=start_color, alpha=0.5, label=labels[0], 
                  edgecolor='black', linewidth=0.5)
        axes.hist(end_trace, bins=bin_edges, density=True, color=end_color, alpha=0.5, label=labels[1], 
                  edgecolor='black', linewidth=0.5)
        # Calculate means
        init_mean = np.mean(init_trace)
        end_mean = np.mean(end_trace)
        
        # Add vertical dashed lines at the means
        axes.axvline(init_mean, color=start_color, linestyle='--', linewidth=1.5)
        axes.axvline(end_mean, color=end_color, linestyle='--', linewidth=1.5)
        axes.set_xlim(right=21)
        #axes.set_ylim(top=25)
        if left is not None:
            axes.set_xlim(left=left)



        if labels[0] or labels[1]:
            axes.legend()

        if self.pf:
            fig.savefig(self.sdir+'Hists'+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'Hists'+name+'.svg',bbox_inches='tight')
        axes.set_title(name)
        plt.show()


    def Plot_input_hist(self, trace0, labels=None,start_color='grey',end_color='black',name='', bin_width=0.2):  
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        init_trace=trace0
        init_trace = init_trace[~np.isnan(init_trace)]

        # Calculate the combined data range across both traces
        min_value = init_trace.min()
        max_value=  init_trace.max()
        # Define bin edges based on the combined range and specified bin width
        bin_edges = np.arange(min_value, max_value + bin_width, bin_width)

        
        axes.hist(init_trace, bins=bin_edges, density=True, color=start_color, alpha=0.5, label=labels, 
                  edgecolor='black', linewidth=0.5)

        # Calculate means
        
        # Add vertical dashed lines at the means
        axes.axvline(0.5, color="#E34A33", linestyle='--', linewidth=1.5)
        axes.axvline(-0.5, color="#E34A33", linestyle='--', linewidth=1.5)

        # axes.set_xlim(right=21)
        # axes.set_ylim(top=25)
    

        if self.pf:
            fig.savefig(self.sdir+'Hists'+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'Hists'+name+'.svg',bbox_inches='tight')
        axes.set_title(name)
        plt.show()



    def Plot_2Snapshots(self,trace1,trace2, name=''):  # This one is to plot snapshot at given time.
        fig, axes=plt.subplots(1,1,figsize=(2.4,2.4))
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace1,'k',label='Expected')
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace2,'r',label='Surprise')
        axes.legend()
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        if self.pf:
            fig.savefig(self.sdir+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+name+'.svg',bbox_inches='tight')
        axes.set_title(name)
    def Plot_3Snapshots(self,trace1,trace2,trace3, name='',
                        colors=['#bdbdbd','#7fcdbb','#2c7fb8'],
                        labels=['Baseline','Stim + Pred','Pred']):  # This one is to plot snapshot at given time.
        fig, axes=plt.subplots(1,1,figsize=(2.4,2.4))
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace1,colors[0],label=labels[0])
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace2,colors[1],label=labels[1])
        axes.plot(np.arange(self.N_eachC,0,-1)-1,trace3,colors[2],label=labels[2])
        axes.legend()
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        if self.pf:
            fig.savefig(self.sdir+name+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+name+'.svg',bbox_inches='tight')
        axes.set_title(name)

    def Plot_FixframeInt(self, mat, int_f, name=''):
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
            axes.plot(np.arange(self.N_eachC, 0, -1) - 1, mat[:, index], color=color, label=label)
        axes.set_ylim(top=60)

        # Add an inset for the color bar
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(axes, width="30%", height="5%", loc='upper right', borderpad=1)

        # Create a ScalarMappable and add the color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=mat.shape[1] - 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=inset_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=6)
        inset_ax.xaxis.set_ticks_position("top")  # Adjust tick position for readability
    
        axes.legend(fontsize=6)

        if self.pf:
            fig.savefig(self.sdir + name + '.jpg', bbox_inches='tight')
            fig.savefig(self.sdir + name + '.svg', bbox_inches='tight')
        axes.set_title(name)


    def Plot_Times_Dict(self,plotobj,name=''):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))
        Width=  self.N_eachC//2

        align_shift= int(self.dev-self.std)
        axes.plot(np.arange(-Width,Width),plotobj[self.std-Width:self.std+Width,self.FirstInd],label='First A')
        axes.plot(np.arange(-Width,Width),plotobj[self.std-Width:self.std+Width,self.LastInd],label='Last A')
        axes.plot(np.arange(-Width,Width),plotobj[self.dev-Width:self.dev+Width,self.OddInd],label='Oddball A')
        axes.legend()


        axes.get_xaxis().set_visible(False)
        if self.pf:
            fig.savefig(self.sdir+'MMN'+name+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMN'+name+ '.svg',bbox_inches='tight')
        axes.set_title('Flashshot'+name)


    def PlotPyrTimes_randomcontrol(self,Output_dict_random,tag=''):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))
        Width=  self.N_eachC//2

        align_shift= int(self.dev-self.std)
        axes.plot(np.arange(-Width,Width),self.Output_dic['E_pop_rate'][self.std-Width:self.std+Width,self.FirstInd],label='First A')
        axes.plot(np.arange(-Width,Width),self.Output_dic['E_pop_rate'][self.std-Width:self.std+Width,self.LastInd],label='Last A')
        axes.plot(np.arange(-Width,Width),self.Output_dic['E_pop_rate'][self.dev-Width:self.dev+Width,self.OddInd],label='Oddball A')
        axes.plot(np.arange(-Width,Width),Output_dict_random['E_pop_rate'][self.dev-Width:self.dev+Width,self.OddInd],label='random control')
        axes.legend()

        axes.get_xaxis().set_visible(False)
        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimesRandomControl'+ tag+ '.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimesRandomControl'+ tag+ '.svg',bbox_inches='tight')
        axes.set_title('Flashshot, First, last, dev')        

    def PlotPyrTimesDif_randomcontrol(self,Output_dict_random,tag=''):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))

        Width=  self.N_eachC//2
        Dif_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        Dif_extra_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        Dif_control_vec= (Output_dict_random['E_pop_rate'][self.dev_neus_vec,self.OddInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        

        axes.plot(np.arange(0,self.N_eachC)-Width,Dif_vec,'k',label='DD')
        axes.plot(np.arange(0,self.N_eachC)-Width,Dif_extra_vec,'r',label='Pred')
        axes.plot(np.arange(0,self.N_eachC)-Width,Dif_control_vec,'b',label='Pred_rand')

        axes.plot(np.arange(0,self.N_eachC)-Width,0*np.arange(self.N_eachC),'k--')
        axes.legend()
        #axes.get_xaxis().set_visible(False)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimesDifRandomControl'+ tag+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimesDifRandomControl'+ tag+'.svg',bbox_inches='tight')
        axes.set_title('DD(MMN) over Input Affinity')

    def PlotPyrTimesDif_randomcontrol_mean(self,trace,tag=''):
        fig, axes=plt.subplots(1,1,figsize=(4,3))

        Width=  self.N_eachC//2

        First_rate_vec= np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd:self.FirstInd+self.shift], 
            axis=1)
        Late_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd:self.LastInd+self.shift], 
            axis=1)
        Odd_rate_vec=np.mean(
            self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd:self.OddInd+self.shift], 
            axis=1)
        Control_rate_vec=np.mean(trace,axis=1)

        DD_vec=Odd_rate_vec-First_rate_vec
        SSA_vec=First_rate_vec-Late_rate_vec

        DD_control_vec=Odd_rate_vec-Control_rate_vec
        SSA_control_vec=Control_rate_vec-Late_rate_vec
        '''
        Dif_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        Dif_extra_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        Dif_control_vec= (trace-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        '''

        axes.plot(np.arange(0,self.N_eachC),DD_vec,'r',label='DD')
        axes.plot(np.arange(0,self.N_eachC),SSA_vec,'k',label='SSA')
        axes.plot(np.arange(0,self.N_eachC),DD_control_vec,'r--',label='DD_ctrl')
        axes.plot(np.arange(0,self.N_eachC),SSA_control_vec,'k--',label='SSA_ctrl')

        axes.plot(np.arange(0,self.N_eachC),0*np.arange(self.N_eachC),'k-.')
        axes.legend()
        #axes.get_xaxis().set_visible(False)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimesDifRandomControl'+ tag+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimesDifRandomControl'+ tag+'.svg',bbox_inches='tight')
        axes.set_title('random control'+tag)

    def PlotPyrTimesDif_randomcontrol(self,Output_dict_random,tag=''):
        fig, axes=plt.subplots(1,1,figsize=(2.7,1.8))

        Width=  self.N_eachC//2
        Dif_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.LastInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        Dif_extra_vec= (self.Output_dic['E_pop_rate'][self.std_neus_vec,self.FirstInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        Dif_control_vec= (Output_dict_random['E_pop_rate'][self.dev_neus_vec,self.OddInd]-
                  self.Output_dic['E_pop_rate'][self.dev_neus_vec,self.OddInd])
        

        axes.plot(np.arange(0,self.N_eachC)-Width,Dif_vec,'k',label='DD')
        axes.plot(np.arange(0,self.N_eachC)-Width,Dif_extra_vec,'r',label='Pred')
        axes.plot(np.arange(0,self.N_eachC)-Width,Dif_control_vec,'b',label='Pred_rand')

        axes.plot(np.arange(0,self.N_eachC)-Width,0*np.arange(self.N_eachC),'k--')
        axes.legend()
        #axes.get_xaxis().set_visible(False)

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)


        if self.pf:
            fig.savefig(self.sdir+'MMNPyrTimesDifRandomControl'+ tag+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'MMNPyrTimesDifRandomControl'+ tag+'.svg',bbox_inches='tight')
        axes.set_title('DD(MMN) over Input Affinity')

    def LearningSnapshot(self,i_frame,tag=''):
        fig, axes=plt.subplots(4,1,figsize=(6,6))

        axes[0].plot(self.Output_dic['E_pop_rate'][:,i_frame],'k--',label='E')
        axes[0].plot(self.Output_dic['P_pop_rate'][:,i_frame],'g',label='P')
        axes[0].plot(self.Output_dic['S_pop_rate'][:,i_frame],'b',label='S')
        axes[0].plot(self.Output_dic['Int_pop_rate'][:,i_frame],'r',label='Int')
        axes[0].legend()
        axes[0].legend(loc='right')
        axes[0].set_ylim(bottom=0)
        axes[0].set_title('Snapshot at frame'+ str(i_frame)+tag)

        axes[1].plot(self.Output_dic['E_pop_Isum'][:,i_frame],'k--',label='E')
        axes[1].plot(self.Output_dic['P_pop_sumI'][:,i_frame],'g',label='P')
        axes[1].plot(self.Output_dic['S_pop_sumI'][:,i_frame],'b',label='S')
        axes[1].set_ylim(bottom=0)
        axes[1].set_title('Current')

        axes[2].plot(self.Output_dic['W_EP'][:,i_frame],'g',label='W\_EP')
        axes[2].plot(self.Output_dic['W_EdS'][:,i_frame],'b',label='W\_EdS')
        axes[2].legend()
        axes[2].set_ylim(bottom=0)

        axes[3].plot(self.Output_dic['dWEP'][:,i_frame],'g',label='dW\_EP')
        axes[3].plot(self.Output_dic['dWEds'][:,i_frame],'b',label='dW\_EdS')
        axes[3].legend()
        axes[3].spines["top"].set_visible(False)
        axes[3].spines["right"].set_visible(False)

        # Adjust layout to prevent overlap
        fig.tight_layout()

        # Display the figure
        plt.show()
        if self.pf:
            fig.savefig(self.sdir+'LearningSnaptshotframe'+str(i_frame)+ tag+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'LearningSnaptshotframe'+str(i_frame)+ tag+'.svg',bbox_inches='tight')
            
    def Learning_change(self,data,tag=''):
        n=12    # Sliding window size
        data_flip=np.flipud(data)
        num_rows, num_cols = data_flip.shape
        # If the number of columns is not divisible by n, truncate the extra columns
        num_cols_to_keep = (num_cols // n) * n
        data_flip = data_flip[:, :num_cols_to_keep]  # Truncate extra columns

        t_axis=self.t_axis[:num_cols_to_keep]
        
        W_reshaped = data_flip.reshape(num_rows, num_cols_to_keep // n, n)
        W_slide = W_reshaped.mean(axis=2)
        change_W=np.abs(W_slide[:,1:]-W_slide[:,:-1])
        norm_2_change = np.linalg.norm(change_W, axis=0)
        
        fig, axes=plt.subplots(1,1,figsize=(4,4))
 #       axes.plot(norm_2_change)
        axes.plot(t_axis[0:num_cols_to_keep-n:n],norm_2_change)
        if self.pf:
            fig.savefig(self.sdir+'WeightChange'+tag+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'WeightChange'+tag+'.svg',bbox_inches='tight')
        axes.set_title('WeightChange'+tag)


    def ModulationIndex(self,data,start_color='green', end_color='red',labels=('', ''), tag=''):
        bin_width = 0.1
        bin_edges = np.arange(-1.5, 1.5+ bin_width, bin_width)  # Creates bins from -2 to 2 with width 0.2
        # Reponding flag
        #thr=1.645   # reverse of p=0.05
        thr=1.28   # reverse of p=0.1
        #thr=1   
        #thr=-1  

        Res0_flag = (data[:, 0] > 2*(1+thr)) | (data[:, 2] > 2*(1+thr))
        Res1_flag = (data[:, 1] > 2*(1+thr)) | (data[:, 3] > 2*(1+thr))
        R_f0M= data[Res0_flag,0] - data[Res0_flag,4]
        R_f1M= data[Res1_flag,1] - data[Res1_flag,4]
        R_f0= data[Res0_flag,2]- data[Res0_flag,4]
        R_f1= data[Res1_flag,3]- data[Res1_flag,4]
        '''
        R_f0M= data[Res0_flag,0] - data[Res0_flag,4]
        R_f1M= data[Res1_flag,1] - data[Res1_flag,4]
        R_f0= data[Res0_flag,2]- data[Res0_flag,4]
        R_f1= data[Res1_flag,3]- data[Res1_flag,4]
        '''
        #R_baseline=data[:,4]
        # range from -pi to pi. 
        theta_f0 = np.arctan2(R_f0M, R_f0 )
        theta_f1 = np.arctan2(R_f1M, R_f1 )
        
        #transferring the theta_f0 into -2 to 2
        MI_f0= theta_f0/np.pi *4 -1
        MI_f1= theta_f1/np.pi* 4 -1

        MI_f0[MI_f0<-2]=-4-MI_f0[MI_f0<=-2]
        MI_f0[MI_f0>2]=4-MI_f0[MI_f0>2]

        MI_f1[MI_f1<-2]=-4-MI_f1[MI_f1<=-2]
        MI_f1[MI_f1>2]=4-MI_f1[MI_f1>2]

        
        # sanity test
        fig, axes=plt.subplots(1,1,figsize=(2.5,2.5))
        axes.plot(R_f1,R_f1M,'.',color=end_color,label=labels[1],markersize=2)
        axes.plot(R_f0,R_f0M,'.',color=start_color,label=labels[0],markersize=2)
        axes.legend() 

        min_val = min(min(R_f0M), min(R_f1M),-1)
        max_val = max(max(R_f0M), max(R_f1M))
        axes.plot([-max_val/np.sqrt(2),max_val/np.sqrt(2)],[-max_val/np.sqrt(2),max_val/np.sqrt(2)],'--k',linewidth=0.5)
        axes.plot([-max_val/np.sqrt(2),max_val/np.sqrt(2)],[max_val/np.sqrt(2),-max_val/np.sqrt(2)],'--k',linewidth=0.5)
        axes.plot([0,0],[-max_val,max_val],'-k',linewidth=0.5)
        axes.plot([-max_val,max_val],[0,0],'-k',linewidth=0.5)


        # Set the same limits for both axes
        axes.set_xlim(-5, max_val)
        axes.set_ylim(-5, max_val)

        # Set equal scaling for both axes
        axes.set_aspect('equal')

        if self.pf:
            fig.savefig(f"{self.sdir}MIscatter_{tag}.jpg", bbox_inches='tight')
            fig.savefig(f"{self.sdir}MIscatter_{tag}.svg", bbox_inches='tight')
        
        axes.set_title(f'MI scatter {tag}')




        fig, axes=plt.subplots(1,1,figsize=(3,2.5))
 #       axes.plot(norm_2_change)
        axes.hist(MI_f0, bins=bin_edges, color=start_color, density=True,alpha=0.5, label=labels[0], 
                  edgecolor='black', linewidth=0.5)
        axes.hist(MI_f1, bins=bin_edges, color=end_color,density=True, alpha=0.5, label=labels[1], 
                  edgecolor='black', linewidth=0.5)
        # Calculate means
        MI_f0_mean = np.mean(MI_f0)
        MI_f1_mean = np.mean(MI_f1)
        
        # Add vertical dashed lines at the means
        axes.axvline(MI_f0_mean, color=start_color, linestyle='--', linewidth=1.5)
        axes.axvline(MI_f1_mean, color=end_color, linestyle='--', linewidth=1.5)



        axes.legend()  # Optional: Add legend if labels are used

        if self.pf:
            fig.savefig(f"{self.sdir}MIhist_{tag}.jpg", bbox_inches='tight')
            fig.savefig(f"{self.sdir}MIhist_{tag}.svg", bbox_inches='tight')
        
        axes.set_title(f'MI Histogram {tag}')

    def nPE_PersonCorr(self,cdata,start_color='green', end_color='#E65100',labels=('', ''), tag=''):
        # This one calculate specifically the Pearson correlation between neuron response with pred and stim strength
        # Initialize lists to store correlations
        # Create a custom diverging colormap
        from sklearn.decomposition import PCA

        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        max_val = np.max(np.abs(cdata))
        norm = plt.Normalize(vmin=-max_val, vmax=max_val)  # Normalize data symmetrically around zero

        corr_with_rho_stim = []
        corr_with_rho_pred = []
        corr_stim_norm = []
        corr_pred_norm = []

        # Calculate Pearson correlation for each row
        for row in self.Output_dic['E_pop_rate']:
            corr_stim, _ = pearsonr(row, self.Output_dic['rho_stim'])
            corr_pred, _ = pearsonr(row, self.Output_dic['rho_pred'])
            corr_with_rho_stim.append(corr_stim)
            corr_with_rho_pred.append(corr_pred)
            corr_stim_norm.append(corr_stim/np.sqrt(corr_stim**2+corr_pred**2))
            corr_pred_norm.append(corr_pred/np.sqrt(corr_stim**2+corr_pred**2))
        # Plot the scatter plot

        # Ensure input lists are NumPy arrays
        corr_with_rho_pred = np.array(corr_with_rho_pred)  # Convert to NumPy array
        corr_with_rho_stim = np.array(corr_with_rho_stim)  # Convert to NumPy array
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        # Plot the scatter plot
        scatter =axes.scatter(corr_with_rho_stim, corr_with_rho_pred, c=cdata, cmap=cmap, norm=norm,
                              alpha=0.8,s=8,edgecolors='black',linewidths=0.2)
        axes.set_aspect('equal')
        # Convert data to numpy array (assuming these are lists)
        data = np.vstack((corr_with_rho_stim, corr_with_rho_pred)).T  # Shape (N,2)

        center = np.mean(data, axis=0)  # Mean of the data



        axes.plot([-1,1],[0,0],'--k',linewidth=0.5)
        axes.plot([0,0],[-1,1],'--k',linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=axes)
        #axes.set_facecolor('#045a8d')  # Light cyan
        #axes.set_facecolor('black')  # Light cyan
   # Add colorbar

        # Adjust layout for better spacing
        plt.tight_layout()

        if self.pf:
            fig.savefig(f"{self.sdir}nPEcorr_{tag}.svg", bbox_inches='tight')
            axes.set_xlabel("Correlation with rho_stim", fontsize=8)
            axes.set_ylabel("Correlation with rho_pred", fontsize=8)
            cbar.set_label('Mismatch dRate', rotation=270, labelpad=15)
            fig.savefig(f"{self.sdir}nPEcorr_{tag}.jpg", bbox_inches='tight')
        axes.set_title(f'nPEcorr{tag}')

        #theta_corr = np.arctan2(corr_pred_norm, corr_stim_norm )

        dist_all = np.sqrt(corr_with_rho_pred**2 + corr_with_rho_stim**2)  # Equivalent to MATLAB dist_all = norm_bCT.^2 + norm_aCT.^2

        theta_corr = np.arctan2(corr_with_rho_pred, corr_with_rho_stim )
        theta_corr[theta_corr<-3/4*np.pi]+=2*np.pi
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))

        bin_width = np.pi/10.0
        bin_edges = np.arange(-3 * np.pi / 4,  5/4*np.pi+ bin_width, bin_width)  # Creates bins from -2 to 2 with width 0.2


        axes.hist(theta_corr, bins=bin_edges,  weights=dist_all, alpha=0.7, density=True, color='blue', edgecolor='black')
        # Add custom x-ticks
        xticks_positions = [-3 * np.pi / 4, -np.pi / 4, np.pi / 4, 3/4*np.pi, 5/4*np.pi ]
        xticks_labels = [r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{4}$', r'$\frac{\pi}{4}$',r'$\frac{3\pi}{4}$',r'$-\frac{5\pi}{4}$']
        axes.set_xticks(xticks_positions)
        axes.set_xticklabels(xticks_labels)

        if self.pf:
            fig.savefig(f"{self.sdir}nPEcorrhist_{tag}.svg", bbox_inches='tight')
            axes.set_xlabel("Augular theta", fontsize=8)
            axes.set_ylabel("Density", fontsize=8)
            fig.savefig(f"{self.sdir}nPEcorrhist_{tag}.jpg", bbox_inches='tight')
        axes.set_title(f'nPEcorrhist {tag}')

    def nPE_PersonCorr_meannorm(self,cdata,start_color='green', end_color='#E65100',labels=('', ''), tag=''):
        # This one calculate specifically the Pearson correlation between neuron response with pred and stim strength
        # Initialize lists to store correlations
        # Create a custom diverging colormap
        from sklearn.decomposition import PCA

        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        max_val = np.max(np.abs(cdata))
        norm = plt.Normalize(vmin=-max_val, vmax=max_val)  # Normalize data symmetrically around zero

        corr_with_rho_stim = []
        corr_with_rho_pred = []
        corr_stim_norm = []
        corr_pred_norm = []

        # Calculate Pearson correlation for each row
        for row in self.Output_dic['E_pop_rate']:
            corr_stim, _ = pearsonr(row, self.Output_dic['rho_stim'])
            corr_pred, _ = pearsonr(row, self.Output_dic['rho_pred'])
            corr_with_rho_stim.append(corr_stim)
            corr_with_rho_pred.append(corr_pred)
            corr_stim_norm.append(corr_stim/np.sqrt(corr_stim**2+corr_pred**2))
            corr_pred_norm.append(corr_pred/np.sqrt(corr_stim**2+corr_pred**2))
        # Plot the scatter plot
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        # Plot the scatter plot
        scatter =axes.scatter(corr_with_rho_stim, corr_with_rho_pred, c=cdata, cmap=cmap, norm=norm,
                              alpha=0.8,s=8,edgecolors='black',linewidths=0.2)
        axes.set_aspect('equal')
        # Convert data to numpy array (assuming these are lists)
        data = np.vstack((corr_with_rho_stim, corr_with_rho_pred)).T  # Shape (N,2)

        center = np.mean(data, axis=0)  # Mean of the data
        corr_with_stim_norm=corr_with_rho_stim-np.mean(corr_with_rho_stim)
        corr_with_pred_norm=corr_with_rho_pred-np.mean(corr_with_rho_pred)
        dist_all = np.sqrt(corr_with_stim_norm**2 + corr_with_pred_norm**2)  # Equivalent to MATLAB dist_all = norm_bCT.^2 + norm_aCT.^2


        axes.plot([-1,1],[0,0],'--k',linewidth=0.5)
        axes.plot([0,0],[-1,1],'--k',linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=axes)
        #axes.set_facecolor('#045a8d')  # Light cyan
        #axes.set_facecolor('black')  # Light cyan
   # Add colorbar

        # Adjust layout for better spacing
        plt.tight_layout()

        if self.pf:
            fig.savefig(f"{self.sdir}nPEcorr_{tag}.svg", bbox_inches='tight')
            axes.set_xlabel("Correlation with rho_stim", fontsize=8)
            axes.set_ylabel("Correlation with rho_pred", fontsize=8)
            cbar.set_label('Mismatch dRate', rotation=270, labelpad=15)
            fig.savefig(f"{self.sdir}nPEcorr_{tag}.jpg", bbox_inches='tight')
        axes.set_title(f'nPEcorr{tag}')

        #theta_corr = np.arctan2(corr_pred_norm, corr_stim_norm )


        theta_corr = np.arctan2(corr_with_rho_pred, corr_with_rho_stim )
        theta_corr[theta_corr<-3/4*np.pi]+=2*np.pi
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))

        bin_width = np.pi/10.0
        bin_edges = np.arange(-3 * np.pi / 4,  5/4*np.pi+ bin_width, bin_width)  # Creates bins from -2 to 2 with width 0.2


        # Directly plot weighted histogram
        axes.hist(theta_corr, bins=bin_edges, weights=dist_all, density=True, color='blue', edgecolor='black', alpha=0.7)
        
        # Add custom x-ticks
        xticks_positions = [-3 * np.pi / 4, -np.pi / 4, np.pi / 4, 3/4*np.pi, 5/4*np.pi ]
        xticks_labels = [r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{4}$', r'$\frac{\pi}{4}$',r'$\frac{3\pi}{4}$',r'$-\frac{5\pi}{4}$']
        axes.set_xticks(xticks_positions)
        axes.set_xticklabels(xticks_labels)

        if self.pf:
            fig.savefig(f"{self.sdir}nPEcorrhist_{tag}.svg", bbox_inches='tight')
            axes.set_xlabel("Augular theta", fontsize=8)
            axes.set_ylabel("Density", fontsize=8)
            fig.savefig(f"{self.sdir}nPEcorrhist_{tag}.jpg", bbox_inches='tight')
        axes.set_title(f'nPEcorrhist {tag}')


    def nPE_PersonCorr_backup(self,cdata,start_color='green', end_color='#E65100',labels=('', ''), tag=''):
        # This one calculate specifically the Pearson correlation between neuron response with pred and stim strength
        # Initialize lists to store correlations
        # Create a custom diverging colormap
        colors = [start_color, 'white', end_color]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        max_val = np.max(np.abs(cdata))
        norm = plt.Normalize(vmin=-max_val, vmax=max_val)  # Normalize data symmetrically around zero

        corr_with_rho_stim = []
        corr_with_rho_pred = []

        # Calculate Pearson correlation for each row
        for row in self.Output_dic['E_pop_rate']:
            corr_stim, _ = pearsonr(row, self.Output_dic['rho_stim'])
            corr_pred, _ = pearsonr(row, self.Output_dic['rho_pred'])
            corr_with_rho_stim.append(corr_stim)
            corr_with_rho_pred.append(corr_pred)

        # Plot the scatter plot
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))

        # Plot the scatter plot
        scatter =axes.scatter(corr_with_rho_stim, corr_with_rho_pred, c=cdata, cmap=cmap, norm=norm,
                              alpha=0.8,s=8,edgecolors='black',linewidths=0.2)
        axes.set_aspect('equal')
        axes.plot([-1,1],[0,0],'--k',linewidth=0.5)
        axes.plot([0,0],[-1,1],'--k',linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=axes)
        #axes.set_facecolor('#045a8d')  # Light cyan
        #axes.set_facecolor('black')  # Light cyan
   # Add colorbar

        # Adjust layout for better spacing
        plt.tight_layout()

        if self.pf:
            fig.savefig(f"{self.sdir}nPEcorr_{tag}.svg", bbox_inches='tight')
            axes.set_xlabel("Correlation with rho_stim", fontsize=8)
            axes.set_ylabel("Correlation with rho_pred", fontsize=8)
            cbar.set_label('Mismatch dRate', rotation=270, labelpad=15)
            fig.savefig(f"{self.sdir}nPEcorr_{tag}.jpg", bbox_inches='tight')
        axes.set_title(f'nPEcorr{tag}')



        theta_corr = np.arctan2(corr_with_rho_pred, corr_with_rho_stim )
        theta_corr[theta_corr<-3/4*np.pi]+=2*np.pi
        fig, axes=plt.subplots(1,1,figsize=(2.8,2.1))

        bin_width = np.pi/10.0
        bin_edges = np.arange(-3 * np.pi / 4,  5/4*np.pi+ bin_width, bin_width)  # Creates bins from -2 to 2 with width 0.2


        axes.hist(theta_corr, bins=bin_edges, alpha=0.7, density=True, color='blue', edgecolor='black')
        # Add custom x-ticks
        xticks_positions = [-3 * np.pi / 4, -np.pi / 4, np.pi / 4, 3/4*np.pi, 5/4*np.pi ]
        xticks_labels = [r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{4}$', r'$\frac{\pi}{4}$',r'$\frac{3\pi}{4}$',r'$-\frac{5\pi}{4}$']
        axes.set_xticks(xticks_positions)
        axes.set_xticklabels(xticks_labels)

        if self.pf:
            fig.savefig(f"{self.sdir}nPEcorrhist_{tag}.svg", bbox_inches='tight')
            axes.set_xlabel("Augular theta", fontsize=8)
            axes.set_ylabel("Density", fontsize=8)
            fig.savefig(f"{self.sdir}nPEcorrhist_{tag}.jpg", bbox_inches='tight')
        axes.set_title(f'nPEcorrhist {tag}')



    def Plot_nRep(self,left=50, right=60,thr= 1):
        import seaborn as sns
        from scipy.stats import ttest_1samp

        n_stim= self.Paras['nbr_rep_std']+ self.Paras['nbr_rep_dev']
        n_trial_frame=round((self.Paras['Tinter']+self.Paras['Tstim'])/self.dtframe)
        Pop_average=np.mean(self.Output_dic['E_pop_rate'], axis=0) 
        choices= self.Paras['choices'].copy()    # 1 for std, 2 for dev
        pPE_id=self.std+self.N_eachC//2-1
        nPE_id=self.dev-self.N_eachC//2
        pPEx= self.Output_dic['E_pop_rate'][pPE_id,:]
        nPEy= self.Output_dic['E_pop_rate'][nPE_id,:]

        
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))  # sanity test on the firing rate
        axes.plot(self.t_axis[self.drawstart:]+self.t0,Pop_average[self.drawstart:],'k')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,pPEx[self.drawstart:],'g')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,nPEy[self.drawstart:],'b')
        for i_stim in np.arange(left,right):
            start=i_stim
            end= i_stim+self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            if round(choices[i_stim])==2:    
                axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           
            if round(choices[i_stim])==6:    
                axes.axvspan(start, end, color='red', alpha=0.3,label='dev')  # Adjust `alpha` for transparency           



        axes.set_xlim([left-0.5,right])
        #axes.set_xlim([-0.5,10])
        if self.pf:
            fig.savefig(self.sdir+'nRepMMN'+'PopAverage'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'nRepMMN'+'PopAverage'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'PopAverage')



        Pyr_trial_averages = []
        Int_trial_averages= []
        PyrInd_trial_averages=[]
        pPEx_trial_averages = []
        nPEy_trial_averages = []

        for i in range(self.FirstInd, self.n_frames, n_trial_frame):
            if i + self.shift <= self.n_frames:  # Ensure there are at least two frames to average
                avg = np.mean(Pop_average[i:i+self.shift ])  # Average of frames [i, i+1]
                Pyr_trial_averages.append(avg)
                ave_int= np.mean(self.Output_dic['Int_pop_rate'][2,i:i+self.shift ]) # average of the 2nd integrator
                Int_trial_averages.append(ave_int)
                avg_pPE = np.mean(pPEx[i:i+self.shift ])  # Average of frames [i, i+1]
                pPEx_trial_averages.append(avg_pPE)
                avg_nPE = np.mean(nPEy[i:i+self.shift ])  # Average of frames [i, i+1]
                nPEy_trial_averages.append(avg_nPE)


        # Need change.  format here has a data type problem
        choices= self.Paras['choices'].copy()    # 1 for std, 2 for dev

        Ind_test= choices == 2
        Ind_test[0]= False

        # Now let me plot some bar chat. 
        Pyr_trial_averages = np.array(Pyr_trial_averages)  # Convert to array if it's not already
        Int_trial_averages = np.array(Int_trial_averages) 
        pPEx_trial_averages = np.array(pPEx_trial_averages)
        nPEy_trial_averages = np.array(nPEy_trial_averages)

        # Create the violin plot, population average
        fig, axes = plt.subplots(1, 1, figsize=(0.9, 1.8))
        sns.violinplot(data=Pyr_trial_averages[Ind_test], inner="point", linewidth=1, ax=axes,palette=["#9ecae1"])
        axes.axhline(Pyr_trial_averages[0],color='red', linestyle='--', linewidth=0.5)
        mean_value = np.mean(Pyr_trial_averages[Ind_test])

        axes.scatter(0, mean_value, color='red', marker='^', s=100)
        
        # Add p-value as text to the plot
        population = Pyr_trial_averages[Ind_test]
        test_value = Pyr_trial_averages[0]
        t_stat, p_value = ttest_1samp(population, test_value)
        axes.text(
            0.2, 
            axes.get_ylim()[1] ,  # Position it near the top of the y-axis
            f"p: {p_value:.3f}",  # Format the p-value
            fontsize=8,
            color="black",
            ha="center"
        )


        # Customize the plot
        axes.set_ylabel("Pop Rate", fontsize=12)
        if self.pf:
            fig.savefig(self.sdir+'nRepMMN'+'ViolinPyrPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'nRepMMN'+'ViolinPyrPop'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'ViolinPyrPop')

        # Create the violin plot, integrator average
        fig, axes = plt.subplots(1, 1, figsize=(0.9, 1.8))
        sns.violinplot(data=Int_trial_averages[Ind_test], inner="point", linewidth=1, ax=axes,palette=["#9ecae1"])
        axes.axhline(Int_trial_averages[0],color='red', linestyle='--', linewidth=0.5)
        mean_value = np.mean(Int_trial_averages[Ind_test])
        axes.scatter(0, mean_value, color='red', marker='^', s=100)
        # Add p-value as text to the plot
        population = Int_trial_averages[Ind_test]
        test_value = Int_trial_averages[0]
        t_stat, p_value = ttest_1samp(population, test_value)
        axes.text(
            0.2, 
            axes.get_ylim()[1] ,  # Position it near the top of the y-axis
            f"p: {p_value:.3f}",  # Format the p-value
            fontsize=8,
            color="black",
            ha="center"
        )

        # Customize the plot
        axes.set_title("Violin Plot of Integrator Trial Averages", fontsize=14)
        #axes.set_xlabel("Trial Index", fontsize=12)
        axes.set_ylabel("Int Rate", fontsize=12)
        if self.pf:
            fig.savefig(self.sdir+'nRepMMN'+'ViolinInt'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'nRepMMN'+'ViolinInt'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'ViolinInt')


        PyrInd_trial_averages = []  # Initialize as an empty list

        for i in range(self.FirstInd, self.n_frames, n_trial_frame):
            if i + self.shift <= self.n_frames:  # Ensure there are enough frames to average
                ave_ind = np.mean(self.Output_dic['E_pop_rate'][:, i:i + self.shift], axis=1)  # Compute average across frames
                PyrInd_trial_averages.append(ave_ind)  # Append the column (1D array) as a new row

        PyrInd_trial_averages = np.array(PyrInd_trial_averages)
        #print(f"Shape of PyrInd_trial_averages: {PyrInd_trial_averages.shape}")
        PyrInd_trial_averages = PyrInd_trial_averages.T

        # Apply the Res_flag condition
        Res_flag = np.mean(PyrInd_trial_averages[:, Ind_test], axis=1) > (1 + thr) * self.First_pre_rate_vec

        # Compute the average of rows where Res_flag is True
        Pyr_res_trial_average = np.mean(PyrInd_trial_averages[Res_flag, :], axis=0)

        # Create the violin plot, population average
        fig, axes = plt.subplots(1, 1, figsize=(0.9, 1.8))
        sns.violinplot(data=Pyr_res_trial_average[Ind_test], inner="point", linewidth=1, ax=axes,palette=["#9ecae1"])

        axes.axhline(Pyr_res_trial_average[0],color='red', linestyle='--', linewidth=0.5)
        mean_value = np.mean(Pyr_res_trial_average[Ind_test])

        axes.scatter(0, mean_value, color='red', marker='^', s=100)
        # Add p-value as text to the plot
        population = Pyr_res_trial_average[Ind_test]
        test_value = Pyr_res_trial_average[0]
        t_stat, p_value = ttest_1samp(population, test_value)
        axes.text(
            0.2, 
            axes.get_ylim()[1] ,  # Position it near the top of the y-axis
            f"p: {p_value:.3f}",  # Format the p-value
            fontsize=8,
            color="black",
            ha="center"
        )

        # Customize the plot
        axes.set_title("Violin Plot of Pyr Trial Averages", fontsize=14)
        axes.set_ylabel("Res. Pop rate", fontsize=12)
        if self.pf:
            fig.savefig(self.sdir+'nRepMMN'+'ViolinRespPop'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'nRepMMN'+'ViolinRespPop'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'ViolinRespPop')

        # Plot the activity during the many-standard control.
        pPEx_control_mean= np.mean(pPEx_trial_averages[Ind_test])    
        pPEx_control_std= np.std(pPEx_trial_averages[Ind_test]) 
        nPEy_control_mean= np.mean(nPEy_trial_averages[Ind_test])    
        nPEy_control_std= np.std(nPEy_trial_averages[Ind_test]) 

        fig, axes=plt.subplots(1,1,figsize=(1.2,2.4))
        rates=[pPEx_control_mean]-self.baseline_rate_vec[pPE_id]
        errors = [pPEx_control_std]

        rep = ['C']
        axes.bar(rep,rates, yerr=errors, capsize=5,color=['#808080']) # grey, blue, red

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartpPEcontrol'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartpPEcontrol'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartpPEcontrol')


        fig, axes=plt.subplots(1,1,figsize=(1.2,2.4))
        rates=[nPEy_control_mean]-self.baseline_rate_vec[nPE_id]
        errors = [nPEy_control_std]

        rep = ['C']
        axes.bar(rep,rates, yerr=errors, capsize=5,color=['#808080']) # grey, blue, red

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartnPEcontrol'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartnPEcontrol'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartnPEcontrol')






    def Plot_real_odd(self,left=50, right=60,thr= 1):
        # still need to check
        # Hope this is the last I did. Let's find out the first, many control, and whatever. 
        import seaborn as sns
        from scipy.stats import ttest_1samp

        n_stim= self.Paras['nbr_rep_std']+ self.Paras['nbr_rep_dev']
        n_trial_frame=round((self.Paras['Tinter']+self.Paras['Tstim'])/self.dtframe)
        Pop_average=np.mean(self.Output_dic['E_pop_rate'], axis=0) 

        pPE_id=self.std+self.N_eachC//2-1
        nPE_id=self.dev-self.N_eachC//2
        pPEx= self.Output_dic['E_pop_rate'][pPE_id,:]
        nPEy= self.Output_dic['E_pop_rate'][nPE_id,:]

        pPE_dev_id=self.dev+self.N_eachC//2-1
        nPE_dev_id=self.std-self.N_eachC//2
        pPEx_dev= self.Output_dic['E_pop_rate'][pPE_dev_id,:]
        nPEy_dev= self.Output_dic['E_pop_rate'][nPE_dev_id,:]
        
        choices= self.Paras['choices']    # 1 for std, 2 for dev
        
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))  # sanity test on the firing rate
        axes.plot(self.t_axis[self.drawstart:]+self.t0,Pop_average[self.drawstart:],'k')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,pPEx[self.drawstart:],'g')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,nPEy[self.drawstart:],'b')

        for i_stim in np.arange(left,right):
            start=i_stim
            end= i_stim+self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            if round(choices[i_stim])==1:    
                axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           
        axes.set_xlim([left-0.5,right])
        #axes.set_xlim([-0.5,10])
        if self.pf:
            fig.savefig(self.sdir+'nRepMMN'+'PopAverage'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'nRepMMN'+'PopAverage'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'PopAverage')
        Pyr_trial_averages = []
        Int_trial_averages= []
        pPEx_trial_averages = []
        nPEy_trial_averages = []
        pPEx_dev_trial_averages = []
        nPEy_dev_trial_averages = []


        for i in range(self.FirstInd, self.n_frames, n_trial_frame):
            if i + self.shift <= self.n_frames:  # Ensure there are at least two frames to average
                avg = np.mean(Pop_average[i:i+self.shift ])  # Average of frames [i, i+1]
                Pyr_trial_averages.append(avg)
                ave_int= np.mean(self.Output_dic['Int_pop_rate'][2,i:i+self.shift ]) # average of the 2nd integrator
                Int_trial_averages.append(ave_int)
                avg_pPE = np.mean(pPEx[i:i+self.shift ])  # Average of frames [i, i+1]
                pPEx_trial_averages.append(avg_pPE)
                avg_nPE = np.mean(nPEy[i:i+self.shift ])  # Average of frames [i, i+1]
                nPEy_trial_averages.append(avg_nPE)
                avg_pPE_dev = np.mean(pPEx_dev[i:i+self.shift ])  # Average of frames [i, i+1]
                pPEx_dev_trial_averages.append(avg_pPE_dev)
                avg_nPE_dev = np.mean(nPEy_dev[i:i+self.shift ])  # Average of frames [i, i+1]
                nPEy_dev_trial_averages.append(avg_nPE_dev)           
                
                
                

        pPEx_trial_averages = np.array(pPEx_trial_averages)
        nPEy_trial_averages = np.array(nPEy_trial_averages)
        pPEx_dev_trial_averages = np.array(pPEx_dev_trial_averages)
        nPEy_dev_trial_averages = np.array(nPEy_dev_trial_averages)



        # Need change.  format here has a data type problem
        choices= self.Paras['choices'].copy()    # 1 for std, 2 for dev 
        choices[0:5]  =  0                # Remove the first 5 in testing 
        choices_oneshift= choices.copy()    # Here I need to change to copy. Modify later.
        choices_oneshift[1:]=choices[:-1]
        choices_twoshift= choices.copy()
        choices_twoshift[2:]=choices[:-2]


        Ind_dev= choices == 2    
        Ind_first= (choices == 1)  & (choices_oneshift != 1    )         # Need a way to get the repetion.  
        Ind_rdnt= (choices == 1)  & (choices_oneshift == 1)  & (choices_twoshift == 1)
        

        pPEx_first_mean= np.mean(pPEx_trial_averages[Ind_first])    
        pPEx_first_std= np.std(pPEx_trial_averages[Ind_first]) 
        pPEx_rdnt_mean= np.mean(pPEx_trial_averages[Ind_rdnt])    
        pPEx_rdnt_std= np.std(pPEx_trial_averages[Ind_rdnt])
        pPEx_dev_mean= np.mean(pPEx_dev_trial_averages[Ind_dev])    
        pPEx_dev_std= np.std(pPEx_dev_trial_averages[Ind_dev]) 


        nPEy_first_mean= np.mean(nPEy_trial_averages[Ind_first])    
        nPEy_first_std= np.std(nPEy_trial_averages[Ind_first]) 
        nPEy_rdnt_mean= np.mean(nPEy_trial_averages[Ind_rdnt])    
        nPEy_rdnt_std= np.std(nPEy_trial_averages[Ind_rdnt])
        nPEy_dev_mean= np.mean(nPEy_dev_trial_averages[Ind_dev])    
        nPEy_dev_std= np.std(nPEy_dev_trial_averages[Ind_dev]) 

        fig, axes=plt.subplots(1,1,figsize=(1.6,3.2))
        rates=[pPEx_first_mean,pPEx_rdnt_mean,pPEx_dev_mean]-self.baseline_rate_vec[pPE_id]
        errors = [pPEx_first_std, pPEx_rdnt_std, pPEx_dev_std]

        rep = ['F', 'R', 'D']
        axes.bar(rep,rates, yerr=errors, capsize=5,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartpPE'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartpPE'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartpPE')


        fig, axes=plt.subplots(1,1,figsize=(3,6))
        rates=[pPEx_first_mean,pPEx_rdnt_mean,pPEx_dev_mean]-self.baseline_rate_vec[pPE_id]
        errors = [pPEx_first_std, pPEx_rdnt_std, pPEx_dev_std]

        rep = ['F', 'R', 'D']
        axes.bar(rep,rates, yerr=errors, capsize=5,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.get_xaxis().set_visible(True)

        axes.set_title('RealOddballBarchartpPE')


        fig, axes=plt.subplots(1,1,figsize=(1.6,3.2))
        rates=[nPEy_first_mean,nPEy_rdnt_mean,nPEy_dev_mean]-self.baseline_rate_vec[nPE_id]
        errors = [nPEy_first_std, nPEy_rdnt_std, nPEy_dev_std]

        rep = ['C', 'R', 'D']
        axes.bar(rep,rates, yerr=errors, capsize=5,color=['#808080', '#3182BD', '#E34A33']) # grey, blue, red

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartnPE'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartnPE'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartnPE')




    def Plot_real_odd_combine(self,self_ctrl,left=50, right=60,thr= 1):
        # still need to check
        # Hope this is the last I did. Let's find out the first, many control, and whatever. 
        import seaborn as sns
        from scipy.stats import ttest_1samp
        from scipy.stats import ttest_ind

        baseline_int_rate_vec=np.mean(
            self.Output_dic['Int_pop_rate'][:,self.drawstart:self.drawstart+self.shift], 
            axis=1)          
        self.baseline_int_rate_vec=baseline_int_rate_vec

        baseline_int_ctrl_rate_vec=np.mean(
            self_ctrl.Output_dic['Int_pop_rate'][:,self.drawstart:self.drawstart+self.shift], 
            axis=1)          
        self.baseline_int_ctrl_rate_vec=baseline_int_ctrl_rate_vec


        #n_stim= self.Paras['nbr_rep_std']+ self.Paras['nbr_rep_dev']
        n_trial_frame=round((self.Paras['Tinter']+self.Paras['Tstim'])/self.dtframe)
        Pop_average=np.mean(self.Output_dic['E_pop_rate'], axis=0) 

        id_shift=-10
        pPE_id=self.std+self.N_eachC//2-1+ id_shift
        pPE_dev_id=self.dev+self.N_eachC//2-1+ id_shift

        nPE_id=self.dev-self.N_eachC//2
        nPE_dev_id=self.std-self.N_eachC//2



        pPEx= self.Output_dic['E_pop_rate'][pPE_id,:]
        nPEy= self.Output_dic['E_pop_rate'][nPE_id,:]
        
        pPEx_ctrl= self_ctrl.Output_dic['E_pop_rate'][pPE_id,:]
        nPEy_ctrl= self_ctrl.Output_dic['E_pop_rate'][nPE_id,:]

        Int_average=np.mean(self.Output_dic['Int_pop_rate'], axis=0) 
        Int_std= np.std(self.Output_dic['Int_pop_rate'], axis=0) 
        Intx=self.Output_dic['Int_pop_rate'][self.stdcn,:]
        Inty=self.Output_dic['Int_pop_rate'][self.devcn,:]

        Int_ctrl_average=np.mean(self.Output_dic['Int_pop_rate'], axis=0) 
        Int_ctrl_std= np.std(self.Output_dic['Int_pop_rate'], axis=0) 
        Intx_ctrl=self_ctrl.Output_dic['Int_pop_rate'][self.stdcn,:]
        Inty_ctrl=self_ctrl.Output_dic['Int_pop_rate'][self.devcn,:]


        pPEx_dev= self.Output_dic['E_pop_rate'][pPE_dev_id,:]
        nPEy_dev= self.Output_dic['E_pop_rate'][nPE_dev_id,:]
        
        choices= self.Paras['choices']    # 1 for std, 2 for dev
        
        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))  # sanity test on the firing rate
        axes.plot(self.t_axis[self.drawstart:]+self.t0,Pop_average[self.drawstart:],'k')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,pPEx[self.drawstart:],'g')
        axes.plot(self.t_axis[self.drawstart:]+self.t0,nPEy[self.drawstart:],'b')

        for i_stim in np.arange(left,right):
            start=i_stim
            end= i_stim+self.Paras['Tstim']
            axes.axvline(start, color='grey', linestyle='--', linewidth=0.5)
            axes.axvline(end, color='grey', linestyle='--', linewidth=0.5)

            if round(choices[i_stim])==1:    
                axes.axvspan(start, end, color='grey', alpha=0.3,label='std')  # Adjust `alpha` for transparency           
        axes.set_xlim([left-0.5,right])
        #axes.set_xlim([-0.5,10])
        if self.pf:
            fig.savefig(self.sdir+'nRepMMN'+'PopAverage'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'nRepMMN'+'PopAverage'+'.svg',bbox_inches='tight')
        axes.set_title('ProbMMN'+'PopAverage')
        Pyr_trial_averages = []


        pPEx_trial_averages = []
        nPEy_trial_averages = []
        pPEx_ctrl_trial_averages = []
        nPEy_ctrl_trial_averages = []
        pPEx_dev_trial_averages = []
        nPEy_dev_trial_averages = []

        Int_mean_trial_averages= []
        Int_std_trial_averages= []        
        Int_ctrl_mean_trial_averages= []
        Int_ctrl_std_trial_averages= []        

        Intx_trial_averages=[]
        Inty_trial_averages=[]

        Intx_ctrl_trial_averages=[]
        Inty_ctrl_trial_averages=[]

        for i in range(self.FirstInd, self.n_frames, n_trial_frame):
            if i + self.shift <= self.n_frames:  # Ensure there are at least two frames to average
                avg = np.mean(Pop_average[i:i+self.shift ])  # Average of frames [i, i+1]
                Pyr_trial_averages.append(avg)

                avg_pPE = np.mean(pPEx[i:i+self.shift ])  # Average of frames [i, i+1]
                pPEx_trial_averages.append(avg_pPE)
                avg_nPE = np.mean(nPEy[i:i+self.shift ])  # Average of frames [i, i+1]
                nPEy_trial_averages.append(avg_nPE)

                avg_pPE_ctrl = np.mean(pPEx_ctrl[i:i+self.shift ])  # Average of frames [i, i+1]
                pPEx_ctrl_trial_averages.append(avg_pPE_ctrl)
                avg_nPE_ctrl = np.mean(nPEy_ctrl[i:i+self.shift ])  # Average of frames [i, i+1]
                nPEy_ctrl_trial_averages.append(avg_nPE_ctrl)

                avg_pPE_dev = np.mean(pPEx_dev[i:i+self.shift ])  # Average of frames [i, i+1]
                pPEx_dev_trial_averages.append(avg_pPE_dev)
                avg_nPE_dev = np.mean(nPEy_dev[i:i+self.shift ])  # Average of frames [i, i+1]
                nPEy_dev_trial_averages.append(avg_nPE_dev)           
                
                ave_int= np.mean(Int_average[i:i+self.shift ]) # average of the integrators
                Int_mean_trial_averages.append(ave_int)
                ave_int_std= np.mean(Int_std[i:i+self.shift ]) # average of the integrators std
                Int_std_trial_averages.append(ave_int_std)

                ave_int_ctrl= np.mean(Int_ctrl_average[i:i+self.shift ]) # average of the integrators
                Int_ctrl_mean_trial_averages.append(ave_int_ctrl)
                ave_int_ctrl_std= np.mean(Int_ctrl_std[i:i+self.shift ]) # average of the integrators std
                Int_ctrl_std_trial_averages.append(ave_int_ctrl_std)

                ave_intx= np.mean(Intx[i:i+self.shift ]) # average of the 2nd integrator x
                Intx_trial_averages.append(ave_intx)
                ave_inty= np.mean(Inty[i:i+self.shift ]) # average of the 6nd integrator y
                Inty_trial_averages.append(ave_inty)

                ave_intx_ctrl= np.mean(Intx_ctrl[i:i+self.shift ]) # average of the 2nd integrator x
                Intx_ctrl_trial_averages.append(ave_intx_ctrl)
                ave_inty_ctrl= np.mean(Inty_ctrl[i:i+self.shift ]) # average of the 6nd integrator y
                Inty_ctrl_trial_averages.append(ave_inty_ctrl)                

        pPEx_trial_averages = np.array(pPEx_trial_averages)
        nPEy_trial_averages = np.array(nPEy_trial_averages)

        pPEx_ctrl_trial_averages = np.array(pPEx_ctrl_trial_averages)
        nPEy_ctrl_trial_averages = np.array(nPEy_ctrl_trial_averages)

        pPEx_dev_trial_averages = np.array(pPEx_dev_trial_averages)
        nPEy_dev_trial_averages = np.array(nPEy_dev_trial_averages)

        Int_mean_trial_averages = np.array(Int_mean_trial_averages)
        Int_std_trial_averages = np.array(Int_std_trial_averages)

        Int_ctrl_mean_trial_averages = np.array(Int_ctrl_mean_trial_averages)
        Int_ctrl_std_trial_averages = np.array(Int_ctrl_std_trial_averages)

        Intx_trial_averages = np.array(Intx_trial_averages)
        Inty_trial_averages = np.array(Inty_trial_averages)
        Intx_ctrl_trial_averages = np.array(Intx_ctrl_trial_averages)
        Inty_ctrl_trial_averages = np.array(Inty_ctrl_trial_averages)

        # Need change.  format here has a data type problem
        choices= self.Paras['choices'].copy()    # 1 for std, 2 for dev 
        choices[0:5]  =  0                # Remove the first 5 in testing 
        choices_oneshift= choices.copy()    # Here I need to change to copy. Modify later.
        choices_oneshift[1:]=choices[:-1]
        choices_twoshift= choices.copy()
        choices_twoshift[2:]=choices[:-2]


        Ind_dev= choices == 2    
        Ind_first= (choices == 1)  & (choices_oneshift != 1    )         # Need a way to get the repetion.  
        Ind_rdnt= (choices == 1)  & (choices_oneshift == 1)  & (choices_twoshift == 1)
        
        Ind_ctrl= self_ctrl.Paras['choices'].copy()== 2
        Ind_ctrl[0]= False
        pPEx_first_mean= np.mean(pPEx_trial_averages[Ind_first])    -self.baseline_rate_vec[pPE_id]
        pPEx_first_std= np.std(pPEx_trial_averages[Ind_first]) 
        pPEx_rdnt_mean= np.mean(pPEx_trial_averages[Ind_rdnt])    -self.baseline_rate_vec[pPE_id]
        pPEx_rdnt_std= np.std(pPEx_trial_averages[Ind_rdnt])
        pPEx_dev_mean= np.mean(pPEx_dev_trial_averages[Ind_dev])   -self.baseline_rate_vec[ pPE_dev_id] 
        pPEx_dev_std= np.std(pPEx_dev_trial_averages[Ind_dev]) 

        pPEx_control_mean= np.mean(pPEx_ctrl_trial_averages[Ind_ctrl])   -self_ctrl.baseline_rate_vec[pPE_id]  
        pPEx_control_std= np.std(pPEx_ctrl_trial_averages[Ind_ctrl]) 

        nPEy_first_mean= np.mean(nPEy_trial_averages[Ind_first])    -self.baseline_rate_vec[nPE_id]
        nPEy_first_std= np.std(nPEy_trial_averages[Ind_first]) 
        nPEy_rdnt_mean= np.mean(nPEy_trial_averages[Ind_rdnt])    -self.baseline_rate_vec[nPE_id]
        nPEy_rdnt_std= np.std(nPEy_trial_averages[Ind_rdnt])
        nPEy_dev_mean= np.mean(nPEy_dev_trial_averages[Ind_dev])    -self.baseline_rate_vec[nPE_dev_id]
        nPEy_dev_std= np.std(nPEy_dev_trial_averages[Ind_dev]) 

        nPEy_control_mean= np.mean(nPEy_ctrl_trial_averages[Ind_ctrl])   -self_ctrl.baseline_rate_vec[nPE_id] 
        nPEy_control_std= np.std(nPEy_ctrl_trial_averages[Ind_ctrl]) 

        pPEx_means=[pPEx_control_mean, pPEx_first_mean,pPEx_rdnt_mean,pPEx_dev_mean]
        nPE_means=[nPEy_control_mean,nPEy_first_mean,nPEy_rdnt_mean,nPEy_dev_mean]
 

        # Perform Welch's t-test (assuming unequal variances)
        p_values_pPEx = [
            ttest_ind(pPEx_ctrl_trial_averages[Ind_ctrl], pPEx_trial_averages[Ind_first], equal_var=False)[1], 
            ttest_ind(pPEx_ctrl_trial_averages[Ind_ctrl], pPEx_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(pPEx_ctrl_trial_averages[Ind_ctrl], pPEx_trial_averages[Ind_dev], equal_var=False)[1]
        ]
        p2_values_pPEx = [
            ttest_ind(pPEx_trial_averages[Ind_first], pPEx_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(pPEx_trial_averages[Ind_first], pPEx_trial_averages[Ind_dev], equal_var=False)[1]
        ]
        p_values_nPEy = [
            ttest_ind(nPEy_ctrl_trial_averages[Ind_ctrl], nPEy_trial_averages[Ind_first], equal_var=False)[1], 
            ttest_ind(nPEy_ctrl_trial_averages[Ind_ctrl], nPEy_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(nPEy_ctrl_trial_averages[Ind_ctrl], nPEy_trial_averages[Ind_dev], equal_var=False)[1]
        ]
        p2_values_nPEy = [
            ttest_ind(nPEy_trial_averages[Ind_first], nPEy_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(nPEy_trial_averages[Ind_first], nPEy_trial_averages[Ind_dev], equal_var=False)[1]
        ]
        # Function to annotate p-values
        def annotate_p_values(ax, bars, p_values):
            for i, (bar, p) in enumerate(zip(bars[1:], p_values)):  # Skip control (index 0)
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.text(x, y + 0.05 * max(pPEx_means), f"p={p:.3f}", ha='center', fontsize=8, color='black')

        def annotate_p2_values(ax, bars, p_values):
            for i, (bar, p) in enumerate(zip(bars[2:], p_values)):  # Skip control and first (index 0,1 )
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.text(x, y + 0.2 * max(pPEx_means), f"p={p:.3f}", ha='center', fontsize=8, color='black')




        # --- Plot pPEx Bar Chart ---

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        
        errors = [pPEx_control_std, pPEx_first_std, pPEx_rdnt_std, pPEx_dev_std]

        rep = ['C', 'F', 'R', 'D']
        bars=axes.bar(rep,pPEx_means, yerr=errors, capsize=5,color=['#808080','#FF8C00', '#3182BD', '#E34A33']) # grey, blue, red
        annotate_p_values(axes, bars, p_values_pPEx)
        annotate_p2_values(axes, bars, p2_values_pPEx)

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartpPE'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartpPE'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartpPE')

        # --- Plot nPEy Bar Chart ---

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        errors = [nPEy_control_std, nPEy_first_std, nPEy_rdnt_std, nPEy_dev_std]

        rep = ['C', 'F', 'R', 'D']
        bars=axes.bar(rep,nPE_means, yerr=errors, capsize=5,color=['#808080','#FF8C00', '#3182BD', '#E34A33']) # grey, blue, red
        annotate_p_values(axes, bars, p_values_nPEy)
        annotate_p2_values(axes, bars, p2_values_nPEy)

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartnPE'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartnPE'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartnPE')


        # Include here for Int bar chart  
        Intx_first_mean= np.mean(Intx_trial_averages[Ind_first])    -self.baseline_int_rate_vec[self.stdcn]
        Intx_first_std= np.std(Intx_trial_averages[Ind_first]) 
        Intx_rdnt_mean= np.mean(Intx_trial_averages[Ind_rdnt])    -self.baseline_int_rate_vec[self.stdcn]
        Intx_rdnt_std= np.std(Intx_trial_averages[Ind_rdnt])


        Intx_dev_mean= np.mean(Inty_trial_averages[Ind_dev])   -self.baseline_int_rate_vec[ self.devcn] 
        Intx_dev_std= np.std(Inty_trial_averages[Ind_dev]) 

        Intx_control_mean= np.mean(Intx_ctrl_trial_averages[Ind_ctrl])   -self.baseline_int_ctrl_rate_vec[self.stdcn]  
        Intx_control_std= np.std(Intx_ctrl_trial_averages[Ind_ctrl]) 

        Inty_first_mean= np.mean(Inty_trial_averages[Ind_first])    -self.baseline_int_rate_vec[self.devcn]
        Inty_first_std= np.std(Inty_trial_averages[Ind_first]) 
        Inty_rdnt_mean= np.mean(Inty_trial_averages[Ind_rdnt])    -self.baseline_int_rate_vec[self.devcn]
        Inty_rdnt_std= np.std(Inty_trial_averages[Ind_rdnt])
        Inty_dev_mean= np.mean(Intx_trial_averages[Ind_dev])    -self.baseline_int_rate_vec[self.stdcn]
        Inty_dev_std= np.std(Intx_trial_averages[Ind_dev]) 

        Inty_control_mean= np.mean(Inty_ctrl_trial_averages[Ind_ctrl])   -self.baseline_int_ctrl_rate_vec[self.devcn] 
        Inty_control_std= np.std(Inty_ctrl_trial_averages[Ind_ctrl]) 

        Intx_means=[Intx_control_mean, Intx_first_mean,Intx_rdnt_mean,Intx_dev_mean]
        Inty_means=[Inty_control_mean,Inty_first_mean,Inty_rdnt_mean,Inty_dev_mean]
 

        # Perform Welch's t-test (assuming unequal variances)
        p_values_Intx = [
            ttest_ind(Intx_ctrl_trial_averages[Ind_ctrl], Intx_trial_averages[Ind_first], equal_var=False)[1], 
            ttest_ind(Intx_ctrl_trial_averages[Ind_ctrl], Intx_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(Intx_ctrl_trial_averages[Ind_ctrl], Intx_trial_averages[Ind_dev], equal_var=False)[1]
        ]
        p2_values_Intx = [
            ttest_ind(Intx_trial_averages[Ind_first], Intx_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(Intx_trial_averages[Ind_first], Intx_trial_averages[Ind_dev], equal_var=False)[1]
        ]

        p_values_Inty = [
            ttest_ind(Inty_ctrl_trial_averages[Ind_ctrl], Inty_trial_averages[Ind_first], equal_var=False)[1], 
            ttest_ind(Inty_ctrl_trial_averages[Ind_ctrl], Inty_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(Inty_ctrl_trial_averages[Ind_ctrl], Inty_trial_averages[Ind_dev], equal_var=False)[1]
        ]
        p2_values_Inty = [
            ttest_ind(Inty_trial_averages[Ind_first], Inty_trial_averages[Ind_rdnt], equal_var=False)[1], 
            ttest_ind(Inty_trial_averages[Ind_first], Inty_trial_averages[Ind_dev], equal_var=False)[1]
        ]

        # --- Plot Intx Bar Chart ---

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        
        errors = [Intx_control_std, Intx_first_std, Intx_rdnt_std, Intx_dev_std]

        rep = ['C', 'F', 'R', 'D']
        bars=axes.bar(rep,Intx_means, yerr=errors, capsize=5,color=['#808080','#FF8C00', '#3182BD', '#E34A33']) # grey, blue, red
        annotate_p_values(axes, bars, p_values_Intx)
        annotate_p2_values(axes, bars, p2_values_Intx)

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartIntx'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartIntx'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartIntx')

        # --- Plot Inty Bar Chart ---

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        errors = [Inty_control_std, Inty_first_std, Inty_rdnt_std, Inty_dev_std]

        rep = ['C', 'F', 'R', 'D']
        bars=axes.bar(rep,Inty_means, yerr=errors, capsize=5,color=['#808080','#FF8C00', '#3182BD', '#E34A33']) # grey, blue, red
        annotate_p_values(axes, bars, p_values_Inty)
        annotate_p2_values(axes, bars, p2_values_Inty)

        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartInty'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartInty'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartInty')



        # Include here for Integrator mean and std bar chart  
        Int_mean_first_mean= np.mean(Int_mean_trial_averages[Ind_first])    -np.mean(self.baseline_int_rate_vec)
        Int_mean_first_std= np.std(Int_mean_trial_averages[Ind_first]) 
        Int_mean_rdnt_mean= np.mean(Int_mean_trial_averages[Ind_rdnt])    -np.mean(self.baseline_int_rate_vec)
        Int_mean_rdnt_std= np.std(Int_mean_trial_averages[Ind_rdnt])


        Int_mean_dev_mean= np.mean(Int_mean_trial_averages[Ind_dev])   -np.mean(self.baseline_int_rate_vec)
        Int_mean_dev_std= np.std(Int_mean_trial_averages[Ind_dev]) 

        Int_mean_control_mean= np.mean(Int_ctrl_mean_trial_averages[Ind_ctrl])   -np.mean(self.baseline_int_ctrl_rate_vec)
        Int_mean_control_std= np.std(Int_ctrl_mean_trial_averages[Ind_ctrl]) 

        Int_std_first_mean= np.mean(Int_std_trial_averages[Ind_first])  
        Int_std_first_std= np.std(Int_std_trial_averages[Ind_first]) 
        Int_std_rdnt_mean= np.mean(Int_std_trial_averages[Ind_rdnt])   
        Int_std_rdnt_std= np.std(Int_std_trial_averages[Ind_rdnt])
        Int_std_dev_mean= np.mean(Int_std_trial_averages[Ind_dev])    
        Int_std_dev_std= np.std(Int_std_trial_averages[Ind_dev]) 

        Int_std_control_mean= np.mean(Int_ctrl_std_trial_averages[Ind_ctrl])   
        Int_std_control_std= np.std(Int_ctrl_std_trial_averages[Ind_ctrl]) 

        Int_mean_means=[Int_mean_control_mean, Int_mean_first_mean,Int_mean_rdnt_mean,Int_mean_dev_mean]
        Int_std_means=[Int_std_control_mean,Int_std_first_mean,Int_std_rdnt_mean,Int_std_dev_mean]

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        
        errors = [Int_mean_control_std, Int_mean_first_std, Int_mean_rdnt_std, Int_mean_dev_std]

        rep = ['C', 'F', 'R', 'D']
        bars=axes.bar(rep,Int_mean_means, yerr=errors, capsize=5,color=['#808080','#FF8C00', '#3182BD', '#E34A33']) # grey, orange, blue, red
        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartIntsMean'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartIntsMean'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartIntsMean')

        fig, axes=plt.subplots(1,1,figsize=(2.4,1.8))
        
        errors = [Int_std_control_std, Int_std_first_std, Int_std_rdnt_std, Int_std_dev_std]

        rep = ['C', 'F', 'R', 'D']
        bars=axes.bar(rep,Int_std_means, yerr=errors, capsize=5,color=['#808080','#FF8C00', '#3182BD', '#E34A33']) # grey, orange, blue, red
        axes.get_xaxis().set_visible(True)
        if self.pf:
            fig.savefig(self.sdir+'RealOddballBarchartIntsStd'+'.jpg',bbox_inches='tight')
            fig.savefig(self.sdir+'RealOddballBarchartIntsStd'+'.svg',bbox_inches='tight')
        axes.set_title('RealOddballBarchartIntsStd')


    def Plot_weight_diagram(self):
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.5))
        N_lines = 6
        segments_total = 200
        base_count = segments_total // N_lines  # Base number of segments per iteration
        remainder = segments_total % N_lines  # Extra segments to distribute

        all_segments = []  # To store all line segments (as coordinate arrays)
        all_colors = []    # To store colors for each segment
        dash_segments = [] # To store dashed line segments
        dash_colors = []   # Stores colors for dashed lines

        # Colormap for unique target colors (e.g., viridis)
        cmap = plt.cm.get_cmap('viridis')
        end_time = self.n_frames  

        start_points = []  # Store start points for triangles
        end_points = []    # Store end points for circles
        marker_colors = []  # Colors for the start and end markers
        Neu_indices = np.linspace(0, self.Nneu - 1, N_lines, dtype=int)  # Guarantees min and max indices are included

        for i in range(N_lines):
            # Determine how many segments this iteration gets
            seg_count = base_count + (1 if i < remainder else 0)
            if seg_count < 1:
                continue  # Skip if no segments (shouldn't happen unless N_lines > 100)

            # Select the data index for this iteration
            index = Neu_indices[i]
            x_line = self.Output_dic['W_EdS'][index, :end_time]
            y_line = self.Output_dic['W_EP'][index, :end_time]

            # Clustered sampling at start
            t = np.linspace(0.0, 1.0, seg_count + 1) ** 4  # Bias clustering towards start
            positions = t * (len(x_line) - 1)
            x_points = np.interp(positions, np.arange(len(x_line)), x_line)
            y_points = np.interp(positions, np.arange(len(y_line)), y_line)

            # Determine the target color for this iteration
            target_color = np.array(cmap(i / (N_lines+2 - 1)))  # RGBA color
            target_rgb = target_color[:3]
            ax.plot(x_line, y_line, color=target_rgb, linewidth=0.7)

            # Store start and end points for markers
            start_points.append((x_points[0], y_points[0]))   # First point (triangle)
            end_points.append((x_points[-1], y_points[-1]))   # Last point (circle)
            marker_colors.append(target_rgb)  # Store color for both markers

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
            '''
            Sum_input = Input_Stim[index] + Input_Pred[index] - E_baseline
            dash_segments.append(np.array([[Sum_input, 0], [0, Sum_input]]))
            dash_colors.append((target_rgb[0], target_rgb[1], target_rgb[2], 1.0))  # Last color from gradient
            '''

        # Create a LineCollection for all colored segments
        line_coll = LineCollection(all_segments, colors=all_colors, linewidths=1)
        #dash_coll = LineCollection(dash_segments, colors=dash_colors, linestyles='--', linewidths=0.5)

        #ax.add_collection(line_coll)
        #ax.add_collection(dash_coll)

        # Convert lists to NumPy arrays for scatter plotting
        start_points = np.array(start_points)
        end_points = np.array(end_points)
        marker_colors = np.array(marker_colors)

        # Plot start markers (triangles)
        ax.scatter(start_points[:, 0], start_points[:, 1], color=marker_colors, marker='x', s=12, label="Start")

        # Plot end markers (circles)
        ax.scatter(end_points[:, 0], end_points[:, 1], color=marker_colors, marker='o', s=8,  label="End")
        ax.set_yticks([0, 50])
        ax.set_xticks([0, 50])

        ax.autoscale()  # Adjust axes to show all segments
        ax.set_aspect('equal')
        plt.show()
        name='WeightDiagram'
        fig.savefig(self.sdir + name + '.jpg', bbox_inches='tight')
        fig.savefig(self.sdir + name + '.svg', bbox_inches='tight')
    
    def column_vs_matrix_corr(self,column_vec, matrix):
        """
        Calculate Pearson correlation between a single vector and each column of a matrix.

        Parameters:
        column_vec (array-like): 1D array of shape (N,)
        matrix (array-like): 2D array of shape (N, M)

        Returns:
        np.ndarray: 1D array of Pearson correlations of length M
        """
        column_vec = np.asarray(column_vec)
        matrix = np.asarray(matrix)

        # Ensure shapes match
        if column_vec.ndim != 1 or matrix.shape[0] != column_vec.shape[0]:
            raise ValueError("Dimension mismatch: column_vec must be 1D and match matrix rows.")

        # Center data
        col_centered = column_vec - np.mean(column_vec)
        mat_centered = matrix - np.mean(matrix, axis=0)

        # Compute numerator and denominator
        numerator = np.sum(col_centered[:, np.newaxis] * mat_centered, axis=0)
        denominator = np.linalg.norm(col_centered) * np.linalg.norm(mat_centered, axis=0)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = np.where(denominator != 0, numerator / denominator, 0)

        return correlation
    

    def column_vs_matrix_fisher(self,column_vec, matrix):
        """
        Parameters:
        labels (array-like): 1D array of 0s and 1s of shape (N,)
        matrix (array-like): 2D array of shape (N, M)
        Using the firing rate as the variance.
        Returns:
        np.ndarray: 1D array of d-prime values for each column
        """
        column_vec = np.asarray(column_vec)
        matrix = np.asarray(matrix)

        # Ensure shapes match
        if column_vec.ndim != 1 or matrix.shape[0] != column_vec.shape[0]:
            raise ValueError("Dimension mismatch: column_vec must be 1D and match matrix rows.")

        fisher_vec=[]
        for i_col in range(matrix.shape[1]):
            col_data = matrix[:, i_col]
            dprime= (column_vec-col_data)**2/ (column_vec+ col_data)  
            fisher_icol=np.sum(dprime)
            fisher_vec.append(fisher_icol)

        return np.array(fisher_vec)



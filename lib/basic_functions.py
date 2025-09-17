import numpy as np
from numba import njit
from scipy.linalg import block_diag
from scipy.stats import truncnorm

import logging
logger = logging.getLogger(__name__)

@njit(cache=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

@njit(cache=True)
def relu(x):
    return np.maximum(x, 0)

@njit(cache=True)
def relu_deriv(x):
    return np.heaviside(x, 0)

@njit(cache=True)
def abott_fI_curve(x, a, b,d):    # Notice this formula has a singular point that needs to be handled. 
        temp=0.0*x
        ind_flag= a*x-b==0
        temp[ind_flag]=1.0/d
        temp[~ ind_flag]=np.divide(a*x[~ ind_flag] - b, 1.0 - np.exp(-d*(a*x[~ ind_flag] - b)))
        return temp

@njit(cache=True)
def abott_fI_curve2(x, gl=20, tau=0.015):    #   Needs to check the units. From Abbott and Chance 2005
        Vl=-70
        Vss=Vl+ x/gl    # check the units later
        Vth=-50
        # Vr=-60 nu=1 simplified in the calculation
        temp=0.0*x
        diff_v= Vss- Vth
        ind_flag= diff_v==0
        
        #temp[ind_flag]=1.0/(tau*(Vth-Vr))
        temp[ind_flag]=1.0/(tau*10)     # nu/(tau*(Vth-Vr))  with nu=1 mv, Vth-Vr=10 mV  Vreset=-60mV
        temp[~ ind_flag]=np.divide(diff_v[~ ind_flag], tau*10*(1.0 - np.exp(-diff_v[~ ind_flag] )    )   )
        return temp    
@njit(cache=True)
def abott_fI_curve3(x, gl=20, tau=0.015):    #   fI curve with a saturating term. Now it saturates at 15 Hz
        Vl=-70
        Vss=Vl+ x/gl    # check the units later
        Vth=-50
        # Vr=-60 nu=1 simplified in the calculation
        temp=0.0*x
        diff_v= Vss- Vth
        ind_flag= diff_v==0
        
        #temp[ind_flag]=1.0/(tau*(Vth-Vr))
        temp[ind_flag]=1.0/(tau*10)     # nu/(tau*(Vth-Vr))  with nu=1 mv, Vth-Vr=10 mV
        temp[~ ind_flag]=np.divide(diff_v[~ ind_flag], tau*10*(1.0 - np.exp(-diff_v[~ ind_flag] )    )   )
        ind_flag=temp>0 
        temp[ind_flag]= 10*(1-np.exp(-temp[ind_flag]/10))
        return temp   
@njit(cache=True)
def nl_NMDA(v):    #   Calculate the nonlinearity part of NMDA current
    c_Mg=1
    return np.divide(1.0, 1.0+ np.exp(-0.062*v)*c_Mg/3.57)

@njit(cache=True)
def current_from_dend(Iinh,Iexc,c1,c2,c3,c4,c5,c6):  # not use it anymroe.
        return c1 * (-0.5 + sigmoid((Iexc - c2 * Iinh + c6 ) / (c3 * Iinh + c4))) + c5

@njit(cache=True)
def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

@njit(cache=True)
def gaussian_connectivity(x,mu, Jmin, Jmax, sigma):
    return Jmin + Jmax*gaussian(x, mu, sigma)

   
@njit(cache=True)    
def deriv_integrator(x,alpha,fI,tau):
    return (-x + alpha*x + fI )/tau   
    
    
@njit(cache=True)
def euler_update(x, f, dt, tau = 1.0):
    return relu(x + dt * f / tau)

@njit(cache=True)
def leaky_integrator(x, f, dt, tau = 1.0, alpha = 1.0):
    return euler_update(x, (-alpha*x + f), dt, tau)



@njit(cache=True)
def soft_bound(x, xmax,beta = 1.0):
    return np.power(np.sign(xmax -x)*np.abs((xmax -x)),beta)


@njit(cache=True)
def threshold_matrix(m,threshold):
    ''' Return the matrix with saturated values above threshold
    '''
    return np.minimum(m,threshold)


# @njit(cache=True)
def compute_list_time_input(dt, ttotal, tinter, tresting, tstim, delay = 0.1):
    # temp = [t for t in np.arange(tresting, ttotal, dt)]
    
    max_iter = int(np.floor((ttotal - tresting)/(tinter+tstim)))
    temp = np.arange(tresting + tinter + delay,ttotal,tinter + tstim)/dt
    temp = temp.astype(int)
    return temp


def generate_ring_stim(center, sigma, N, tau):
    ''' Fucntion that generates an exponential stimulus
    '''
    logger.info('Generating ring stimulus')
    theta_range1 = np.linspace(0.0, 360.0 - 360.0/N, N)
    center = center*360.0/N
    stim = np.zeros(N)
    min_dir = np.abs(theta_range1 - center)
    min_dir[min_dir>180.0] = min_dir[min_dir>180.0] - 360.0
  
    stim = tau*np.exp(-(min_dir)**2/(2*sigma**2))
    return stim 




def create_ring_connectivity(NCells1, NCells2, Jmax, sigma, N_column= 8, Jmin = 0.0, Nref=8):
    ''' Create the ring connectivity from Layer 1 to Layer 2, W_21
    '''
    logger.info('Creating ring connectivity')
    theta_range1 = np.linspace(0.0, 360.0 - 360.0/NCells1, NCells1)
    theta_range2 = np.linspace(0.0, 360.0 - 360.0/NCells2, NCells2)

    the_rings = np.zeros((NCells2, NCells1))
    for count,theta0 in enumerate(theta_range2):
        min_dir = np.abs(theta_range1 - theta0)
        min_dir[min_dir>180.0] = min_dir[min_dir>180.0] - 360.0
        gauss_con = np.exp(-0.5*(min_dir)**2/sigma**2)
        ring  = Jmax * gauss_con 
        #the_rings[count,:] = ring/np.sqrt(2.0*np.pi*sigma**2) + Jmin #NCells1  # Now the integrate over the ring is Jmax+ Jmin*Ncells. Suggest Jmin kept as 0.
        the_rings[count,:] = ring + Jmin
    the_rings=the_rings*Nref/N_column    
    return the_rings
    

def create_global_competition(Ncells, Jtotal):
     
    logger.info('Creating global competiton, default integrater')
    return Jtotal/Ncells

#@njit(cache=True)
def generate_block_stim(side, v_l, v_r, N):
    ''' Function that generates an linear stimulus. This can vary of course. 
    side== 0: 0~N/2 receive input. Designed for input as 0, or 1, but works 0 to 1 
    '''

    stim= np.zeros(N)
    start=int(side*N//2)
    end= int(start+N//2)

    stim[start:end]= np.linspace(v_l,v_r,int(N//2))
    return stim 

def generate_ringblock_stim(center, N_col, N_eachc, v_l, v_r, sigma):
    '''
    Function that generates a 2d where within each column, affinity to specific stimuli varies from v_l to v_r
     different column has different tuning curve
     after generating this 2D array, change that back to 1D array for further use
     center is from 0 to <360 degree
    '''
    stim_2d=np.zeros((N_eachc,N_col))
    #stim_2d=np.zeros((int(round(N_eachc)),int(round(N_col))))
    stim_maxc= np.linspace(v_l,v_r,N_eachc)
    theta_range1 = np.linspace(0.0, 360.0 - 360.0/N_col, N_col)
    min_dir = np.abs(theta_range1 - center)
    min_dir[min_dir>180.0] = min_dir[min_dir>180.0] - 360.0
  
    stim_scale = np.exp(-(min_dir)**2/(2*sigma**2))

    stim_2d= np.outer(stim_maxc,stim_scale)
    stim_1d= stim_2d.T.flatten()
    #index_1d = j * N_eachc + i  for stim_2d[i,j]  so each column is close to each other. 

    #stim_1d= stim_2d.flatten()
    ##index_1d = i * N_col + j   for stim_2d[i,j]

    return stim_1d, stim_2d    # Can include stim_2d for debugging. 



@njit(cache=True)
def create_delta_connectivity(J,N,ref_N=100):   # So it seems every post-neuron received 100 synapses input in the bulk design.
    return np.diag(np.full(N,J))*ref_N

#@njit(cache=True)
def create_bulk_connectivity(v_l,v_r,N, ref_N=100):
    ''' Generate a bulk connectivity such that top A to V1 A with linear, but not distinguish the units in A. 
    Can be used for Bottom to top as well
    ref_N: suggested synapses post neuron received. The weight is notmalized by that.
    '''
    
    vec1=generate_block_stim(0,v_l,v_r,N)
    vec2=generate_block_stim(1,v_l,v_r,N)
    vec1_cols=vec1[:,np.newaxis]
    vec2_cols=vec2[:,np.newaxis]
    W= np.zeros((N,N))
    W[:,:N//2]=vec1_cols
    W[:,N//2:]=vec2_cols
    W=W*ref_N*(2/N)       # So it seems every post-neuron received 100 synapses input in the bulk design.       
    return W


def create_ringbulk_connectivity(v_l, v_r, N_col=90, N_eachc=40, ref_N=100,type='BulktoBulk',jitter=None, shuffle_inds=None):
    ''' Generate a bulk connectivity such that top A to V1 A with linear, but not distinguish the units in A. 
    Can be used for Bottom to top as well
    Actually need 3 differnet type. bulktobulk, bulktoInt,InttoBulk
    ref_N: suggested synapses post neuron received. The weight is notmalized by that.
    '''
    Conn_eachC= np.linspace(v_l,v_r,N_eachc)
    if shuffle_inds is not None:
        Conn_eachC=Conn_eachC[shuffle_inds]
    
    Conn_eachC_cols=Conn_eachC[:,np.newaxis]
    if type=='BulktoBulk':
        W_bulk=np.zeros((N_eachc,N_eachc))
        W_bulk[:,:]=Conn_eachC_cols
        bulk_list= [W_bulk]* N_col
        W_full=block_diag(*bulk_list)* ref_N/N_eachc

    elif type=='BulktoInt': # In this scenario, v_l and v_r should be the same.
        W_bulk=np.ones((1,N_eachc))*(v_l+v_r)/2.0
        bulk_list= [W_bulk]* N_col
        W_full=block_diag(*bulk_list)* ref_N/N_eachc
    elif type=='InttoBulk':
        if jitter is not None:
            pert_vec = np.random.uniform(1 - jitter, 1 + jitter, size=N_eachc)
            Conn_eachC_cols=Conn_eachC_cols*pert_vec[:,np.newaxis]
        bulk_list= [Conn_eachC_cols]* N_col
#        W_full=block_diag(*bulk_list)* ref_N/N_eachc
        W_full=block_diag(*bulk_list)* ref_N/40    # Shouldn't vary here. Continue tomorrow
      
    return W_full


def create_ringCompte_stimuli(center,N_col,stim,kappa):
    theta_vec_orig= np.linspace(0, 2*np.pi, N_col, endpoint=False)
    theta_vec= theta_vec_orig-center
    v = np.exp(kappa * np.cos(theta_vec))
    v = v / np.sum(v)
    stimulus = stim * v*(N_col/256)        # Make it consistent with the Compte orig model. Make it independent of N_col, too.   
    return stimulus


def create_random_delta_connectivity(mu, n, sigma_r=0.5,rng=0,ref_N=100):
    # Generate a random matrix for the initial connectivity. 
    # Define the lower and upper bounds for truncation.
     
    lower_bound = 0
    upper_bound = 2 * mu
    sigma=sigma_r*mu
    # Calculate the bounds in terms of standard deviation
    lower_bound_std = (lower_bound - mu) / sigma
    upper_bound_std = (upper_bound - mu) / sigma

    # Create the truncated normal distribution
    distribution = truncnorm(lower_bound_std, upper_bound_std, loc=mu, scale=sigma)
    if rng==0:
        diagonal_values= distribution.rvs(n)
    else:
        diagonal_values= distribution.rvs(size=n,random_state=rng)
    # Generate the matrix
    matrix =  np.diag(diagonal_values*ref_N)
    return matrix


def load_ringbulk_connectivity(v_vec,N_col,type='BulktoBulk_diag'):
    # Generate a connection matrix from the trained network. 
    Conn_eachC=v_vec
    if type=='InttoBulk':
        Conn_eachC_cols=Conn_eachC[:,np.newaxis]
        bulk_list= [Conn_eachC_cols]* N_col
        W_full=block_diag(*bulk_list)
    elif type=='BulktoBulk_diag':
        Conn_eachC_cols=np.tile(Conn_eachC,N_col)
        W_full=np.diag(Conn_eachC_cols)
    return W_full


def load_ringblock_stim(center, N_col, N_eachc, v_vec, sigma):
    '''
    Function that generates a 2d where within each column, affinity to specific stimuli varies from v_l to v_r
    Utilziing the trained v_vec to do so. 
      different column has different tuning curve
     after generating this 2D array, change that back to 1D array for further use
     center is from 0 to <360 degree
    '''
    stim_2d=np.zeros((N_eachc,N_col))
    #stim_2d=np.zeros((int(round(N_eachc)),int(round(N_col))))
    stim_maxc= v_vec/np.mean(v_vec)
    theta_range1 = np.linspace(0.0, 360.0 - 360.0/N_col, N_col)
    min_dir = np.abs(theta_range1 - center)
    min_dir[min_dir>180.0] = min_dir[min_dir>180.0] - 360.0
  
    stim_scale = np.exp(-(min_dir)**2/(2*sigma**2))

    stim_2d= np.outer(stim_maxc,stim_scale)
    stim_1d= stim_2d.T.flatten()
    #index_1d = j * N_eachc + i  for stim_2d[i,j]  so each column is close to each other. 

    #stim_1d= stim_2d.flatten()
    ##index_1d = i * N_col + j   for stim_2d[i,j]

    return stim_1d, stim_2d    # Can include stim_2d for debugging. 


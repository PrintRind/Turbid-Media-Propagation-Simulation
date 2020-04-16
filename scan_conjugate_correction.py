
# coding: utf-8

# In[1]:


def scan_conjugate_correction(u, sample, conj_mask, z_conj, grating_spacing_in, stepsize_in, FOV_in):
    '''to apply the correction, we let the uncorrected beam propagate to the conjugate plane,
    apply the correction phase mask, and then to the sample'''
    
    from grating1D import grating1D
    from fun_propagate import fun_propagate
    from input_wavefront import input_wavefront
    import numpy as np
    NA = 0.5
    N = sample.shape
    #make spherical input wavefront
    E0, phase, amp = input_wavefront(NA, N, u)
    #define x axis
    x=np.arange(-N[1]/2,N[1]/2)*u[1]
    #define grating
    grating = np.fft.fftshift(grating1D(grating_spacing_in, x, u)) 
    #size of steps during beam scan
    m = int(stepsize_in)  
    #number of steps in image
    num_steps = int(FOV_in/m) 
    offset=int(-FOV_in/2)
    #initialize array to hold 2 photon intensities for each line of the scan 
    lines = np.zeros((num_steps, N[1])) 
    #loop to step beam across sample
    for q in range(num_steps):  
        #calculate electric field at conjugate plane
        E = fun_propagate(u,E0,np.roll(sample[0:z_conj,:],offset+q*m,axis=1))  
        #apply AO mask at conjugate plane
        E1 = abs(E[-1,:])*np.roll(conj_mask,offset+q*m)/abs(np.roll(conj_mask,offset+q*m))  
        #finish propagating to sample
        E2 = fun_propagate(u,E1,np.roll(sample[z_conj:,:],offset+q*m,axis=1))  
        #calculate 2 photon intensity on grating
        lines[q,:] = abs(E2[-1,:])**4*np.roll(grating,offset+q*m)  
   
    image=np.sum(lines,1)
    return image


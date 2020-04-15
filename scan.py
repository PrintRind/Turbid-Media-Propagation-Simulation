
# coding: utf-8

# In[1]:


def scan(u,E_0, sample, grating_spacing_in, stepsize_in, FOV_in):
    '''Steps beam propagated through 'sample' across grating with spacing 'grating_spacing' with a step size 'stepsize' and field of view FOV in units of pixels.''' 
    
    from grating1D import grating1D
    from fun_propagate import fun_propagate
    import numpy as np
    N = sample.shape
    x=np.arange(-N[1]/2,N[1]/2)*u[1]
    grating = np.fft.fftshift(grating1D(grating_spacing_in, x, u)) #create sample grating with spacing grating_spacing in units of pixels
    m = int(stepsize_in)  #size of steps during beam scan
    num_steps = int(FOV_in/m) #number of steps in image
    offset=int(-FOV_in/2)
    lines = np.zeros((num_steps, N[1])) #make array to hold each line of the scan 
    for q in range(num_steps):  #loop to step beam and add up intensities
        E = fun_propagate(u,E_0,np.roll(sample,offset+q*m,axis=1))  #calculate electric field at focus modulated by sample
        lines[q,:] = abs(E[-1,:])**4*np.roll(grating,offset+q*m)
   
    image=np.sum(lines,1)
    return image


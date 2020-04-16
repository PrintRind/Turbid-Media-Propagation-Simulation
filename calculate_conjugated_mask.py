
# coding: utf-8

# In[ ]:


def calculate_conjugated_mask(u, z_conj, sample, FOV, stepsize_in):
    '''calculate the correction field for the many image points 
    and average them to create a conjugated correction mask'''
    
    import numpy as np
    from fun_propagate import fun_propagate
    from input_wavefront import input_wavefront
    
    NA = 0.5
    N = sample.shape
    num_steps = int(FOV/stepsize_in)
    E_conj_corr = np.zeros([num_steps,N[1]], dtype='complex')
    #shift center of sample
    offset=int(-FOV/2)
    #create clear sample
    sample_clear=np.ones(sample.shape)*1.33
    #create spherical input wavefront
    E_0, phase, amp = input_wavefront(NA, N, u)
    #create guide star field
    E_gs=fun_propagate(u,E_0,sample_clear) 
    
    for i in range(num_steps):
        #first propagate guidestar at this sample position
        E1=fun_propagate(u,np.conj(E_gs[-1,:]),np.roll(np.flip(sample,0),offset+i*int(stepsize_in),axis=1))
        #now propagate back through a clear sample to get the correction field for a single imaging point
        E2 = fun_propagate(u, np.conj(E1[-1,:]), sample_clear[0:z_conj,:])
        E_conj_corr[i,:] = E2[-1,:]
    
    E_corr = np.average(E_conj_corr, 0)
    return E_corr


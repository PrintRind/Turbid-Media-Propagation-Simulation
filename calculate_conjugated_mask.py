
# coding: utf-8

# In[ ]:

def calculate_conjugated_mask(u, z_conj, sample, FOV, num_steps,NA):
    '''calculate the correction field for the many image points 
    and average them to create a conjugated correction mask'''
    
    import numpy as np
    from fun_propagate import fun_propagate
    from input_wavefront import input_wavefront
    
    #NA = 0.5
    N = sample.shape
    stepsize=np.floor(FOV/num_steps)
    E_conj_corr = np.zeros([num_steps,N[1]], dtype='complex')
    #shift center of sample
    offset=int(-FOV/2)
    #create clear sample
    sample_clear=np.ones(sample.shape)*1.33
    #create spherical input wavefront
    E_0, phase, amp = input_wavefront(NA, N, u)
    #create guide star field
    E_gs=fun_propagate(u,E_0,sample_clear) 
    
    #calculate curved reference phase; this phase must be subtracted from the corr-patterns in the conjugate plane
    E_ref=E_gs[z_conj,:]/abs(E_gs[z_conj,:])
    
    for i in range(num_steps):
        #first propagate guidestar at this sample position
        rollval=offset+int(stepsize/2)+i*int(stepsize) #sample shift
        E1=fun_propagate(u,np.conj(E_gs[-1,:]),np.roll(np.flip(sample,0),rollval,axis=1))
        #now propagate back through a clear sample to get the correction field for a single imaging point
        E2 = fun_propagate(u, np.conj(E1[-1,:]), sample_clear[0:z_conj,:])
        #plt.figure(1)
        #plt.plot(np.angle(E2))
        #time.sleep(3)
        E_conj_corr[i,:] = np.roll(E2[-1,:]/E_ref,-rollval) #the correction phase pattern is "fixed" with the sample, therefore we must also consider the sample-shift here
    
    E_corr = np.average(E_conj_corr, 0)
    return E_corr


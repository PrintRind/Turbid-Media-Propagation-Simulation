
# coding: utf-8

# In[ ]:


def fun_propagate(u,E_in,sample):
    
    import numpy as np
    RI=np.unique(sample) 
    N=sample.shape       #size of sample (and field)
    E=np.zeros(N,dtype='complex') #initializing field 
    Ek=np.zeros((len(RI),N[1]),dtype='complex') #init.
    mask=np.zeros((len(RI),N[0],N[1])) #initializing mask
    E[0,:]=E_in 
    
    uk=2*np.pi/u/N   #increment in frequency-space is 1/(size of grid in x-space); increment in k-space is 2*pi/(size of grid in x-space)
    kx=np.arange(-N[1]/2,N[1]/2)*uk[1] 
    k = 2 * np.pi * RI/1.33 # k-vectors of light that traverses the different refractive indices, not accounting for objective NA here
    
    #defining propagator for each refractive index 
    prop = np.zeros((len(RI), N[1]), dtype='complex')  # initializing propagators
    for q in range(len(RI)):
        prop[q, :] = np.fft.ifftshift(np.exp(1j * u[0] * np.sqrt(0j+k[q] ** 2 - kx ** 2)))
    
    #constructing binary masks for every RI
    for q in range(len(RI)):
        mask[q,:,:]=sample==RI[q] 
    
    #propagation loop
    for m in range(N[0]-1):
        
        for q in range(len(RI)):
            Ek[q,:]=np.fft.fft(E[m,:]*mask[q,m,:])*prop[q,:]  #for each RI: transform field to k-space and multiply with correct propagator
    
        E[m+1,:]=np.fft.ifft(sum(Ek,0)) #calculate next depth-slice of field
    
    return E


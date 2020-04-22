
# coding: utf-8

# In[4]:


def input_wavefront(NA, N, u, RI):
    '''defining input wavefront: a spherical, converging wave'''
    
    import numpy as np
    x=np.arange(-N[1]/2,N[1]/2)*u[1]
    NA_geo=NA/RI #geometric NA of our objective lens
    # phase
    R=N[0]*u[0] #radius of curvature; define it such that focus lies at last layer
    #phase = 2*np.pi*(np.sqrt(0j+R**2-x**2)-R)  # spherical phase
    phase = -2*np.pi*np.sqrt((N[0]*u[0])**2+x**2) #phase according to pythargoras
    
    # amplitude
    amp0=np.abs(x)<R*np.tan(np.arcsin(NA_geo)) #wave amplitude is 1 for x less than R*NA_geo, didn't know this function of the < operator
    sigma_amp=10 #we filter the amplitude a bit to make the edges less sharp
    amp_filter=np.exp(-x**2/sigma_amp**2)
    F_amp0=np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(amp0))))
    amp=np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(F_amp0*amp_filter))))
    #full wavefront
    E=np.zeros(N,dtype='complex') #initializing field 
    E_in=np.fft.ifftshift(amp*np.exp(1j*phase))  #already apply ifftshift (loop later is faster)
    
    return E_in, phase, amp


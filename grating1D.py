
# coding: utf-8

# In[2]:


def grating1D(spacing, x, u):
    "Creates a grating structure with frequency 1/spacing where 'spacing' is in units of pixels"
    grating=((x/u[1]+1/2) % spacing)>=spacing/2
    return grating


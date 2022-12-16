import numpy as np
from pyfftw.interfaces.numpy_fft import rfft, irfft, irfft2, rfft2, ifft, fft, rfftn, irfftn, fft2, ifft2
import time

#grid size
Nx = 128
Ny = 128
Nz = 128

######################################
def Get_kx(lx):
	return lx
	
def Get_ky(ly):
	return ly

def Get_kz(lz):
	return lz

##################################### 


## ref: http://www-personal.umich.edu/~mejn/computational-physics/dcst.py
######################################################################
# 1D DST Type-I

def dst(y):
    N = len(y)
    y2 = np.zeros(2*N,dtype=float)
    y2[0] = y2[N] = 0.0
    y2[1:N] = y[1:]
    y2[:N:-1] = -y[1:]
    a = -np.imag(rfft(y2))[:N]
    a[0] = 0.0

    return a


######################################################################
# 1D inverse DST Type-I

def idst(a):
    N = len(a)
    c = np.zeros(N+1,dtype=complex)
    c[0] = c[N] = 0.0
    c[1:N] = -1j*a[1:]
    y = irfft(c)[:N]
    y[0] = 0.0

    return y


######################################################################

######################################################################
# 2D DST

def dst2(y):
    M = y.shape[0]
    N = y.shape[1]
    P = y.shape[2]

    a = np.zeros([M,N,P],dtype=float)
    b = np.zeros([M,N,P],dtype=float)
   
    
    for j in range(N):
    	for k in range(P):
        	a[:,j,k] = dst(y[:,j,k])

    for i in range(M):
        for k in range(P):
            b[i,:,k] = dst(a[i,:,k])
    
    return b


######################################################################
# 2D inverse DST

def idst2(b):
    M = b.shape[0]
    N = b.shape[1]
    P = b.shape[2]

    y = np.zeros([M,N,P],dtype=float)
    c = np.zeros([M,N,P],dtype=float)
    
    for j in range(N):
    	for k in range(P):
        	c[:,j,k] = idst(b[:,j,k])

    for i in range(M):
        for k in range(P):
            y[i,:,k] = idst(c[i,:,k])
    
    return y

#main part

#initial condition
Lx = np.pi
Ly = np.pi
Lz = 2*np.pi

dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

f = np.zeros((Nx, Ny, Nz))

# example function
def init_cond(f):
   
   x = np.linspace(0,Nx-1, Nx)
   y = np.linspace(0,Ny-1, Ny)
   z = np.linspace(0, Nz-1, Nz)

   x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z,indexing = 'ij')
   
   k0=1
   
   f = 8*np.sin(k0*x_mesh*dx)*np.sin(k0*y_mesh*dy)*np.cos(k0*z_mesh*dz)
   
   return f

f = init_cond(f)


f_copy = np.zeros_like(f)

f_copy[:,:,:] = f[:,:,:]


## forward transform-- 
temp =  dst2(f)
fk = rfft(temp, axis=2)/(Nx*Ny*Nz)

## inverse transform --
temp =  irfft(fk, axis=2)
f = idst2(temp)*(Nx*Ny*Nz)


print (np.max(np.abs(f-f_copy)))


def max_element_of_array(fk):
	for i in range (0, np.shape(fk)[0]):
		for j in range (0, np.shape(fk)[1]):
			for k in range (0, np.shape(fk)[2]):
				if (abs(fk[i, j, k])> 1e-6):
					print ("vect(k) = (", Get_kx(i), ",",  Get_ky(j),  ",", Get_kz(k) , "); f(k) = " , fk[i, j, k])

	return 


## print spectral modes with nonzero amplitudes
max_element_of_array(fk)


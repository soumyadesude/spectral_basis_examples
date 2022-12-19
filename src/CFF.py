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
	if ly <= Ny //2:
		return ly
	else:
		return ly-Ny

def Get_kz(lz):
	return lz

##################################### 


## ref: http://www-personal.umich.edu/~mejn/computational-physics/dcst.py
######################################################################
# 1D DCT Type-I

def dct(y):
    N = len(y)
    y2 = np.zeros(2*N, dtype=float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = rfft(y2)
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))
    return np.real(phi*c[:N])

######################################################################
# 1D inverse DCT Type-I

def idct(a):
    N = len(a)
    c = np.zeros(N+1,dtype=complex)

    phi = np.exp(1j*np.pi*np.arange(N)/(2*N))
    c[:N] = phi*a
    c[N] = 0.0
    return irfft(c)[:N]


######################################################################

######################################################################
# 2D DST

def dct2(y):
    M = y.shape[0]
    N = y.shape[1]
    P = y.shape[2]

    a = np.zeros([M,N,P],dtype=float)
    
    for j in range(N):
    	for k in range(P):
        	a[:,j,k] = dct(y[:,j,k])
    
    return a


######################################################################
# 2D inverse DST

def idct2(a):
    M = a.shape[0]
    N = a.shape[1]
    P = a.shape[2]

    y = np.zeros([M,N,P],dtype=float)
    
    for j in range(N):
    	for k in range(P):
        	y[:,j,k] = idct(a[:,j,k])
    
    return y

#main part

#initial condition
Lx = np.pi
Ly = 2*np.pi
Lz = 2*np.pi

dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

f = np.zeros((Nx, Ny, Nz))

# example function
def init_cond(f):
   
   x = np.linspace(0,Nx-1, Nx)+0.5
   y = np.linspace(0,Ny-1, Ny)
   z = np.linspace(0, Nz-1, Nz)

   x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z,indexing = 'ij')
   
   k0 = 1
   k1 = 2
   k2 = 3
   k3 = 4
   k4 = 5
   k5 = 6

   
   f = 8*np.cos(k0*x_mesh*dx)*np.cos(k1*y_mesh*dy)*np.cos(k2*z_mesh*dz) + 8*np.cos(k3*x_mesh*dx)*np.cos(k4*y_mesh*dy)*np.cos(k5*z_mesh*dz)
   
   return f

f = init_cond(f)


f_copy = np.zeros_like(f)

f_copy[:,:,:] = f[:,:,:]


## forward transform-- 
temp =  dct2(f)
fk = rfft2(temp, axes=(1, 2))/(Nx*Ny*Nz)

## inverse transform --
temp =  irfft2(fk, axes=(1, 2))
f = idct2(temp)*(Nx*Ny*Nz)


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


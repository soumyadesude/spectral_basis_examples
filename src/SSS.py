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
# 3D DST

def dst3(y):
    M = y.shape[0]
    N = y.shape[1]
    P = y.shape[2]

    a = np.zeros([M,N,P],dtype=float)
    
    
    for j in range(N):
    	for k in range(P):
        	a[:,j,k] = dst(y[:,j,k])

    for i in range(M):
        for k in range(P):
            a[i,:,k] = dst(a[i,:,k])

    for i in range(M):
        for j in range(N):
            a[i,j,:] = dst(a[i,j,:])
    
    return a


######################################################################
# 3D inverse DST

def idst3(a):
    M = a.shape[0]
    N = a.shape[1]
    P = a.shape[2]

    y = np.zeros([M,N,P],dtype=float)
    
    for j in range(N):
    	for k in range(P):
        	y[:,j,k] = idst(a[:,j,k])

    for i in range(M):
        for k in range(P):
            y[i,:,k] = idst(y[i,:,k])

    for i in range(M):
        for j in range(N):
            y[i,j,:] = idst(y[i,j,:])
    
    return y

#main part

#initial condition
Lx = np.pi
Ly = np.pi
Lz = np.pi

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
   
   k0 = 1
   k1 = 2
   k2 = 3
   k3 = 4
   k4 = 5
   k5 = 6

   
   f = 8*np.sin(k0*x_mesh*dx)*np.sin(k1*y_mesh*dy)*np.sin(k2*z_mesh*dz) + 8*np.sin(k3*x_mesh*dx)*np.sin(k4*y_mesh*dy)*np.sin(k5*z_mesh*dz)
   
   return f

f = init_cond(f)


f_copy = np.zeros_like(f)

f_copy[:,:,:] = f[:,:,:]


## forward transform-- 
fk =  dst3(f)/(Nx*Ny*Nz)

## inverse transform --
f = idst3(fk)*(Nx*Ny*Nz)


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


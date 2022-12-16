import numpy as np
from pyfftw.interfaces.numpy_fft import rfft, irfft, irfft2, rfft2, ifft, fft, rfftn, irfftn, fft2, ifft2
import time

#grid size
Nx = 128
Ny = 128
Nz = 128

######################################
def Get_kx(lx):
	if lx <= Nx // 2:
		return lx
	else:
		return lx-Nx

def Get_ky(ly):
	if ly <= Ny //2:
		return ly
	else:
		return ly-Ny

def Get_kz(lz):
	return lz

##################################### 

#main part

#initial condition
Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2*np.pi

dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

k0 = 1
k1 = 2
k2 = 3
k3 = 4
k4 = 5
k5 = 6

f = np.zeros((Nx, Ny, Nz))

# example function
def init_cond(f):
   
   x = np.linspace(0,Nx-1, Nx)
   y = np.linspace(0,Ny-1, Ny)
   z = np.linspace(0, Nz-1, Nz)

   x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z,indexing = 'ij')
   f = 8*np.sin(k0*x_mesh*dx)*np.sin(k1*y_mesh*dy)*np.sin(k2*z_mesh*dz) + 8*np.sin(k3*x_mesh*dx)*np.sin(k4*y_mesh*dy)*np.sin(k5*z_mesh*dz)
   
   return f

f = init_cond(f)


f_copy = np.zeros_like(f)

f_copy[:,:,:] = f[:,:,:]

'''
##### type - I ### 
## forward transform-- 
fk =  rfftn(f)/(Nx*Ny*Nz)


## inverse transform --
f =  irfftn(fk)*(Nx*Ny*Nz)

#####
'''

##### type - II ### 
## forward transform-- 
temp =  rfft(f, axis=2)
fk = fft2(temp, axes=(0, 1))/(Nx*Ny*Nz)

## inverse transform --
temp =  ifft2(fk, axes=(0, 1))
f = irfft(temp, axis=2)*(Nx*Ny*Nz)

#####



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


import sys
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt

# read meta data header
fread = open(sys.argv[1],'rb')			# open first file of data set


nChanSize = 51200 
nPol = int(sys.argv[2])
sigCom = np.zeros(int(nChanSize),dtype=np.complex)
array = np.zeros((128,nChanSize),dtype=np.complex)
R = np.zeros((128,128,nChanSize),dtype=np.complex)
	
if nPol < 0 and nPol > 1:
	print('Polarization number is between 0 and 1')
	exit()

nblock = 0
for nAnt in range(128):
	fread.seek(4096 + 26214400 + 26214400*nblock + 102400*nAnt*2 + 102400*nPol)
	sig = np.fromfile(fread, dtype = np.int8, count = nChanSize*2)
	sig = np.reshape(sig,(2,int(nChanSize)), order='F')
	sigCom.real = sig[0,:]
	sigCom.imag = sig[1,:]
	array[nAnt,:] = sigCom

for k in range(nChanSize):
	#R[:,:,k] = np.outer(array[:,k],np.conj(array[:,k]))
	R[:,:,k] = np.outer(array[:,k],(array[:,k]))

Rcyc = np.fft.fft(R,axis=2)

spec = np.zeros((nChanSize))
for k in range(nChanSize):
	spec[k] = LA.norm(Rcyc[:,:,k], 'fro')
	
#plt.figure()
#plt.plot(np.linspace(0.,1.,nChanSize),spec)
#plt.xlabel('cyclic frequency [normalized]')
#plt.ylabel('cyclic energy [arbitrary]')
#plt.grid()
#plt.show()

plt.figure()
plt.plot(np.linspace(0.,1.,nChanSize-1),10.*np.log10(spec[1:]))
plt.xlabel('cyclic frequency [normalized]')
plt.ylabel('cyclic energy [dB]')
plt.grid()
plt.show()


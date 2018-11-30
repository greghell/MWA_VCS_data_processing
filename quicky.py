# quicky.py filename resolution nBlocks
# gives a quick snapshot of the content of a sub file

import sys
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt

fname = sys.argv[1]
nResol = int(sys.argv[2])


nTotBlocks = int(sys.argv[3])
nTotAnts = 128
nTotPols = 2
nChanSize = 51200

if nResol > nChanSize:
	print('Resolution must be < ' + str(nChanSize))
	exit()

sigCom = np.zeros(int(nChanSize),dtype=np.complex)
specgram = np.zeros((nResol,nTotAnts,nTotPols))	# nResol, 128 ants, 2 pols, 200 blocks
AntsSigs = np.zeros((nTotAnts,int(nChanSize)),dtype=np.complex)
Rs = np.zeros((nTotAnts,nTotAnts,nTotPols),dtype=np.complex)

fread = open(fname,'rb')
print('opening ' + fname + ' and processing...')
for nblock in range(nTotBlocks):
	print('block #' + str(nblock+1) + ' / ' + str(nTotBlocks))
	for nPol in range(nTotPols):
		for nAnt in range(nTotAnts):
			fread.seek(4096 + 26214400 + 26214400*nblock + (nChanSize*2)*nAnt*nTotPols + (nChanSize*2)*nPol)
			sig = np.fromfile(fread, dtype = np.int8, count = 2*nChanSize)
			sig = np.reshape(sig,(2,int(nChanSize)), order='F')
			sigCom = np.zeros(int(nChanSize),dtype=np.complex)
			sigCom.real = sig[0,:]
			sigCom.imag = sig[1,:]
			AntsSigs[nAnt,:] = sigCom

			spec = np.reshape(sigCom[0:int(math.floor(int(nChanSize)/nResol)*nResol)],(nResol,int(math.floor(int(nChanSize)/nResol))), order='F')
			spec = np.abs(np.fft.fft(spec,axis=0))**2

			specgram[:,nAnt,nPol] = specgram[:,nAnt,nPol] + (1/nTotBlocks) * np.mean(spec,axis=1)
		Rs[:,:,nPol] = Rs[:,:,nPol] + (1/nTotBlocks) * np.inner(AntsSigs,np.conjugate(AntsSigs))


plt.figure()

minvalSpec = np.min(10.*np.log10(specgram))
maxvalSpec = np.max(10.*np.log10(specgram))

plt.subplot(221)
plt.imshow(10.*np.log10(specgram[:,:,0]),aspect='auto',vmin=minvalSpec, vmax=maxvalSpec)
plt.xlabel('antenna #')
plt.ylabel('frequency channel')
plt.title('polarization 0 [dB]')
plt.colorbar()

plt.subplot(222)
plt.imshow(10.*np.log10(specgram[:,:,1]),aspect='auto',vmin=minvalSpec, vmax=maxvalSpec)
plt.xlabel('antenna #')
plt.ylabel('frequency channel')
plt.title('polarization 1 [dB]')
plt.colorbar()

a0,b = LA.eigh(Rs[:,:,0])
a1,b = LA.eigh(Rs[:,:,1])
minvalEV = np.min([10.*np.log10(a0),10.*np.log10(a1)])
maxvalEV = np.max([10.*np.log10(a0),10.*np.log10(a1)])

plt.subplot(223)
plt.plot(10.*np.log10(a0))
plt.ylim([minvalEV,maxvalEV])
plt.grid()
plt.xlabel('Eigenvalue number')
plt.ylabel('Eigen value [dB]')
plt.title('polarization 0')

plt.subplot(224)
plt.plot(10.*np.log10(a1))
plt.ylim([minvalEV,maxvalEV])
plt.grid()
plt.xlabel('Eigenvalue number')
plt.ylabel('Eigen value [dB]')
plt.title('polarization 1')

plt.show()

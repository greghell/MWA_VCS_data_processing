import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plots spectra and spectrograms of both polarization for one antenna for one .sub file')
parser.add_argument('filename', help='indicate file name')
parser.add_argument("nResol", type=int, help="spectral resolution (< 51200)")
parser.add_argument("nAnt", type=int, help="antenna number (0-127)")
parser.add_argument("-nBlocks", type=int, help="number of data blocks to process (< 200)", default = 200)
args = parser.parse_args()

fread = open(args.filename,'rb')			# open first file of data set

nChanSize = 51200
nBloks = int(args.nBlocks)
nPols = 2
nResol = int(args.nResol)
nAnt = int(args.nAnt)
specgram = np.zeros((nResol,nBloks,nPols))
TotnSam = 0

if nAnt < 0 or nAnt > 127:
	print('Antenna number is between 0 and 127')
	exit()
	
if nBloks > 200 or nBloks < 1:
	print('Block number is between 1 and 200')
	exit()

for k in range(nBloks):
	for nPol in range(2):
		fread.seek(4096 + 26214400 + 26214400*k + (2*nChanSize)*nAnt*2 + (2*nChanSize)*nPol)
		sig = np.fromfile(fread, dtype = np.int8, count = 2*nChanSize)
		sig = np.reshape(sig,(2,int(nChanSize)), order='F')
		sigCom = np.zeros(int(nChanSize),dtype=np.complex)
		sigCom.real = sig[0,:]
		sigCom.imag = sig[1,:]

		spec = np.reshape(sigCom[0:int(math.floor(len(sigCom)/nResol)*nResol)],(nResol,int(math.floor(len(sigCom)/nResol))), order='F')
		spec = np.fft.fftshift(np.abs(np.fft.fft(spec,axis=0))**2,axes=0)

		specgram[:,k,nPol] = specgram[:,k,nPol] + (1./nBloks)*np.mean(spec,axis=1)
	
	TotnSam = TotnSam + len(sigCom)
	
	print('block #' + str(k+1))

plt.figure()
plt.subplot(131)
spec0 = specgram[:,:,0]
spec1 = specgram[:,:,1]
plt.plot(10.*np.log10(np.mean(spec0,axis=1)),'k',label='Pol0')
plt.plot(10.*np.log10(np.mean(spec1,axis=1)),'b',label='Pol1')
plt.legend()
plt.xlabel('frequency [MHZ] - place holder')
plt.ylabel('Power [dB]')
plt.title(sys.argv[1]+' - ' + str(TotnSam) + ' samples - Antenna #' + str(nAnt))
plt.grid()

valminSpec = np.min(10.*np.log10(specgram))
valmaxSpec = np.max(10.*np.log10(specgram))

plt.subplot(132)
plt.imshow(10.*np.log10(specgram[:,:,0]),aspect='auto',vmin=valminSpec, vmax=valmaxSpec)
plt.colorbar()
plt.xlabel('time')
plt.ylabel('frequency [MHZ] - place holder')
plt.title('Polarization 0 [dB]')

plt.subplot(133)
plt.imshow(10.*np.log10(specgram[:,:,1]),aspect='auto',vmin=valminSpec, vmax=valmaxSpec)
plt.colorbar()
plt.xlabel('time')
plt.ylabel('frequency [MHZ] - place holder')
plt.title('Polarization 1 [dB]')
plt.show()

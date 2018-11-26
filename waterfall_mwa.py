import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# read meta data header
fread = open(sys.argv[1],'rb')			# open first file of data set


nChanSize = 102400
nResol = int(sys.argv[2])
nAnt = int(sys.argv[3])
nPol = int(sys.argv[4])
#specavg = np.zeros(nResol)
specgram = np.zeros((nResol,200))
TotnSam = 0

if nAnt < 0 and nAnt > 127:
	print('Antenna number is between 0 and 127')
	exit()
	
if nPol < 0 and nPol > 1:
	print('Polarization number is between 0 and 1')
	exit()

for k in range(200):
	fread.seek(4096 + 26214400 + 26214400*k + 102400*nAnt*2 + 102400*nPol)
	sig = np.fromfile(fread, dtype = np.int8, count = nChanSize)
	sig = np.reshape(sig,(2,int(nChanSize/2)), order='F')
	sigCom = np.zeros(int(nChanSize/2),dtype=np.complex)
	sigCom.real = sig[0,:]
	sigCom.imag = sig[1,:]

	spec = np.reshape(sigCom[0:int(math.floor(len(sigCom)/nResol)*nResol)],(nResol,int(math.floor(len(sigCom)/nResol))), order='F')
	spec = np.power(abs(np.fft.fft(spec,axis=0)),2)

#	specavg = specavg + (1./200.)*np.mean(spec,axis=1)
	specgram[:,k] = np.mean(spec,axis=1)
	TotnSam = TotnSam + len(sigCom)
	print('block #' + str(k+1))

plt.figure()
plt.subplot(121)
plt.plot(10*np.log10(np.fft.fftshift(np.mean(specgram,axis=1))),'k')
plt.xlabel('frequency [MHZ] - place holder')
plt.ylabel('Power [dB]')
plt.title(sys.argv[1]+' - ' + str(TotnSam) + ' samples - Antenna #' + str(nAnt) + ' - Polarization #' + str(nPol))
plt.grid()
plt.subplot(122)
plt.imshow(10*np.log10(np.fft.fftshift(specgram,axes=(0,))),aspect='auto')
plt.xlabel('time')
plt.ylabel('frequency [MHZ] - place holder')
plt.show()

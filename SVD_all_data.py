import sys
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import glob

all_files = glob.glob("*.sub")
nTotFiles = len(all_files)

nChanSize = 51200 
sigCom = np.zeros(int(nChanSize),dtype=np.complex)
array = np.zeros((128,nChanSize),dtype=np.complex)
vals = np.zeros((128,200))

idx = 0
for currfile in all_files:
	fread = open(currfile,'rb')
	print('processing ' + currfile)
	idx = idx + 1
	for nPol in range(2):
		for nblock in range(200):
			for nAnt in range(128):
				fread.seek(4096 + 26214400 + 26214400*nblock + 102400*nAnt*2 + 102400*nPol)
				sig = np.fromfile(fread, dtype = np.int8, count = nChanSize*2)
				sig = np.reshape(sig,(2,int(nChanSize)), order='F')
				sigCom.real = sig[0,:]
				sigCom.imag = sig[1,:]
				array[nAnt,:] = sigCom
			R = np.inner(array,np.conjugate(array))
			u, s, vh = np.linalg.svd(R)
			vals[:,nblock] = s
			print("file #"+str(idx)+"/"+str(nTotFiles)+" - block #" + str(nblock)+ " processed")

		plt.figure()
		plt.imshow(np.log10(vals),aspect = 'auto')
		plt.savefig(currfile+"_pol"+str(nPol)+".png")
		plt.close()

import sys
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import argparse
import astropy.io.fits
from astropy.io import fits

parser = argparse.ArgumentParser(description='Produces dirty image and eigen value analysis')
parser.add_argument('filename', help='indicate file name')
parser.add_argument('freq', help='central frequency in MHz')
parser.add_argument("-nResol", type=int, help="Number of pixels", default = 200)
parser.add_argument("-nBlocks", type=int, help="number of data blocks to process (< 200)", default = 200)
parser.add_argument("-xmin", type=float, help="x min in skymap", default = -1.)
parser.add_argument("-xmax", type=float, help="x max in skymap", default = 1.)
parser.add_argument("-ymin", type=float, help="y min in skymap", default = -1.)
parser.add_argument("-ymax", type=float, help="y max in skymap", default = 1.)
parser.add_argument("-show", help="show plot before saving it",action="store_true")
args = parser.parse_args()

fname = args.filename

nTotBlocks = int(args.nBlocks)
nTotAnts = 128
nTotPols = 2
nChanSize = 51200

sigCom = np.zeros(int(nChanSize),dtype=np.complex)
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
		Rs[:,:,nPol] = Rs[:,:,nPol] + (1/nTotBlocks) * np.inner(AntsSigs,np.conjugate(AntsSigs))

xmin = float(args.xmin); xmax = float(args.xmax)
ymin = float(args.ymin); ymax = float(args.ymax)

hdulist = fits.open('1223270792_metafits.fits')
hdu = hdulist[1]

cR = np.zeros((nTotAnts,2))
AntIdx = np.zeros((nTotAnts))
for k in range(nTotAnts):
    cR[k,0] = hdu.data[k*2][10]
    cR[k,1] = hdu.data[k*2][9]
    AntIdx[k] = hdu.data[k*2][1]

Fc = float(args.freq)*1e6   # central frequency
d = Fc / 3e8

plt.figure()
plt.plot(cR[:,0],cR[:,1],'.')
plt.title('MWA tiles')
plt.grid()
plt.xlabel('<- West -- East ->')
plt.ylabel('<- South -- North ->')
plt.show()

Nx = int(args.nResol)    # number of horizontal pixels in sky map
Ny = int(args.nResol)    # number of vertical pixels in sky map
skymap = np.zeros((6,Nx,Ny),dtype=complex);

x = np.linspace(xmin,xmax,Nx)
y = np.linspace(ymin,ymax,Ny)

# try all matrices
R0 = Rs[:,:,0]
R1 = Rs[:,:,1]
R2 = np.zeros((nTotAnts,nTotAnts),dtype=np.complex)
R3 = np.zeros((nTotAnts,nTotAnts),dtype=np.complex)
R4 = np.zeros((nTotAnts,nTotAnts),dtype=np.complex)
R5 = np.zeros((nTotAnts,nTotAnts),dtype=np.complex)
for k in range(nTotAnts):
    for kk in range(nTotAnts):
        R2[int(AntIdx[k]),int(AntIdx[kk])] = R0[k,kk]
        R3[int(AntIdx[k]),int(AntIdx[kk])] = R1[k,kk]
        R4[k,kk] = R0[int(AntIdx[k]),int(AntIdx[kk])]
        R5[k,kk] = R1[int(AntIdx[k]),int(AntIdx[kk])]

sb = np.zeros((2,1))
for nx in range(Nx):
    for ny in range(Ny):
        sb[0] = x[nx]; sb[1] = y[ny];
        a = np.exp(-2*np.pi*1j*d*np.inner(cR,np.transpose(sb)))
        w = a / np.sqrt(np.inner(np.transpose(a),np.conjugate(np.transpose(a))))
        if (x[nx]**2+y[ny]**2) <= 1:
            skymap[0,nx,ny] = (np.transpose(np.conjugate(w)).dot(R0)).dot(w)
            skymap[1,nx,ny] = (np.transpose(np.conjugate(w)).dot(R1)).dot(w)
            skymap[2,nx,ny] = (np.transpose(np.conjugate(w)).dot(R2)).dot(w)
            skymap[3,nx,ny] = (np.transpose(np.conjugate(w)).dot(R3)).dot(w)
            skymap[4,nx,ny] = (np.transpose(np.conjugate(w)).dot(R4)).dot(w)
            skymap[5,nx,ny] = (np.transpose(np.conjugate(w)).dot(R5)).dot(w)
        else:
            skymap[0,nx,ny] = float('nan')
            skymap[1,nx,ny] = float('nan')
            skymap[2,nx,ny] = float('nan')
            skymap[3,nx,ny] = float('nan')
            skymap[4,nx,ny] = float('nan')
            skymap[5,nx,ny] = float('nan')
            
#cR2 = cR
#for k in range(nTotAnts):
#    cR2[k,0] = cR[k,1]
#    cR2[k,1] = cR[k,0]
#cR = cR2
#
#sb = np.zeros((2,1))
#for nx in range(Nx):
#    for ny in range(Ny):
#        sb[0] = x[nx]; sb[1] = y[ny];
#        a = np.exp(-2*np.pi*1j*d*np.inner(cR,np.transpose(sb)))
#        w = a / np.sqrt(np.inner(np.transpose(a),np.conjugate(np.transpose(a))))
#        if (x[nx]**2+y[ny]**2) <= 1:
#            skymap[6,nx,ny] = (np.transpose(np.conjugate(w)).dot(R0)).dot(w)
#            skymap[7,nx,ny] = (np.transpose(np.conjugate(w)).dot(R1)).dot(w)
#            skymap[8,nx,ny] = (np.transpose(np.conjugate(w)).dot(R2)).dot(w)
#            skymap[9,nx,ny] = (np.transpose(np.conjugate(w)).dot(R3)).dot(w)
#            skymap[10,nx,ny] = (np.transpose(np.conjugate(w)).dot(R4)).dot(w)
#            skymap[11,nx,ny] = (np.transpose(np.conjugate(w)).dot(R5)).dot(w)
#        else:
#            skymap[6,nx,ny] = float('nan')
#            skymap[7,nx,ny] = float('nan')
#            skymap[8,nx,ny] = float('nan')
#            skymap[9,nx,ny] = float('nan')
#            skymap[10,nx,ny] = float('nan')
#            skymap[11,nx,ny] = float('nan')
            
myfig = plt.figure()
myfig.add_subplot(2,3,1)
plt.imshow(10.*np.log10(abs(skymap[0,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
myfig.add_subplot(2,3,2)
plt.imshow(10.*np.log10(abs(skymap[1,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
myfig.add_subplot(2,3,3)
plt.imshow(10.*np.log10(abs(skymap[2,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
myfig.add_subplot(2,3,4)
plt.imshow(10.*np.log10(abs(skymap[3,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
myfig.add_subplot(2,3,5)
plt.imshow(10.*np.log10(abs(skymap[4,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
myfig.add_subplot(2,3,6)
plt.imshow(10.*np.log10(abs(skymap[5,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
#myfig.add_subplot(3,4,7)
#plt.imshow(10.*np.log10(abs(skymap[6,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
#myfig.add_subplot(3,4,8)
#plt.imshow(10.*np.log10(abs(skymap[7,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
#myfig.add_subplot(3,4,9)
#plt.imshow(10.*np.log10(abs(skymap[8,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
#myfig.add_subplot(3,4,10)
#plt.imshow(10.*np.log10(abs(skymap[9,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
#myfig.add_subplot(3,4,11)
#plt.imshow(10.*np.log10(abs(skymap[10,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
#myfig.add_subplot(3,4,12)
#plt.imshow(10.*np.log10(abs(skymap[11,:,:])**2),extent=[-1,1,-1,1],aspect='auto')
plt.show()

plt.figure()
a0,b = LA.eigh(R0)
a1,b = LA.eigh(R1)
a2,b = LA.eigh(R2)
a3,b = LA.eigh(R3)
a4,b = LA.eigh(R4)
a5,b = LA.eigh(R5)

myfig = plt.figure()
myfig.add_subplot(2,3,1)
plt.plot(10.*np.log10(a0));plt.grid()
myfig.add_subplot(2,3,2)
plt.plot(10.*np.log10(a1));plt.grid()
myfig.add_subplot(2,3,3)
plt.plot(10.*np.log10(a2));plt.grid()
myfig.add_subplot(2,3,4)
plt.plot(10.*np.log10(a3));plt.grid()
myfig.add_subplot(2,3,5)
plt.plot(10.*np.log10(a4));plt.grid()
myfig.add_subplot(2,3,6)
plt.plot(10.*np.log10(a5));plt.grid()
plt.show()
plt.close()

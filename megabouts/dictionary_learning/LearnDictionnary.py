import platform
import os
import json

# Data Wrangling
import h5py
import numpy as np
import pandas as pd
import pickle

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40

from sporco.dictlrn import cbpdndl
from sporco import util
from sporco import signal
from sporco import plot
from sporco.dictlrn import prlcnscdl

import click

def alignDict(D):
    alignedD = np.copy(D)
    for i in range(D.shape[-1]):
        intensity = np.abs(D[:,-1,i])
        idmax = np.argmax(intensity)
        if idmax>0:#30:
            if True:#np.mean(intensity[idmax:min(len(intensity),idmax+40)])>np.mean(intensity[max(0,idmax-40):idmax]):
                alignedD[:,:,i]=np.roll(D[:,:,i],-idmax+25,axis=0)
                alignedD[:,:,i]=alignedD[:,:,i]*np.sign(alignedD[25,:,i])
                print('rect')
                print(idmax)
    return alignedD


@click.command()
@click.option('--num_atoms_max', prompt='Number of atoms in dictionnary:', type=int)
#@click.option('--num_atoms', default=3, help='Number of atoms in dictionnary.')
@click.option('--strain', prompt='Strain to Merge:',
              help='Tu Giant or Danionella')

def main_routine(strain,num_atoms_max):
    print(num_atoms_max)
    if platform.system()=='Linux':
        folder = '/mnt/d/ResultCatchAllBouts3/'+strain+ '/'
    else:
        folder = 'D:/ResultCatchAllBouts3/'+strain+ '/'
    
    data = np.load('TensorTailSample'+strain+'.npy')
    print(data.shape)
    Tensor = data
    # Make Speed Smaller to avoid fitting only this:
    #Tensor= Tensor[:,3:] 
    Tensor[:,[0,1,2],:] = Tensor[:,[0,1,2],:]/50

    np.random.seed(0)
    
    for num_atoms in range(1,num_atoms_max):

        D0 = np.random.randn(100,13,num_atoms)
        Dinit=np.copy(D0)
        lmbdaList = np.linspace(0,6,4)

        k=0
        dicthist =[]
        for i in range(len(lmbdaList)):
            lmbda = lmbdaList[i]
            opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options({'Verbose': True,
                        'MaxMainIter': 400,
                        'CBPDN': {'rho': 50.0*lmbda + 0.5},
                        'CCMOD': {'rho': 1.0, 'ZeroMean': True}})

            print(i)
            #k=np.random.randint(Tensor.shape[-1]-201)
            #d = prlcnscdl.ConvBPDNDictLearn_Consensus(Dinit,Tensor[:,:,k:k+200], lmbda, opt,nproc=64,dimN=1)
            d = prlcnscdl.ConvBPDNDictLearn_Consensus(Dinit,Tensor, lmbda, opt,nproc=64,dimN=1)
    
            # Fit new impulse on d
            D = d.solve()
            #D = D.squeeze()
            D = D[:,:,0,:]
            dicthist.append(D)
            Dinit = np.copy(D)
            Dinit = alignDict(Dinit)
        
        #d = prlcnscdl.ConvBPDNDictLearn_Consensus(Dinit,Tensor, lmbda, opt,nproc=64,dimN=1)
        #D = d.solve()
        #D = D[:,:,0,:]
        #dicthist.append(D)

        # Save Dict
        if platform.system()=='Linux':
            folder = '/mnt/d/ResultCatchAllBouts3/'+strain+ '/'
        else:
            folder = 'D:/ResultCatchAllBouts3/'+strain+ '/'
        
        filename=os.path.join(folder,'strain_'+ strain +'_atoms_'+str(num_atoms)+'_Atoms_Dataset.pickle')

        Atoms_Dataset={'dicthist':dicthist,'D':D}

        with open(filename, 'wb') as handle:
            pickle.dump(Atoms_Dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    main_routine()

import numpy as np
import matplotlib.pyplot as plt
from mrts import MRTS
from tps import TPS
from visualization import display_grid2D
import os

cwd = os.getcwd()

MRTS_SAVE_IMAGE_PATH = cwd + '/mrts.jpg'
TPS_SAVE_IMAGE_PATH = cwd + '/tps.jpg'

def mrts_main():

    control_points = np.linspace(1/50,1,50).reshape(-1,1)
    mrts = MRTS(inv_mode = 'numpy', ed_mode='numpy')
    mrts.fit(control_points, control_points)
    mtx = mrts.basis
    plt.plot(control_points, mtx[:,4])

    fig = plt.figure(figsize=(10,10))
    spec = fig.add_gridspec(ncols=7, nrows=7)

    for i in range(7):
        for j in range(7):
            ax = fig.add_subplot(spec[i, j])
            ax.plot(control_points, mtx[:,i*7+j+1])
            plt.axis('off')

    fig.suptitle('MRTS')

    if MRTS_SAVE_IMAGE_PATH:
        plt.savefig(MRTS_SAVE_IMAGE_PATH)
        print(f'save figure to {MRTS_SAVE_IMAGE_PATH}')

def mrts_test(inv_mode, ed_mode, use_gpu):
    control_points = np.linspace(1/50,1,10000).reshape(-1,1)
    location = np.random.randint(1,255,size=(10000,2))
    mrts = MRTS(inv_mode=inv_mode, ed_mode=ed_mode, use_gpu=use_gpu)
    mrts.fit(control_points, location)
    mtx = mrts.basis

def tps_main():

    tps = TPS()

    gridx, gridy = np.meshgrid(np.arange(30)+1, np.arange(30)+1, indexing='xy')
    control_points = np.array([[1,3,5,7,9,6,8,5],[1.1,2.8,5.7,6.2,3.3,4.4,5,8]]).T
    deformation_points = np.array([[0.8,3.6,5.88,7.6,9.9,5.5,8.8,9.6],[0.8,2.4,5.8,6.3,3.5,4.5,7,8.4]]).T
    newdata = np.vstack([gridx.ravel(), gridy.ravel()]).T

    tps.fit(control_points, deformation_points)
    pred = tps.predict(newdata)

    display_grid2D(tps,0,10,50,0,10,50,TPS_SAVE_IMAGE_PATH)

if __name__ == '__main__':
    mrts_test(inv_mode='cupy', ed_mode='cupy', use_gpu=True)
    mrts_main()
    tps_main()
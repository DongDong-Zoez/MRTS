import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tps import TPS

def display_grid2D(
    tps, 
    xmin=0, 
    xmax=10, 
    xnum=50,
    ymin=0, 
    ymax=10,
    ynum=50,
    save_path = None
    ):

    gridx = np.linspace(xmin, xmax, xnum)
    gridy = np.linspace(ymin, ymax, ynum)
    gridx, gridy = np.meshgrid(gridx, gridy)
    grid = np.vstack([gridx.ravel(), gridy.ravel()]).T

    if not isinstance(tps, TPS):
        raise Exception('Input module should be TPS class.')
    deformation_grid = tps.predict(grid)
    
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.title('Before Deformation')
    plt.axis('off')
    plt.scatter(*grid.T)

    plt.subplot(1,2,2)
    plt.title('After Deformation')
    plt.axis('off')
    plt.scatter(*deformation_grid.T)

    if save_path:
        plt.savefig(save_path)
        print(f'save figure to {save_path}')

def image_warp(
    tps, 
    image_path,
    save_path=None
    ):

    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    if arr.max() > 1:
        arr = arr / 255

    gridx, gridy = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[0])+1, indexing='xy')
    newdata = np.vstack([gridx.ravel(), gridy.ravel()]).T
    pred = tps.predict(newdata)

    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.title('Before Deformation')
    plt.axis('off')
    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)

    plt.subplot(1,2,2)
    plt.title('After Deformation')
    plt.axis('off')
    plt.scatter(pred[:,0], -pred[:,1], c=arr.ravel(), vmin=0, vmax=1, cmap='gray')
    
    if save_path:
        plt.savefig(save_path)
        print(f'save figure to {save_path}')
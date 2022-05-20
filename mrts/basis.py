import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mrts import MRTS
from tps import TPS
from visualization import display_grid2D
import os
import argparse

cwd = os.getcwd()

def parse_args():

    description = 'Welcome to use MRTS python version'
    parser = argparse.ArgumentParser(description=description) 
                                                                     
    parser.add_argument('-i', help='mode of matrix inversion', type=str, default='numpy')
    parser.add_argument('-e', help='mode of eigen decomposition', type=str, default='numpy')
    parser.add_argument('-p', help='pseudo inverse', type=bool, default=False)
    parser.add_argument('-g', help='use gpu or not', type=bool, default=False)
    parser.add_argument('-o', help='the tolerance of devided by zero', type=float, default=1e-6)
    parser.add_argument('-k', help='kernel function ot use', type=str, default='rbf')
    parser.add_argument('-t', help='path to store your logger txt file', type=str, default='logger.txt')
    parser.add_argument('-n', help='scale the mrts basis or not', type=bool, default=True)
    parser.add_argument('-s', help='path to save your mrts csv output', type=str, default='basis.csv')
    parser.add_argument('-c', help='path to your control points csv file', type=str, default=None)
    parser.add_argument('-l', help='path to your location points csv file', type=str, default=None)
    
    args = parser.parse_args()                                        
    return args


def main():

    mrts = MRTS(
        if_scale = args.n,
        inv_mode = args.i,
        ed_mode = args.e,
        pseudo_inv = args.p,
        top_k=None,
        kernel_function=args.k,
        tolerance=args.o,
        use_gpu=args.g,
        txt_path=args.t,
    )
    
    control_points = pd.read_csv(args.c, index_col=0).to_numpy()
    location = pd.read_csv(args.l, index_col=0).to_numpy()

    mrts.fit(control_points, location)
    mtx = mrts.basis

    mrts.to_csv(mtx, args.s)

if __name__ == '__main__':
    args = parse_args()
    main()
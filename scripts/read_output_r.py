#!/bin/env python3

import os
import sys

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def plot_from_file(filename, dataset, fig, ax = plt):
    data_file = h5py.File(filename, 'r')

    print(data_file)
    print(data_file.keys())

    nodes0 = data_file['nodes0'][()]
    nodes1 = data_file['nodes1'][()]
    nodes2 = data_file['nodes2'][()]

    soln = data_file['soln'][()]

    tmp = data_file['soln'][()]
    tmp = tmp.reshape((len(nodes0),len(nodes1),len(nodes2))).transpose();

    fn = RegularGridInterpolator((nodes0,nodes1,nodes2), tmp)
    num_pts = 101
    r_min = 0.0
    r_max = 4.0
    r = np.linspace(0.0,4.0,num_pts)
    a_theta = np.linspace(0.0, np.pi, 12)
    a_phi = np.linspace(0.0, 2.0 * np.pi, 6)

    result = []
    for i,theta in enumerate(a_theta):
        for j,phi in enumerate(a_phi):
            xx = 1. + r*np.sin(theta)*np.cos(phi)
            yy = 1. + r*np.sin(theta)*np.sin(phi)
            zz = 1. + r*np.cos(theta)
            pts = []
            for (x,y,z) in zip(xx,yy,zz):
              pts.append([x,y,z])  
            result = fn(pts)
            plt.plot(r,result)

    ax.set_title("Lenard-Bernstein 3D, t = {}".format(data_file['time'][()]))
    ax.set_xlim(r_min,r_max)
    ax.set_xlabel(r"R")
    ax.set_ylabel(r"U(R,$\theta$,$\phi$)");

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise RuntimeError("Expected a datafile")

    input_fname = sys.argv[1]

    if not os.path.exists(input_fname):
        raise RuntimeError("File '{}' does not exist".format(input_fname))

    fig, ax = plt.subplots()
    plot_from_file(input_fname, 'asgard', fig, ax)
    #plt.show()
    plt.savefig('lenard_bernstein_3_t_1p0.png', dpi=600)


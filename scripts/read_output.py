#!/bin/env python3

import os
import sys

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def plot_from_file(filename, dataset, fig, ax = plt):
    data_file = h5py.File(filename, 'r')

    nodes0 = data_file['nodes0'][()]
    nodes1 = data_file['nodes1'][()]
    nodes2 = data_file['nodes2'][()]

    soln = data_file['soln'][()]

    tmp = data_file['soln'][()]
    tmp = tmp.reshape((len(nodes0),len(nodes1),len(nodes2))).transpose();

    # full grid calculation done before we added use_full_grid to output...
    grid_type = 'full'
    try:
        grid_type = data_file['grid_type'][()]
    except:
        pass
    degree = data_file['degree'][()]
    level = data_file['dim0_level'][()]
    pde = data_file['pde'][()]

    fn = RegularGridInterpolator((nodes0,nodes1,nodes2), tmp)
    num_pts = 401
    r = 0.5
    a_theta = np.linspace(0.0, np.pi, num_pts)
    a_phi = np.linspace(0.0, 2.0 * np.pi, num_pts)

    result = []
    for i,theta in enumerate(a_theta):
        for j,phi in enumerate(a_phi):
            xx = 1. + r*np.sin(theta)*np.cos(phi)
            yy = 1. + r*np.sin(theta)*np.sin(phi)
            zz = 1. + r*np.cos(theta)
            result.append(fn([xx,yy,zz])[0])
            assert(np.abs((xx - 1.0)**2 + (yy-1.0)**2 + (zz-1.0)**2 - r**2) < 1e-5)
    print(np.min(result),np.max(result))
    result = np.array(result).reshape((num_pts,num_pts)) - 0.055504468907057014
    cs = ax.pcolormesh(a_theta/np.pi, a_phi/np.pi, result)
    ax.set_title("{}, r = {}, {} grid, d={}, l={}".format(pde,r, grid_type, degree, level))
    ax.set_xlabel(r"$\frac{\theta}{\pi}$")
    ax.set_ylabel(r"$\frac{\phi}{\pi}$");
    fig.colorbar(cs, format='%.0e')
    plt.savefig('{}_r_{}_d_{}_l_{}_{}_grid_error.png'.format(pde,r,degree,level,grid_type), dpi=600)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise RuntimeError("Expected a datafile")

    input_fname = sys.argv[1]

    if not os.path.exists(input_fname):
        raise RuntimeError("File '{}' does not exist".format(input_fname))

    fig, ax = plt.subplots()
    plot_from_file(input_fname, 'asgard', fig, ax)


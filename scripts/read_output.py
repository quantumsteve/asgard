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
    result = np.array(result).reshape((num_pts,num_pts))
    cs = ax.contourf(a_theta, a_phi, result)
    ax.set_title("Lenard-Bernstein 3D, r = {}, t = {}".format(r,data_file['time'][()]))
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\phi$");
    fig.colorbar(cs)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise RuntimeError("Expected a datafile")

    input_fname = sys.argv[1]

    if not os.path.exists(input_fname):
        raise RuntimeError("File '{}' does not exist".format(input_fname))

    fig, ax = plt.subplots()
    plot_from_file(input_fname, 'asgard', fig, ax)
    plt.savefig('lenard_bernstein_3_r_0p5_t_1p0.png', dpi=600)


#!/bin/env python

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

    r = 0.1
    a_theta = np.linspace(0.0, np.pi, 101)
    a_phi = np.linspace(0.0, 2.0 * np.pi, 101)

    result = []
    for i,theta in enumerate(a_theta):
        for j,phi in enumerate(a_phi):
            xx = r*np.sin(theta)*np.cos(phi)
            yy = r*np.sin(theta)*np.sin(phi)
            zz = r*np.cos(theta)
            result.append(fn([xx,yy,zz])[0])
            #print(i,j)
            #print(fn([xx,yy,zz]))
    #print(result)
    print(np.min(result),np.max(result))
    result = np.array(result).reshape((101,101))
    cs = ax.contourf(a_theta, a_phi, result)
    ax.set_title("t = {}".format(data_file['time'][()]))
    fig.colorbar(cs)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise RuntimeError("Expected a datafile")

    input_fname = sys.argv[1]

    if not os.path.exists(input_fname):
        raise RuntimeError("File '{}' does not exist".format(input_fname))

    fig, ax = plt.subplots()
    plot_from_file(input_fname, 'asgard', fig, ax)

    plt.show()

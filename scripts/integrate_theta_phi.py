#!/bin/env python3

import os
import sys

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

class interpolator:   
    def __init__(self, fn, r):
        self.fn = fn
        self.r = r

    def evaluate(self, phi, theta):
        xx = 1. + self.r*np.sin(theta)*np.cos(phi)
        yy = 1. + self.r*np.sin(theta)*np.sin(phi)
        zz = 1. + self.r*np.cos(theta)
        pts = [xx,yy,zz]
        return self.fn(pts)[0]*np.sin(theta)

def plot_from_file(filename, dataset, fig, ax = plt):
    data_file = h5py.File(filename, 'r')

    #print(data_file)
    #print(data_file.keys())

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
    rr = [0.5]#; np.linspace(0.0,4.0,num_pts)

    th = 1.0;
    prefactor = 1.0 / np.sqrt(2. * np.pi)

    average_value = []
    for r in rr:
        f = interpolator(fn,r)
        #print(r,f.evaluate(np.pi,np.pi/2.))
        y, abserr = dblquad(f.evaluate, 0, np.pi, 0, 2.*np.pi, epsrel=1.e-3, epsabs=1.e-4)
        #print(r, y, abserr)
        theta = 0.
        phi = 0.
        xx = 1. + r*np.sin(theta)*np.cos(phi)
        yy = 1. + r*np.sin(theta)*np.sin(phi)
        zz = 1. + r*np.cos(theta)
        exact = prefactor * np.exp(-1. * (xx - 1.)**2 / (2.0 * th))
        exact *= prefactor * np.exp(-1. * (yy - 1.)**2 / (2.0 * th))
        exact *= prefactor * np.exp(-1. * (zz - 1.)**2 / (2.0 * th))
        diff = np.abs(y/(4.0*np.pi) - exact)
        #print(r,diff,abserr/(4.0*np.pi))
        #average_value.append(diff)
        print(y/(4.*np.pi))

    plt.plot(rr,average_value)
    
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
    plt.show()
    #plt.savefig('lenard_bernstein_3_t_1p0.png', dpi=600)


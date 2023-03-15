#!/bin/env python

import os
import sys

import h5py

import matplotlib.pyplot as plt

def plot_from_file(filename, dataset, ax = plt):
    data_file = h5py.File(filename, 'r')

    print(data_file)
    print(data_file.keys())

    nodes0 = data_file['nodes0'][()]
    nodes1 = data_file['nodes1'][()]
    nodes2 = data_file['nodes2'][()]

    soln = data_file['soln'][()]

    tmp = data_file['soln'][()]
    tmp = tmp.reshape((len(nodes0),len(nodes1),len(nodes2))).transpose();

    ax.contourf(nodes1, nodes2, tmp[7,:,:])
    ax.set_title("t = {}".format(data_file['time'][()]))


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise RuntimeError("Expected a datafile")

    input_fname = sys.argv[1]

    if not os.path.exists(input_fname):
        raise RuntimeError("File '{}' does not exist".format(input_fname))

    fig, ax = plt.subplots()
    plot_from_file(input_fname, 'asgard', ax)

    plt.show()

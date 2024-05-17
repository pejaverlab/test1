import csv
import argparse
import os
import numpy as np
import math
import time
import bisect
from Tavtigian.tavtigian import get_tavtigian_c, get_tavtigian_thresholds
from configmodule import ConfigModule
from LocalCalibration.gaussiansmoothing import *
from multiprocessing.pool import Pool
from LocalCalibration.LocalCalibration import LocalCalibration
import time


def load_labelled_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    x = [float(e[0]) for e in data]
    y = [int(e[1]) for e in data]
    return x,y

def load_unlabelled_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    g = [float(e[0]) for e in data]
    return g


def getParser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--configfile",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tool",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--labelled_data_file",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--unlabelled_data_file",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--reverse",
        action='store_true',
    )

    return parser



def main():

    parser = getParser()
    args = parser.parse_args()
    tool = args.tool
    labeldatafile = args.labelled_data_file
    udatafile = args.unlabelled_data_file
    reverse = args.reverse
    configfile = args.configfile
    outdir = args.outdir

    configmodule = ConfigModule()
    configmodule.load_config(configfile)
    B = configmodule.B
    discountonesided = configmodule.discountonesided
    windowclinvarpoints = configmodule.windowclinvarpoints
    windowgnomadfraction = configmodule.windowgnomadfraction
    gaussian_smoothing = configmodule.gaussian_smoothing
    data_smoothing = configmodule.data_smoothing
    if data_smoothing:
        assert udatafile is not None

    alpha = None
    c = None
    if (configmodule.emulate_tavtigian):
        alpha = 0.1
        c = 350
    elif (configmodule.emulate_pejaver):
        alpha = 0.0441
        c = 1124
    else:
        alpha = configmodule.alpha
        c = get_tavtigian_c(alpha)

    x,y = load_labelled_data(labeldatafile)
    g = load_unlabelled_data(udatafile)

    x = np.array(x)
    y = np.array(y)
    g = np.sort(np.array(g))
    xg = np.concatenate((x,g))

    calib = LocalCalibration(alpha, c, reverse, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing, data_smoothing)
    thresholds, posteriors_p = calib.fit(x,y,g,alpha)
    posteriors_b = 1 - np.flip(posteriors_p)



if __name__ == '__main__':
    main()

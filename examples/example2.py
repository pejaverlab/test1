import csv
import argparse
import os
import numpy as np
import math
import time
import bisect
from LocalCalibration.tavtigian import get_tavtigian_c, get_tavtigian_thresholds
from LocalCalibration.configmodule import ConfigModule
from LocalCalibration.gaussiansmoothing import *
from multiprocessing.pool import Pool
from LocalCalibration.LocalCalibration import *
import time


def load_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    return data


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
        "--data_dir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--labelled_data_file",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--PU_data_file",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--clamp",
        action='store_true',
    )
    parser.add_argument(
        "--reverse",
        action='store_true',
    )

    return parser



def main():

    parser = getParser()
    args = parser.parse_args()
    configmodule = ConfigModule()
    configmodule.load_config(args.configfile)

    B = configmodule.B
    discountonesided = configmodule.discountonesided
    windowclinvarpoints = configmodule.windowclinvarpoints
    windowgnomadfraction = configmodule.windowgnomadfraction

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

    print(c)

    tool = args.tool
    datadir = args.data_dir;
    labeldatafile = args.labelled_data_file
    pudatafile = args.PU_data_file
    reverse = args.reverse
    clamp = args.clamp
    gaussian_smoothing = configmodule.gaussian_smoothing

    labelleddata = load_data(os.path.join(datadir,labeldatafile))
    x = [float(e[0]) for e in labelleddata]
    y = [int(e[1]) for e in labelleddata]
    pudata = load_data(os.path.join(datadir,pudatafile))
    g = [float(e[0]) for e in pudata if e[1] == '0']

    x = np.array(x)
    y = np.array(y)
    g = np.sort(np.array(g))
    xg = np.concatenate((x,g))

    w = ( (1-alpha)*((y==1).sum()) ) /  ( alpha*((y==0).sum()) )    

    calib = LocalCalibration(alpha, c, reverse, clamp, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing)
    thresholds, posteriors_p = calib.fit(x,y,g,alpha)

    print(thresholds, posterior_p)



if __name__ == '__main__':
    main()

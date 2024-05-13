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

def storeResults(outdir, tool, thresholds, posteriors_p, posteriors_b, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB):

    fname = os.path.join(outdir,tool + "-pathogenic.txt")
    tosave = np.array([thresholds,posteriors_p]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,tool + "-benign.txt")
    tosave = np.array([np.flip(thresholds),posteriors_b]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.flip(thresholds),posteriors_b , linewidth=2.0)
    #ax.hlines(bthresh[0],thresholds[-1], thresholds[0], color='r', linestyle='dashed')
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title(tool)
    plt.savefig(os.path.join(outdir,tool+"-benign.png"))
    ax.clear()
    ax.plot(thresholds,posteriors_p , linewidth=2.0, color='b')
    #ax.plot(thresholds,posteriors_p_smooth , linewidth=2.0, color='r')
    #ax.hlines(pthresh[0],thresholds[-1], thresholds[0], color='r', linestyle='dashed')
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title(tool)
    plt.savefig(os.path.join(outdir, tool+"-pathogenic.png"))

    fname = os.path.join(outdir,tool + "-pthresh.txt")
    np.savetxt(fname, pthresh , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,tool + "-bthresh.txt")
    np.savetxt(fname, bthresh , delimiter='\t', fmt='%f')

    fname = os.path.join(outdir,tool + "-pthreshdiscounted.txt")
    np.savetxt(fname, DiscountedThresholdP , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,tool + "-bthreshdiscounted.txt")
    np.savetxt(fname, DiscountedThresholdB , delimiter='\t', fmt='%f')



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
    pu_data_available = configmodule.pu_data_available
    if pu_data_available:
        assert pudatafile is not None

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
    posteriors_b = 1 - np.flip(posteriors_p)

    calib = LocalCalibrateThresholdComputation(alpha, c, reverse, clamp, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing, )
    start = time.time()
    _, posteriors_p_bootstrap = calib.get_both_bootstrapped_posteriors_parallel(x,y, g, 1000, alpha, thresholds)
    end = time.time()
    print("time: " ,end - start)


    Post_p, Post_b = get_tavtigian_thresholds(c, alpha)

    all_pathogenic = np.row_stack((posteriors_p, posteriors_p_bootstrap))
    all_benign = 1 - np.flip(all_pathogenic, axis = 1)

    pthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_pathogenic, thresholds, Post_p)
    bthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_benign, np.flip(thresholds), Post_b) 

    DiscountedThresholdP = LocalCalibrateThresholdComputation.get_discounted_thresholds(pthresh, Post_p, B, discountonesided, 'pathogenic')
    DiscountedThresholdB = LocalCalibrateThresholdComputation.get_discounted_thresholds(bthresh, Post_b, B, discountonesided, 'benign')

    storeResults(args.outdir, tool, thresholds, posteriors_p, posteriors_b, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB)

    print("Discounted Thresholds: ", DiscountedThresholdP)



if __name__ == '__main__':
    main()

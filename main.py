import csv
import argparse
import os
import numpy as np
import math
import time


def load_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    return data


def getIncrementValue(tool):

    if tool == "CADDphred":
        return 0.1

    if tool == "BayesDel-noAF" or tool == "CADDraw" or tool == "phyloP100way" or tool == "FATHMM" or tool == "GERP++":
        return 0.01

    if tool == "EA1.0" or tool == "MutPred" or tool == "MutPred2.0" or tool == "PolyPhen2-HVAR" or tool == "REVEL" or tool == "VEST4" or  tool == "SIFT":
        return 0.001

    return 0.001

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

    return parser


def preprocess_data(x,y,g, toolname):
    if toolname == 'SIFT' or toolname == 'FATHMM':
        x = [-e for e in s]
        g = [-e for e in g]

    if toolname == 'EA1.0':
        g = [e for e in g if e <= 1]

    return x,y,g


def compute_thresholds(l):
    th = np.unique(l).tolist()
    if not math.floor(th[0]) == th[0]:
        th.insert(0, math.floor(th[0]))
    if not math.ceil(th[-1]) == th[-1]:
        th.append(math.ceil(th[-1]))
    th.reverse()
    return th

    
import bisect


def findPosterior(allthrs, thrs, xpos, xneg, g, increment, minpoints, gft, w):
    
    maxthrs = allthrs[0];
    minthrs = allthrs[-1];
    minwindow = 0.0
    maxwindow = (maxthrs - minthrs)
    #print("maxwindow: ", maxwindow)
    lengthgnomad = len(g)

    smallwindow = minwindow
    bigwindow = maxwindow

    # check that bigwindow works else we will have to return false
    # if small window works, return smallwindow

    maxiterations = 100
    iterationNo = 0
    pos = None
    neg = None
    while(bigwindow - smallwindow > increment*0.5):
        #print("bigwindow smallwindow: ", bigwindow, smallwindow)
        iterationNo += 1
        if iterationNo > 100:
            raise Exception("will have to address this\n")
        currentWindow = (bigwindow + smallwindow)/2.0
        #print("currentWindow: ", currentWindow)
        lo = thrs - currentWindow
        hi = thrs + currentWindow

        c = None
        if hi > maxthrs:
            c = (maxthrs-lo)/(hi-lo)
        elif lo < minthrs:
            c = (hi-minthrs)/(hi-lo)
        else:
            c = 1
            
        if c <= 0:
            raise Exception("Problem with computing c")

        pos = bisect.bisect_right(xpos, hi) - bisect.bisect_left(xpos, lo)
        neg = bisect.bisect_right(xneg, hi) - bisect.bisect_left(xneg, lo)
        if pos + neg < c*minpoints:
            smallwindow = currentWindow
            continue
        gfilterlen = bisect.bisect_right(g, hi) - bisect.bisect_left(g, lo)
        if gfilterlen < gft*c*lengthgnomad:
            smallwindow = currentWindow
            continue

        bigwindow = currentWindow

    #print(thrs, bigwindow, smallwindow, maxthrs, minthrs ,pos, neg)
    #xsprint("iterationNo: ",  iterationNo)
    return pos/(pos + w*neg)



def get_both_local_posteriors(x, y, g, thrs, w, minpoints, gft, increment):

    start = time.time()

    xpos = np.sort(x[y==1])  #[x[i] for i in range(len(x)) if y[i] == 1]
    xneg = np.sort(x[y==0])  #[x[i] for i in range(len(x)) if y[i] == 0]

    assert x.size == y.size

    post = np.zeros(len(thrs))

    for i in range(len(thrs)):

        post[i] = findPosterior(thrs, thrs[i], xpos, xneg, g, increment, minpoints, gft, w)
    
    #print(post)
    end = time.time()
    print("time taken: ", end - start)
    return post


from multiprocessing.pool import Pool

def initialize(x_, y_, g_, w_, thrs_, minpoints_, gft_, increment_, B_):
    global x
    global y
    global g
    global w
    global thrs
    global minpoints
    global gft
    global increment
    global B
    x = x_
    y = y_
    g = g_
    w = w_
    thrs = thrs_
    minpoints  = minpoints_
    gft = gft_
    increment = increment_
    B = B_
    


def get_both_bootstrapped_posteriors_parallel(x, y, g, w, thrs, minpoints, gft, increment, B):
    with Pool(192,initializer = initialize, initargs=(x, y, g, w, thrs, minpoints, gft, increment, B,),) as pool:
        #print(pool._processes)
        items = [i for i in range(B)]
        ans = pool.map(get_both_bootstrapped_posteriors, items, 64)
        return np.array(ans)

def get_both_bootstrapped_posteriors(seed):
    np.random.seed(seed)
    qx = np.random.randint(0,len(x),len(x))
    qg = np.random.randint(0,len(g),len(g))

    posteriors_p = get_both_local_posteriors(x[qx], y[qx], g[qg], thrs, w, minpoints, gft, increment)

    return posteriors_p


def get_all_thresholds(posteriors, thrs, Post):
    
    thresh = np.zeros((posteriors.shape[0], len(Post)))

    for i in range(posteriors.shape[0]):
        posterior = posteriors[i]
        for j in range(len(Post)):
            idces = np.where(posterior < Post[j])[0]
            if i == 0:
                print("idces here: ", idces, Post[j])
            if len(idces) > 0 and idces[0] > 0:
                thresh[i][j] = thrs[idces[0]-1]
            else:
                thresh[i][j] = np.NaN

    return thresh
            
            
def get_discounted_thresholds(thresh, Post, B, discountonesided, tp):
    threshT = thresh.T
    DiscountedThreshold = np.zeros(len(Post))
    for j in range(len(Post)):
        threshCurrent = threshT[j]
        invalids = np.count_nonzero(np.isnan(threshCurrent))
        if invalids > discountonesided*B:
            DiscountedThreshold[j] = np.NaN
        else:
            valids = threshCurrent[~np.isnan(threshCurrent)]
            valids = np.sort(valids)
            if tp == 'pathogenic':
                valids = np.flip(valids)
            else:
                assert tp == 'benign'

            DiscountedThreshold[j] = valids[np.floor(discountonesided * B).astype(int) - invalids];

    return DiscountedThreshold


import configmodule


def main():

    parser = getParser()
    args = parser.parse_args()
    configmodule.load_config(args.configfile)

    alpha = configmodule.alpha
    c = configmodule.c
    B = configmodule.B
    discountonesided = configmodule.discountonesided
    windowclinvarpoints = configmodule.windowclinvarpoints
    windowgnomadfraction = configmodule.windowgnomadfraction

    tool = args.tool
    datadir = args.data_dir;
    labeldatafile = args.labelled_data_file
    pudatafile = args.PU_data_file

    labelleddata = load_data(os.path.join(datadir,labeldatafile))
    x = [float(e[0]) for e in labelleddata]
    y = [int(e[1]) for e in labelleddata]
    pudata = load_data(os.path.join(datadir,pudatafile))
    g = [float(e[0]) for e in pudata if e[1] == '0']

    x,y,g = preprocess_data(x,y,g,tool)

    x = np.array(x)
    y = np.array(y)
    g = np.sort(np.array(g))
    xg = np.concatenate((x,g))
    thresholds = compute_thresholds(xg)
    print("threshold size: ", len(thresholds))

    w = ( (1-alpha)*((y==1).sum()) ) /  ( alpha*((y==0).sum()) )


    increment = getIncrementValue(args.tool)
    print("increment: ", increment)
    #print("thresholds: ", thresholds)
    posteriors_p = get_both_local_posteriors(x, y, g, thresholds, w, windowclinvarpoints, windowgnomadfraction, increment)
    posteriors_b = 1 - np.flip(posteriors_p)
    #print("posteriors: ", posteriors_p)

    fname = os.path.join(args.outdir,tool + "-pathogenic.txt")
    tosave = np.array([thresholds,posteriors_p]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f')
    fname = os.path.join(args.outdir,tool + "-benign.txt")
    tosave = np.array([np.flip(thresholds),posteriors_b]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f')

    Post_p = np.zeros(4)
    Post_b = np.zeros(4)
    for j in range(4):
        Post_p[j] = c ** (1 / 2 ** (j)) * alpha / ((c ** (1 / 2 ** (j)) - 1) * alpha + 1);
        Post_b[j] = (c ** (1 / 2 ** (j))) * (1 - alpha) /(((c ** (1 / 2 ** (j))) - 1) * (1 - alpha) + 1);

    print("here")
    print("posteriors_b: ", posteriors_b)
    print("Post_p: ", Post_p)
    print("Post_b: ", Post_b)

    posteriors_p_bootstrap = get_both_bootstrapped_posteriors_parallel(x,y,g,w, thresholds, windowclinvarpoints, windowgnomadfraction, increment, B)

    all_pathogenic = np.row_stack((posteriors_p, posteriors_p_bootstrap))
    all_benign = 1 - np.flip(all_pathogenic, axis = 1)
    

    pthresh = get_all_thresholds(all_pathogenic, thresholds, Post_p)
    bthresh = get_all_thresholds(all_benign, np.flip(thresholds), Post_b) 

    #np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.flip(thresholds),posteriors_b , linewidth=2.0)
    #ax.hlines(bthresh[0],thresholds[-1], thresholds[0], color='r', linestyle='dashed')
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title(tool)
    plt.savefig(os.path.join(args.outdir,tool+"-benign.png"))
    ax.clear()
    ax.plot(thresholds,posteriors_p , linewidth=2.0)
    #ax.hlines(pthresh[0],thresholds[-1], thresholds[0], color='r', linestyle='dashed')
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title(tool)
    plt.savefig(os.path.join(args.outdir, tool+"-pathogenic.png"))

    
    DiscountedThresholdP = get_discounted_thresholds(pthresh[1:], Post_p, B, discountonesided, 'pathogenic')
    DiscountedThresholdB = get_discounted_thresholds(bthresh[1:], Post_b, B, discountonesided, 'benign')

    if tool == "SIFT" or tool == "FATHMM":
        pthresh = -pthresh
        bthresh = -bthresh
        DiscountedThresholdP = -DiscountedThresholdP
        DiscountedThresholdB = -DiscountedThresholdB

    fname = os.path.join(args.outdir,tool + "-pthresh.txt")
    np.savetxt(fname, pthresh , delimiter='\t', fmt='%f')
    fname = os.path.join(args.outdir,tool + "-bthresh.txt")
    np.savetxt(fname, bthresh , delimiter='\t', fmt='%f')

    fname = os.path.join(args.outdir,tool + "-pthreshdiscounted.txt")
    np.savetxt(fname, pthresh , delimiter='\t', fmt='%f')
    fname = os.path.join(args.outdir,tool + "-bthreshdiscounted.txt")
    np.savetxt(fname, bthresh , delimiter='\t', fmt='%f')

        


if __name__ == "__main__":
    main()

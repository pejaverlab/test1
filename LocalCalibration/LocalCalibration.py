import csv
import argparse
import os
import numpy as np
import math
import time
import bisect
from .tavtigian import get_tavtigian_c
from .configmodule import ConfigModule
from .gaussiansmoothing import *
from multiprocessing.pool import Pool



class LocalCalibration:

    def __init__(self, alpha,c, reverse = None, clamp = None, windowclinvarpoints = 100, 
                 windowgnomadfraction = 0.03, gaussian_smoothing = True, unlabelled_data=False):
        self.alpha = alpha
        self.c = c
        self.reverse = reverse
        self.clamp = clamp
        self.windowclinvarpoints = windowclinvarpoints
        self.windowgnomadfraction = windowgnomadfraction
        self.gaussian_smoothing = gaussian_smoothing
        self.unlabelled_data = unlabelled_data


    @staticmethod
    def preprocess_data(x,y,g,reverse, clamp):
        if reverse:
            x = np.negative(x)
            g = np.negative(g)
            
        if clamp:
            g = g[g<=1]

        return x,y,g


    @staticmethod
    def findPosterior(allthrs, thrs, xpos, xneg, g, minpoints, gft, w):
    
        maxthrs = allthrs[0];
        minthrs = allthrs[-1];
        lengthgnomad = len(g)
        
        smallwindow = 0.0
        bigwindow = maxthrs - minthrs
        
        # check that bigwindow works else we will have to return false
        # if small window works, return smallwindow

        maxiterations = 100
        iterationNo = 0
        pos = None
        neg = None
        eps = 0.000001
        
        conditionSatisfied = False

        while(bigwindow - smallwindow > eps):
            iterationNo += 1
            if iterationNo > 100:
                raise Exception("will have to address this\n")
            currentWindow = (bigwindow + smallwindow)/2.0
            lo = thrs - currentWindow - (eps/2)
            hi = thrs + currentWindow + (eps/2)
            
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

            if gft is None:
                continue

            gfilterlen = bisect.bisect_right(g, hi) - bisect.bisect_left(g, lo)
            if gfilterlen < gft*c*lengthgnomad:
                smallwindow = currentWindow
                continue

            bigwindow = currentWindow

        return pos/(pos + w*neg)




    @staticmethod
    def get_both_local_posteriors(x, y, g, thrs, w, minpoints, gft):
        
        xpos = np.sort(x[y==1])  #[x[i] for i in range(len(x)) if y[i] == 1]
        xneg = np.sort(x[y==0])  #[x[i] for i in range(len(x)) if y[i] == 0]
        
        assert x.size == y.size
        gsorted = np.sort(g)
        
        post = np.zeros(len(thrs))
        for i in range(len(thrs)):
            post[i] = LocalCalibration.findPosterior(thrs, thrs[i], xpos, xneg, gsorted, minpoints, gft, w)
            
        return post


    @staticmethod
    def compute_thresholds(l):
        th = np.unique(l).tolist()
        if not math.floor(th[0]) == th[0]:
            th.insert(0, math.floor(th[0]))
        if not math.ceil(th[-1]) == th[-1]:
            th.append(math.ceil(th[-1]))
        th.reverse()
        return th


    def fit(self, X_train, y_train, pu_data, alpha):
        # preprocess
        x,y,g = LocalCalibration.preprocess_data(X_train, y_train, pu_data, self.reverse, self.clamp)
        if alpha is None:
            alpha = self.alpha
        w = ( (1-alpha)*((y==1).sum()) ) /  ( alpha*((y==0).sum()) )
        xg = np.concatenate((x,g))
        thresholds = self.compute_thresholds(xg)

        posteriors_p = self.get_both_local_posteriors(x, y, g, thresholds, w, self.windowclinvarpoints, self.windowgnomadfraction)
        return thresholds, posteriors_p



class LocalCalibrateThresholdComputation:


    def __init__(self, alpha,c, reverse = None, clamp = None, windowclinvarpoints = 100, 
                 windowgnomadfraction = 0.03, gaussian_smoothing = True, unlabelled_data=False):

        self.alpha = alpha
        self.c = c
        self.reverse = reverse
        self.clamp = clamp
        self.windowclinvarpoints = windowclinvarpoints
        self.windowgnomadfraction = windowgnomadfraction
        self.gaussian_smoothing = gaussian_smoothing
        self.unlabelled_data = unlabelled_data


    @staticmethod
    def initialize(x_, y_, g_, w_, thrs_, minpoints_, gft_, B_):
        global x
        global y
        global g
        global w
        global thrs
        global minpoints
        global gft
        global B
        x = x_
        y = y_
        g = g_
        w = w_
        thrs = thrs_
        minpoints  = minpoints_
        gft = gft_
        B = B_


    @staticmethod
    def get_both_bootstrapped_posteriors(seed):
        #print(seed)
        np.random.seed(seed)
        qx = np.random.randint(0,len(x),len(x))
        qg = np.random.randint(0,len(g),len(g))
        posteriors_p = LocalCalibration.get_both_local_posteriors(x[qx], y[qx], g[qg], thrs, w, minpoints, gft)
        return posteriors_p


    def get_both_bootstrapped_posteriors_parallel(self, x, y, g, B, alpha = None, thresholds = None):

        if alpha is None:
            alpha = self.alpha
        w = ( (1-alpha)*((y==1).sum()) ) /  ( alpha*((y==0).sum()) )
        xg = np.concatenate((x,g))
        
        if thresholds is None:
            thresholds = LocalCalibration.compute_thresholds(xg)

        ans = None
        with Pool(192,initializer = self.initialize, initargs=(x, y, g, w, thresholds, self.windowclinvarpoints, self.windowgnomadfraction, B,),) as pool:
            items = [i for i in range(B)]
            ans = pool.map(LocalCalibrateThresholdComputation.get_both_bootstrapped_posteriors, items, 64)

        return thresholds, np.array(ans)



    @staticmethod
    def get_all_thresholds(posteriors, thrs, Post):
    
        thresh = np.zeros((posteriors.shape[0], len(Post)))

        for i in range(posteriors.shape[0]):
            posterior = posteriors[i]
            for j in range(len(Post)):
                idces = np.where(posterior < Post[j])[0]
                if len(idces) > 0 and idces[0] > 0:
                    thresh[i][j] = thrs[idces[0]-1]
                else:
                    thresh[i][j] = np.NaN

        return thresh
            
    @staticmethod
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

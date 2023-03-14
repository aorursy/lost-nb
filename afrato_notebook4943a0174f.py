#!/usr/bin/env python
# coding: utf-8



import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
import copy
from numpy import genfromtxt

class Pocket:
    def __init__(self, N):
        # Random linearly separated data
        self.X = self.generate_points(N)
 
    def generate_points(self, N):
        # read digits data & split it into X (training input) and y (target output)
        dataset = genfromtxt('../input/train.csv', delimiter =' ')
        y = dataset[:, 0]
        X = dataset[:, 1:]
        y[y!=1] = -1    #rest of numbers are negative class
        y[y==1] = +1    #number zero is the positive class
        bX = []
        for k in range(0,N) :
            bX.append((np.concatenate(([1], X[k,:])), y[k]))
        
        # this will calculate linear regression at this point
        X = np.concatenate((np.ones((N,1)), X),axis=1); # adds the 1 constant
        self.linRegW = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)   # lin reg
        print(self.linRegW)

        return bX
    
 
    def plot(self, mispts=None, vec=None, save=False):
        # fig = plt.figure(figsize=(5,5))
        # plt.xlim(-1.5,2.5)
        # plt.ylim(-2.0,1.5)
        l = np.linspace(-1.5,2.5)
        V = self.linRegW
        a, b = -V[1]/V[2], -V[0]/V[2]
        plt.plot(l, a*l+b, 'k-')
        V = self.bestW    # for Pocket
        a, b = -V[1]/V[2], -V[0]/V[2]
        plt.plot(l, a*l+b, 'r-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'.')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'x')
        if vec.size:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points'                           % (str(len(self.X)),str(len(mispts))))
            plt.savefig('midterm.plot.pdf', bbox_inches='tight')

 
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        #myErr = 0
        for x,s in pts:
            #myErr += abs(s - int(np.sign(vec.T.dot(x))))
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        #print error
        #print myErr
        return error
 
    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False):
        # Initialize the weigths to zeros
        # w = np.zeros(3)

        w = np.array([-0.17892634,  0.13887177, -1.07425448])
        self.bestW = copy.deepcopy(w);     # for Pocket
        self.plaError = []
        self.pocketError = []   # for Pocket
        X, N = self.X, len(self.X)
        it = 0
        # Iterate until all points are correctly classified
        self.plaError.append(self.classification_error(w))
        self.pocketError.append(self.plaError[it])    # for Pocket
        oldw = np.zeros(3)
        while self.plaError[it] != 0 and oldw.all < w.all:
            it += 1
            oldw = w
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x

            self.plaError.append(self.classification_error(w))
            if (self.pocketError[it-1] > self.plaError[it]):  # for Pocket
                self.pocketError.append(self.plaError[it])
                self.bestW = copy.deepcopy(w);
            else:
                self.pocketError.append(self.pocketError[it-1])

            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n'                           % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)),                             dpi=200, bbox_inches='tight')
                plt.close()
            print(it)
        self.w = w
        print(self.classification_error(self.linRegW))
        print(self.classification_error(self.bestW))
        print(self.plaError)
        print(self.pocketError)

        return it
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

def main():
    # read digits data & split it into X (training input) and y (target output)
    ds = genfromtxt('../input/train.csv', delimiter =' ')
    row_count = sum(1 for row in ds)
    it = np.zeros(1)
    for x in range(0, 1):
        p = Pocket(row_count)
        it[x] = p.pla(save=False)
        # p.plot(save=True)
        print(it)

    #n, bins, patches = plt.hist(it, 50, normed=1, facecolor='green', alpha=0.75)
    plt.show()


    #p.plot()


main()


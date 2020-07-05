# The following are snap functions for finding a best approximated integer or rational number for a real number:

import numpy as np
from sympy import Rational

def bestApproximation(x,imax):
    # The input is a numpy parameter vector p.
    # The output is an integer specifying which parameter to change,
    # and a float specifying the new value.
    def float2contfrac(x,nmax):
        x = float(x)
        c = [np.floor(x)];
        y = x - np.floor(x)
        k = 0
        while np.abs(y)!=0 and k<nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c
    
    def contfrac2frac(seq):
        ''' Convert the simple continued fraction in `seq`
            into a fraction, num / den
            '''
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num*u, num
        return num, den
    
    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[:i+1]) for i in range(len(c))))
    
    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:,0] / float(q[:,1])
    
    def truncateContFrac(q,imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k,0]), q[k,1]) <= imax:
            k = k + 1
        return q[:k]
    
    def pval(p):
        p = p.astype(float)
        return 1 - np.exp(-p ** 0.87 / 0.36)
    
    xsign = np.sign(x)
    q = truncateContFrac(contFracRationalApproximations(float2contfrac(abs(x),20)),imax)
    
    if len(q) > 0:
        p = np.abs(q[:,0] / q[:,1] - abs(x)).astype(float) * (1 + np.abs(q[:,0])) * q[:,1]
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i,0] / float(q[i,1]), xsign* q[i,0], q[i,1], p[i])
    else:
        return (None, 0, 0, 1)

def integerSnap(p, top=1):
    p = np.array(p)
    metric = np.abs(p - np.round(p.astype(np.double)))
    chosen = np.argsort(metric)[:top]
    return dict(list(zip(chosen, np.round(p.astype(np.double))[chosen])))


def zeroSnap(p, top=1):
    p = np.array(p)
    metric = np.abs(p)
    chosen = np.argsort(metric)[:top]
    return dict(list(zip(chosen, np.zeros(len(chosen)))))


def rationalSnap(p, top=1):
    """Snap to nearest rational number using continued fraction."""
    p = np.array(p)
    snaps = np.array(list(bestApproximation(x,10) for x in p))
    chosen = np.argsort(snaps[:, 3])[:top]    
    d = dict(list(zip(chosen, snaps[chosen, 1:3])))
    d = {k:  f"{val[0]}/{val[1]}" for k,val in d.items()}
    
    return d




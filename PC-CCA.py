# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:47:36 2017

@author: admin
"""

from numpy import arange, where, array, dot, outer, zeros, concatenate, ones, tile, mean, cov, eye, real, sqrt,diag, fliplr,flipud
import numpy as np
from numpy.linalg import inv, norm
from numpy import linalg as LA
from scipy.io import loadmat

def KennardStone(X, Num):
    nrow = X.shape[0]
    CalInd = zeros((Num), dtype=int)-1
    vAll = arange(0, nrow)
    D = zeros((nrow, nrow))
    for i in range(nrow-1):
        for j in range(i+1, nrow):
            D[i, j] = norm(X[i, :]-X[j, :])
    ind = where(D == D.max())
    CalInd[0] = ind[1]
    CalInd[1] = ind[0]
    for i in range(2, Num):
        vNotSelected = array(list(set(vAll)-set(CalInd)))
        vMinDistance = zeros(nrow-i)
        for j in range(nrow-i):
            nIndexNotSelected = vNotSelected[j]
            vDistanceNew = zeros((i))
            for k in range(i):
                nIndexSelected = CalInd[k]
                if nIndexSelected <= nIndexNotSelected:
                    vDistanceNew[k] = D[nIndexSelected,nIndexNotSelected]
                else:
                    vDistanceNew[k] = D[nIndexNotSelected, nIndexSelected]
            vMinDistance[j] = vDistanceNew.min()
        nIndexvMinDistance = where(vMinDistance == vMinDistance.max())
        CalInd[i] = vNotSelected[nIndexvMinDistance]
    ValInd = array(list(set(vAll)-set(CalInd)))
    return CalInd, ValInd

def plscvfold(X, y, A, K):
    sort_index = np.argsort(y, axis = 0)
    y = np.sort(y, axis = 0)
    X = X[sort_index[:, 0]]
    M = X.shape[0]
    yytest = zeros([M, 1])
    YR = zeros([M, A])
    groups = np.asarray([i % K + 1 for i in range(0, M)])
    group = np.arange(1, K+1)
    for i in group:
        Xtest = X[groups == i]
        ytest = y[groups == i]
        Xcal = X[groups != i]
        ycal = y[groups != i]
        index_Xtest = np.nonzero(groups == i)
        index_Xcal = np.nonzero(groups != i)

        (Xs, Xp1, Xp2) = pretreat(Xcal)
        (ys, yp1, yp2) = pretreat(ycal)
        PLS1 = pls1_nipals(Xs, ys, A)
        W, T, P, Q = PLS1['W'],  PLS1['T'],  PLS1['P'], PLS1['Q']
        yp = zeros([ytest.shape[0], A])
        for j in range(1, A+1):
            B = dot(W[:, 0:j], Q.T[0:j])
            C = dot(B, yp2) / Xp2
            coef = concatenate((C, yp1-dot(C.T, Xp1)), axis = 0)
            Xteste = concatenate((Xtest, ones([Xtest.shape[0], 1])), axis = 1)
            ypred = dot(Xteste, coef)
            yp[:, j-1:j] = ypred

        YR[index_Xtest, :] = yp
        yytest[index_Xtest, :] = ytest
        print("The %sth group finished" %i )

    error =YR - tile(y, A)
    errs = error * error
    PRESS = np.sum(errs, axis=0)
    RMSECV_ALL = np.sqrt(PRESS/M)
    index_A = np.nonzero(RMSECV_ALL == min(RMSECV_ALL))
    RMSECV_MIN = min(RMSECV_ALL)
    SST = np.sum((yytest - mean(y))**2)
    Q2_all = 1-PRESS/SST
    return {'index_A': index_A[0] + 1, 'RMSECV_ALL': RMSECV_ALL, 'Q2_all': Q2_all}

def pls1_nipals(X, y, a):
    T = zeros((X.shape[0], a))
    P = zeros((X.shape[1], a))
    Q = zeros((1, a))
    W = zeros((X.shape[1], a))
    for i in range(a):
        v = dot(X.T, y[:, 0])
        
        W[:, i] = v/norm(v)
        T[:, i] = dot(X, W[:, i])
        P[:, i] = dot(X.T, T[:, i])/dot(T[:, i].T, T[:, i])
        Q[0, i] = dot(T[:, i].T, y[:, 0])/dot(T[:, i].T, T[:, i])
        X = X-outer(T[:, i], P[:, i])
    W = dot(W, inv(dot(P.T, W)))
    B = dot(W[:, 0:a], Q[:, 0:a].T)
    return {'B': B, 'T': T, 'P': P, 'Q': Q, 'W': W}

def plspredtest(B, Xtest, xp1, xp2, yp1, yp2):
    C = dot(B, yp2) / xp2
    coef = concatenate((C, yp1-dot(C.T, xp1)), axis = 0)
    Xteste = concatenate((Xtest, ones([Xtest.shape[0], 1])), axis = 1)
    ypred = dot(Xteste, coef)
    return ypred

def RMSEP(ypred, Ytest):
    error = ypred - Ytest
    errs = error ** 2
    PRESS = np.sum(errs)
    RMSEP = np.sqrt(PRESS/Ytest.shape[0])
    SST = np.sum((Ytest - np.mean(Ytest))**2)
    Q2 = 1-PRESS/SST
    return RMSEP, Q2

    
def error(pre, signal):
    err = pre - signal
    err = err * err
    print ("sum of err^2", err.sum())
    return err.sum()


def pretreat(X):
    [M, N] = X.shape
    p1 =np.mean(X, axis=0).reshape(N, 1)
    p2 = np.ones([N, 1])
    Xs = np.zeros([M, N])
    for i in range(0, N):
        Xs[:, i:i+1] = ((X[:, i:i+1] - p1[i])/p2[i])
    return Xs, p1, p2

def cca(X, Y):
    z = concatenate((X,Y), axis = 0)  
    C = cov(z)
    Wx = zeros([9,9])
    sx = X.shape[0]
    sy = Y.shape[0]
    Cxx = C[0:sx, 0:sx] + 0.00000001*eye(sx)
    Cxy = C[0:sx, sx:sx+sy]
    Cyx = Cxy.T
    Cyy = C[sx:sx+sy, sx:sx+sy] + 0.00000001*eye(sy);
    a=dot(dot(dot(inv(Cxx),Cxy),inv(Cyy)),Cyx)
    r, Wxx = LA.eig(a)      
    r = sqrt(real(r))
    V = fliplr(Wxx)  
    r = flipud(r)	
    I = np.argsort(real(r))
    r = np.sort(r)    
    r = flipud(r)
    for j in range(len(I)):
      Wx[:,j] = V[:,I[j]] 
    Wx = fliplr(Wx)	
    Wy = dot(dot(inv(Cyy),Cyx),Wx)
    Wy = Wy/tile(sqrt(sum(abs(Wy)**2)),(sy,1))
    return Wx, Wy, r
    
def ctcca(TM,TS,Ttest):
     
    TM = TM.T; TS = TS.T
    [Wx, Wy, corr] = cca(TM,TS)  
    TM = TM.T; TS = TS.T
    Lm = dot(TM,Wx)
    Ls = dot(TS,Wy)
  
    F1 = dot(dot( inv(dot(Ls.T, Ls)), Ls.T), Lm)
    F2 = dot(dot( inv(dot(Lm.T, Lm)), Lm.T), TM)
   
    T_trans = dot(dot(dot(Ttest,Wy),F1),F2)

    return T_trans

def PC_PCA(SXtrain,SXtest):
    [u, s, v]=LA.svd(SXtrain,full_matrices=False) 
    sumvar=sum(s)
    sum1=0
    for i in range(SXtrain1.shape[0]):
      if sum1<=0.999*sumvar:
        sum1=sum1+s[i]
      else:
        break
    S = diag(s)
    TS = dot(u[:,0:i+1],S[0:i+1,0:i+1])    
    Ttest = dot(SXtest,(v[0:i+1,:]).T)
   
    return TS,Ttest

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    datafile =u'C:/Users/admin/Desktop/PC-CCA master-python/CORN.mat'
    dataset = loadmat(datafile)
    XS = dataset['X_M5']; XM = dataset['X_MP5'];Y = dataset['Y']

    CalInd, ValInd = KennardStone(XM, 64)
    MXcal = XM[CalInd]; MXtest = XM[ValInd]
    SXcal = XS[CalInd]; SXtest = XS[ValInd]
    Ycal = Y[CalInd]; Ytest = Y[ValInd]

#################PLS model###########################
    CV = plscvfold(MXcal, Ycal, 15, 10)
    index_A, RMSECV_ALL = CV['index_A'], CV['RMSECV_ALL']

    [mX, xp1, xp2] = pretreat(MXcal)
    [my, yp1, yp2] = pretreat(Ycal)
    
    PLS = pls1_nipals(mX, my, index_A)
    coef = PLS['B']; P =  PLS['P']; W =  PLS['W']

    MXtestypred = plspredtest(coef, MXtest, xp1, xp2, yp1, yp2)
    SXtestypred = plspredtest(coef, SXtest, xp1, xp2, yp1, yp2)
    RMSEP_M, Q2_M = RMSEP(MXtestypred, Ytest)
   
################# Recalibration ###########################
#    CV2 = plscvfold(SXcal, Ycal, 15, 10)
#    index_A2, RMSECV_ALL2 = CV2['index_A'], CV2['RMSECV_ALL']
#
#    [sX, sxp1, sxp2] = pretreat(SXcal)
#    [sy, syp1, syp2] = pretreat(Ycal)
#    PLS2 = pls1_nipals(sX, sy, index_A2)
#    coef2 = PLS2['B']
#    recal_ypred = plspredtest(coef2, SXtest, sxp1, sxp2, syp1, syp2)
#    RMSEP_recal, Q2_recal = RMSEP(recal_ypred, Ytest)    
   
#################calibration transfer###########################
    CalInd1, ValInd1 = KennardStone(MXcal, 40)
    MXtrain = MXcal[CalInd1]; SXtrain = SXcal[CalInd1]; Ytrain = Ycal[CalInd1]

    MXtrain1 = MXtrain - np.mean( MXtrain,axis=0)
    SXtrain1 = SXtrain - np.mean( SXtrain,axis=0)
    SXtest1 = SXtest - tile(np.mean( SXtrain,axis=0),(SXtest.shape[0],1))
   
    T_direction =dot(W,inv(dot(P.T,W)))      
    TM = dot(MXtrain1, T_direction)
    [TS, Ttest] = PC_PCA (SXtrain1,SXtest1)

    T_trans = ctcca(TM,TS,Ttest)    
    SX_trans = dot(T_trans,P.T)
    xm = SX_trans + tile(np.mean( MXtrain,axis=0),(MXtest.shape[0],1))
    err = error(xm, MXtest)

################ PLS prediction ###########################
    trans_ypred = plspredtest(coef, xm, xp1, xp2, yp1, yp2)
    RMSEP_trans, Q2_trans = RMSEP(trans_ypred, Ytest)  
    
####################### Output ###########################    
    
    x = np.arange(9, 13)
    y = np.arange(9, 13)

    plt.figure()
    plt.plot(x,y)
    plt.plot(Ytest, MXtestypred, 'ro', label="MXtestypred" )
    plt.plot(Ytest, SXtestypred, 'b+', label="SXtestypred")
    plt.plot(Ytest, trans_ypred, 'g^', label="trans_ypred")
    plt.xlabel("Reference values")
    plt.ylabel("predicted values")
    plt.legend()

    wavelength = np.arange(1100, 2500, 2)
    diff = xm-MXtest
    diff2 = SXtest-MXtest
    plt.figure()
    plt.subplot(211)
    plt.plot(wavelength, diff.T)
    plt.title('PC-CCA-SXtestypred-MXtest')
    plt.axis([1100, 2500, -0.08, 0.08])
    plt.subplot(212)
    plt.plot(wavelength, diff2.T)
    plt.title('SXtest-MXtest')
    plt.axis([1100, 2500, -0.08, 0.08])
    plt.show()
    
    print( "RMSEP_M:", RMSEP_M, "Q2_mp5:", Q2_M)
#   print( "RMSEP_recal:", RMSEP_recal, "Q2_recal:", Q2_recal)
    print( "RMSEP_trans:", RMSEP_trans, "Q2_trans:", Q2_trans)
    print( "sum_err:", err)
      
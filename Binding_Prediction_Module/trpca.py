import numpy as np
from numpy import linalg as LA
from numpy.linalg import multi_dot
#import cv2
import sys
import pickle


def prox_tnn(Y, rho):
    #print(Y.shape)
    n1, n2, n3 = Y.shape
    X = np.zeros((n1, n2, n3), dtype=complex)
    Y = np.fft.fft(Y)
    tnn = 0
    trank = 0
    U, S, V = np.linalg.svd(Y[:, :, 0], full_matrices=False)
    r = np.sum(S > rho)
    if r >= 1:
        S = S[:r] - rho
        X[:, :, 0] = multi_dot([U[:, :r], np.diag(S), V[:r, :]])
        tnn += np.sum(S)
        trank = max(trank, r)
    halfn3 = round(n3/2)
    for i in range(1, halfn3):
        U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
        r = np.sum(S > rho)
        if r >= 1:
            S = S[:r] - rho
            X[:, :, i] = multi_dot([U[:, :r], np.diag(S), V[:r, :]])
            tnn += np.sum(S)*2
            trank = max(trank, r)
        X[:, :, n3 - i] = np.conjugate(X[:, :, i])
    if n3 % 2 == 0:
        i = halfn3
        U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
        r = np.sum(S > rho)
        if r >= 1:
            S = S[:r] - rho
            X[:, :, i] = multi_dot([U[:, :r], np.diag(S), V[:r, :]])
            tnn += np.sum(S)
            trank = max(trank, r)
    tnn /= 3
    X = np.fft.ifft(X)
    return X, tnn, trank


def prox_l1(b, plambda):
    return np.maximum(0, b - plambda) + np.minimum(0, b + plambda)


def trpca(X, plambda, tol=1e-8, max_iter=500, rho=1.1, mu=1e-4, max_mu=1e10):
    L = np.zeros(X.shape)
    S = L
    Y = L
    for i in range(max_iter):
        Lk = L
        Sk = S
        L, tnnL, _ = prox_tnn(-S + X - Y / mu, 1 / mu)
        S = prox_l1(-L + X - Y / mu, plambda / mu)
        dY = L + S - X
        chgL = np.max(abs(Lk - L))
        chgS = np.max(abs(Sk - S))
        chg = max(chgL, chgS, np.max(abs(dY)))
        if chg < tol:
            break
        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)
        obj = tnnL + plambda * LA.norm(S.flatten(), ord=1)
        err = LA.norm(dY.flatten(), ord=2)
    print("Iter: %d/%d     Err: %.4f" % (i + 1, max_iter, err))

    return L, S, obj, err, i



# M always means matrix
import numpy as np
import pandas as pd
import os
import numpy.linalg as la
import copy
from itertools import permutations


def RMSE(pred, truth):
    sample_size = len(truth)
    fn = la.norm(np.asarray(np.asarray(pred).squeeze() - np.asarray(truth).squeeze()), 2) / np.sqrt(
        sample_size)  # RMSPE from Hongtu ZHU
    return fn


def RMSE_C(C_hat, C):
    # estimation error in statistics
    if type(C_hat) is list and type(C) is list:
        d1, d2 = C[0].shape
        res = np.mean([la.norm(C_hat[i]-C[i], 2) / np.sqrt(d1*d2) for i in range(len(C))])
    elif type(C_hat) is list and type(C) != list:
        d1, d2 = C.shape
        res = np.mean([la.norm(C_hat[i]-C, 2) / np.sqrt(d1*d2) for i in range(len(C_hat))])
    elif type(C_hat) != list and type(C) is list:
        d1, d2 = C_hat.shape
        res = np.mean([la.norm(C_hat - C[i], 2) / np.sqrt(d1 * d2) for i in range(len(C))])
    else:
        d1, d2 = C.shape
        res = la.norm(C_hat - C, 2) / np.sqrt(d1 * d2)
    return res


# def error(beta_hat,beta, th = 0.3):
# 	## th is threshold
# 	## return typeI error and Power
# 	m,n = beta.shape
# 	beta_hat = np.asarray(beta_hat)
# 	tmp = np.where(np.abs(beta_hat) > th,beta_hat,0)
# 	tmp = np.where(np.abs(tmp) < th,tmp,1)

# 	diff = beta - tmp
# 	mul = beta * tmp
# 	typeI = np.where(diff == -1)[0].shape[0]/np.where(beta == 0)[0].shape[0]
# 	Power = np.where(mul == 1)[0].shape[0]/np.where(beta == 1)[0].shape[0]
# 	return typeI,Power


def error(C_hat, C, th=0):
    C_hat_ind = np.where(np.abs(C_hat) > th, 1, 0)
    C_ind = np.where(np.abs(C) != 0, 1, 0)
    if type(C_hat) is list and type(C) != list:
        n = len(C_hat)
        typeI = np.mean([np.sum(C_hat_ind[i] * (1 - C_ind)) / np.sum(1 - C_ind) for i in range(n)])
        Power = np.mean([np.sum(C_hat_ind[i] * C_ind) / np.sum(C_ind) for i in range(n)])
    elif type(C_hat) != list and type(C) is list:
        n = len(C)
        typeI = np.mean([np.sum(C_hat_ind * (1 - C_ind[i])) / np.sum(1 - C_ind[i]) for i in range(n)])
        Power = np.mean([np.sum(C_hat_ind * C_ind[i]) / np.sum(C_ind[i]) for i in range(n)])
    else:
        typeI = np.sum(C_hat_ind * (1 - C_ind)) / np.sum(1 - C_ind)
        Power = np.sum(C_hat_ind * C_ind) / np.sum(C_ind)
    # non-threshold version
    # return typeI error and Power
    return typeI, Power


def fun_block_diff(B):
    # input B is a matrix
    # first = copy.deepcopy(B[:,0])
    # tmp_B = copy.deepcopy(B)
    # tmp_B[:,:-1] = B[:,1:]
    # tmp_B[:,-1] = first
    m, n = np.shape(B)
    mean_block = np.mean(B, axis=1).squeeze()
    tmp = np.tile(mean_block, n).reshape(n, m).T
    return la.norm(B - tmp, 2, axis=0)


def decom_error(C_hat, C_true, th):
    TPR, FPR = error(C_hat, C_true, th=th)
    rmse_c = RMSE_C(C_hat, C_true)
    print("For C | RMSE: %.5f" % rmse_c, end=' | ')
    print("TPR: %.10f" % (TPR * 100), end=' | ')
    print("FPR: %.10f" % (FPR * 100))
    return TPR, FPR, rmse_c


def pred_error(Y_hat, Y_true, trY_hat=None, trY_true=None, insmp=False):
    rmse_y = RMSE(Y_hat, Y_true)
    if insmp:
        trrmse_y = RMSE(trY_hat, trY_true)
        print("For Y | Train RMSE: {:.5f}, Test RMSE: {:.5f}".format(trrmse_y, rmse_y))
        return trrmse_y, rmse_y
    else:
        print("RMSE for Y: %.5f" % rmse_y)
        return rmse_y


def Acc_cls(y, y_hat):
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    n = y.shape[0]
    freq = np.sum(y == y_hat) / n
    return max(freq, 1 - freq)


def Acc_pmtt(y, y_hat):
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    n = y.shape[0]
    labels = np.unique(y)
    k = labels.shape[0]
    idx_lst = []
    for l in labels:
        idx_lst.append(np.where(y_hat == l)[0])
    perms = list(permutations(range(k)))
    y_perm_lst = []
    Acc_lst = []
    for p in perms:
        y_perm = np.zeros(n)
        for l in labels:
            y_perm[idx_lst[l]] = p[l]
        y_perm_lst.append(y_perm)
        Acc_lst.append(np.sum(y_perm == y))
    return max(Acc_lst) / n


def Acc_class(y, y_hat, bound=0.5):
    pred = (y_hat > bound) + 0
    y = y.reshape(-1)
    pred = pred.reshape(-1)
    return np.sum(y == pred) / len(y)


def my_log_loss(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(mask(y_pred))
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss.mean()


def mask(y, eps=1e-15):
    idx_0 = np.where(np.abs(y - 0) < eps)[0]
    y[idx_0] = y[idx_0] + eps
    idx_0 = np.where(np.abs(1 - y) < eps)[0]
    y[idx_0] = y[idx_0] - eps
    return y

import numpy as np
from operators import *
from criterion import my_log_loss, Acc_class
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, log_loss
from scipy.special import expit
import time


class DKN_T():
    def __init__(self, blockshape_list, task='reg', R=2):
        self.task = task
        self.R = R
        self.L = len(blockshape_list)
        self.blockshapes = np.array(blockshape_list)
        bs_ext = blockshape_list.copy()
        bs_ext.insert(0, [1, 1, 1])
        bs_ext.append([1, 1, 1])
        self.bs_ext = np.array(bs_ext)
        self.B_list = None
        self.C_hat = None
        self.b2l_list = None
        self.bl2_list = None
        self.from_init = True

    def Kron_k(self, mat_list, reverse=True):
        Ms = mat_list.copy()
        if reverse:
            Ms.reverse()
        prod = 1
        for m in Ms:
            prod = np.kron(prod, m)
        return prod

    def TXl(self, X, b_left, b_right, l):
        shape_w_1 = np.prod(self.bs_ext[:l+1, :], axis=0)
        shape_w_l = self.bs_ext[l, :]
        shape_w_L = np.prod(self.bs_ext[l+1:, :], axis=0)

        R1 = Rearrange_T_pro(X, shape_w_L)
        M1 = Vec_inv(b_left.T.dot(R1), shape_w_1)
        X_tilde = Rearrange_T_pro(M1, shape_w_l).dot(b_right)
        return X_tilde.reshape(-1)

    def Xbar(self, X, b_left, b_right, l):
        R = b_left.shape[1]
        TX_list = []
        for r in range(R):
            TX = self.TXl(X, b_left[:, r], b_right[:, r], l)
            TX_list.append(TX)
        BX_mat = np.array(TX_list)
        return BX_mat

    def Init(self, X, Y, R, right=False):
        Y = Y.reshape(-1)
        b2l_list = [np.array([])] * self.L  # from b:2 to b:(L+1), and b:(L+1) = (1)
        if R > np.prod(self.bs_ext[-2, :], axis=0):
            raise ValueError('By SVD initialization, R cannot be greater than the number of elements of last factor.')
        for l in range(2, self.L + 1):
            shape_w_L = np.prod(self.bs_ext[l:, :], axis=0)
            M = np.sum([Y[i] * Rearrange_T_pro(X[i], shape_w_L) for i in range(len(Y))], axis=0)
            U, S, Vt = np.linalg.svd(M)
            b_l = U[:, :R]
            b2l_list[l - 2] = b_l
        b2l_list[-1] = np.ones([1, R])

        bl2_list = [np.array([])] * self.L  # from b0: to b(L-1):, and b0: = (1)
        bl2_list[0] = np.ones([1, R])
        if right:
            for l in range(1, self.L):
                shape_w_L = np.prod(self.bs_ext[l + 1:, :], axis=0)
                M = np.sum([Y[i] * Rearrange_T_pro(X[i], shape_w_L) for i in range(len(Y))], axis=0)
                U, S, Vt = np.linalg.svd(M)
                b_l = Vt.T[:, :R]
                bl2_list[l] = b_l

        return b2l_list, bl2_list

    def Init_random(self, R, right=False, seed=0):
        np.random.seed(seed)
        b2l_list = [np.array([])] * self.L  # from b:2 to b:(L+1), and b:(L+1) = (1)
        for l in range(2, self.L + 1):
            d_L, p_L = np.prod(self.bs_ext[l:, :], axis=0)
            b_l = np.random.normal(size=(d_L*p_L, R))
            b2l_list[l - 2] = b_l
        b2l_list[-1] = np.ones([1, R])

        bl2_list = [np.array([])] * self.L  # from b0: to b(L-1):, and b0: = (1)
        bl2_list[0] = np.ones([1, R])
        if right:
            for l in range(1, self.L):
                d_L, p_L = np.prod(self.bs_ext[l + 1:, :], axis=0)
                b_l = np.random.normal((d_L*p_L, R))
                bl2_list[l] = b_l

        return b2l_list, bl2_list

    def fit(self, train, valid, init='svd', alpha=0, max_itr=10, tol=1e-5, timing=False):
        print('Start at', time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())) if timing else None
        t0 = time.time()

        if self.task == 'reg':
            model = LinearRegression(fit_intercept=False) if alpha == 0 else Ridge(alpha=alpha, fit_intercept=False)
        elif self.task == 'cla':
            model = LogisticRegression(fit_intercept=False) if alpha == 0 else LogisticRegression(fit_intercept=False, C=alpha)
        else:
            raise
        X, Y = train
        X_val, Y_val = valid
        Y = Y.reshape(-1)
        b_list = [np.array([])] * self.L
        b_list.insert(0, np.array([[[1]]] * self.R))
        b_list.append(np.array([[[1]]] * self.R))  # length of b_list: L+2, from 0 to L+1.
        if self.from_init:
            print('Initializing (' + init + ')' + '...', end='\r')
            if init == 'svd':
                b2l_list, bl2_list = self.Init(X, Y, self.R, right=False)
            else:
                b2l_list, bl2_list = self.Init_random(self.R, right=False)
            print('Initialization Finished. Start Computing...')
        else:
            b2l_list, bl2_list = self.b2l_list, self.bl2_list

        loss_former = 1e10
        loss_train_list, loss_valid_list = [], []
        acc_train_list, acc_valid_list = [], [] if self.task == 'cla' else None
        for itr in range(max_itr):
            for l in range(1, self.L + 1):
                b_left, b_right = b2l_list[l - 1], bl2_list[l - 1]
                M = np.array([Vec(self.Xbar(Xi, b_left, b_right, l)).reshape(-1) for Xi in X])
                bl = model.fit(M, Y).coef_.reshape(-1, 1)
                b_list[l] = bl.reshape(-1, self.R, order='F')
                if l < self.L:
                    bl2_r_list = []
                    for r in range(self.R):
                        Bl_r = Vec_inv(b_list[l][:, r], shape=self.bs_ext[l])
                        Blm12_r = Vec_inv(bl2_list[l - 1][:, r], shape=np.prod(self.bs_ext[:l], axis=0))
                        bl2_r = Vec(np.kron(Bl_r, Blm12_r)).reshape(-1)
                        bl2_r_list.append(bl2_r)
                    bl2_list[l] = np.array(bl2_r_list).T
            # update all b2l
            for j in range(self.L, 1, -1):  # j: L to 2
                b2l_r_list = []
                for r in range(self.R):
                    B2lp1 = Vec_inv(b2l_list[j - 1][:, r], shape=np.prod(self.bs_ext[j+1:], axis=0))
                    Bl = Vec_inv(b_list[j][:, r], shape=self.bs_ext[j])
                    b2l_r = Vec(np.kron(B2lp1, Bl)).reshape(-1)
                    b2l_r_list.append(b2l_r)
                b2l_list[j-2] = np.array(b2l_r_list).T

            ''' Calculate error '''
            C_hat_list = []
            for r in range(self.R):
                B_list_r = [Vec_inv(b_list[i + 1][:, r], self.bs_ext[i + 1]) for i in range(self.L)]
                C_hat_r = self.Kron_k(B_list_r)
                C_hat_list.append(C_hat_r)
            C_hat = np.array(C_hat_list).sum(axis=0)
            if self.task == 'reg':
                Y_hat = model.predict(M)
                loss_new = mean_squared_error(Y, Y_hat, squared=False)
                loss_train_list.append(loss_new)
            if self.task == 'cla':
                Y_hat = model.predict_proba(M)[:, 1]
                loss_new = my_log_loss(Y, Y_hat)
                acc_train = Acc_class(Y, Y_hat)
                loss_train_list.append(loss_new)
                acc_train_list.append(acc_train)
            change_rate = np.abs(loss_former - loss_new) / loss_former
            loss_former = loss_new

            # Validation Error
            Y_val_hat = np.array([np.sum(np.vdot(Xi, C_hat)) for Xi in X_val])
            if self.task == 'reg':
                loss_val = mean_squared_error(Y_val, Y_val_hat, squared=False)
                loss_valid_list.append(loss_val)
            if self.task == 'cla':
                Y_val_hat = expit(Y_val_hat)
                loss_val = my_log_loss(Y_val, Y_val_hat)
                acc_val = Acc_class(Y_val, Y_val_hat)
                loss_valid_list.append(loss_val)
                acc_valid_list.append(acc_val)

            if itr % 1 == 0 or itr == max_itr - 1:
                if self.task == 'reg':
                    print('Iteration: {} | Loss | Train: {:.4f}, Valid: {:.4f}'.format(itr, loss_new, loss_val), end='\r')
                if self.task == 'cla':
                    print('Iteration: {} | Loss | Train: {:.4f}, Valid: {:.4f}'.format(itr, loss_new, loss_val), end=' | ')
                    print('Accuracy | Train: {:.4f}, Valid: {:.4f}'.format(acc_train, acc_val), end='\r')
                    # print('Loss | Train: {:.4f}, Valid: {:.4f}'.format(itr, loss_new, loss_val), end=' | ')
                    # print('Accuracy | Train: {:.4f}, Valid: {:.4f}'.format(acc_train, acc_val))
            # Verify early-stopping criterion
            if change_rate < tol:
                print('Training loss converges at iteration {}.'.format(itr+1))
                break
            if loss_val < 10 * tol:
                print('Validation loss vanishes at iteration {}.'.format(itr+1))
                break

        B_list = []
        for l in range(self.L):
            B_r_list = []
            for r in range(self.R):
                B_l_r = Vec_inv(b_list[l + 1][:, r], self.bs_ext[l + 1])
                B_r_list.append(B_l_r)
            B_list.append(B_r_list)
        C_hat_list = []
        for r in range(self.R):
            B_list_r = [B_list[l][r] for l in range(self.L)]
            C_hat_r = Kron_k(B_list_r)
            C_hat_list.append(C_hat_r)
        C_hat = np.array(C_hat_list).sum(axis=0)
        self.B_list = B_list
        self.C_hat = C_hat
        self.b2l_list = b2l_list
        self.bl2_list = bl2_list
        self.from_init = False

        print('End at', time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())) if timing else None
        t1 = time.time()
        print(f'Cost time {round(t1 - t0)}s.') if timing else None

        if self.task == 'reg':
            return self.C_hat, [loss_train_list, loss_valid_list]
        if self.task == 'cla':
            return self.C_hat, [loss_train_list, loss_valid_list], [acc_train_list, acc_valid_list]

    def predict(self, X):
        Y_hat = np.array([np.sum(np.vdot(Xi, self.C_hat)) for Xi in X]).reshape(-1, 1)
        if self.task == 'cla':
            Y_hat = expit(Y_hat)

        return Y_hat


def summary_error(C, train, test, model):
    X_train, Y_train = train
    X_test, Y_test = test
    C_hat = model.C_hat
    if C is None:
        pass
    else:
        rmse = mean_squared_error(C.ravel(), C_hat.ravel(), squared=False)
        print('For C | RMSE: {:.4f}'.format(rmse))
    # Prediction Error
    # Train
    Y_inhat = model.predict(X_train)
    Train_RMSE = mean_squared_error(Y_train, Y_inhat, squared=False)
    # Test
    Y_outhat = model.predict(X_test)
    Test_RMSE = mean_squared_error(Y_test, Y_outhat, squared=False)
    print('For Y | Train RMSE: {:.4f}, Test RMSE: {:.4f}'.format(Train_RMSE, Test_RMSE))
    if C is None:
        return Train_RMSE, Test_RMSE
    else:
        return rmse, Train_RMSE, Test_RMSE


if __name__ == '__main__':
    rng = np.random.RandomState(666)
    ''' Definition '''
    X = rng.normal(size=[8, 8, 8])
    B1 = rng.normal(size=[2, 2, 2])
    B2 = rng.normal(size=[2, 2, 2])
    B3 = rng.normal(size=[2, 2, 2])
    C = Kron_k([B1, B2, B3])
    y = np.sum(X[:] * C[:])

    ''' Rearrange '''
    b1, b2, b3 = Vec(B1), Vec(B2), Vec(B3)
    y1 = b2.T.dot(Rearrange_T(Vec_inv(b3.T.dot(Rearrange_T(X, [2, 2, 2])), [4, 4, 4]), [2, 2, 2])).dot(b1)
    print(y, y1.ravel())

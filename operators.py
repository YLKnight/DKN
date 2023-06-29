''' Operators w.r.t. the model '''
import numpy as np
import cv2


def Vec(M):
    return M.reshape(-1, 1, order='C')


def Vec_inv(M, shape):
    return M.reshape(shape, order='C')


''' R operator for matrix '''


def R_opt(M, idctshape):
    m, n = M.shape
    p1, p2 = idctshape
    assert m % p1 == 0 and n % p2 == 0, "Dimension wrong"
    d1, d2 = m // p1, n // p2
    RM = []
    for i in range(p1):
        for j in range(p2):
            Mij = M[d1*i: d1*(i + 1), d2*j: d2*(j + 1)]
            RM.append(Vec(Mij))
    return np.concatenate(RM, axis=1).T


def R_opt_pro(A, idctshape):
    m, n = A.shape
    p1, p2 = idctshape
    assert m % p1 == 0 and n % p2 == 0, "Dimension wrong"
    d1, d2 = m // p1, n // p2
    strides = A.itemsize * np.array([p2*d2*d1, d2, p2*d2, 1])
    A_blocked = np.lib.stride_tricks.as_strided(A, shape=(p1, p2, d1, d2), strides=strides)
    RA = A_blocked.reshape(-1, d1*d2)
    return RA


def R_inv(RM, blockshape, idctshape):
    m, n = RM.shape
    d1, d2 = blockshape
    p1, p2 = idctshape
    assert m == p1 * p2 and n == d1 * d2, "Dimension wrong"
    M = np.zeros([d1 * p1, d2 * p2])
    for i in range(m):
        Block = Vec_inv(RM[i, :], blockshape)
        ith = i // p2  # quotient
        jth = i % p2  # remainder
        M[d1*ith: d1*(ith+1), d2*jth: d2*(jth+1)] = Block

    return M


''' R operator for tensor '''


def Rearrange_T(T, Adim):
    ''' R operator for tensor: (p1*d1, p2*d2, p3*d3) to (p1*p2*p3, d1*d2*d3) '''
    N1, N2, N3 = T.shape
    p1, p2, p3 = Adim
    assert N1 % p1 == 0 and N2 % p2 == 0 and N3 % p3 == 0, "Dimension wrong"
    d1, d2, d3 = N1 // p1, N2 // p2, N3 // p3
    RC = []
    for i in range(p1):
        for j in range(p2):
            for k in range(p3):
                Tij = T[d1 * i:d1 * (i + 1), d2 * j:d2 * (j + 1), d3 * k:d3 * (k + 1)]
                RC.append(Vec(Tij))
    return np.concatenate(RC, axis=1).T


def Rearrange_T_pro(T, Adim):
    ''' R operator for tensor: (p1*d1, p2*d2, p3*d3) to (p1*p2*p3, d1*d2*d3) '''
    N1, N2, N3 = T.shape
    p1, p2, p3 = Adim
    assert N1 % p1 == 0 and N2 % p2 == 0 and N3 % p3 == 0, "Dimension wrong"
    d1, d2, d3 = N1 // p1, N2 // p2, N3 // p3
    strides = T.itemsize * np.array([N2 * N3 * d1, N3 * d2, d3, N2 * N3, N3, 1])  # 大层，大行，大列，小层，小行，小列
    T_blocked = np.lib.stride_tricks.as_strided(T, shape=(p1, p2, p3, d1, d2, d3), strides=strides)
    RT = T_blocked.reshape(-1, d1 * d2 * d3)
    return RT


def R_inv_T(RT, Adim, Bdim):
    ''' Inverse R operator for tensor: (p1*p2*p3, d1*d2*d3) to (p1*d1, p2*d2, p3*d3) '''
    P, D = RT.shape
    p1, p2, p3 = Adim
    d1, d2, d3 = Bdim
    assert P == p1 * p2 * p3 and D == d1 * d2 * d3, 'Dimension wrong!'
    slices = []
    fibers = []
    for i in range(P):
        fiber = RT[i, ].reshape(Bdim)
        fibers.append(fiber)
        if len(fibers) == p2:
            slice = np.concatenate(fibers, axis=1)
            slices.append(slice)
            fibers = []
    T = np.concatenate(slices, axis=0)
    return T


def Kron_k(mat_list, reverse=True):
    Ms = mat_list.copy()
    if reverse:
        Ms.reverse()
    prod = 1
    for m in Ms:
        prod = np.kron(prod, m)
    return prod


''' (Inv-) One-hot function of the labels '''


def One_hot(Y, cate=2):
    Y_hot = np.zeros((len(Y), cate))
    for i in range(cate):
        Y_hot[np.where(Y == i)[0], i] = 1
    return Y_hot


def Num(Y_hot, cate=2):
    Y = np.zeros((len(Y_hot), 1))
    Y[np.where(Y_hot[:, 0] < Y_hot[:, 1])[0]] = 1
    return Y


''' Resize the sample '''


def Resize(data, shape):
    new = []
    shape = list(shape)
    shape.reverse()
    for pic in data:
        pic_new = cv2.resize(pic, shape)
        new.append(pic_new)
    return np.array(new)


''' Make the signal quasi-sparse '''


# If the signal is gaussian
def Gaussian_lize(C, mu=1, sigma=1, seed=666):
    rng = np.random.RandomState(seed)
    idx = np.where(C != 0)
    C[idx] = rng.normal(mu, sigma, len(idx[0]))
    return C


# If the pixels around signal are gaussian
def Gaussian_round(C, mu=0, sigma=1, seed=666):
    rng = np.random.RandomState(seed)
    idx = np.where(C == 0)
    C[idx] = rng.normal(mu, sigma, len(idx[0]))
    return C


if __name__ == '__main__':

    rng = np.random.RandomState(666)
    ''' Definition '''
    X = rng.normal(size=[6, 6, 6])
    A = rng.normal(size=[3, 3, 3])
    B = rng.normal(size=[2, 2, 2])
    C = np.kron(A, B)
    y = np.sum(X[:] * C[:])

    print(y)

    ''' Rearrange '''
    RX = Rearrange_T(X, A.shape)
    RC = Rearrange_T(C, A.shape)
    a, b = Vec(A), Vec(B)
    abT = a.dot(b.T)
    y1 = np.sum(RX[:] * abT)
    y2 = a.T.dot(RX).dot(b)

    ''' Pro '''
    # shape = (64, 48)
    # X = np.arange(1, np.prod(shape)+1).reshape(shape)
    # Adim = [2, 3]
    #
    # RX = R_opt(X, Adim)
    # RX_pro = R_opt_pro(X, Adim)
    # print(np.sum(RX != RX_pro))

    # shape = (48, 60, 48)
    # X = np.arange(1, np.prod(shape)+1).reshape(shape)
    # Adim = [12, 12, 12]
    #
    # N1, N2, N3 = shape
    # p1, p2, p3 = Adim
    # d1, d2, d3 = N1 // p1, N2 // p2, N3 // p3
    #
    # RX = Rearrange_T(X, Adim)
    # RX_pro = Rearrange_T_pro(X, Adim)
    # print(np.sum(RX != RX_pro))

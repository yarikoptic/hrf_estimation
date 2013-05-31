import numpy as np
from scipy import linalg, optimize
from scipy.sparse import linalg as splinalg

def khatri_rao(A, B):
    """
    Compute the Khatri-rao product, where the partition is taken to be
    the vectors along axis one.

    This is a helper function for rank_one

    Parameters
    ----------
    A : array, shape (n, p)
    B : array, shape (m, p)
    AB : array, shape (nm, p), optimal
        if given, result will be stored here

    Returns
    -------
    a*b : array, shape (nm, p)
    """
    num_targets = A.shape[1]
    assert B.shape[1] == num_targets
    return (A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
    ).reshape(num_targets, len(B) * len(A)).T

def matmat2(X, a, b, n_task):
    """
    X (b * a)
    """
    uv0 = khatri_rao(b, a)
    return X.matvec(uv0)


def rmatmat1(X, a, b, n_task):
    """
    (I kron a^T) X^T b
    """
    b1 = X.rmatvec(b[:X.shape[0]]).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='F')
    res = np.einsum("ijk, ik -> ij", B, a.T).T
    return res


def rmatmat2(X, a, b, n_task):
    """
    (a^T kron I) X^T b
    """
    b1 = X.rmatvec(b).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='C')
    tmp = np.einsum("ijk, ik -> ij", B, a.T).T
    return tmp


def obj(X_, Y_, Z_, a, b, c, alpha, u0):
    uv0 = khatri_rao(b, a)
    # u0 = u0.reshape((a.size, -1), order='C')
    cost = .5 * linalg.norm(Y_ - X_.matvec(uv0) - Z_.matmat(c), 'fro') ** 2
    reg = .5 * alpha * linalg.norm(a - u0, 'fro') ** 2
    return cost + reg

def f(w, X_, Y_, Z_, size_u, alpha, u0):
    n_task = Y_.shape[1]
    size_v = X_.shape[1] / size_u
    X_ = splinalg.aslinearoperator(X_)
    Z_ = splinalg.aslinearoperator(Z_)
    W = w.reshape((-1, n_task), order='F')
    u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    return obj(X_, Y_, Z_, u, v, c, alpha, u0)


def fprime(w, X_, Y_, Z_, size_u, alpha, u0):
    X_ = splinalg.aslinearoperator(X_)
    Z_ = splinalg.aslinearoperator(Z_)
    size_v = X_.shape[1] / size_u
    # u0 = u0.reshape((size_u, -1), order='C')
    n_task = Y_.shape[1]
    W = w.reshape((-1, n_task), order='F')
    u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    tmp = Y_ - matmat2(X_, u, v, n_task) - Z_.matmat(c)
    grad = np.empty((size_u + size_v + Z_.shape[1], n_task))  # TODO: do outside
    grad[:size_u] = rmatmat1(X_, v, tmp, n_task) - alpha * (u - u0)
    grad[size_u:size_u + size_v] = rmatmat2(X_, u, tmp, n_task)
    grad[size_u + size_v:] = Z_.rmatvec(tmp)
    # print('Gradient ', linalg.norm(grad.ravel(), np.inf))
    return - grad.reshape((-1,), order='F')


def hess(s, x0, X_, Y_, Z_, size_u, alpha, u0):
    # TODO: regularization
    X_ = splinalg.aslinearoperator(X_)
    Z_ = splinalg.aslinearoperator(Z_)
    n_task = Y_.shape[1]
    size_v = X_.shape[1] / size_u
    W = x0.reshape((-1, n_task), order='F')
    S = s.reshape((-1, n_task), order='F')
    XY = X_.rmatvec(Y_)
    u, v, w = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    s1, s2, s3 = S[:size_u], S[size_u:size_u + size_v], S[size_u + size_v:]
    W2 = X_.rmatvec(matmat2(X_, u, v, n_task))

    W2 = W2.reshape((size_v, size_u, n_task), order='C')
    XY = XY.reshape((size_v, size_u, n_task), order='C')
    C = X_.rmatvec(Z_.matvec(w))
    C = C.reshape((size_v, size_u, n_task), order='C')

    tmp = matmat2(X_, s1, v, n_task)
    As1 = rmatmat1(X_, v, tmp, n_task)
    tmp = matmat2(X_, u, s2, n_task)
    Ds2 = rmatmat2(X_, u, tmp, n_task)
    tmp = Z_.matvec(s3)

    Cs3 = rmatmat1(X_, v, tmp, n_task)
    tmp = matmat2(X_, s1, v, n_task).T
    Cts1 = Z_.rmatvec(tmp.T)

    tmp = matmat2(X_, u, s2, n_task)
    W2t = W2.transpose((1, 0, 2))
    XYT = XY.transpose((1, 0, 2))
    CT = C.transpose((1, 0, 2))
    Bs2 = rmatmat1(X_, v, tmp, n_task) + (W2t * s2).sum(1) \
        - (XYT * s2).sum(1) + (CT * s2).sum(1)

    tmp = matmat2(X_, s1, v, n_task)
    Bts1 = rmatmat2(X_, u, tmp, n_task) + (W2 * s1).sum(1) - \
        (XY * s1).sum(1) + (C * s1).sum(1)

    tmp = Z_.matvec(s3)
    Es3 = rmatmat2(X_, u, tmp, n_task)

    tmp = matmat2(X_, u, s2, n_task)
    Ets2 = Z_.rmatvec(tmp)

    Fs3 = Z_.rmatvec(Z_.matvec(s3))

    line0 = As1 + Bs2 + Cs3
    line1 = Bts1 + Ds2 + Es3
    line2 = Cts1 + Ets2 + Fs3

    return np.vstack((line0, line1, line2)).ravel('F')


def rank_one(X, Y, alpha, size_u, u0=None, v0=None, Z=None,
             rtol=1e-6, verbose=False, maxiter=1000):
    X = splinalg.aslinearoperator(X)
    if Z is None:
        # create identity operator
        Z_ = splinalg.LinearOperator(shape=(X.shape[0], 1),
                                     matvec=lambda x: np.zeros((X.shape[0], x.shape[1])),
                                     rmatvec=lambda x: np.zeros((1, x.shape[1])), dtype=np.float)
    else:
        Z_ = splinalg.aslinearoperator(Z)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if u0 is None:
        u0 = np.ones((size_u, n_task))
    if u0.size == size_u:
        u0 = u0.reshape((-1, 1))
        u0 = np.repeat(u0, n_task, axis=1)
    if v0 is None:
        v0 = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    size_v = X.shape[1] / size_u
    u0 = u0.reshape((-1, n_task))
    v0 = v0.reshape((-1, n_task))
    w0 = np.zeros((size_u + size_v + Z_.shape[1], n_task))
    w0[:size_u] = u0
    w0[size_u:size_u + size_v] = v0
    w0 = w0.reshape((-1,), order='F')

#    f(w0.ravel(), X, Y, Z_, size_u, alpha, u0)
#    fprime(w0.ravel(), X, Y, Z_, size_u, alpha, u0)

    def callback(x0):
        x0 = np.asarray(x0)
        print(optimize.check_grad(f, fprime, x0, X, Y, Z_, size_u, 0., u0))
        import numdifftools as nd
        import pylab as pl
        H = nd.Hessian(lambda x: f(x, X, Y, Z_, size_u, 0., u0))
        pl.matshow(H(x0))
        pl.title('numdifftools')
        pl.colorbar()

        n_target = Y.shape[1]
        E = np.eye(n_target * (size_u + size_v + Z_.shape[1]))
        out = []
        for i in range(E.shape[0]):
            ei = E[i]
            ei = ei.reshape((-1, 1))
            tmp = hess(ei, x0, X, Y, Z_, size_u, 0., u0)
            out.append(tmp.ravel())
        true_H = np.array(out)
        pl.matshow(true_H)
        pl.colorbar()
        pl.show()

    def grad_hess(x0, X_, Y_, Z_, size_u, alpha, u0):
        grad = fprime(x0, X_, Y_, Z_, size_u, alpha, u0)
        hessp = lambda x: hess(x, x0, X_, Y_, Z_, size_u, alpha, u0)
        return grad, hessp

#    import pytron
#    res = pytron.minimize(f, grad_hess, w0.ravel(),
#        args=(X, Y, Z_, size_u, alpha, u0), tol=1e-3)

    res = optimize.minimize(f, w0.ravel(), jac=fprime,
            args=(X, Y, Z_, size_u, alpha, u0), tol=1e-3,
            method='TNC', options={'disp' : True},
            callback=callback)

    W = res.x.reshape((-1, Y.shape[1]), order='F')
    U = W[:size_u]
    V = W[size_u:size_u + size_v]
    C = W[size_u + size_v:]

    if Z is None:
        return U, V
    else:
        return U, V, C

# if __name__ == '__main__':
    # n_target = 2
    # np.random.seed(0)
    # X1 = np.random.randn(12, 10)
    # Z1 = np.random.randn(12, 10)
    # Y1 = np.random.randn(12, n_target)
    # size_u = 5
    # size_v = 2
    # canonical = np.random.randn(size_u)
    # x0 = np.random.randn(size_u + size_v + Z1.shape[1],
    #                      Y1.shape[1]).ravel()
    # print(optimize.check_grad(f, fprime, x0, X1, Y1, Z1, size_u, 1.,
    #                           canonical))
    #
    # import numdifftools as nd
    # import pylab as pl
    # H = nd.Hessian(lambda x: f(x, X1, Y1, Z1, size_u, 0., canonical))
    # pl.matshow(H(x0))
    # pl.title('numdifftools')
    # pl.colorbar()
    #
    # E = np.eye(n_target * (size_u + size_v + Z1.shape[1]))
    # out = []
    # for i in range(E.shape[0]):
    #     ei = E[i]
    #     ei = ei.reshape((-1, 1))
    #     tmp = hess(ei, x0, X1, Y1, Z1, size_u, 0., canonical)
    #     out.append(tmp.ravel())
    # true_H = np.array(out)
    # pl.matshow(true_H)
    # pl.colorbar()
    # pl.show()

if __name__ == '__main__':
    size_u, size_v = 9, 4
    from scipy import sparse
    X = sparse.csr_matrix(np.random.randn(100, size_u * size_v))
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .1 * np.random.randn(X.shape[0])
    y = np.array([i * y for i in range(1, 5)]).T

    u, v, w = rank_one(X.A, y, 0., size_u, Z=np.random.randn(X.shape[0], 3),
                       verbose=True, rtol=1e-10)

    # import pylab as plt
    # plt.matshow(B)
    # plt.title('Groud truth')
    # plt.colorbar()
    # plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    # plt.title('Estimated')
    # plt.colorbar()
    # plt.show()
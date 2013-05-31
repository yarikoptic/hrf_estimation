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
    (a^T kron I) X^T b
    """
    b1 = X.rmatvec(b[:X.shape[0]]).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='F')
    res = np.einsum("ijk, ik -> ij", B, a.T).T
    return res


def rmatmat2(X, a, b, n_task):
    """
    (I kron a^T) X^T b
    """
    b1 = X.rmatvec(b).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='C')
    tmp = np.einsum("ijk, ik -> ij", B, a.T).T
    return tmp

def obj(X_, Y_, Z_, a, b, c, alpha, u0):
    uv0 = khatri_rao(b, a)
    cost = .5 * linalg.norm(Y_ - X_.matvec(uv0) - Z_.matmat(c), 'fro') ** 2
    reg = .5 * alpha * linalg.norm(a - u0[:, None], 'fro') ** 2
    return cost + reg

def f(w, X_, Y_, Z_, size_u, alpha, u0):
    n_task = Y_.shape[1]
    size_v = X_.shape[1] / size_u
    X_ = splinalg.aslinearoperator(X_)
    Z_ = splinalg.aslinearoperator(Z_)
    W = w.reshape((-1, n_task), order='F')
    u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    return obj(X_, Y_, Z_, u, v, c, alpha, u0)

def fprime(x0, X_, Y_, Z_, size_u, alpha, u0):
    X_ = splinalg.aslinearoperator(X_)
    Z_ = splinalg.aslinearoperator(Z_)
    n_task = Y_.shape[1]
    size_v = X1.shape[1] / size_u
    W = x0.reshape((-1, n_task), order='F')
    u, v, w = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    tmp = Y_ - matmat2(X_, u, v, n_task) - Z_.matmat(w)
    grad = np.empty((size_u + size_v + Z_.shape[1], n_task))  # TODO: do outside
    grad[:size_u] = rmatmat1(X_, v, tmp, n_task) - alpha * (u - u0[:, None])
    grad[size_u:size_u + size_v] = rmatmat2(X_, u, tmp, n_task)
    grad[size_u + size_v:] = Z_.rmatvec(tmp)
    return - grad.reshape((-1,), order='F')


def hess(s, x0, X_, Y_, Z_, size_u, alpha, u0):
    # TODO: regularization
    X_ = splinalg.aslinearoperator(X_)
    Z_ = splinalg.aslinearoperator(Z_)
    n_task = Y_.shape[1]
    size_v = X1.shape[1] / size_u
    W = x0.reshape((-1, n_task), order='F')
    S = s.reshape((-1, n_task), order='F')
    XY = X_.rmatvec(Y_)
    u, v, w = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    s1, s2, s3 = S[:size_u], S[size_u:size_u + size_v], S[size_u + size_v:]
    W2 = X_.rmatvec(matmat2(X_, u, v, n_task))
    W2 = W2.reshape((n_task, size_v, size_u), order='C')
    XY = XY.reshape((n_task, size_v, size_u), order='C')
    C = X_.rmatvec(Z_.matvec(w))
    C = C.reshape((n_task, size_v, size_u), order='C')

    tmp = matmat2(X_, s1, v, n_task)
    As1 = rmatmat1(X_, v, tmp, n_task)
    tmp = matmat2(X_, u, s2, n_task)
    Ds2 = rmatmat2(X_, u, tmp, n_task)
    tmp = Z_.matvec(s3)

    Cs3 = rmatmat1(X_, v, tmp, n_task)
    tmp = matmat2(X_, s1, v, n_task).T
    Cts1 = Z_.rmatvec(tmp.T)

    tmp = matmat2(X_, u, s2, n_task)
    Bs2 = rmatmat1(X_, v, tmp, n_task) + W2.T.dot(s2) - XY.T.dot(s2) + \
        C.T.dot(s2)

    tmp = matmat2(X_, s1, v, n_task)
    Bts1 = rmatmat2(X_, u, tmp, n_task) + W2.dot(s1) - XY.dot(s1) + C.dot(s1)

    tmp = Z_.matvec(s3)
    Es3 = rmatmat2(X_, u, tmp, n_task)

    tmp = matmat2(X_, u, s2, n_task)
    Ets2 = Z_.rmatvec(tmp)

    Fs3 = Z_.rmatvec(Z_.matvec(s3))

    line0 = As1 + Bs2 + Cs3
    line1 = Bts1 + Ds2 + Es3
    line2 = Cts1 + Ets2 + Fs3

    return np.concatenate((line0, line1, line2))

if __name__ == '__main__':
    n_target = 2
    np.random.seed(0)
    X1 = np.random.randn(12, 10)
    Z1 = np.random.randn(12, 10)
    Y1 = np.random.randn(12, n_target)
    size_u = 5
    size_v = 2
    canonical = np.random.randn(size_u)
    x0 = np.random.randn(size_u + size_v + Z1.shape[1],
                         Y1.shape[1]).ravel()
    print(optimize.check_grad(f, fprime, x0, X1, Y1, Z1, size_u, 1.,
                              canonical))

    import numdifftools as nd
    import pylab as pl
    H = nd.Hessian(lambda x: f(x, X1, Y1, Z1, size_u, 0., canonical))
    pl.matshow(H(x0))
    pl.title('numdifftools')
    pl.colorbar()
    pl.show()

    E = np.eye(n_target * (size_u + size_v + Z1.shape[1]))
    out = []
    for i in range(E.shape[0]):
        ei = E[i]
        ei = ei.reshape((-1, 1))
        tmp = hess(ei, x0, X1, Y1, Z1, size_u, 0., canonical)
        out.append(tmp.ravel())
    true_H = np.array(out)
    pl.matshow(true_H)
    pl.colorbar()
    pl.show()
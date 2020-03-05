import numpy as np
import math
import graspy
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize_scalar
from sklearn.utils import check_array
from skp import SinkhornKnopp
from joblib import Parallel, delayed
from graspy.simulations import er_np

class FastApproximateQAP:
    def __init__(
        self,
        n_init=10,
        init_method="rand",
        max_iter=30,
        shuffle_input=True,
        eps=0.1,
        gmp=False,
    ):

        if n_init > 0 and type(n_init) is int:
            self.n_init = n_init
        else:
            msg = '"n_init" must be a positive integer'
            raise TypeError(msg)
        if init_method == "rand":
            self.init_method = "rand"
        elif init_method == "barycenter":
            self.init_method = "barycenter"
            self.n_init = 1
        else:
            msg = 'Invalid "init_method" parameter string'
            raise ValueError(msg)
        if max_iter > 0 and type(max_iter) is int:
            self.max_iter = max_iter
        else:
            msg = '"max_iter" must be a positive integer'
            raise TypeError(msg)
        if type(shuffle_input) is bool:
            self.shuffle_input = shuffle_input
        else:
            msg = '"shuffle_input" must be a boolean'
            raise TypeError(msg)
        if eps > 0 and type(eps) is float:
            self.eps = eps
        else:
            msg = '"eps" must be a positive float'
            raise TypeError(msg)
        if type(gmp) is bool:
            self.gmp = gmp
        else:
            msg = '"gmp" must be a boolean'
            raise TypeError(msg)

    def fit(self, A, B):
        A = check_array(A, copy=True, ensure_2d=True)
        B = check_array(B, copy=True, ensure_2d=True)

        if A.shape[0] != B.shape[0]:
            msg = "Matrix entries must be of equal size"
            raise ValueError(msg)
        elif A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            msg = "Matrix entries must be square"
            raise ValueError(msg)
        elif (A >= 0).all() == False or (B >= 0).all() == False:
            msg = "Matrix entries must be greater than or equal to zero"
            raise ValueError(msg)

        n = A.shape[0]  # number of vertices in graphs

        if self.shuffle_input:
            node_shuffle_input = np.random.permutation(n)
            A = A[np.ix_(node_shuffle_input, node_shuffle_input)]
            # shuffle_input to avoid results from inputs that were already matched

        obj_func_scalar = 1
        if self.gmp:
            obj_func_scalar = -1

        At = np.transpose(A)  # A transpose
        Bt = np.transpose(B)  # B transpose
        score = math.inf
        perm_inds = np.zeros(n)

#        for i in range(self.n_init):
        def forloop(init_num):
            
            # setting initialization matrix
            if self.init_method == "rand":
                sk = SinkhornKnopp()
                K = np.random.rand(
                    n, n
                )  # generate a nxn matrix where each entry is a random integer [0,1]
                for i in range(10):  # perform 10 iterations of Sinkhorn balancing
                    K = sk.fit(K)
                J = np.ones((n, n)) / float(
                    n
                )  # initialize J, a doubly stochastic barycenter
                P = (K + J) / 2
            elif self.init_method == "barycenter":
                P = np.ones((n, n)) / float(n)

            grad_P = math.inf  # gradient of P
            n_iter = 0  # number of FW iterations
            
            # OPTIMIZATION WHILE LOOP BEGINS
            while grad_P > self.eps and n_iter < self.max_iter:

                delta_f = (
                    A @ P @ Bt + At @ P @ B
                )  # computing the gradient of f(P) = -tr(APB^tP^t)
                rows, cols = linear_sum_assignment(
                    obj_func_scalar * delta_f
                )  # run hungarian algorithm on gradient(f(P))
                Q = np.zeros((n, n))
                Q[rows, cols] = 1  # initialize search direction matrix Q

                def f(x):  # computing the original optimization function
                    return obj_func_scalar * np.trace(
                        At
                        @ (x * P + (1 - x) * Q)
                        @ B
                        @ np.transpose(x * P + (1 - x) * Q)
                    )

                alpha = minimize_scalar(
                    f, bounds=(0, 1), method="bounded"
                ).x  # computing the step size
                
                P_1 = alpha * P + (1 - alpha) * Q  # Update P
                grad_P = np.linalg.norm(P - P_1)
                P = P_1
                n_iter += 1
            # end of FW optimization loop

            _, perm_inds_new = linear_sum_assignment(
                -P
            )  # Project onto the set of permutation matrices

            score_new = np.trace(
                np.transpose(A) @ B[np.ix_(perm_inds_new, perm_inds_new)]
            )  # computing objective function value

            return score_new, perm_inds_new
        if self.init_method=='barycenter':
            self.n_init=1
            result=forloop(self.n_init)
        else:
            par = Parallel(n_jobs=8)
            result = par(delayed(forloop)(init_num) for init_num in range(self.n_init))
        result = np.mat(result)
        score_new = np.transpose(result[:,0])
        perm_inds_new = result[:,1].tolist()
        #print(score_new)
        #print(perm_inds_new)

        _, column = score_new.shape# get the matrix of a raw and column
        _positon = np.argmin(score_new)# get the index of max in the a
        _, j = divmod(_positon, column)
        #print(j)
        #print(perm_inds_new[j])
        if score_new.min() < score:  # minimizing
            score = score_new.min()
            if self.shuffle_input:
                perm_inds = np.array([0] * n)
                perm_inds[node_shuffle_input] = perm_inds_new[j]
                #print(perm_inds)
            else:
                perm_inds = perm_inds_new[j]
                #print(perm_inds)

        

        if self.shuffle_input:
            node_unshuffle_input = np.array(range(n))
            node_unshuffle_input[node_shuffle_input] = np.array(range(n))
            A = A[np.ix_(node_unshuffle_input, node_unshuffle_input)]
            score = np.trace(np.transpose(A) @ B[np.ix_(perm_inds, perm_inds)])

        self.perm_inds_ = perm_inds  # permutation indices
        self.score_ = score  # objective function value
        return self

    def fit_predict(self, A, B):
        self.fit(A, B)
        return self.perm_inds_


# this is for

from graspy.simulations import er_np
import numpy as np
import copy


def test_er_corr(n, p, rho=0.2):
    G1 = er_np(n, p)
    origin_G1 = copy.deepcopy(G1)
    print(origin_G1.mean())

    for i in range(n):
        for j in range(n):
            if G1[i][j] == 1:
                G1[i][j] = np.random.binomial(1, p+rho*(1-p))
            else:
                G1[i][j] = np.random.binomial(1, p*(1-rho))
    print(G1.mean())

    prob1 = 0
    prob2 = 0

    for i in range(n):
        for j in range(n):
            if origin_G1[i][j] == 1 and G1[i][j] == 1:
                prob1 += 1
            if origin_G1[i][j] == 0 and G1[i][j] == 1:
                prob2 += 1
    exp_prob1 = p + rho * (1 - p)
    real_prob1 = prob1/(origin_G1.mean()*n**2)
    exp_prob2 = p*(1-rho)
    real_prob2 = prob2 / (G1.mean() * n ** 2)
    # ratio = ((real_prob1/exp_prob1)+ (real_prob2/exp_prob2))/2
    var = np.sqrt((exp_prob1 - real_prob1)**2 + (exp_prob1 - real_prob1)**2)
    print('expected prob1 = ', exp_prob1)
    print('real prob1 = ', real_prob1)
    print('expected prob2 = ', exp_prob2)
    print('real prob2 = ', real_prob2)
    print('the variance between estimation and real values =', var)
    return origin_G1, G1


if __name__ == '__main__':

    G1, G2 = test_er_corr(100, 0.5, 0.2)

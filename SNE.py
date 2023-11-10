import numpy as np
from functools  import reduce
import illustrator as ill
import eigen_module as em
import binary_search as bs

#データの取得
X = np.loadtxt('data.csv')
row = 5
col = 5

#取得データの確認
print("inputed data:")
print(X)

Perplexity = 1.0001
log_perp = np.log2(Perplexity)
print("log_perp: " + str(log_perp))

sigma = 500.0
compressed_data = np.random.rand(row, 2)

def d_pow2(vec1, vec2):
    return np.dot((vec1 - vec2), (vec1 - vec2))

def p_ij(X, i, j, sigma):
    d_ij = np.exp( -d_pow2(X[i], X[j])/(2*sigma) )

    def p_ij_sum(l, r):
        if l[0] == r[0]:
            return l
        else:
            i, j = l[0], r[0]
            r_val = np.exp( -d_pow2(X[i], X[j])/(2*sigma) )
            return (l[0], l[1] + r_val)
        
    d_sum =  reduce(p_ij_sum, enumerate(X), (i, 0.0))
    return d_ij / d_sum[1]

def q_ij(X, i, j):
    d_ij = np.exp( -d_pow2(X[i], X[j]) )

    def q_ij_sum(l, r):
        if l[0] == r[0]:
            return l
        else:
            i, j = l[0], r[0]
            r_val = np.exp( -d_pow2(X[i], X[j]) )
            return (l[0], l[1] + r_val)
    
    d_sum =  reduce(q_ij_sum, enumerate(X), (i, 0.0))
    return d_ij / d_sum[1]

def Entropy(sigma, X, i):
    entropies = [p_ij(X,i,j,sigma) for j in range(row) if i!=j]
    return sum([-p * np.log2(p) for p in entropies])

def KLdivergence(X):
    pass

sigma = bs.binary_search(Entropy, (X, 0), log_perp, 1.0e-7, [1.0001, 100000.0], 1000)
print("sigma: " + str(sigma))

entropy = [Entropy(sigma,X,i) for i in range(row)]
print(entropy)

def grad(X, i, j, sigma):
    pass
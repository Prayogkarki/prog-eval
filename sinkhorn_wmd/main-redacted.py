#!/usr/bin/env python

"""
    sinkhorn_wmd/main.py
"""

import os
import sys
import argparse
import numpy as np
from time import time
from scipy import sparse

# --
# Sinkhorn

def sinkhorn_wmd(r, c, vecs, lamb, max_iter):
    """
        r (np.array):          query vector (sparse, but represented as dense)
        c (sparse.csr_matrix): data vectors, in CSR format.  shape is `(dim, num_docs)`
        vecs (np.array):       embedding vectors, from which we'll compute a distance matrix
        lamb (float):          regularization parameter
        max_iter (int):        maximum number of iterations
    """
    i = r>0
    r = r [i]
    print('shape of r')
    print(np.shape(r))
    row_vecs = vecs[i]
    print(np.shape(row_vecs))
    m = np.sqrt((np.square(row_vecs[:,np.newaxis]-vecs).sum(axis=2)))
    #m = np.sqrt(((row_vecs-vecs)**2).sum(axis = 1))
    print(np.shape(m))
    print('............')

   
    k = np.exp(-lamb * lm)
    print('Value of k')
    print(np.shape(k))
    x = np.ones((100000,5000))/100000
    print('shappppe of x')
    print(np.shape(x))
    s = 0
    while s < 1:
        w1 = np.diag(np.linalg.inv(r))
        print('shape of w1')

        print(np.shape(w1))
        print('shape of c')
        print(np.shape(c))
        print(np.shape(np.transpose(k)))
        invx = np.linalg.inv(x)
        print('shape of inverse x')
        print(np.shape(invx))
        w2 = np.dot(w1,k)
        w3 = np.dot(invx, np.transpose(k))
        w4 = c * w3
        x = np.dot(w2, w4)
        
        s = s + 1
        
    u = np.linalg.inv(x)
    w5 = np.dot(np.transpose(k),u)
    v = c * (np.linalg.inv(w5))
    w = u * np.dot((k * m),v)
    scores = w.sum(axis = 0)
    return scores


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='/data/test/sinkhorn_wmd/data/cache')
    parser.add_argument('--n-docs', type=int, default=5000)
    parser.add_argument('--query-idx', type=int, default=100)
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--max_iter', type=int, default=15)
    args = parser.parse_args()
    print("!st Breakpopint")
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.inpath == '/data/test/sinkhorn_wmd/data/cache'
    assert args.n_docs == 5000
    assert args.query_idx == 100
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    # vecs: (num_words, word_embedding_dim) matrix of word embeddings
    # mat:  (num_words, num_docs) sparse term-document matrix in CSR format
    print("!st Breakpopint")
    vecs = np.load(args.inpath + '-vecs.npy')
    #v1 = np.load(args.inpath + '-mat/data.npy')
    #v2 = np.load(args.inpath + '-mat/format.npy')
    #v3 = np.load(args.inpath + '-mat/indices.npy')
    #v4 = np.load(args.inpath + '-mat/indptr.npy')
    #v5 = np.load(args.inpath + '-mat/shape.npy')
    #mat = sparse.csr_matrix((v1, v3, v4), shape=(100000, 558535))
    mat  = sparse.load_npz(args.inpath + '-mat.npz')
    # --
    # Prep
    
    # Maybe subset docs
    if args.n_docs:
        mat  = mat[:,:args.n_docs]
    
    # --
    # Run
    
    # Get query vector
    r = np.asarray(mat[:,args.query_idx].todense()).squeeze()
    
    t = time()
    scores = sinkhorn_wmd(r, mat, vecs, lamb=args.lamb, max_iter=args.max_iter)
    elapsed = time() - t
    print('elapsed=%f' % elapsed, file=sys.stderr)
    
    # --
    # Write output
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/scores', scores, fmt='%.8e')
    open('results/elapsed', 'w').write(str(elapsed))

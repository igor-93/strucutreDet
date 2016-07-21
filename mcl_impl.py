#!/usr/bin/env python

import sys
import numpy as np
import time
from optparse import OptionParser
from scipy.sparse import csr_matrix
from scipy.sparse import *
import scipy 
import sklearn.preprocessing as skpp
import logging

def inflate(A, inflate_factor):
    A.data **= inflate_factor
    return skpp.normalize(A, norm='l1', axis=0, copy=True)#.toarray()

def expand(A, expand_factor):
    A **= expand_factor
    return A

def regularize(M, M_g):
    return np.dot(M,M_g)

def add_diag(A, mult_factor):
    return A + mult_factor * eye(A.shape[0])


def prune(M):
    
    threshold_start = 1e-3
    M_temp = M.multiply(M > threshold_start)
    column_sums = M_temp.sum(axis=0)
    ave_sum_left = column_sums.sum() / float(M_temp.shape[0])
    print 'ave_sum_left: ', ave_sum_left

    # we have to pune more
    threshold_lst = [5*1e-4, 1e-4]
    for threshold in threshold_lst:
        if ave_sum_left >= 0.85:
            break
        else:
            M_temp = M.multiply(M > threshold)
            column_sums = M_temp.sum(axis=0)
            ave_sum_left = column_sums.sum() / float(M_temp.shape[0])
            print '     ave_sum_left: ', ave_sum_left
    if ave_sum_left > 0.5:     
        M = M_temp   
        M = skpp.normalize(M, norm='l1', axis=0, copy=False)

    return M

def get_clusters(A):
    clusters = []
    for i, r in enumerate((A>0).tolist()):
        if r[i]:
            clusters.append(A[i,:]>0)

    clust_map  ={}
    for cn , c in enumerate(clusters):
        for x in  [ i for i, x in enumerate(c) if x ]:
            clust_map[cn] = clust_map.get(cn, [])  + [x]
    return clust_map

def draw(A, cluster_map = {}, colors = []):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.from_numpy_matrix(A)
    if colors == []:
        if cluster_map == {}:
            print 'ERROR in draw: at least of two attributes for coloring has to be provided.'
        clust_map = {}
        for k, vals in cluster_map.items():
            for v in vals:
                clust_map[v] = k

        colors = []
        for i in range(len(G.nodes())):
            colors.append(clust_map.get(i, 100))

    if len(colors) <> A.shape[0]:
        print 'ERROR in draw: len(colors) <> A.shape[0].'
    pos = nx.spring_layout(G)

    from matplotlib.pylab import matshow, show, cm
    plt.figure(2)
    nx.draw_networkx_nodes(G, pos,node_size = 200, node_color =colors , cmap=plt.cm.Blues )
    nx.draw_networkx_edges(G,pos, alpha=0.5)
    #matshow(A, fignum=1, cmap=cm.gray)
    plt.show()
    show()


def stop(M, i):
    # this saves time, so we dont have to do multiplication in the first 7 iterations
    if i > 6:
        m = ( M**2 - M).max() - ( M**2 - M).min()
        #m = ( M.dot(M) - M).max() - ( M.dot(M) - M).min()
        if abs(m) < 1e-8:
            return True

    return False

def mcl_implementation(M, expand_factor = 2, inflate_factor = 2, max_loop = 20 , mult_factor = 1):

    M = add_diag(M, mult_factor)
    M = skpp.normalize(M, norm='l1', axis=0, copy=False)

    M = prune(M)

    for i in range(max_loop):
        
        M = inflate(M, inflate_factor)
        M = expand(M, expand_factor)
        M = prune(M)

        test_counter = i  
        if stop(M, i): break
          

    print 'MCL_STOPPED at iteration %i ' %test_counter 
    M = M.toarray()
    print M
    clusters = get_clusters(M)
    return clusters

def r_mcl_implementation(M, inflate_factor = 2, max_loop = 20 , mult_factor = 1):

    M = add_diag(M, mult_factor)
    M = skpp.normalize(M, norm='l1', axis=0, copy=False)

    M = prune(M)
    
    M_g = np.copy(M)

    for i in range(max_loop):
        
        M = inflate(M, inflate_factor)
        M = regularize(M, M_g) 

        M = prune(M)

        test_counter = i 
        if stop(M, i): break
        

    print 'R-MCL_STOPPED at iteration %i' %test_counter 
    M = M.toarray()
    clusters = get_clusters(M)
    return clusters

def networkx_mcl(G, expand_factor = 2, inflate_factor = 2, max_loop = 10 , mult_factor = 1):
    import networkx as nx
    A = nx.adjacency_matrix(G)
    return mcl(A, expand_factor, inflate_factor, max_loop, mult_factor)

def get_graph(csv_filename):
    import networkx as nx

    M = []
    for r in open(csv_filename):
        r = r.strip().split(",")
        M.append( map( lambda x: float(x.strip()), r))

    G = nx.from_numpy_matrix(np.matrix(M))
    return np.array(M), G

def clusters_to_output(clusters, options):
    if options.output and len(options.output)>0:
        f = open(options.output, 'w')
        for k, v in clusters.items():
            f.write("%s|%s\n" % (k, ", ".join(map(str, v)) ))
        f.close()
    else:    
        print "Clusters:"
        for k, v in clusters.items():
            print k, v

if __name__ == '__main__':

    options, filename = get_options()
    print_info(options)
    M, G = get_graph(filename)

    print " number of nodes: %s\n" % M.shape[0]

    print time.time(), "evaluating clusters..."
    M, clusters = networkx_mcl(G, expand_factor = options.expand_factor,
                               inflate_factor = options.inflate_factor,
                               max_loop = options.max_loop,
                               mult_factor = options.mult_factor)
    print time.time(), "done\n"

    clusters_to_output(clusters, options)

    if options.draw:
        print time.time(), "drawing..."
        draw(G, M, clusters)
        print time.time(), "done"

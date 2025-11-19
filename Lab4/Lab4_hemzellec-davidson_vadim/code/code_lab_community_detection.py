"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3

def spectral_clustering(G, k):

    a=nx.adjacency_matrix(G)
    degrees = np.array([deg for _, deg in G.degree()])
    d_inv = diags(1.0 / degrees)
    lrw = eye(G.number_of_nodes()) - d_inv.dot(a)

    eigenvalues, eigenvectors = eigs(lrw, k=k, which='SM')
    idx = np.argsort(eigenvalues)[:k]
    U=eigenvectors.real[:, idx]

    kmeans = KMeans(n_clusters=k, random_state=2).fit(U)
    clustering=kmeans.labels_

    return clustering

############## Task 4

g=nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t", create_using=nx.Graph(), nodetype=int)
largest_cc = max(nx.connected_components(g), key=len)
print(spectral_clustering(g.subgraph(largest_cc), 50))


############## Task 5

def modularity(G, clustering):

    m = G.number_of_edges()
    communities = {}

    for node, c in zip(G.nodes(), clustering):
        if c not in communities:
            communities[c] = []
        communities[c].append(node)

    Q = 0.0 

    for c, nodes in communities.items():
        lc = 0
        for u in nodes:
            for v in G.neighbors(u):
                if v in nodes:
                    lc += 1
        lc = lc / 2  

        dc = 0
        for u in nodes:
            dc += G.degree(u)

        Q += (lc / m) - (dc / (2*m))**2

    return Q

############## Task 6

print("For k = 50: Q =", modularity(g.subgraph(largest_cc), spectral_clustering(g.subgraph(largest_cc), 50)))

nodes = list(g.subgraph(largest_cc).nodes())
labels_random = [randint(0, 49) for _ in nodes]
print("For random clustering: Q =", modularity(g.subgraph(largest_cc), labels_random))








"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
from sklearn.cluster import KMeans


# Loads the karate network
import matplotlib.pyplot as plt
import os

# Loads the karate network
file_path = os.path.join(os.path.dirname(__file__), '../data/karate.edgelist')
G = nx.read_weighted_edgelist(file_path, delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
labels_file_path = os.path.join(os.path.dirname(__file__), '../data/karate_labels.txt')
class_labels = np.loadtxt(labels_file_path, delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network 

nx.draw_networkx(G, node_color=y, cmap=plt.cm.Set1)
plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


############## Task 8
# Generates spectral embeddings

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

print(accuracy_score(y, spectral_clustering(G, 2)))

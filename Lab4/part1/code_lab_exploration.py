"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# The answer to the questions of the lab can be found in the "ALTEGRAD LAB 4 QUESTIONS.pdf" file.

############## Task 1

g=nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t", create_using=nx.Graph(), nodetype=int)
print(g)

############## Task 2

print("Number of connected components:", nx.number_connected_components(g))
largest_cc = max(nx.connected_components(g), key=len)
print("Number of nodes in the largest connected component:", len(largest_cc))
print("Number of edges in the largest connected component:", g.subgraph(largest_cc).number_of_edges())
print("Fraction of nodes in the largest connected component:", len(largest_cc) / g.number_of_nodes())
print("Fraction of edges in the largest connected component:", g.subgraph(largest_cc).number_of_edges() / g.number_of_edges())





    



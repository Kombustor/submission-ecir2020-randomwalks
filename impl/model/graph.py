import networkx as nx


def read_adjlist(filename):
    G = nx.read_adjlist(filename, create_using=nx.DiGraph())
    for i, j in G.edges():
        G[i][j]['weight'] = 1.0
    return G


def read_edgelist(filename, weighted=False):
    func = nx.read_edgelist
    if weighted:
        func = nx.read_weighted_edgelist

    return func(path=filename, comments='%', nodetype=int)

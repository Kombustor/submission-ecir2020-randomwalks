from impl.model.node2vec import Node2Vec, DefaultWalker, MethodOpts
import random
from impl.utils import debug
import networkx as nx
import numpy as np


class Node2VecHubs(Node2Vec):

    def __init__(self, opts: MethodOpts, h=1):
        debug("Node2Vec opts: ", opts.__dict__)
        self.opts = opts
        self.h = h

    def init_walker(self, graph):
        self.walker = HubsWalker(graph, self.h)


class HubsWalker(DefaultWalker):

    def __init__(self, G, h):
        self.G = G
        self.h = h
        self.detect_hubs()

    def detect_hubs(self):
        self.hubs = []

        # [(node1, node2...), (degree1, degree2, ...)]
        node_degrees = list(zip(*self.G.degree()))
        degrees = node_degrees[1]
        nodes = node_degrees[0]

        avg_degree = np.mean(degrees)
        std_degree = np.std(degrees)

        for idx, node in enumerate(nodes):
            deg = degrees[idx]
            if(deg > (avg_degree + std_degree)):
                self.hubs.append(node)

        debug('Hubs: ', self.hubs, ", avg degree: ", avg_degree, ", std degree: ", std_degree)

    def is_hub(self, node):
        return node in self.hubs

    def get_alias_edge(self, src, dst):
        G = self.G
        h = self.h

        unnormalized_probs = []
        # for every (sorted) neighbor of destination node
        for dst_nbr in sorted(G.neighbors(dst)):
            # if the destination neighbor is a hub, modify probability with h (=> likelihood of revisiting a hub increased)
            if self.is_hub(dst_nbr):
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/h)
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])

        return self.alias_probs(unnormalized_probs)

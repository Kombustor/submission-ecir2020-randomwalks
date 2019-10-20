from impl.model.node2vec import Node2Vec, DefaultWalker, MethodOpts
import random
from impl.utils import debug


class Node2VecJumps(Node2Vec):

    def __init__(self, opts: MethodOpts, jump_prob=0):
        debug("Node2Vec opts: ", opts.__dict__)
        self.opts = opts
        self.jump_prob = jump_prob

    def init_walker(self, graph):
        self.walker = JumpsWalker(graph, self.jump_prob)


class JumpsWalker(DefaultWalker):

    def __init__(self, G, jump_prob):
        self.G = G
        self.jump_prob = jump_prob

    def walk_node(self, alias_nodes, alias_edges, walk, cur, cur_nbrs):
        # jump probability
        if(random.random() <= self.jump_prob):
            # draw random node
            random_node = random.choice(list(alias_nodes.keys()))
            return random_node

        # on start or if theres no edge between previous and current, walk to random neighbor node
        if len(walk) == 1 or ((walk[-2], cur) not in alias_edges):
            # draw random neighbor of current node
            random_neighbor = cur_nbrs[self.alias_draw(
                alias_nodes[cur][0], alias_nodes[cur][1])]

            return random_neighbor
        # then, get previous node, and set next node to random neighbor based on edges
        else:
            prev = walk[-2]
            # draw random edge of get_alias_edge(prev, cur)
            random_edge = self.alias_draw(alias_edges[(prev, cur)][0],
                                     alias_edges[(prev, cur)][1])
            # get neighbor of random edge
            next = cur_nbrs[random_edge]
            return next

    def get_alias_edge(self, src, dst):
        G = self.G

        unnormalized_probs = []
        # for every (sorted) neighbor of destination node
        for dst_nbr in sorted(G.neighbors(dst)):
            unnormalized_probs.append(G[dst][dst_nbr]['weight'])

        return self.alias_probs(unnormalized_probs)
import numpy as np
import random
from gensim.models import Word2Vec
import os
import psutil
import itertools
import multiprocessing as mp
from tqdm import tqdm
from impl.utils import debug, DEBUG


class MethodOpts:

    def __init__(self, dim=128, walk_length=80, num_walks=10, window=10, min_count=0):
        """
        dim - dimensions to learn, default 128
        walk_length -- length of each random walk, default 80
        num_walks -- number of random walks at each node, default 10
        workers -- number of parallel processes, default num of cores
        window -- window size for word2vec, default 10
        min_count -- min_count for word2vec, default 0
        """
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = os.cpu_count()
        self.window = window
        self.min_count = min_count


class Node2Vec(object):

    def __init__(self, opts: MethodOpts, p=1.0, q=1.0):
        debug("Node2Vec opts: ", opts.__dict__)
        self.opts = opts
        self.p = p
        self.q = q

    def init_walker(self, graph):
        self.walker = DefaultWalker(graph, self.p, self.q)

    def run(self):
        debug("Preprocessing transition probs...")
        self.walker.preprocess_transition_probs()

        debug("Simulating walks...")
        walks = self.walker.simulate_walks(
            self.opts.num_walks, self.opts.walk_length)

        debug("Learning model...")
        # Converts every walk value (= node) from int to str for word2vec (=> words)
        walks_str = tqdm([list(map(str, walk)) for walk in walks], disable=(not DEBUG))
        self.model = Word2Vec(walks_str, workers=self.opts.workers, size=self.opts.dim,
                              min_count=self.opts.min_count, window=self.opts.window, sg=1)

    def load_embeddings(self, filename):
        self.model = Word2Vec.load(filename)

    def save_embeddings(self, filename):
        assert self.model
        self.model.save(filename)

    def get_vectors(self):
        vectors = {}
        for word in self.model.wv.vocab:
            vectors[word] = self.model.wv[word]

        return vectors


class DefaultWalker:
    def __init__(self, G, p, q):
        self.G = G
        self.p = p
        self.q = q

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        debug("Preprocessing nodes...")
        alias_nodes = {}
        for node in tqdm(G.nodes(), disable=(not DEBUG)):
            # get all weights of node's neighbors
            unnormalized_probs = [G[node][nbr]['weight']
                                  for nbr in sorted(G.neighbors(node))]
            # normalize weights with sum
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        debug("Preprocessing edges...")
        alias_edges = {}

        # Split edges up in chunks
        workers = os.cpu_count()
        edge_divisor = workers ** 2
        chunk_size = int(G.size() / edge_divisor) + 1
        chunks = list(self.split_chunks(G.edges(), chunk_size))

        # Creating worker pool & running
        #maxtasks = int(len(chunks) / 10)

        #with mp.Pool(processes=workers, maxtasksperchild=maxtasks) as pool:
        with mp.Pool(processes=workers) as pool:
            debug(
                f"Workers {workers}, chunks {len(chunks)} of size {chunk_size}")

            result = list(
                tqdm(pool.imap(self.preprocess_edge_chunk, chunks), total=len(chunks), disable=(not DEBUG)))

            # Combining results into one
            for d in result:
                alias_edges.update(d)

            del result
            pool.close()
            pool.join()

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return

    def preprocess_edge_chunk(self, edges):
        alias_edges = {}
        for edge in edges:
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(
                edge[1], edge[0])
        return alias_edges

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in tqdm(nodes, desc=f"Walk {walk_iter+1}/{num_walks}", disable=(not DEBUG)):
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))

        return walks

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        # contains the transition probability to every neighbor for every node
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        # initialize walk to start node
        walk = [start_node]

        while len(walk) < walk_length:
            # get the current node
            cur = walk[-1]
            # get neighbors of the current node
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(self.walk_node(
                    alias_nodes, alias_edges, walk, cur, cur_nbrs))
            else:
                break

        return walk

    def walk_node(self, alias_nodes, alias_edges, walk, cur, cur_nbrs):
        # on start, walk to random neighbor node
        if len(walk) == 1:
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
        p = self.p
        q = self.q

        unnormalized_probs = []
        # for every (sorted) neighbor of destination node
        for dst_nbr in sorted(G.neighbors(dst)):
            # if the destination neighbor is the source node,increase probability with p (=> likelihood of revisiting a node increased)
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            # if destination neighbor has an edge to the source node, set probability to base weight
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            # if the destination neighbor is not the source node and has no edge to the source, increase probability with q (q > 1 => more close to src, q < 1, more outward)
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)

        return self.alias_probs(unnormalized_probs)

    def alias_probs(self, unnormalized_probs):
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)


    def split_chunks(self, l, n):
        """Divide a list of nodes `l` in `n` chunks"""
        l_c = iter(l)
        while 1:
            x = tuple(itertools.islice(l_c, n))
            if not x:
                return
            yield x


    def alias_setup(self, probs):
        '''
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        q = np.zeros(K, dtype=np.float32)
        J = np.zeros(K, dtype=np.int32)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q


    def alias_draw(self, J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)

        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

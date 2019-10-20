import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
import time
import os

from impl.model import node2vec, graph, classify
from impl.utils import debug




class Options(object):

    def __init__(self, input, graph_format="adjlist", output=None, weighted=True, label_file=None, training_ratio=0.5, clf_iterations=10):
        """
        input -- the input graph, required
        graph_format -- format of the input graph, default adjlist, one of "adjlist" "edgelist"
        weighted -- whether the graph is weighted, default True
        output -- output path of the embeddings, default None
        label_file -- file containing the node labels, optional, default None
        training_ratio -- ratio of training data in the classification, default 0.5
        clf_iterations -- number of iterations to evaluate training results, default 10
        """
        self.input = input
        self.graph_format = graph_format
        self.weighted = weighted
        self.output = output
        self.label_file = label_file
        self.training_ratio = training_ratio
        self.clf_iterations = clf_iterations


def main(opts: Options, model: node2vec.Node2Vec):
    debug("Options: ", opts.__dict__)
    random.seed(32)
    np.random.seed(32)

    # Reading graph
    debug("Reading graph...")
    G = None
    if opts.graph_format == 'adjlist':
        G = graph.read_adjlist(filename=opts.input)
    elif opts.graph_format == 'edgelist':
        G = graph.read_edgelist(filename=opts.input, weighted=opts.weighted)

    debug(f"Graph: {len(G.nodes())} nodes, {G.size()} edges")

    # Loading/learning embeddings
    if(opts.output and os.path.isfile(opts.output)):
        debug(f"Model {opts.output} exists, loading from file...")
        model.load_embeddings(opts.output)
    else:
        debug("Training model...")
        model.init_walker(G)
        model.run()

        if opts.output:
            debug("Saving embeddings...")
            model.save_embeddings(opts.output)

    # Classification
    clf_results = None
    if opts.label_file:
        X, Y = classify.read_node_label(opts.label_file)
        debug("Training classifier using {:.2f}% nodes...".format(
            opts.training_ratio*100))
        clf = classify.Classifier(vectors=model.get_vectors(
        ), clf=LogisticRegression(solver='liblinear'))
        clf_results = clf.split_train_evaluate(
            X, Y, opts.training_ratio, iter=opts.clf_iterations)

    return G, clf_results

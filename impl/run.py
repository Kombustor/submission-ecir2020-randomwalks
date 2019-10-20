import importlib
import networkx as nx
from impl.main import main, Options
from impl.visualize import cluster, tsne, drawing, utils
from impl.model.node2vec import MethodOpts, Node2Vec
from impl.utils import debug

def run(opts: Options, method: Node2Vec, title=None, cluster2color=['green', 'red', 'blue', 'orange', 'purple', 'cyan'], 
        num_clusters=6, zero_indexed=False, tsne_perplexity=12, draw=["graph", "tsne"], save=False, save_path=None):

    G, clf_results = main(opts, method)
    model = method.model

    if(draw or save):
        debug("Generating colors...")
        colors = cluster.cluster_colors(graph=G, model=model, cluster2color=cluster2color, num_clusters=num_clusters)
        colors_idx = utils.color_by_index(colors, list(model.wv.vocab), zero_indexed)

    if(draw):
        debug("Drawing...")
        if("graph" in draw):
            debug("Drawing graph...")
            layout = nx.drawing.layout.spring_layout(G)
            drawing.graph(graph=G, model=model, colors=colors, layout=layout, title=title)

        if("tsne" in draw):
            debug("Drawing TSNE...")
            tsne_model = tsne.model(model, perplexity=tsne_perplexity)
            drawing.tsne(tsne_model, colors_idx, title=title)

    if(save):
        assert title, "title must be set if save=True"
        assert save_path, "save_path must be set if save=True"

        utils.save_gefx(G, colors, title, save_path, zero_indexed)
        
    return clf_results
import networkx as nx
import matplotlib.pyplot as plt


def graph(graph, model, colors, layout, title="", filterFunc=None):
    if(filterFunc):
        if(filterFunc(words, membership, colors)):
            return

    plt.title(title)
    nx.draw_networkx(graph, node_color=colors, with_labels=True, pos=layout)
    # plt.savefig("./graphs/png/{}.png".format(title), format="PNG")
    plt.show()

    # save_gefx(G, colors, title)


def tsne(tsne_model, colors, title=""):
    # plt.figure(figsize=(16, 9))
    x = tsne_model[:, 0]
    y = tsne_model[:, 1]

    plt.title(title)
    plt.scatter(x, y, c=colors)
    plt.grid(True)
    plt.show()

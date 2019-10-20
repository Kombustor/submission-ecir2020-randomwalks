from webcolors import name_to_rgb
import numpy as np
import networkx as nx


def color_by_index(color_map, words, zero_indexed=False):
    color_map_index = [None] * len(color_map)
    for node, color in enumerate(color_map, start=1):
        index = (int(node) - 1) if zero_indexed else node
        color_map_index[words.index(str(index))] = color

    return color_map_index


def maprange(a, b, s):
    (a1, a2), (b1, b2) = a, b
    return b1 + ((s - a1) * (b2 - b1) / (a2 - a1))


def save_gefx(G, colors, name, path, zero_indexed=False):
    degree = G.degree()
    for node in G:
        #size=maprange((0, 36), (10, 100), degree[node])
        index = int(node) if zero_indexed else int(node) - 1
        r, g, b = name_to_rgb(colors[index])
        # , 'size': int(size)}
        G.node[node]['viz'] = {'color': {'r': r, 'g': g, 'b': b, 'a': 0}}
    nx.write_gexf(G, path + name + ".gexf")


def compare_colors(colors1, colors2):
    colors1 = np.array(normalize_colors(colors1))
    colors2 = np.array(normalize_colors(colors2))

    matching = np.where(colors1 == colors2)[0]

    # debug('C1\n', colors1, '\nC2\n', colors2, '\nM\n', matching)

    return len(matching)/len(colors1)


def normalize_colors(colors):
    used = set()
    unique = [x for x in colors if x not in used and (used.add(x) or True)]

    return [unique.index(color) for color in colors]


def plot_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    stdAboveMean = mean + std
    stdBelowMean = mean - std

    plt.plot(scores)
    plt.axhline(mean, color='g', linestyle='--', label="Mean")
    plt.axhline(stdAboveMean, color='r', linestyle='--', label="Std")
    plt.axhline(stdBelowMean, color='r', linestyle='--', label="Std")
    plt.legend()
    plt.show()

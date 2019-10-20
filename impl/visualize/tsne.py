from sklearn.manifold import TSNE


def model(model, perplexity):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=perplexity, n_components=2,
                      init='pca', n_iter=3500, random_state=23, method='exact')
    return tsne_model.fit_transform(tokens)

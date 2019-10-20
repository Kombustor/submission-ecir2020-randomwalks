import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from time import time
from tqdm import tqdm
import numpy as np
from impl.utils import DEBUG

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self._clf = clf
        self.averages = ["micro", "macro", "samples", "weighted"]
    
    def reset(self):
        self.clf = TopKRanker(self._clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        
    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        results = []
        for average in self.averages:
            results.append(f1_score(Y, Y_, average=average) * 100)
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_percent, iter):
        results = []
        for i in tqdm(range(iter), disable=(not DEBUG), ):
            self.reset()

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=1-train_percent, random_state=i)

            self.train(X_train, Y_train, Y)
            results.append(self.evaluate(X_test, Y_test))
        
        return self.mean_result(results)

    def mean_result(self, results):
        zipped = zip(*results)
        return dict(zip(self.averages, [{"mean": np.mean(i), "std": np.std(i)} for i in zipped]))


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors

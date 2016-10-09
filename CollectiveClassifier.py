from __future__ import division
from statistics import mode
from nltk.classify import ClassifierI


class Collective_Classifier(ClassifierI):
    def __init__(self, *CLASSIFIERS):
        self._CLASSIFIERS = CLASSIFIERS

    def classify(self, FEATURES):
        CUM_TOTAL_AMOUNT = []
        for c in self._CLASSIFIERS:
            t = c.classify(FEATURES)
            CUM_TOTAL_AMOUNT.append(t)
        return mode(CUM_TOTAL_AMOUNT)

    def CONFIDENCE(self, FEATURES):
        CUM_TOTAL_AMOUNT = []
        for c in self._CLASSIFIERS:
            t = c.classify(FEATURES)
            CUM_TOTAL_AMOUNT.append(t)

        CUM_TOTAL = CUM_TOTAL_AMOUNT.count(mode(CUM_TOTAL_AMOUNT))
        CONF = CUM_TOTAL / len(CUM_TOTAL_AMOUNT) * 100
        return CONF
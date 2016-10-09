import nltk


def FIND_FEATURES(DOCUMENT, WORD_FEATURES):
    """

    :param DOCUMENT: DOCUMENT CONTAINING TEXT
    :param WORD_FEATURES: LIST OF SPECIFIC FEATURES
    :return: FEATURES FOUND WITHIN DOCUMENT
    """
    WORDS = set(DOCUMENT)
    FEATURES = {}
    for w in WORD_FEATURES:
        FEATURES[w] = (w in WORDS)
    return FEATURES


def OUTPUT_ACCURACY(FUNC_NAME, CLASSI, TESTING_SET0):
    print(FUNC_NAME + " Accuracy [%]: " + str(nltk.classify.accuracy(CLASSI, TESTING_SET0) * 100))

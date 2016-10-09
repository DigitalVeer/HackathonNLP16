#Importing Methods and Libraries Handled and Implemented by External Libraries
import random, pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from nltk.corpus import movie_reviews


#Importing methods and class created for specificity
from CollectiveClassifier import *
import CollectiveClassifier
from NaiveBayes import *


DOCUMENTS = [(list(movie_reviews.words(fileid)), CATEGORY)
             for CATEGORY in movie_reviews.categories()
             for fileid in movie_reviews.fileids(CATEGORY) ]

#SHUFFLE RESULTS TO ENSURE NO BIAS
random.shuffle(DOCUMENTS)


#OBTAIN WORDS
COLLECTIVE_WORD_SET = [w.lower() for w in movie_reviews.words()]
COLLECTIVE_WORD_SET = nltk.FreqDist(COLLECTIVE_WORD_SET)
WORD_FEATURES = list(COLLECTIVE_WORD_SET.keys())[:3000]

FEATURE_SETS = [(FIND_FEATURES(REV, WORD_FEATURES), CATEGORY) for (REV, CATEGORY) in DOCUMENTS]

LEARNING_SET = FEATURE_SETS[:1900]
TEST_SET = FEATURE_SETS[1900:]


#ALLOCATE PRIOR CLASSIFIER TO SAVE MEMORY
CLASSIFIER_f = open("naivebayes.pickle", "rb")
CLASSIFIER = pickle.load(CLASSIFIER_f)
CLASSIFIER_f.close()


OUTPUT_ACCURACY("Original Naive Bayes", CLASSIFIER, TEST_SET)
CLASSIFIER.show_most_informative_features(15)


#CREATE CLASSIFIER CLASSES TO HANDLE NB VARIATIONS & OTHER STATISTICAL CALCULATIONS
MULTINOMIA_NB_CLASSIFIER = SklearnClassifier(MultinomialNB())
BERNOULLI_NB_CLASSIFIER = SklearnClassifier(BernoulliNB())
LOGISTIC_REGRESSION_CLASSIFIER = SklearnClassifier(LogisticRegression())
LINEAR_SVC_CLASSIFIER = SklearnClassifier(LinearSVC())
SGD_CLASSIFIER = SklearnClassifier(SGDClassifier())
NU_SVC_CLASSIFIER = SklearnClassifier(NuSVC())

#TRAIN BASED OFF LEARNING SET | OUTPUT ACCURACY
MULTINOMIA_NB_CLASSIFIER.train(LEARNING_SET)
OUTPUT_ACCURACY("Multinomial Naive Bayes", MULTINOMIA_NB_CLASSIFIER, TEST_SET)

BERNOULLI_NB_CLASSIFIER.train(LEARNING_SET)
OUTPUT_ACCURACY("Bernoulli Naive Bayes", BERNOULLI_NB_CLASSIFIER, TEST_SET)

LOGISTIC_REGRESSION_CLASSIFIER.train(LEARNING_SET)
OUTPUT_ACCURACY("LogisticRegression", LOGISTIC_REGRESSION_CLASSIFIER, TEST_SET)

SGD_CLASSIFIER.train(LEARNING_SET)
OUTPUT_ACCURACY("SGD", SGD_CLASSIFIER, TEST_SET)

LINEAR_SVC_CLASSIFIER.train(LEARNING_SET)
OUTPUT_ACCURACY("Linear SVC", LINEAR_SVC_CLASSIFIER, TEST_SET)

NU_SVC_CLASSIFIER.train(LEARNING_SET)
OUTPUT_ACCURACY("NuSVC", NU_SVC_CLASSIFIER, TEST_SET)

#COMBINE PRIOR CLASSIFIERS FOR ULTIMATE ACCURACY
COLLECTIVE_CLASSIFIER = Collective_Classifier(CLASSIFIER, MULTINOMIA_NB_CLASSIFIER, BERNOULLI_NB_CLASSIFIER,
                                  LOGISTIC_REGRESSION_CLASSIFIER, SGD_CLASSIFIER,
                                  LINEAR_SVC_CLASSIFIER, NU_SVC_CLASSIFIER)


OUTPUT_ACCURACY("Collective Classifier", COLLECTIVE_CLASSIFIER, TEST_SET)

TEST_SET_CASE_ONE = TEST_SET[0][0]
print("Classification: ", COLLECTIVE_CLASSIFIER.classify(TEST_SET_CASE_ONE), "Confidence [%]: ",
      COLLECTIVE_CLASSIFIER.CONFIDENCE(TEST_SET_CASE_ONE))

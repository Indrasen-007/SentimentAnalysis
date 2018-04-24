import csv
import nltk
import random

from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from sklearn.tree import DecisionTreeClassifier
from statistics import mode
from nltk.tokenize import word_tokenize
import re
import sqlite3
from nltk.corpus import stopwords
import sys
from collections import Counter
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import twitter_samples
import json


neg_tweets = twitter_samples.strings('negative_tweets.json')
pos_tweets = twitter_samples.strings('positive_tweets.json')


#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 

all_words = []
documents = []
neg_words=[]
pos_words=[]


#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]



stop_words = set(stopwords.words('english'))    
for row in neg_tweets:
    documents.append( (row,"negative") )
    row1 = processTweet(row)
    tokenized_words = word_tokenize(row1)
    words = [word for word in tokenized_words if not word in stop_words]
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            neg_words.append(w[0].lower())

for row in pos_tweets:
    documents.append( (row,"positive") )
    row1 = processTweet(row)
    tokenized_words = word_tokenize(row1)
    words = [word for word in tokenized_words if not word in stop_words]
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            pos_words.append(w[0].lower())



save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_classifier = open("pickled_algos/featuresets5k.pickle","wb")
pickle.dump(featuresets, save_classifier)
save_classifier.close()


random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[9000:]
training_set = featuresets[:9000]

accuricies_classifier=[]
######## own classifier begins ##
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



pos_prob=len(pos_tweets)/(len(pos_tweets)+len(neg_tweets))
neg_prob=len(neg_tweets)/(len(pos_tweets)+len(neg_tweets))

positive_wordlist = Counter(pos_words)
wordlist = Counter(all_words)
negative_wordlist = Counter(neg_words)

v=len(set(all_words))

writeto=open('pickled_algos/own_classifier.pickle','wb')
pickle.dump(pos_prob,writeto)
pickle.dump(neg_prob,writeto)
pickle.dump(positive_wordlist,writeto)
pickle.dump(negative_wordlist,writeto)
pickle.dump(pos_words,writeto)
pickle.dump(neg_words,writeto)
pickle.dump(v,writeto)
writeto.close()


predictions = []
for test in testing_set:
    pos=math.log(pos_prob)
    neg=math.log(neg_prob)        

    for word in test[0].keys():
            pos=pos+math.log((positive_wordlist[word]+1)/(len(pos_words)+v))
            neg=neg+math.log((negative_wordlist[word]+1)/(len(neg_words)+v))
                                              
    #print("pos : ",pos)
    #print("neg : ",neg)
    
    if(pos>neg):
            #print(test, " : Positive")
            predictions.append("positive")
    else:
            #print(test," : Negative")
            predictions.append("negative")

accuracy = getAccuracy(testing_set, predictions)
print('Accuracy: ',accuracy)
accuricies_classifier.append(accuracy)


#### own classifier ends##

classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy = (nltk.classify.accuracy(classifier, testing_set))*100
print("Original Naive Bayes Algo accuracy percent:",accuracy )
accuricies_classifier.append(accuracy)
classifier.show_most_informative_features(15)

###############
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
accuracy = (nltk.classify.accuracy(MNB_classifier, testing_set))*100
print("MNB_classifier accuracy percent:",accuracy )
accuricies_classifier.append(accuracy)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
accuracy =(nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100
print("BernoulliNB_classifier accuracy percent:",accuracy )
accuricies_classifier.append(accuracy)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
accuracy = (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100
print("LogisticRegression_classifier accuracy percent:",accuracy )
accuricies_classifier.append(accuracy)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
accuracy = (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100
print("LinearSVC_classifier accuracy percent:",accuracy )
accuricies_classifier.append(accuracy)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
accuracy = nltk.classify.accuracy(SGDC_classifier, testing_set)*100
print("SGDClassifier accuracy percent:",accuracy)
accuricies_classifier.append(accuracy)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

DecisionTree_classifier = SklearnClassifier(DecisionTreeClassifier())
DecisionTree_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(DecisionTree_classifier, testing_set))*100
print("DecisionTree_classifier accuracy percent:", accuracy)
accuricies_classifier.append(accuracy)

save_classifier = open("pickled_algos/DecisionTree_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()





objects = ( 'OWN','NB','MNB', 'BNB','LRC',"SVC","SDGC" , 'DTC');
y_pos = np.arange(len(objects))
 
plt.bar(y_pos, accuricies_classifier,facecolor='#9999ff', align='center', alpha=0.5)
plt.plot(y_pos, accuricies_classifier,y_pos, accuricies_classifier,'ro')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Classifiers')
 
plt.show()



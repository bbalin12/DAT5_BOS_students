# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:34:54 2015

@author: jeppley
"""
import sys
print("Python: {}".format(sys.version))

import numpy as np
import sklearn
print("scikit-learn: {}".format(sklearn.__version__))
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer

newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'soc.religion.christian', 'talk.politics.guns'])
pprint(list(newsgroups_train.target_names))


# Let's split the dataset into two sets: training and testing
docs_train, docs_test, y_train, y_test = train_test_split(newsgroups_train.data, newsgroups_train.target)

print("Number of training documents: {}".format(len(docs_train)))
print("Number of testing documents: {}".format(len(docs_test)))

# First: how are we going to evaluate?
# F-score -- related to accuracy, based on precision and recall
# Hard to "fake" for unbalanced datasets

# Fit on training data
model = CountVectorizer()
X_train = model.fit_transform(docs_train)

# Vocabulary is the words used, and is a dict, available in model.vocabulary_
# They map the words to the indices

pprint(list(model.vocabulary_.items())[:10])

# X_train gives us our bag of words matrix: X_train[i][j] is the value of word with index j for document with index i
# It is a sparse matrix, which we will get to later on
print(type(X_train))

keyword = "believe"
documents_containing_keyword = [index for index in range(len(docs_train)) if keyword in docs_train[index]]
assert len(documents_containing_keyword) > 0
keyword_index = model.vocabulary_[keyword]
document_index = documents_containing_keyword[0]
print(keyword_index in X_train[document_index].nonzero()[1])
print("The keyword {} appears {} times in document {}".format(keyword, X_train[document_index,keyword_index], document_index))



# Let's compare some words, and how they differ between categories
words = ["believe", "right", "bible"]
word_indices = [model.vocabulary_[word] for word in words]
classes = sorted(set(y_train))
categories = newsgroups_train.target_names
#print(X_train[y_train == 0, word_indices[0]])
frequency = np.array([[X_train[y_train == category,wi].mean() for wi in word_indices]
                       for category in classes]).T

assert frequency.shape == (len(word_indices), sum(set(y_train))), frequency.shape
print(frequency)

%matplotlib inline
# Setup the plot
from matplotlib import pyplot as plt
ind = np.arange(frequency.shape[0])
width = 0.2

colors = "rgbyck"

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111)
for column in range(frequency.shape[1]):
    ax.barh(ind + (width * column), frequency[:,column], width, color=colors[column], label=categories[column])
ax.set(yticks=ind + width, yticklabels=words, ylim=[len(words)*width - 1, frequency.shape[0]])
ax.legend(bbox_to_anchor=(0.9, 0.8))
r = plt.xlim((0, 1))
plt.show()

# We can use our existing model to transform the test documents in the same way
# Because we don't fit again, the indices match with the previouc
X_test = model.transform(docs_test)

# Then we build a basic classifier and test it out
from sklearn.svm import SVC
clf = SVC().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("F1-score: {:.3f}".format(f1_score(y_test, y_pred)))
print("Accuracy: {:.3f}".format(np.mean(y_test == y_pred)))

# Let's put that all into a short snippet:
text_model = CountVectorizer()
clf_model = SVC()

# Convert documents to vectors
X_train = text_model.fit_transform(docs_train)
X_test = text_model.transform(docs_test)

# Train classifier
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

# Evaluate
print("F1-score: {:.3f}".format(f1_score(y_test, y_pred)))
print("Accuracy: {:.3f}".format(np.mean(y_test == y_pred)))
# The results will change, as there is some randomness.
# We can usually address that using random_state, but that is out of scope for today.





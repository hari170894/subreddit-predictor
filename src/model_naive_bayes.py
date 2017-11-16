#!/usr/bin/python2
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import logistics_pickler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("loading pickled feature representation")
X_train, Y_train = logistics_pickler.load_obj("../pickle_files/feature_all_train_bow.pkl")
X_test, Y_test = logistics_pickler.load_obj("../pickle_files/feature_all_test_bow.pkl")

print("training classifier")
clf = MultinomialNB()
clf.fit(X_train, Y_train)

print("evaluating model")
y_true = Y_test
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_true, y_pred)
print("accuracy: {}".format(accuracy))

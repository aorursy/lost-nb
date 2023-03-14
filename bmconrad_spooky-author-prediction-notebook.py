#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Feed: 
#  1. A list of sentences
#  2. A pandas dataframe that represents a list of sentences
# E.g., ["This is the first sentence, yes.",
#        "Now youre getting the idea, aren't you?",
#        ...]
def get_bag_of_words(X):
    
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(X)
    #X_counts.shape

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
    X_tf = tf_transformer.transform(X_counts)
    #X_tf.shape

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    #X_tfidf.shape
    #X_tfidf.data
    
    print("Bag of words created!")
    return X_tfidf




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train["text"],
                                                    df_train['author'],
                                                    test_size=0.33, random_state=42)

X_train_tfidf = get_bag_of_words(pd.concat([X_train, X_test]))
X_test_tfidf = get_bag_of_words(pd.concat([X_test, X_train]))




from sklearn.naive_bayes import MultinomialNB

# Fit on our term frequency inverse document frequency
clf = MultinomialNB().fit(X_train_tfidf[:len(X_train)], y_train)

# Build the test data set
y_pred = clf.predict(X_test_tfidf[:len(X_test)])
print("Top 5 Predictions on X_test: ", y_pred[:5])




#import entropy/log loss as a metric
from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

def generate_results(y_true, y_pred):
    
    print ('Accuracy:\n', accuracy_score(y_test, y_pred))
    print ('F1 score:\n', f1_score(y_test, y_pred, average='macro'))
    print ('Recall:\n', recall_score(y_test, y_pred, average='macro'))
    print ('Precision:\n', precision_score(y_test, y_pred, average='macro'))
    print ('clasification report:\n', classification_report(y_test,y_pred))
    print ('confussion matrix:\n',confusion_matrix(y_test, y_pred))
    #print ('log loss:\n',log_loss(y_test, y_pred))
    #print entropy/log_loss as a metric

generate_results(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=clf.classes_))




X_train_tfidf = get_bag_of_words(pd.concat([df_train["text"], df_test['text']]))
y_train = df_train["author"]
X_test_tfidf = get_bag_of_words(pd.concat([df_test["text"], df_train["text"]]))
y_pred = []

clf = MultinomialNB().fit(X_train_tfidf[:len(df_train)], y_train)
y_pred = clf.predict_proba(X_test_tfidf[:len(df_test)])
results = pd.DataFrame({'id':df_test["id"]})
results[clf.classes_] = pd.DataFrame(y_pred)

# For the results, I need a table like the following
#
# id | P(author1) | P(author2) | P(author3)
#
results.head()




from sklearn.model_selection import KFold

# Create 10 folds to test our data on
kf = KFold(n_splits=10)
scores=[]
for train_index, test_index in kf.split(df_train):
    
    # Foldi Train/Test Data
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    X_traini, X_testi = df_train.loc[train_index,"text"], df_train.loc[test_index,"text"]
    y_traini, y_testi = df_train.loc[train_index,"author"], df_train.loc[test_index,"author"]
    
    # Foldi Train/Test Bag of words
    X_train_tfidfi = get_bag_of_words(pd.concat([X_traini, X_testi]))
    X_test_tfidfi = get_bag_of_words(pd.concat([X_testi, X_traini]))
    
    # Foldi Model
    clfi = MultinomialNB().fit(X_train_tfidfi[:len(X_traini)], y_traini)

    # Test Foldi Model on Foldi held out data
    y_predi = clfi.predict(X_test_tfidfi[:len(X_testi)])
    
    # Append results, iterate
    scores.append(accuracy_score(y_testi, y_predi))
print("Accuracy Scores After 10-Fold Cross Validation:")
print(scores)
print("Average Accuracyy After 10 Folds:")
print(np.mean(scores))









from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

nb_parameters = {#'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
                 #'tfidf__use_idf': (True, False),
                 'alpha': (1,0,0.01, 0.001, 0.0001)}

svm_parameters = {#'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
                 #'tfidf__use_idf': (True, False),
                 'C': (1, 10, 100, 1000),
                 'kernel': ('linear', 'rbf'),
                 'gamma': (0.0001, 0.001, 0.01, 0.1)}
xgb_parameters = {#'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
                 #'tfidf__use_idf': (True, False),
                 'n_estimators':[1000,1500,2000],
                 'max_depth':[3],
                 'subsample':[0.5],
                 'learning_rate':[0.01, 0.02, 0.03, 0.04, 0.05],
                 'min_samples_leaf': [1],
                 'random_state': [3]}


from sklearn import ensemble
from sklearn.svm import SVC

def getXGBPipe():
    clf_xgb_pipe = Pipeline([('vect', CountVectorizer()), 
                              ('tfidf', TfidfTransformer()),
                              ('clf', ensemble.GradientBoostingClassifier())])
def getNaiveBayesPipe():
    clf_nb_pipe = Pipeline([('vect', CountVectorizer()), 
                              ('tfidf', TfidfTransformer()),
                              ('clf', MultinomialNB())])
    return clf_nb_pipe


def getSVMPipe():
    clf_svm_pipe = Pipeline([('vect', CountVectorizer()), 
                              ('tfidf', TfidfTransformer()),
                              ('clf', SVC())])
    return clf_svm_pipe





gs_clf_nb = GridSearchCV(MultinomialNB(), nb_parameters, n_jobs=-1, cv=10)
gs_clf_nb_fit = gs_clf_nb.fit(X_train_tfidf[:len(X_train)], y_train)
best_nb_clf_fit = gs_clf_nb_fit.best_estimator_
print("The best alpha: ", best_nb_clf_fit.alpha)
y_pred = best_nb_clf_fit.predict_proba(X_test_tfidf[:len(X_test)])
generate_results(y_pred, y_test)




X_train_tfidf = get_bag_of_words(pd.concat([df_train["text"], df_test['text']]))
y_train = df_train["author"]
X_test_tfidf = get_bag_of_words(pd.concat([df_test["text"], df_train["text"]]))
y_pred = []

clf = best_nb_clf_fit.fit(X_train_tfidf[:len(df_train)], y_train)
y_pred = clf.predict_proba(X_test_tfidf[:len(df_test)])
results = pd.DataFrame({'id':df_test["id"]})
results[clf.classes_] = pd.DataFrame(y_pred)

# For the results, I need a table like the following
#
# id | P(author1) | P(author2) | P(author3)
#
results.to_csv("11102017_2_bestNB.csv")




from sklearn.cluster import KMeans
#log_loss(y_pred, y_true)

kmeans_column_train = KMeans(n_clusters=3, random_state=0).fit(X_train_tfidf[:len(X_train)]).labels_
kmeans_column_test = KMeans(n_clusters=3, random_state=0).fit(X_test_tfidf[:len(X_test)]).labels_

# Cast into a usable form to append as a column
kmeans_column_train = np.matrix(kmeans_column_train).T
kmeans_column_test = np.matrix(kmeans_column_test).T

print("Kmeans columns created!")




# Append the columns
X_train_raw = np.matrix(X_train_tfidf[:len(X_train)].toarray())
X_test_raw = np.matrix(X_test_tfidf[:len(X_test)].toarray())

X_train_new = np.concatenate((X_train_raw, kmeans_column_train), axis=1)
X_test_new = np.concatenate((X_test_raw, kmeans_column_test), axis=1)

clf = best_nb_clf_fit.fit(X_train_new, y_train)
y_pred = clf.predict_proba(X_test_new)

generate_results(y_pred, y_test)





# 1. PCA on X_train_tfidf and X_test_tfidf
from sklearn.decomposition import PCA
pca = PCA().fit(np.matrix(X_train_tfidf[:len(X_train)].toarray()))
pca

# 2. Visualize to pick the top K components that maximize variance

# 3. Kmeans on our top K columns

# 4. with PCA + Kmeans run NB on it

# 5. Predict on NB and report out how well we did




from sklearn.cluster import KMeans

X_train_tfidf = get_bag_of_words(pd.concat([df_train["text"], df_test['text']]))
y_train = df_train["author"]
X_test_tfidf = get_bag_of_words(pd.concat([df_test["text"], df_train["text"]]))
y_pred = []
print("Data matrix created!")

kmeans_column_train = KMeans(n_clusters=3, random_state=0).fit(X_train_tfidf[:len(df_train["text"])]).labels_
kmeans_column_test = KMeans(n_clusters=3, random_state=0).fit(X_test_tfidf[:len(df_test["text"])]).labels_
kmeans_column_train = np.matrix(kmeans_column_train).T
kmeans_column_test = np.matrix(kmeans_column_test).T

print("Kmeans columns created!")


X_train_raw = np.matrix(X_train_tfidf[:len(df_train["text"])].toarray())
X_train_new = np.concatenate((X_train_raw, kmeans_column_train), axis=1)
X_test_raw = np.matrix(X_test_tfidf[:len(X_test)].toarray())
X_test_new = np.concatenate((X_test_raw, kmeans_column_test), axis=1)
print("Column appended!")

clf = best_nb_clf_fit.fit(X_train_new, y_train)
print("Estimator built!")
y_pred = clf.predict_proba(X_test_new)
print("Predictions casted!")


results = pd.DataFrame({'id':df_test["id"]})
results[clf.classes_] = pd.DataFrame(y_pred)

# For the results, I need a table like the following
#
# id | P(author1) | P(author2) | P(author3)
#
results.to_csv("11102017_2_kmeans.csv")
print("Writing out!")




gs_clf_xgb = GridSearchCV(ensemble.GradientBoostingClassifier(), xgb_parameters, n_jobs=-1, cv=3)
gs_clf_xgb_fit = gs_clf_xgb.fit(np.matrix(X_train_tfidf[:len(X_train)], y_train)
best_xgb_clf_fit = gs_clf_xgb_fit.best_estimator_
print("The best xgb: ", best_nb_clf_fit)
y_pred = best_xgb_clf_fit.predict_proba(X_test_tfidf[:len(X_test)])
generate_results(y_pred, y_test)




gs_clf_xgb = ensemble.GradientBoostingClassifier().fit(np.matrix(X_train_tfidf[:len(X_train)].toarray()),
                                                       y_train)
print("Classifier fit!")
y_pred = best_xgb_clf_fit.predict_proba(np.matrix(X_test_tfidf[:len(X_test)].toarray()))
print("Classifier predictions casted!!")
generate_results(y_pred, y_test)












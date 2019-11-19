import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(1)

# Load the data
data = pd.read_csv('spam.csv', usecols = [0,1], encoding='latin-1')
data.columns = ['label', 'text']
print("The data has", data.shape[0], "row(s) and ", data.shape[1], "column(s)")
print(data.head(), '\n')


# Get a word count and unique word count of the text
print(data.groupby('label').describe())


# print(data.groupby('label').describe())

# data['length'] = data.text.str.len()
# data['cat'] = data['label'].map({'ham' : 1, 'spam' : 0})
# print(data)

stop_words = set(stopwords.words('english'))

##remove punc and stop
# def remove_punc_stop(text):
#     sen = [i for i in text if i not in string.punctuation]
#     word = "".join(sen).split()
#     word = [i.lower() for i in word if i.lower() not in stopwords.words("english")]
#     word = " ".join(word)
#     return word
#
# data['cleaned text'] = data.text.apply(remove_punc_stop)
# print(data['cleaned text'])


##counting the occurrences of tokens in each document.
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['text']).toarray()

print(x)
le = LabelEncoder()
y = le.fit_transform(data['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, shuffle = True)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=15,criterion='entropy')
classifier.fit(x_train,y_train)
predRF=classifier.predict(x_test)
# for i in range(len(x_test)):
#     print(x_test[i], predRF[i])
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))
print('Precision score: {}'.format(precision_score(y_test, predRF)))
print('Recall score: {}'.format(recall_score(y_test, predRF)))
print('F1 score: {}'.format(f1_score(y_test, predRF)))


##To avoid these potential discrepancies it suffices to divide the number of occurrences
##of each word in a document by the total number of words in the document:
##these new features are called tf for Term Frequencies.
# vectorizer2 = TfidfVectorizer()
# data_X2 = vectorizer2.fit_transform(data['text'])
#
dictionary = np.array(vectorizer.get_feature_names())
print(dictionary)
# dictionary2 = np.array(vectorizer2.get_feature_names())
#


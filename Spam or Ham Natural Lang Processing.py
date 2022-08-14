#spam or ham NLP data
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

#get data

messages = pd.read_csv("smsspamcollection/SMSSpamCollection", sep="\t", names = ['label', 'message'])

messages['length'] = messages['message'].apply(len)

#Data analysis
messages['length'].plot.hist(bins = 50)
plt.show()

messages.hist(column = 'length', by = 'label', bins = 60)
plt.show()

#Process data
nopunc = [c for c in mess if c not in string.punctuation]

from nltk.corpus import stopwords
stopwords = stopwords.words('english')
nopunc = ''.join(nopunc)

clean_mess = [word for word in nopunc if word.lower() not in stopwords]
def text_process(mess):
    '''
    1.remove punc
    2.remove stop words
    3. return list of clean text words
    '''
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords]


messages['message'].head().apply(text_process)

#Count Vectorizer - bag of words
#TfidfTransformer - term-frequency times inverse document-frequency
#MultinomialNB - classification for discrete features
 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split




msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size = 0.3)


#create pipeline
from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer = text_process)),
    ('tfdidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

#evaluate model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(label_test, predictions))
print(confusion_matrix(label_test, predictions))





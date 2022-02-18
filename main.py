# VAIVO19256

# Imports
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords


def readCSV(path):
    data = pd.read_csv(path)
    return data


def preprocess(X):
    documents = []

    stemmer = WordNetLemmatizer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Removing unnecessary whitespace
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    return documents


if __name__ == '__main__':
    # Get data form csv
    data = readCSV("IMDB Dataset.csv")

    X, y = data.review, data.sentiment


    # Data preprocessing - remove punctuation, lowercase, remove break tags, remove numbers, lemminzation
    print("CLEANING DATA")
    documents = preprocess(X)

    # Vectorize the data and remove stopwords

    print("VECTORIZING")
    vectorizer = CountVectorizer(max_features=300, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()


    # Calculate tfidf
    print("CALCULATING TFIDF")
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()


    # Split into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Training Text Classification Model and Predicting Sentiment
    print("FITTING MODEL")
    clf = RandomForestClassifier(n_estimators=1500, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("PREDICTING SENTIMENT")
    y_pred = clf.predict(X_test)

    # Testing accuracy
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # Save model for later use
    with open('text_classifier', 'wb') as picklefile:
        pickle.dump(clf, picklefile)

    # print(classifier.feature_importances_)

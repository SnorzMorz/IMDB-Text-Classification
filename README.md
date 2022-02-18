# IMDB-Text-Classification
Classifying the sentiment of IMDB Reviews using a Random Forest Classifier

1.	To read the file form CSV I use the pandas library, This allows to immediately turn the csv data into a dataset for easier use

2.	For pre-processing I apply multiple data cleaning methods. I firstly remove all special characters (any non-word characters), then I remove all single character values, remove useless whitespace, lowercase the text and finally I perform lemmatisation (only leave the root of the word), by splitting text into a list of words and then for each calling the stemmer from the nltk library

3.	The model I’m using requires that the input data to be vectorized, so I use the CountVectorizer from sklearn to vectorize the data. To improve the speed of runtime I have set max_features to 300, also I remove all the stop words using the stop words from nltk.corpus.

4.	After vectorizing I calculate the tf-idf(term frequency-inverse document frequency) for the words

5.	Split the dataset into training and testing datasets (80/20 split)

6.	Train the model using the training dataset. For this project I use the RandomForestClassifier with n_estimators at 1500

7.	Predict the sentiment of the testing dataset

8.	Test the accuracy of the model using sklearns libraries - classification report and accuracy score.

9.	Save the model for later use using the pickle library

## Conclusion / Future improvements
In summary, the model has an accuracy of around  0.8, which I think  isn’t very good. To improve the model, I tried playing around with different values for max_features and n_estimators. This improves the accuracy but also expectedly made the program run much longer. So, for the demo I’ve set the values lower than you might do in production. Also, changing the max_depth of the classifier could improve performance in the expense of runtime. To find the best values for these it would be useful to use the GridSearchCV library from sklearn, it allows to search for optimal parameters. 

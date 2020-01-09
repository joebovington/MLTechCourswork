import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import re
import nltk
import unidecode
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
from nltk.corpus import stopwords
from langdetect import detect_langs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

trainingData = pd.read_csv("Data/trainingData.csv")
testingData = pd.read_csv("Data/testingData.csv")
trainingData = trainingData.sample(frac=1)
trainingData.reset_index(drop=True, inplace=True)

training_x = trainingData['tweetText']
training_labels = trainingData['label']

testing_x = testingData['tweetText']
testing_labels = testingData['label']


def label(labels):
    y = []
    for k in range(0, len(labels)):
        if labels[k] == 'real':
            y.append(1)
        else:
            y.append(0)
    return y


training_y = label(training_labels)
testing_y = label(testing_labels)


def preprocess(dat, langs, hashtags):
    processed_dat = []
    for i in range(0, len(dat)):
        # len(data)):
        tweet = dat[i]
        # removes links
        tweet = re.sub('http([^\s])*', '', tweet)
        # strips accents
        tweet = unidecode.unidecode(tweet)
        # removes special characters
        tweet = re.sub(r'[^\w#\s]', '', tweet)
        # gets tweet language
        try:
            lang = detect_langs(tweet)
            langs.append(lang[0].lang)
        except:
            # print("no lang detected")
            lang = "null"
            langs.append(lang)
        # gets hashtags
        hasht = re.findall(r'(\s#\w+)', tweet)
        hashtag = ""

        for j in range(0, len(hasht)):
            hasht[j] = re.sub(r'#', '', hasht[j])
            hashtag = hashtag + hasht[j]

        hashtags.append(hashtag)
        # removes hashtags from tweet
        tweet = re.sub(r'(\s#[a-zA-Z]+)', '', tweet)
        # removes single characters
        tweet = re.sub(r'\s+[a-zA-Z]\s+', '', tweet)
        # removes multiple whitespaces
        tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
        # converts to lowercase
        tweet = tweet.lower()
        # tweet = lang[0].lang + hashtag + tweet

        processed_dat.append(tweet)
    return processed_dat


training_hashtags = []
training_language = []
training_tweets = preprocess(training_x, training_language, training_hashtags)
df_training = pd.DataFrame(
    {'tweets': training_tweets, 'hashtags': training_hashtags, 'language': training_language, 'labels': training_y})

testing_hashtags = []
testing_language = []
testing_tweets = preprocess(testing_x, testing_language, testing_hashtags)
df_testing = pd.DataFrame(
    {'tweets': testing_tweets, 'hashtags': testing_hashtags, 'language': testing_language, 'labels': testing_y})

v = TfidfVectorizer(max_features=1500, stop_words=stopwords.words('english'))
df_training['tweets'] = list(v.fit_transform(df_training['tweets']).toarray())
df_testing['tweets'] = list(v.transform(df_testing['tweets']).toarray())

v2 = TfidfVectorizer()
df_training['hashtags'] = list(v2.fit_transform(df_training['hashtags']).toarray())
df_testing['hashtags'] = list(v2.transform(df_testing['hashtags']).toarray())

v3 = CountVectorizer()
df_training['language'] = list(v3.fit_transform(df_training['language']).toarray())
df_testing['language'] = list(v3.transform(df_testing['language']).toarray())


def concat_features(tweets, language, hashtags):
    vector = []
    for i in range(0, len(tweets)):
        tweet = tweets[i]
        lan = language[i]
        hash = hashtags[i]
        v = np.concatenate([tweet, lan, hash])
        vector.append(v)
    return vector


df_training['vector'] = concat_features(df_training['tweets'], df_training['language'], df_training['hashtags'])
df_testing['vector'] = concat_features(df_testing['tweets'], df_testing['language'], df_testing['hashtags'])
# data_train, data_test, label_train, label_test = train_test_split(df['vector'], df['labels'], test_size=0.2)
data_train = np.asarray(df_training['vector'].tolist())
label_test = df_testing['labels']


def svc_param_selection(X, y, nfolds):
    Cs = [0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    print('param select')
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search.best_params_


# params = svc_param_selection(data_train, label_train, 5)
# svm = svm.SVC(C=params['C'], gamma=params['gamma'])
# svm = svm.SVC(C=11, gamma=0.075)
svm = svm.SVC()
svm.fit(data_train, df_training['labels'])
predicted = svm.predict(np.asarray(df_testing['vector'].tolist()))

# dt = DecisionTreeClassifier(max_depth=50)
# dt.fit(data_train, df['labels'])
# predicted = dt.predict(np.asarray(df_testing['vector'].tolist()))

# clf = RandomForestClassifier(max_depth=5)
# clf.fit(data_train, df['labels'])
# predicted = clf.predict(np.asarray(df_testing['vector'].tolist()))

# gb = GaussianNB()
# gb.fit(data_train, label_train)
# predicted = gb.predict(np.asarray(data_test.tolist()))

precision, recall, fscore, support = precision_recall_fscore_support(label_test, predicted, zero_division=1,
                                                                     average='binary')
# precision_score = precision_score(label_test, predicted)
# recall_score = recall_score(label_test, predicted)
f1 = f1_score(label_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('f1score: {}'.format(f1))

# dt = DecisionTreeClassifier(max_depth=2)
# dt.fit(df, labels)

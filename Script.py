import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import re
import nltk
import unidecode
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn_pandas import DataFrameMapper
from langdetect import detect_langs
from sklearn.feature_extraction.text import TfidfVectorizer

trainingData = pd.read_csv("Data/mediaeval-2015-trainingset.txt", sep="	")
testingData = pd.read_csv("Data/mediaeval-2015-testset.txt", sep="	")
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


def preprocess(dat):
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
            lang = lang[0].lang
        except:
            # print("no lang detected")
            lang = "null"
        # gets hashtags
        hasht = re.findall(r'(\s#\w+)', tweet)
        hashtag = ""

        for j in range(0, len(hasht)):
            hashtag = hashtag + hasht[j]

        # removes hashtags from tweet
        tweet = re.sub(r'(\s#[a-zA-Z]+)', '', tweet)
        # removes single characters
        tweet = re.sub(r'\s+[a-zA-Z]\s+', '', tweet)
        # removes multiple whitespaces
        tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
        # converts to lowercase
        tweet = tweet.lower()
        # concatenates features
        tweet = tweet + hashtag + lang
        processed_dat.append(tweet)
    return processed_dat


training_tweets = preprocess(training_x)

df_training = pd.DataFrame(
    {'tweets': training_tweets, 'labels': training_y})

testing_tweets = preprocess(testing_x)

df_testing = pd.DataFrame(
    {'tweets': testing_tweets, 'labels': testing_y})

v = TfidfVectorizer(max_features=10000)
mapper = DataFrameMapper([('tweets', v)], sparse=True)

training_data = mapper.fit_transform(df_training[['tweets']])
testing_data = mapper.transform(df_testing[['tweets']])

# def concat_features(tweets, language, hashtags):
#   vector = []
#  for i in range(0, len(tweets)):
#     tweet = tweets[i]
#    lan = language[i]
#   hash = hashtags[i]
#  v = np.concatenate([tweet, lan, hash])
# vector.append(v)
# return vector


data_train = df_training[['tweets']]
label_test = df_testing['labels']

# def svc_param_selection(X, y, nfolds):
#    Cs = [0.5, 1, 2]
#    gammas = [0.00001, 0.0001, 0.001]
#    param_grid = {'C': Cs, 'gamma': gammas}
#   print('param select')
#  grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# return grid_search.best_params_


# params = svc_param_selection(training_data, df_training['labels'], 5)
# svm = svm.SVC(C=params['C'], gamma=params['gamma'])
# svm = svm.SVC(kernel='linear')
svm = svm.SVC(kernel='linear')
svm.fit(data_train, df_training['labels'])
predicted = svm.predict(testing_data)

# dt = DecisionTreeClassifier(max_depth=50)
# dt.fit(training_data, df_training['labels'])
# predicted = dt.predict(testing_data)

# RF = RandomForestClassifier(max_depth=50)
# RF.fit(training_data, df_training['labels'])
# predicted = RF.predict(testing_data)

# gb = GaussianNB()
# gb.fit(training_data.toarray(), df_training['labels'])
# predicted = gb.predict(testing_data.toarray())

precision, recall, fscore, support = precision_recall_fscore_support(label_test, predicted,
                                                                     average='weighted')
f1 = f1_score(label_test, predicted, average='weighted')

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('f1score: {}'.format(f1))


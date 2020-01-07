import pandas as pd
import numpy as np
from sklearn import svm
import re
import nltk
import unidecode
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

nltk.download('stopwords')
from nltk.corpus import stopwords
from langdetect import detect_langs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

trainingData = pd.read_csv("Data/trainingData.csv")
trainingData = trainingData.sample(frac=1)
trainingData.reset_index(drop=True, inplace=True)

x = trainingData['tweetText']
labels = trainingData['label']

y = []
for k in range(0, len(labels)):
    if labels[k] == 'real':
        y.append(1)
    else:
        y.append(0)


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
            print("no lang detected")
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


# def merge(list1, list2, list3):
#    merged_list = [(list1[i], list2[i], list3[i]) for i in range(0, len(list1))]
#    return merged_list

# narray = np.array([tweets, hashtags, langs])
# tup = merge(tweets, hashtags, langs)

# nTweets = np.array(tweets)
# nHashtags = np.array(hashtags)
# nLanguages = np.array(langs)
hashtags = []
language = []
tweets = preprocess(x, language, hashtags)
df = pd.DataFrame({'tweets': tweets, 'hashtags': hashtags, 'language': language, 'labels': y})

# vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
# max_features=1500, min_df=5, max_df=0.7,
# vectorizer.fit(data)
# data = vectorizer.transform(data)
# vectorizer.fit(df)
# df = vectorizer.transform(df)
v = TfidfVectorizer(max_features=1500, stop_words=stopwords.words('english'))
df['tweets'] = list(v.fit_transform(df['tweets']).toarray())
v2 = TfidfVectorizer()
df['hashtags'] = list(v2.fit_transform(df['hashtags']).toarray())
v3 = CountVectorizer()
df['language'] = list(v3.fit_transform(df['language']).toarray())


def concat_features(tweets, language, hashtags):
    vector = []
    for i in range(0, len(tweets)):
        tweet = tweets[i]
        lan = language[i]
        hash = hashtags[i]
        v = np.concatenate([tweet, lan, hash])
        vector.append(v)
        # print(vector)
    return vector


df['vector'] = concat_features(df['tweets'], df['language'], df['hashtags'])

data_train, data_test, label_train, label_test = train_test_split(df['vector'], df['labels'], test_size=0.2)

data_train = np.asarray(data_train.tolist())
# print(x)

# for j in range(0, len(x)):
#    print(x[j])

svm = svm.SVC()
svm.fit(data_train, label_train)

predicted = svm.predict(np.asarray(data_test.tolist()))
# print(predicted)

precision, recall, fscore, support = precision_recall_fscore_support(label_test, predicted)
f1 = f1_score(label_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print('f1score: {}'.format(f1))

# dt = DecisionTreeClassifier(max_depth=2)
# dt.fit(df, labels)



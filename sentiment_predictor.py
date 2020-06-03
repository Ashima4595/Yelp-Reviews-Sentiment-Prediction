"""
Project - Sentiment Predictor
Developer - Ashima Munjal
"""
# Loading dependencies.
import sys
import time
import re
import pickle
import pandas as pd
import numpy as np

# Natural Language Processing Libraries.
from nltk.stem import PorterStemmer
from textblob import TextBlob

# Machine Learning Libraries.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

def clean_text(txt):
    '''
    This function is used for text processing. It is used for cleaning the
    textual information by removing URLs, mentions, hyperlinks, punctuations,
    tags, special characters and digits.
    Parameters:
    txt -- String. String containing text data to be cleaned.
    Return:
    txt -- String. String containing cleaned text.
    '''
    # Compile Regex Information.
    # URL Regex.
    url_reg = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')

    # Regex to treat "@mentions".
    mention_reg = re.compile(r'@(\w+)')

    # Remove hyperlinks.
    txt = url_reg.sub(' ', txt)

    # Remove text containing "@mentions".
    txt = mention_reg.sub(' ', txt)

    # Removing punctuations.
    txt = re.sub('[^a-zA-Z]', ' ', txt)

    # Convert to lower case.
    txt = txt.lower()

    # Remove tags.
    txt = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", txt)

    # Remove special characters and digits.
    txt = re.sub("(\\d|\\W)+", " ", txt)

    # Convert to list from string by splitting on "space" character.
    txt = txt.split(" ")

    # Remove empty strings.
    txt = [wrd for wrd in txt if wrd != ""]

    # Form sentences from words.
    txt = " ".join(txt)

    return txt

def filter_stopwords(txt):
    '''
    This function is used in removing words from the text that do not represent
    a sentiment and therefore,are useless and can be removed while doing text
    processing. In natural language processing useless words(data), are
    referred to as stop words.
    Parameters:
    txt -- String. String containing text data to be filtered.
    Return:
    txt -- String. Filtered text data.
    '''
    # Stop Words list from "NLTK" library.
    stop_wrds = ["out", "we", "was", "how", "myself", "for", "they", "about", "hasn't", "doesn't",
                 "then", "both", "so", "re", "don", "m", "as", "any", "mightn", "after",
                 "you", "wouldn", "why", "been", "where", "by", "isn't", "just", "she's",
                 "yourself", "wasn", "a", "haven't", "did", "hadn't", "their", "hasn", "him",
                 "doing", "be", "further", "ours", "now", "am", "her", "you'll", "and", "that'll",
                 "yourselves", "that", "my", "what", "to", "d", "not", "won't", "couldn't",
                 "own", "there", "this", "each", "all", "haven", "more", "me", "ve", "weren",
                 "which", "himself", "nor", "other", "shouldn't", "who", "should've", "same",
                 "at", "such", "t", "up", "than", "can", "you've", "too", "these", "while",
                 "wasn't", "ourselves", "before", "i", "he", "didn't", "our", "its", "but", "with",
                 "wouldn't", "those", "because", "the", "y", "shouldn", "it", "mustn", "hers",
                 "doesn", "ain", "between", "over", "had", "aren", "mightn't", "does", "have",
                 "or", "some", "mustn't", "only", "won", "when", "needn", "below", "in", "if",
                 "theirs", "needn't", "aren't", "isn", "again", "his", "whom", "ll", "hadn",
                 "above", "should", "itself", "themselves", "until", "are", "she", "no", "from",
                 "into", "will", "your", "few", "here", "is", "s", "don't", "shan't", "during",
                 "herself", "of", "has", "down", "were", "once", "ma", "having", "them", "under",
                 "shan", "couldn", "do", "on", "an", "you\'d", "yours", "being", "off", "o",
                 "very", "weren't", "didn", "through", "you're", "most", "against", "it's"]

    # Convert to list from string by splitting on "space" character.
    txt = txt.split(" ")

    # Words describing relationships.
    rl_wrds = ['guy', 'spokesman', 'chairman', "men's", 'men', 'him', "he's", 'his',
               'boy', 'boyfriend', 'boyfriends', 'boys', 'brother', 'brothers', 'dad',
               'dads', 'dude', 'father', 'fathers', 'fiance', 'gentleman', 'gentlemen',
               'god', 'grandfather', 'grandpa', 'grandson', 'groom', 'he', 'himself',
               'husband', 'husbands', 'king', 'male', 'man', 'mr', 'nephew', 'nephews',
               'priest', 'prince', 'son', 'sons', 'uncle', 'uncles', 'waiter', 'widower',
               'widowers', 'heroine', 'spokeswoman', 'chairwoman', "women's", 'actress',
               "she's", 'her', 'aunt', 'aunts', 'bride', 'daughter', 'daughters', 'female',
               'fiancee', 'girl', 'girlfriend', 'girlfriends', 'girls', 'goddess',
               'granddaughter', 'grandma', 'grandmother', 'herself', 'ladies', 'lady',
               'lady', 'mom', 'moms', 'mother', 'mothers', 'mrs', 'ms', 'niece', 'nieces',
               'priestess', 'princess', 'queens', 'she', 'sister', 'sisters', 'waitress',
               'widow', 'widows', 'wife', 'wives', 'woman', 'women']

    # Words representing utterances.
    utterance_wrds = ["um", "huh"]

    # Complete Stop Word List.
    stop_wrds += rl_wrds + utterance_wrds

    # Removing stop words.
    txt = [word for word in txt if word not in stop_wrds]

    # Form sentences from words.
    txt = " ".join(txt)

    return txt

def stemmer(txt):
    '''
    This function is used to apply stemming to words in text data.
    Parameters:
    txt -- String. String containing unprocessed text data.
    Return:
    txt -- String. Stemmed text data.
    '''
    port_stmr = PorterStemmer()
    # Convert to list from string by splitting on "space" character.
    txt = txt.split(" ")
    stemmed = []
    for ele in txt:
        stemmed += [port_stmr.stem(ele)]

    # Form sentences from words.
    txt = " ".join(stemmed)
    return txt

def sentiment_scr(txt):
    '''
    This function is used to calculate sentiment score on the text data.
    Parameters:
    txt -- String. String containing text data.
    Return:
    scr -- Float. Sentiment score for the text.
    '''
    scr = TextBlob(txt).sentiment.polarity

    return scr

def label_threshold(scr):
    '''
    This function defines label threshold for sentiment score.
    Parameters:
    scr -- Float. Sentiment score for the text.
    Return:
    label -- Integer. Sentiment label 1 - Positive, 0 - Negative.
    '''
    if scr >= 0.5:
        label = 1
    else:
        label = 0

    return label

def training_classifier(data_df):
    '''
    This function is used to train the model on training data and store the model as pickle files.
    Parameters:
    data_df -- Pandas Dataframe Object. Dataframe containing review data and star rating.
    Store:
    count_vctrzr -- Model. Model containing Count Vectorizer is stored.
    clf -- Model. Model containing the GBM Classifer is stored.
    '''

    # Apply text processing to text data.
    print("\nCleaning Text.")
    data_df["Cleaned_Text"] = data_df["text"].apply(clean_text)
    print("\nFiltering Text.")
    data_df["Filtered_Text"] = data_df["Cleaned_Text"].apply(filter_stopwords)
    # Apply stemming to identify root words.
    print("\nStemming Words.")
    data_df["Stemmed_Text"] = data_df["Filtered_Text"].apply(stemmer)
    # Get Sentiment Score.
    print("\nCompute Sentiment Score.")
    data_df["Sentiment_Score"] = data_df["Stemmed_Text"].apply(sentiment_scr)
    # Get Weighted Score.
    data_df["Score"] = data_df["Sentiment_Score"] * data_df["stars"]
    # Assign Labels.
    data_df["Label"] = data_df["Score"].apply(label_threshold)
    print("\nLabel Sentiment.")
    ml_df = data_df[["Stemmed_Text", "Label"]]
    split = np.random.rand(len(ml_df)) < 0.8
    train = ml_df[split]
    test = ml_df[~split]
    clean_train_corpus = train["Stemmed_Text"].values.tolist()
    clean_test_corpus = test["Stemmed_Text"].values.tolist()
    train_label = train["Label"].values.tolist()
    test_label = test["Label"].values.tolist()

    # Training Label Distribution.
    print("\nTraining Label Distribution:\n")
    print(train['Label'].value_counts())

    # Testing Label Distribution.
    print("\nTesting Label Distribution:\n")
    print(test['Label'].value_counts())

    # Creating Vectorizer.
    count_vctrzr = CountVectorizer()
    count_vctrzr.fit(clean_train_corpus)

    # Storing Vectorizer model.
    pickle.dump(count_vctrzr, open("sentiment_analyzer_count_vector.pickle", "wb"))

    # Fit transform with the Training data.
    train_vctr = count_vctrzr.fit_transform(clean_train_corpus)
    print("\nTraining Vector Shape:", train_vctr.shape)

    # Transform the testing data.
    test_vctr = count_vctrzr.transform(clean_test_corpus)
    print("\nTesting Vector Shape:", test_vctr.shape)

    # Train Classifier.
    clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500, max_depth=24,
                                     min_samples_split=3, min_samples_leaf=3,
                                     max_features='sqrt', random_state=42)
    # Train all Classifier on the training Data.
    clf.fit(train_vctr, train_label)

    # Storing Classifier Model.
    pickle.dump(clf, open("sentiment_analyzer_gbm.pickle", 'wb'))

    # Predict Test Data.
    predictions = clf.predict(test_vctr)

    # Accuracy.
    print("\nClassifier Accuracy for our data:",
          accuracy_score(predictions, test_label)*100, "%")

def testing(data_df):
    '''
    This function is used to predict sentiment for the testing data and store it in a text file.
    Parameters:
    data_df -- Pandas Dataframe Object. Dataframe containing text data to predict sentiment.
    Store:
    predictions -- List. Prediction for text data stored in text file.
    '''
    run_time = time.time()
    try:
        # Load Vectorizer Model.
        count_vctrzr = pickle.load(open("sentiment_analyzer_count_vector.pickle", 'rb'))
        # Load Classifier Model.
        clf = pickle.load(open("sentiment_analyzer_gbm.pickle", 'rb'))

    except FileNotFoundError:
        # Loading training data.
        print("\nLoading the Training Data and Label.")
        training_file = open("training_file.txt", "r")
        data_train = []
        for line in training_file.readlines():
            data_train.append(line)

        # Loading labels.
        training_labl_file = open("training_label_file.txt", "r")
        data_label = []
        for line in training_labl_file.readlines():
            data_label.append(int(line.strip("\n")))

        # Creating dataframe.
        data_df = pd.DataFrame({"text":data_train,
                                "stars":data_label})
        print('\nLoading Complete.')
        print("\nTraining Data Shape:", data_df.shape)
        print("\nTraining the model.")
        training_classifier(data_df)
        print('\nTraining complete.')

    # Load Vectorizer Model.
    count_vctrzr = pickle.load(open("sentiment_analyzer_count_vector.pickle", 'rb'))
    # Load Classifier Model.
    clf = pickle.load(open("sentiment_analyzer_gbm.pickle", 'rb'))

    # Apply text processing to text data.
    print("\nCleaning Text.")
    data_df["Cleaned_Text"] = data_df["text"].apply(clean_text)
    print("\nFiltering Text.")
    data_df["Filtered_Text"] = data_df["Cleaned_Text"].apply(stemmer)
    # Apply stemming to identify root words.
    print("\nStemming Words.")
    data_df["Stemmed_Text"] = data_df["Filtered_Text"].apply(filter_stopwords)
    clean_test_corpus = data_df["Stemmed_Text"].values.tolist()

    # Transform the testing data.
    test_vctr = count_vctrzr.transform(clean_test_corpus)
    print("\nTesting Vector Shape:", test_vctr.shape)

    # Predict Test Data.
    predictions = clf.predict(test_vctr)

    # Output File.
    file = open('output.txt', 'w')
    for ele in predictions:
        file.write(str(ele) + '\n')
    file.close()
    print("\nTotal Testing Runtime:", int(time.time()-run_time), "seconds.")


def main():
    '''
    This function is used to run the entire project.
    '''
    strt_time = time.time()
    print("\nInput to this Script is as follows: ")
    print("\nModes: 'testing', 'training'")
    print("\nExample Input:")
    print("\n'training' 'training_file.txt' 'training_label_file.txt'")
    print("\n'testing' 'sample_testing.txt'")
    try:
        if sys.argv[1] == "training":
            # Loading training data.
            print("\nTraining the model.")
            training_file = open(sys.argv[2], "r")
            data_train = []
            for line in training_file.readlines():
                data_train.append(line)

            # Loading labels.
            training_labl_file = open(sys.argv[3], "r")
            data_label = []
            for line in training_labl_file.readlines():
                data_label.append(int(line.strip("\n")))

            # Creating dataframe.
            data_df = pd.DataFrame({"text":data_train,
                                    "stars":data_label})
            print("\nTraining Data Shape:", data_df.shape)
            training_classifier(data_df)
        else:
            print("\nTesting the model.")
            file1 = open(sys.argv[2], "r")
            data = []
            for line in file1.readlines():
                data.append(line)
            data_df = pd.DataFrame({"text":data})
            testing(data_df)
    except IndexError:
        print("\nPlease provide the input as requested.")
        print("\nMake sure all the data files are present in the directory.")
        print("\nDefault run.")
        print("\nTraining the model.")

        # Loading training data.
        print("\nLoading File : training_file.txt")
        training_file = open("training_file.txt", "r")
        data_train = []
        for line in training_file.readlines():
            data_train.append(line)

        # Loading labels.
        print("\nLoading File : training_label_file.txt")
        training_labl_file = open("training_label_file.txt", "r")
        data_label = []
        for line in training_labl_file.readlines():
            data_label.append(int(line.strip("\n")))

        # Creating dataframe.
        data_df = pd.DataFrame({"text":data_train,
                                "stars":data_label})
        print("\nTraining Data Shape:", data_df.shape)
        training_classifier(data_df)
    print("\nTotal Runtime:", int(time.time()-strt_time), "seconds.")

if __name__ == "__main__":
    main()

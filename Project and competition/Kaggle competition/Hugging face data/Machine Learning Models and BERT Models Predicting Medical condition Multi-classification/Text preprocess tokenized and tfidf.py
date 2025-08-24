import pandas as pd
import scispacy
import spacy
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

train_df= pd.read_csv('/mnt/d/kaggle_trainset.csv')
test_df= pd.read_csv('/mnt/d/kaggle_testset.csv')

nlp = spacy.load('en_core_sci_md')
nlp = spacy.load('en_core_sci_lg')
nlp = spacy.load('en_ner_bionlp13cg_md')

def process_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]      
    pos_tags = [token.pos_ for token in doc]   
    lemmas = [token.lemma_ for token in doc]    
    return tokens, pos_tags, lemmas

def dataPreprocessing_no_stopwords_remove_parentheses_with_numbers(x):
    # x = x.lower()
    x = re.sub('\s+', ' ', x)
    x = re.sub(r'\(\d+\)', '', x)
    x = re.sub('\d+', '', x)
    x = re.sub(r'\.+', '.', x)
    x = re.sub(r'\,+', ',', x)
    x = x.strip()
    return x

train_df['condition'] = train_df['condition'].apply(dataPreprocessing_no_stopwords_remove_parentheses_with_numbers)
test_df['condition'] = test_df['condition'].apply(dataPreprocessing_no_stopwords_remove_parentheses_with_numbers)

label_map = {"neoplasms": 1,
             "digestive system diseases": 2,
             "nervous system diseases": 3,
             "cardiovascular diseases": 4,
             "general pathological conditions": 5}
train_df["label"] = train_df["label"].map(label_map)

processed = train_df['condition'].apply(process_text)
train_df['tokens'], train_df['pos_tags'], train_df['lemmas'] = zip(*processed)
processed = test_df['condition'].apply(process_text)
test_df['tokens'], test_df['pos_tags'], test_df['lemmas'] = zip(*processed)

stop_words = set(stopwords.words('english'))

train_df['filtered_tokens'] = train_df['lemmas'].apply(lambda x: [token for token in x if token.lower() not in stop_words])
test_df['filtered_tokens'] = test_df['lemmas'].apply(lambda x: [token for token in x if token.lower() not in stop_words])

train_df.drop_duplicates(subset=['condition'], keep=False, inplace=True)
train_df.duplicates(subset=['condition'], keep=False, inplace=True)

processed = train_df['condition'].apply(process_text)
train_df['tokens'], train_df['pos_tags'], train_df['lemmas'] = zip(*processed)
processed = test_df['condition'].apply(process_text)
test_df['tokens'], test_df['pos_tags'], test_df['lemmas'] = zip(*processed)

stop_words = set(stopwords.words('english'))

train_df['filtered_tokens'] = train_df['lemmas'].apply(lambda x: [token for token in x if token.lower() not in stop_words])
test_df['filtered_tokens'] = test_df['lemmas'].apply(lambda x: [token for token in x if token.lower() not in stop_words])

train_df.to_csv('/mnt/d/train_tokenized_1.csv', index= False)
test_df.to_csv('/mnt/d/test_tokenized_1.csv', index= False)

train_df['processed_text'] = train_df['filtered_tokens'].apply(lambda x: ' '.join(x))
test_df['processed_text'] = test_df['filtered_tokens'].apply(lambda x: ' '.join(x))

vectorizer = TfidfVectorizer(min_df= 0.001,
                             max_df= 0.9,
                             use_idf= True,
                             ngram_range=(1, 2))
train_tfidf_matrix = vectorizer.fit_transform(train_df['processed_text'])
test_tfidf_matrix = vectorizer.transform(test_df['processed_text'])

feature_names = vectorizer.get_feature_names_out()
final_train = pd.DataFrame(train_tfidf_matrix.toarray(), columns=feature_names)
final_test = pd.DataFrame(test_tfidf_matrix.toarray(), columns=feature_names)

final_train= pd.concat([train_df['label'], final_train],axis=1)

final_train.to_pickle('/mnt/d/final_train_use.pkl')
final_test.to_pickle('/mnt/d/final_test_use.pkl')

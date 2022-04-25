import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
import string
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

NUM_WORDS = 4366
MAX_LENGTH = 25

def remove_columns_from_dataframe(dataset: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        try:
            dataset = dataset.drop(columns=[col])
        except:
            continue
    return dataset

class Dataset:

  def __init__(self,dataset,stop_words):
    self.dataset = dataset
    self.stop_words = stop_words
    self.tokenizer = RegexpTokenizer(r'\w+')
    self.blacklist = ['https','http','co','www']

  def calculate_df(self):
    df = {}
    for tweet in self.dataset['text'].tolist():
      for word in tweet.split(" "):
        word = word.lower()
        try:
          df[word]+=1
        except:
          df[word] = 1
    return df
  
  def remove_URL(self, text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)
  
  def remove_html(self, text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)
  
  def remove_emojis(self, text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

  def remove_low_freq(self, text, threshold=2):
    s = ""
    for word in text.split(" "):
      if self.df[word]>threshold:
        s+=word+" "
    return s.strip()
  
  def remove_punct(self,text):
    return text.translate(str.maketrans('', '', string.punctuation))

  def clean_text(self,text):
    tokens = self.tokenizer.tokenize(text.strip())
    filtered_sentence = [w.lower() for w in tokens if not w.lower() in self.stop_words]
    filtered_sentence = [i for i in filtered_sentence if i not in self.blacklist]
    filtered_sentence = " ".join(filtered_sentence)
    filtered_sentence = re.sub(r'[^\x00-\x7f]',r'',filtered_sentence)
    filtered_sentence = self.remove_URL(filtered_sentence)
    filtered_sentence = self.remove_html(filtered_sentence)
    filtered_sentence = self.remove_emojis(filtered_sentence)
    filtered_sentence = self.remove_punct(filtered_sentence)
    return filtered_sentence

  def create_embedding_matrix(self, train=True):
    self.dataset['text'] = self.dataset['text'].apply(lambda x: self.clean_text(x))
    if train:
        self.df = self.calculate_df()
        self.dataset['text'] = self.dataset['text'].apply(lambda x: self.remove_low_freq(x))
    return self.dataset

class CustomTokenizer:

  def __init__(self, train_texts):
    self.tokenizer = Tokenizer(num_words=NUM_WORDS)
    self.train_texts = train_texts

  def train_tokenizer(self):
    max_length = len(max(self.train_texts , key=len))
    self.max_length = min(max_length, MAX_LENGTH)
    self.tokenizer.fit_on_texts(self.train_texts)
  
  def vectorize_input(self, tweets):
    tweets = self.tokenizer.texts_to_sequences(tweets)
    tweets = sequence.pad_sequences(tweets, maxlen=self.max_length, truncating='post',padding='post')
    return tweets
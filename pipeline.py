import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow import keras
import methods
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global Variables
VARIABLES = {
    'COLUMNS_TO_REMOVE' : ['id','keyword','location'],
    'TOKENIZER_FILE'    : './saved/tokenizer-24-04.pkl',
    'MODEL_FILE'        : './saved/dis_mgmt_lstm-24-04.h5'
}

STOP_WORDS = stopwords.words('english')

# Loading pkl files

tokenizer = pickle.load(open(VARIABLES['TOKENIZER_FILE'],'rb'))
model = keras.models.load_model(VARIABLES['MODEL_FILE'])

# Creating classes for pipeline
class DataTransformer(BaseEstimator,TransformerMixin, methods.Dataset):

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = STOP_WORDS
        self.blacklist = ['https','http','co','www']

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        X = self.clean_text(X)
        return [X]

class TokenizerTransformer(BaseEstimator,TransformerMixin):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        X = self.tokenizer.vectorize_input(X)
        return X

predict_pipeline = Pipeline(steps= [
    ('clean_text', DataTransformer()),
    ('tokenize', TokenizerTransformer(tokenizer)),
    ('model', model)
])

def predict(text):
    return float(predict_pipeline.predict(text)[0][0])
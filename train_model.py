# Importing required libraries
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.optimizers import Adam
import methods
import pickle
from datetime import date
from os import path as Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

stop_words = stopwords.words('english')

# Creating required global variables
FILES = {
    'TRAIN_DATA' : './data/train.csv',
    'TEST_DATA'  : './data/test.csv'
}

VARIABLES = {
    'COLUMNS_TO_REMOVE' : ['id','keyword','location']
}

SEED = 100

# Loading the data
dataset = pd.read_csv(FILES['TRAIN_DATA'])

# Removing columns from dataframe
dataset = methods.remove_columns_from_dataframe(dataset, VARIABLES['COLUMNS_TO_REMOVE'])

# Preprocessing the data
data = methods.Dataset(dataset, stop_words)
dataset = data.create_embedding_matrix()
dataset = dataset.drop(dataset[(dataset['text'] == '')].index)

# Splitting data into train and val (since separate test data is available)
X_train, X_val, y_train, y_val = train_test_split(dataset['text'], dataset['target'], test_size=0.2, random_state = SEED)

# Creating and training the tokenizer
tokenizer = methods.CustomTokenizer(X_train)
tokenizer.train_tokenizer()
tokenized_train = tokenizer.vectorize_input(X_train)
tokenized_val = tokenizer.vectorize_input(X_val)

# Saving the tokenizer in pkl format
date_obj = date.today()
str_date = date_obj.strftime('%d-%m')
tokenizer_name = 'tokenizer-'+str_date+'.pkl'
file = open(Path.join('./saved',tokenizer_name),'wb')
pickle.dump(tokenizer, file)

# Defining the RNN model
model = Sequential()
embedding = Embedding(4366, 100, input_length = 25)
model.add(embedding)
model.add(LSTM(2,recurrent_dropout=0.3, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(2, recurrent_dropout=0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimzer = Adam()
model.compile(optimizer=optimzer, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print(model.summary())

# Training the model
history = model.fit(tokenized_train, y_train, 
                    batch_size=16, 
                    epochs=5, 
                    validation_data=(tokenized_val,y_val), 
                    verbose=2)

# Saving the model in pkl format
model_name = "dis_mgmt_lstm-"+str_date+".h5"
model.save(Path.join('./saved',model_name))
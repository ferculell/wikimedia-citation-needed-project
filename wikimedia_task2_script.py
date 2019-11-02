#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import pickle
import types
import wikipedia
import nltk
from bs4 import BeautifulSoup

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))




def obtain_data(title_input):
    # Find the actual article title
    title = wikipedia.search(title_input, results=1)
    
    # Get the full HTML page
    html_page = wikipedia.page(title).html()
    
    # Make a BeautifulSoup object
    soup = BeautifulSoup(html_page)
    
    # Select the Wikipedia page content
    content = soup.find('div', class_="mw-parser-output")
    paragraphs = content('p')
    
    # Construct the dataframe
    data = []
    
    for paragraph in paragraphs:
        citation = 0
        prev_h = paragraph.find_previous_sibling('h2')
        section = "MAIN_SECTION"
        
        if prev_h == None:
            section = section
        else:
            section = prev_h.get_text()
    
        text = paragraph.get_text()
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            row = [None, None, None]
            row[0] = sentence
            row[1] = section
            row[2] = citation
            data.append(row)
            
    df = pd.DataFrame(data)
    # Rename columns to match the previous columns names setting
    df.columns = ['statement', 'section', 'citation']
    
    return df


'''
######    Begin the citation-needed-paper modified code    ######
'''

'''
    Parse and construct the word representation for a sentence.
'''


def text_to_word_list(text):
    # check first if the statements is longer than a single sentence.
    sentences = re.compile('\.\s+').split(str(text))
    if len(sentences) != 1:
        # text = sentences[random.randint(0, len(sentences) - 1)]
        text = sentences[0]

    text = str(text).lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()

    return text


'''
    Create the instances from our datasets
'''


def construct_instance_reasons(statement_path, section_dict_path, vocab_w2v_path, max_len=-1):
    # Load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'))

    # load the section dictionary.
    section_dict = pickle.load(open(section_dict_path, 'rb'))

    # Load the statements.
    statements = statement_path

    # construct the training data
    X = []
    sections = []
    #y = []
    outstring=[]
    for index, row in statements.iterrows():
        try:
            statement_text = text_to_word_list(row['statement'])

            X_inst = []
            for word in statement_text:
                if max_len != -1 and len(X_inst) >= max_len:
                    continue
                if word not in vocab_w2v:
                    X_inst.append(vocab_w2v['UNK'])
                else:
                    X_inst.append(vocab_w2v[word])

            # extract the section, and in case the section does not exist in the model, then assign UNK
            section = row['section'].strip().lower()
            sections.append(np.array([section_dict[section] if section in section_dict else 0]))

            #label = row['citations']

            # some of the rows are corrupt, thus, we need to check if the labels are actually boolean.
            #if type(label) != types.BooleanType:
                #continue

            #y.append(label)
            X.append(X_inst)
            outstring.append(str(row["statement"]))
            #entity_id  revision_id timestamp   entity_title    section_id  section prg_idx sentence_idx    statement   citations

        except Exception as e:
            print row
            print e.message
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

    #encoder = LabelBinarizer()
    #y = encoder.fit_transform(y)
    #y = to_categorical(y)

    return X, np.array(sections), outstring


if __name__ == '__main__':
    # Input the Wikipedia article title
    article_title = raw_input("Article title: ")
    # Input the path to the model
    model_path = raw_input("Path to the model: ")
    # Input the path to the vocabulary of sections
    sections_voc = raw_input("Path to the sections vocabulary: ")
    # Input the path to the vocabulary of words
    words_voc = raw_input("Path to the words vocabulary: ")
    
    # obtain statement data
    statement_input = obtain_data(article_title)
    
    # load the model
    model = load_model(model_path)
    
    # load the data
    max_seq_length = model.input[0].shape[1].value
    
    X, sections, outstring = construct_instance_reasons(statement_input, sections_voc, words_voc, max_seq_length)
    
    # classify the data
    pred = model.predict([X, sections])

    # store the predictions: printing out the sentence text sorted by prediction score.
    out = []
    for idx, y_pred in enumerate(pred):
        row = [None, None]
        row[0] = y_pred[0]
        row[1] = outstring[idx]
        out.append(row)
    
    out = pd.DataFrame(out)
    out.columns = ['score', 'sentence']
        
    print out.sort_values('score', ascending=False)


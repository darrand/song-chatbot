# %%
# Sumber dataset https://www.kaggle.com/imuhammad/audio-features-and-lyrics-of-spotify-songs
# https://docs.google.com/document/d/1DPdsfIHn1nFRBsNYk1oId3KooaP8uQfek6noMv_3g3g/edit
import numpy as np
import nltk
import pandas as pd
import re
import os
import json
from flask import Flask, request
from flask_ngrok import run_with_ngrok
from gensim.models import Word2Vec
import multiprocessing
from time import time
from spellchecker import SpellChecker
print("CPU Count")
print(multiprocessing.cpu_count())

app = Flask(__name__)
run_with_ngrok(app)
# %%
nltk.download('punkt')

# %% [markdown]
# # Pre-processing

# %% [markdown]
# ## **[Filter based on language]**
# ---

# %%
df = pd.read_csv(os.getcwd() + '/spotify_songs.csv')
sorted = df.loc[df['language'] == 'en']
sorted = sorted[['track_id','track_name','track_artist','lyrics','playlist_genre','playlist_subgenre','track_popularity']]
sorted = sorted.dropna()
sorted.to_csv(os.getcwd() +'/data.csv',index=False)

# %% [markdown]
# ## **[Tokenization]**
# ---

# %% [markdown]
# ### Cleansing lyric from contractions

# %%
# Dictionary of english Contractions from https://www.analyticsvidhya.com/blog/2020/08/information-retrieval-using-word2vec-based-vector-space-model/ 
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not","can't": "can not","can't've": "cannot have",
"'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
"didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
"he'll've": "he will have","how'd": "how did","how'd'y": "how do you","how'll": "how will","i'd": "i would",
"i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
"isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
"mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
"needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
"oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
"shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will",
"she'll've": "she will have","should've": "should have","shouldn't": "should not",
"shouldn't've": "should not have","so've": "so have","that'd": "that would","that'd've": "that would have",
"there'd": "there would","there'd've": "there would have",
"they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
"they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
"we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
"weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
"what've": "what have","when've": "when have","where'd": "where did",
"where've": "where have","who'll": "who will","who'll've": "who will have","who've": "who have",
"why've": "why have","will've": "will have","won't": "will not","won't've": "will not have",
"would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would","you'd've": "you would have","you'll": "you will","you'll've": "you will have",
"you're": "you are","you've": "you have"}
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def cleansed_docs(doc):
  x = pd.Series([i.lower() for i in doc])
  return x.apply(lambda x:expand_contractions(x))

lyrics = cleansed_docs(sorted['lyrics'])
titles = cleansed_docs(sorted['track_name'])

# %%
def tokenize_docs(doc):
  output = []
  for i in doc:
    token = nltk.tokenize.word_tokenize(i)
    clean_token = []
    for i in token:
      if str.isalnum(i):
        clean_token.append(i)
    output.append(clean_token)
  return output

# %% [markdown]
# ### Tokenize lyric and titles

# %%
tok_lyric = tokenize_docs(lyrics)

# %% [markdown]
# ### Tokenize and cleansing artist name

# %%
subt_dict = { "\$": "s","&":" and ","-": " ","!": "i","\/":" ", "'n": " and", "n'": "and", "\.":""}
temp_artist = sorted[['track_id','track_artist']].values.tolist()
new_artists = [] 
for words in temp_artist:
  result = words[1]
  for keys in subt_dict.keys():
    result = re.sub(r'{}'.format(keys),subt_dict[keys],result)
  new_artists.append({words[0]: result})

# %%
tok_artist = []
for dict_word in new_artists:
  for key,values in dict_word.items():
    token = nltk.tokenize.word_tokenize(values.lower())
    clean_token = []
    for i in token:
      if str.isalnum(i):
        clean_token.append(i)
    tok_artist.append({key: clean_token})


try:
  w2v_model = Word2Vec.load(os.getcwd() + "/spotify_songs_en.model")
  print('Existing model detected')
except (FileNotFoundError):
  w2v_model = Word2Vec(min_count=1, vector_size=300,window=10, workers=2, sg=1) 
  t = time()
  w2v_model.build_vocab(tok_lyric)
  print('Time elapsed: {} mins'.format(round((time() - t) / 60, 2)))
  t = time()
  w2v_model.train(tok_lyric, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)
  w2v_model.save(os.getcwd() + '/spotify_songs_en.model')
  print('Time elapsed: {} mins'.format(round((time() - t) / 60, 2)))

# %% [markdown]
# ### Word Embeddings for lyric

# %%
# Get the embeddings for lyrics
def get_embeddings(tokens):
  tok_val = []
  if len(tokens) < 1:
    return np.zeros(300)
  for tok in tokens:
    if tok in w2v_model.wv.key_to_index:
      tok_val.append(w2v_model.wv.get_vector(tok))
    else:
      tok_val.append(np.random.rand(300))
  return np.mean(tok_val, axis=0)

# %% [markdown]
# ### Finding song based on lyrics

# %%
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_input(q):
  query = q.lower()
  q_tok = nltk.tokenize.word_tokenize(query)
  q_tok = pd.Series(q_tok).apply(lambda x:expand_contractions(x))
  q_clean = []
  for i in q_tok:
    if str.isalnum(i):
      q_clean.append(i)
  
  return q_clean

def find_lyrics(lyric, doc=None):
  # Get embeddings for input
  q_vec = preprocess_input(lyric)
  q_vec = get_embeddings(q_vec)

  # Get lyric|embed
  embed_lyric = sorted.copy()
  embed = []
  for i in tok_lyric:
    embed_val = get_embeddings(i)
    embed.append(embed_val)
  embed_lyric['embed_val'] = embed
  
  # Get Cosine Similarity
  sim = []
  for i in embed_lyric['embed_val']:
    similar = cosine_similarity(q_vec.reshape(1,-1), i.reshape(1, -1))
    sim.append(similar)

  embed_lyric['similarity'] = sim 

  if doc is not None:
    new_doc = pd.DataFrame(columns=doc.columns)
    for i in doc['track_id']:
      ranked_el = embed_lyric.query("track_id == '{}'".format(str(i)))
      new_doc = new_doc.append(ranked_el)
      
    new_doc.sort_values(by='similarity', ascending=False, inplace=True)
    return new_doc.head(10).reset_index(drop=True)

  embed_lyric.sort_values(by='similarity',ascending=False, inplace=True)


  top_songs = embed_lyric.head(10).reset_index(drop=True)
  top_songs.sort_values(by='track_popularity', ascending=False, inplace=True)
  return top_songs[['track_name', 'track_artist', 'lyrics', 'track_popularity','playlist_genre']].head(10).reset_index(drop=True)


# %% [markdown]
# ## **[Boolean Model for artists]**
# ---

# %% [markdown]
# ### Checking artist name's mispelling

# %%
'''
Spell Checker using norvig spell checker for artist name, some problems:
Its accuracy is worrying
'''


def build_frequency_list_artist():
  freq = {}
  for list_a in tok_artist:
      for key,values in list_a.items():
        for value in values:
          if value not in freq.keys():
            freq[value] = 1
          else :
            freq[value] += 1

  with open(os.getcwd() + '//freq_list_artis.txt','w', encoding='utf-8') as file:
    for key,values in freq.items():
      file.write('{} {}\n'.format(key, values))
build_frequency_list_artist()

def spell_check_artist(kalimat):
  # turn off loading a built language dictionary
  spell = SpellChecker(language=None)
  spell.word_frequency.load_text_file(os.getcwd() + '/freq_list_artis.txt', encoding="utf-8")

  # find those words that may be misspelled
  result = []
  for word in kalimat:
    misspelled = spell.unknown([word])
    if misspelled:
      for mis_words in misspelled:
        result.append(spell.correction(mis_words))
    else:
      result.append(word)

  separator = ' '
  result = separator.join(result)
  return result

# %% [markdown]
# ### Indexing artists' name

# %%
def create_index(token):
  hasil = {}
  for docs in token:
    i = list(docs.keys())[0]
    for j,word in enumerate(docs[i]):
      if word in hasil.keys():
        if i not in hasil[word].keys():
          hasil[word][i] = [j]  
        else:
          hasil[word][i].append(j)
      else:
        hasil[word] = {i:[j]}
  return hasil

# %%
index_artist = create_index(tok_artist)

# %% [markdown]
# ### Find song based on artist

# %%
def positional_intersect(prev_word,next_word):
  intersect_id = set()
  for doc_id_1 in prev_word.keys():
    if doc_id_1 in next_word.keys():
      for doc_pos in prev_word[doc_id_1]:
        if doc_pos+1 in next_word[doc_id_1]:
          intersect_id.add(doc_id_1)
  return intersect_id

def get_query(kalimat,token_dict):
  token = nltk.tokenize.word_tokenize(kalimat.lower())
  if len(token) == 1:
    hasil = token_dict[token[0]].keys()
    return hasil  
  token_hasil = token[1:]
  hasil = token_dict[token[0]].keys()
  temp_word = token[0]
  for word in token_hasil:
      hasil = hasil & positional_intersect(token_dict[temp_word],token_dict[word])
      temp_word = word
  return hasil

def find_artist(kalimat,token=index_artist, lyrics=False):
  kalimat = spell_check_artist(nltk.tokenize.word_tokenize(kalimat))
  id_set = get_query(kalimat,token)
  artist_result = pd.DataFrame()
  artist_result = sorted[sorted['track_id'].isin(id_set)]
  artist_result = artist_result.sort_values(by='track_popularity', ascending=False)
  if lyrics:
    return artist_result
  return artist_result.head(10)

# %% [markdown]
# ## **[Boolean Model for song title]**

# %% [markdown]
# ### Indexing song title

# %%
def indexer(list_of_lists):
  #for making a index with the type of dictionary, made for title
  index = dict()
  for i in range(len(list_of_lists)):
    title = list_of_lists[i]
    for no in range(len(title)):
      token = title[no]
      if token not in index.keys():
        temp = dict()
        temp[i] = [no] 
        index[token] = temp
      else:
        if i in index[token].keys():
          index[token][i] += [no]
        else:
          index[token][i] = [no]
  return index

# %% [markdown]
# ### Spelling correction for song title

# %%
from autocorrect import Speller
spell = Speller(lang='en')

def preprocess_input_with_spelling(q):
  query = q.lower()
  q_tok = nltk.tokenize.word_tokenize(query)
  q_tok = pd.Series(q_tok).apply(lambda x:expand_contractions(x))
  q_clean = []
  for i in q_tok:
    if str.isalnum(i):
      autocorrect = spell(i)
      q_clean.append(autocorrect)
  
  return q_clean

# %% [markdown]
# ### Find the intersection between query and title index

# %%
def find_intersection(query, index):
  keys = index.keys()
  q_clean = preprocess_input_with_spelling(query)
  set_of_indexes = []
  answer_indexes = []
  for token in q_clean:
    if token in keys:
      set_of_indexes.append(set(list(index[token])))

  if len(set_of_indexes) > 0:
    answer_indexes = set_of_indexes[0]
    for ind_set in set_of_indexes:
      answer_indexes = answer_indexes.intersection(ind_set)
  
  answer_indexes = list(answer_indexes)
  return answer_indexes[:10]

# %%
tok_title = tokenize_docs(titles)
title_index = indexer(tok_title)

# %% [markdown]
# ### Find song based on title

# %%
def find_title(query, title_index=title_index, data=sorted):
  indx = find_intersection(query, title_index)
  answer = pd.DataFrame()
  if indx == []:
    return answer
  answer = sorted.iloc[indx]
  answer = answer.sort_values(by='track_popularity', ascending=False)                 
  return answer

# Functions for exporting to ports down below
@app.route('/')
def default_page():
  return "Hello"

@app.route('/artist-title', methods = ['POST'])
def find_artist_title():
  q_artist = request.get_json()['artist']
  q_title = request.get_json()['title']
  artist_filter = find_artist(q_artist, lyrics=True)
  title_filter = find_title(q_title)
  full_filter = pd.merge(artist_filter, title_filter, how='right', on=['track_artist', "track_name"])
  full_filter = full_filter.iloc[:, 0:7].dropna()
  full_filter.columns = artist_filter.columns
  return json.dumps(full_filter.to_json())

@app.route('/artist-lyrics', methods = ['POST'])
def find_artist_lyrics():
  q_artist = request.get_json()['artist']
  q_lyrics = request.get_json()['lyrics']

  artist_filter = find_artist(q_artist, lyrics=True)
  full_filter = find_lyrics(q_lyrics, artist_filter)
  return json.dumps(full_filter.to_json())

@app.route('/title-lyrics', methods = ['POST'])
def find_title_lyrics():
  q_title = request.get_json()['title'] 
  q_lyrics = request.get_json()['lyrics']
  title_idx = indexer(tok_title)
  title_doc = find_title(q_title, title_idx, sorted)
  full_doc = find_lyrics(q_lyrics, title_doc)
  return json.dumps(full_doc.to_json())

@app.route('/title', methods=['POST'])
def find_title_json():
  q_title = request.get_json()['title']
  title_df = find_title(q_title)
  return json.dumps(title_df.to_json())

@app.route('/artist', methods=['POST'])
def find_artist_json():
  q_artist = request.get_json()['artist']
  artist_df = find_artist(q_artist, lyrics=True)
  return json.dumps(artist_df.to_json())

@app.route('/lyrics', methods=['POST'])
def find_lyrics_json():
  q_lyrics = request.get_json()['lyrics']
  lyrics_df = find_lyrics(q_lyrics)
  return json.dumps(lyrics_df.to_json())

if __name__ == "__main__":
  print('starting server')
  app.run()
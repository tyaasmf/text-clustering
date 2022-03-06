# read data
import pandas as pd
tempo = pd.read_csv("/content/drive/MyDrive/tempo_tugas.csv")
tempo.head()

tempo = tempo["content"]
tempo
print("Jumlah amatan NaN adalah",tempo.isnull().sum())

# Noise Removal 
import string
def noise_removal(words):
    words = words.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = words.strip()
    return words
    
tempo_clean = tempo.str.lower()
tempo_clean = tempo_clean.apply(noise_removal)
tempo_clean

# Tokenization
import nltk 
nltk.download('punkt')

from nltk.tokenize import word_tokenize
def tokenize_fun(words):
  return word_tokenize(words)
  
tempo_clean = tempo_clean.apply(tokenize_fun)
tempo_clean

# Normalization
indo_slang_words = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
indo_slang_words

def replace_slang_word(words):
  for index in range(0,len(words)-1):
    index_slang = indo_slang_words.slang==words[index]
    formal = list(set(indo_slang_words[index_slang].formal))
    if len(formal)==1:
      words[index]=formal[0]
      return words
      
tempo_clean = tempo_clean.apply(replace_slang_word) #lama banget run-nya 50 menit 
tempo_clean.head()

print("Jumlah amatan NaN adalah",tempo_clean.isnull().sum())
tempo_clean = tempo_clean.dropna()
tempo_clean

# Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
indo_stopwords = stopwords.words('indonesian')
indo_stopwords.append("nya") # adding stopwords

def stopwords_removal(words):
    return [word for word in words if word not in indo_stopwords]

tempo_clean2 = tempo_clean.apply(stopwords_removal)
tempo_clean2.head()

# Stemming
!pip install Pysastrawi

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmer_func(word):
    return stemmer.stem(word)

word_dict = {}

for document in tempo_clean:
    for word in document:
        if word not in word_dict:
            word_dict[word] = ' '

for word in word_dict:
    word_dict[word] = stemmer_func(word)
    
def get_stemmer_word(document):
    return [word_dict[word] for word in document]
  
tempo_clean3 = tempo_clean2.apply(get_stemmer_word)
tempo_clean3.head()

# Convert to text for Wordcloud Visualization
def list_to_text(token):
    text = " "
    return text.join(token)
    
tempo_clean_fin = tempo_clean3.apply(list_to_text)
tempo_clean_fin.head()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

tempo_text = tempo_clean_fin.to_string()
wordcloud = WordCloud(background_color="white")
wordcloud.generate(tempo_text)

plt.imshow(wordcloud, interpolation='bilinear') # Display the generated image
plt.axis("off")
plt.show()

# Removing tempoco as a stopwords
def remove_certain_word(words,text):
    return [word for word in words if word not in text]

tempo_clean3 = tempo_clean3.apply(lambda x: remove_certain_word(x,"tempoco"))
tempo_clean_fin2 = tempo_clean3.apply(list_to_text)
tempo_text2 = tempo_clean_fin2.to_string()

wordcloud2 = WordCloud(background_color="white")
wordcloud2.generate(tempo_text2)

# TF IDF Weighting
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectorizer = TfidfVectorizer()
tf_idf_result = tf_idf_vectorizer.fit_transform(tempo_clean_fin2)
tf_idf_result_df = pd.DataFrame(tf_idf_result.toarray(),columns=tf_idf_vectorizer.get_feature_names())
tf_idf_result_df.sum(axis=0).T.sort_values(ascending=False)

# K-Means Clustering
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer 

kmeans = MiniBatchKMeans()
kmeans_vis_sil = KElbowVisualizer(kmeans, k=(2,15),metric="silhouette")
kmeans_vis_sil.fit(tf_idf_result)

kmeans = MiniBatchKMeans(n_clusters=10) # optimal clusters
kmeans.fit(tf_idf_result)

# Visualizing Clusters
cluster_label = pd.DataFrame(kmeans.predict(tf_idf_result),columns=["k"])
tf_idf_df_lab = pd.concat([tf_idf_result_df,cluster_label],axis=1)
tf_idf_df_lab.head()

import numpy as np
import seaborn as sns

for i in range(10):
    cluster_df = tf_idf_df_lab.groupby("k").sum().T.sort_values(ascending=False,by=i).head(10)[i].to_frame().reset_index()
    cluster_df.columns = ["words","TF_IDF"]
    sns.barplot(x="TF_IDF",y="words",data=cluster_df).set_title('Cluster '+str(i+1))
    plt.show()
    




    
    

## Written for the use of students in UGCA class ###
## Queries should be directed to aditya(aditya_ks@utexas.edu) ###

import csv
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import PunktSentenceTokenizer,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import warnings

#to ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Please use xlsx file format to read the data
#I faced issues with csv
reviews_df=pd.read_excel("yelp_reviews.xlsx",encoding='utf8', errors='ignore')

#checking for nulls if present any
print("Number of rows with null values:")
print(reviews_df.isnull().sum().sum())
reviews_df=reviews_df.dropna() 

#reading the attributes file
#check into the "attributes.txt" file for the proper format
#each attribute has to be listed in a new line.
attributes=list(line.strip() for line in open('attributes.txt'))
attributes=" ".join(attributes)


# #merging attibutes to the review
## Restaurant_review is the name of the column with review text.

tempDataFrame=pd.DataFrame({'Restaurant_review':[attributes]})
tempDataFrame=tempDataFrame.transpose()
description_list1=reviews_df['Restaurant_review']
frames = [tempDataFrame, description_list1]
result = pd.concat(frames)
result.columns = ['review']
result=result.reset_index()


#building bag of words using frequency

vec_words = CountVectorizer(decode_error='ignore')
total_features_words = vec_words.fit_transform(result['review'])
print("The size of the vocabulary space:")
print(total_features_words.shape)

#Calculating pairwise cosine similarity

subset_sparse = sparse.csr_matrix(total_features_words)
total_features_review=subset_sparse
total_features_attr=subset_sparse[0,]
similarity=1-pairwise_distances(total_features_attr,total_features_review, metric='cosine')


#Assigning the similarity score to dataframe
# similarity=np.array(similarities[0]).reshape(-1,).tolist()
similarity=pd.DataFrame(similarity)
similarity=similarity.transpose()
similarity.columns = ['similarity']
similarity=similarity.drop(similarity.index[[0]])
reviews_df=reviews_df.assign(similarity=similarity.values)

reviews_df['reviews_text_new']=reviews_df['Restaurant_review'].apply(lambda x: "".join(line.rstrip("\n") for line in x))



#writing to an output file
reviews_df.to_excel("similarity_score.xlsx",index=False)
import math
import re
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

#.....................................................................................................#

def vectorize_feature(new_feat, feat_type, features_array, df_features, vectorizer=None):
    # This function vertorize new feature, a series "new_feat".
    # "feat_type" is the type of new feature, specified by the user
    # "features_array" contains all previous vectorized features  
    # "df_features" contains all previous features information

    if vectorizer==None: 
        # if "vectorizer" is not specified, use basic count.
        vectorizer = CountVectorizer(min_df = 0)
        
    # vectorize new feature
    new_array = vectorizer.fit_transform(new_feat)
    new_array = new_array.toarray() 
    
    # add new feature to the features_array (append horizontally: axis = 1)
    if features_array.all() == None:
        features_array_updated = new_array
    else:
        features_array_updated = np.concatenate((features_array, new_array), axis=1)
    
    # update feature name and its location to "df_features"
    columns = ['feat_type','feat_name','location'] # column names of "df_features" 
    df_features_updated = df_features.copy() # make a copy of the input
    n_feats = len(df_features) # current number of features, before adding new ones.

    for vocab in vectorizer.vocabulary_.items():
        feat_name = vocab[0] # "feat_name" refers to the actual words  
        feat_loc = vocab[1] + n_feats # "feat_loc" refers to the location in the array
        new_feature = [feat_type,feat_name,feat_loc]
        df_new = pd.DataFrame([new_feature], columns = columns)
        df_features_updated = df_features_updated.append(df_new, ignore_index=True)
    
    df_features_updated = df_features_updated.sort_values('location', axis=0, ascending=True)
    
    return features_array_updated, df_features_updated

#.....................................................................................................#

def selected_features(df, list_of_features, vectorizer_jobtitle=None):

    columns = ['feat_type','feat_name','location']
    df_features = pd.DataFrame(columns = columns)
    features_array = np.array(None)
    
    # vectorizing states
    if 'state' in list_of_features: 
        features_array, df_features = vectorize_feature(df.state, 
                                                        'state', 
                                                        features_array, 
                                                        df_features)

    # vectorizing titles
    if 'jobtitle' in list_of_features:
        if vectorizer_jobtitle == None:
            vectorizer_jobtitle = TfidfVectorizer(stop_words = 'english',
                                                  min_df = 5, 
                                                  ngram_range = (1, 2))
            
        features_array, df_features = vectorize_feature(df.jobtitle, 
                                                        'jobtitle',
                                                        features_array, 
                                                        df_features,
                                                        vectorizer_jobtitle)

    # vectorizing naics (industry code)
    if 'naics' in list_of_features:
        features_array, df_features = vectorize_feature(df.naics, 
                                                        'naics', 
                                                        features_array, 
                                                        df_features)

    # vectorizing onet (occupation code)
    if 'onet' in list_of_features:
        features_array, df_features = vectorize_feature(df.onet, 
                                                        'onet', 
                                                        features_array, 
                                                        df_features)

    # vectorizing education
    if 'edu' in list_of_features:
        list_edu = ['high_school', 'associate', 'bachelor', 'master', 'phd']
        
        # transform education variables into numpy array
        edu_array = df[list_edu].values

        # concatenate education variables to job titles
        features_array = np.concatenate((features_array, edu_array), axis=1)

        # get current total number of features
        feat_loc = df_features.location.max()

        # add education variables into list of features dataframe
        for edu in list_edu:
            feat_type = 'edu'
            feat_name = edu
            feat_loc += 1
            new_feature = [feat_type,feat_name,feat_loc]
            df_new = pd.DataFrame([new_feature], columns = columns)
            df_features = df_features.append(df_new, ignore_index=True)
    
    if 'titleloc' in list_of_features:
        feat_name = 'titleloc'
        feat_type = 'titleloc'
        feat_loc += 1
        new_feature = [feat_type,feat_name,feat_loc]
        df_new = pd.DataFrame([new_feature], columns = columns)
        df_features = df_features.append(df_new, ignore_index=True)
        
        features_array = np.concatenate((features_array, df['titleloc'].values.reshape(-1,1)), axis=1)

    if 'countword' in list_of_features:
        feat_name = 'countword'
        feat_type = 'countword'
        feat_loc += 1
        new_feature = [feat_type,feat_name,feat_loc]
        df_new = pd.DataFrame([new_feature], columns = columns)
        df_features = df_features.append(df_new, ignore_index=True)
        
        features_array = np.concatenate((features_array, df['countword'].values.reshape(-1,1)), axis=1)
        
    return features_array, df_features

#.....................................................................................................#

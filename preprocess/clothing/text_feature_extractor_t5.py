import os
import re
import yaml
import math
import torch
import pickle
import numpy as np
import json
import pandas as pd
import gzip
from tqdm import tqdm

from transformers import AutoTokenizer, T5EncoderModel
import nltk
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import IncrementalPCA

import sys
sys.path.append("../")
from utils import build_config, extract_raw_text_feature, decompose_feature_by_pca, decompose_feature_by_ae, cluster_features


def get_text(meta_df):
    '''
    get the title description from meta_df
    '''
    title_dict = {} # asin => title
    drop_item_num = 0
    for i in range(len(meta_df)):
        row = meta_df.iloc[i]
        title = row['title']
        if not pd.isna(title):
            title_dict[row['asin']]=row['title']
        else:
            drop_item_num += 1
    print(drop_item_num,'item dropped')
    asin_list = [i[0] for i in title_dict.items()]
    title_corpus = [i[1].lower() for i in title_dict.items()]
    return asin_list, title_corpus


def preprocess_text(tokenizer, title_corpus):
    stopwords = list(nltk.corpus.stopwords.words('english'))
    stopwords.extend(['!','"','#','$','%','&','(',')','*','+',',','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','\t','\n'])
    for i in range(len(title_corpus)):
        sentence = title_corpus[i]
        tokens = tokenizer.tokenize(sentence)
        tokens_without_stopwords = [token for token in tokens if token.lower() not in stopwords]
        preprocessed_sentence = ' '.join(tokens_without_stopwords)
        title_corpus[i] = preprocessed_sentence
    return title_corpus


def extract_text_feature(config, title_corpus, asin_list, tokenizer, batch_size=512):
    text_feature_list_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'raw-feature512.npy')
    if os.path.exists(text_feature_list_path):
        print('skip t-5 extractor')
        return np.load(text_feature_list_path) 
    model = T5EncoderModel.from_pretrained("t5-base")
    features_array = extract_raw_text_feature(title_corpus, asin_list, tokenizer, model, batch_size) # from utils.py
    # 将特征数组保存到文件中
    np.save(text_feature_list_path, features_array)
    print('saved ', text_feature_list_path)
    return features_array


def get_dense_features(features_array, config, batch_size = 512, method='ae'):
    text_feature_list_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'method-feature{}.npy'.format(config['text_dimension']))
    if os.path.exists(text_feature_list_path):
        print('skip decomposition.')
        return np.load(text_feature_list_path)
    if method == 'pca':
        features_reduced = decompose_feature_by_pca(features_array, epoch_num=5, encoding_dimension=config['text_dimension'], batch_size = 512) # from utils.py
    elif method == 'ae':
        features_reduced = decompose_feature_by_ae(features_array, epoch_num=500, encoding_dimension=config['text_dimension'], batch_size = 256) # from utils.py
    
    np.save(text_feature_list_path, features_reduced)
    print('saved ', text_feature_list_path)
    return features_reduced


def save_features_dict(asin_list, features_array, feature_prefix='raw', feature_dim=512):
    asin_features_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'asin-{}-feature{}.npy'.format(feature_prefix, feature_dim))
    if os.path.exists(asin_features_path):
        print('found saved asin_features_dict, skip')
        return
    asin_feature_dict = {}
    for idx, asin in enumerate(asin_list):
        feature = features_array[idx]
        asin_feature_dict[asin] = feature
    np.save(asin_features_path, np.array(asin_feature_dict))
    print('saved ', asin_features_path)
    return


def cluster_and_save(asin_list, txt_feature_embedding, config, source='raw'):
    '''
    cluster based on img feature, and save it into asin_text_{}_c{}_k{}_dict.pickle [kmeans/cluster_num/link-k]
    '''
    cluster_num = config['text_cluster_num']
    k_num = config['link_k']
    cluster_method = config['cluster_method']
    feat_dim = 512 if source=='raw' else config['text_dimension']
    asin_text_dict_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'asin_text_feature{}_{}_c{}_k{}_dict.npy'.format(feat_dim, cluster_method, cluster_num, k_num))
    text_center_feature_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'text-center{}-feature{}.npy'.format(config['text_cluster_num'], feat_dim))
    if os.path.exists(asin_text_dict_path) and os.path.exists(text_center_feature_path):
        print('found the saved asin-cluster, and cluster-center feature, end.')
        return
    
    text_center_feature, asin_text_cluster_dict = cluster_features(cluster_num, cluster_method, asin_list, txt_feature_embedding, k_num) # from utils.py

    np.save(text_center_feature_path, text_center_feature)
    print('write ', text_center_feature_path)
    np.save(asin_text_dict_path, asin_text_cluster_dict)
    print('write ', asin_text_dict_path)


if __name__=='__main__':
    config = build_config('../../config/preprocess.yaml')
    config['data_root'] = '/ssd1/holdenhu/Amazon2014_dataset/Amazon_Clothing/'
    meta_df = pd.read_csv('/ssd1/holdenhu/Amazon2014_dataset/Amazon_Clothing/raw/core_meta_Clothing_Shoes_and_Jewelry.csv')
    working_folder = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'])
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    # define T5 Model
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # get text info
    asin_list, title_corpus = get_text(meta_df)
    title_corpus = preprocess_text(tokenizer, title_corpus)

    # encode sentences
    features_array = extract_text_feature(config, title_corpus, asin_list, tokenizer, batch_size=512)

    save_features_dict(asin_list, features_array, feature_prefix='raw', feature_dim=512)
    cluster_and_save(asin_list, features_array, config, source='raw')

    decomposite_methods = ['ae']
    for dec_method in decomposite_methods:
        features_reduced = get_dense_features(features_array, config, batch_size = 512, method=dec_method)
        save_features_dict(asin_list, features_reduced, feature_prefix=dec_method, feature_dim=config['text_dimension'])
        cluster_and_save(asin_list, features_reduced, config, source=dec_method)

    
    
    
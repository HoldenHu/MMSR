import os
import re
import yaml
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


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def build_config(file):
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    file_config_dict = dict()
    if file:
        with open(file, 'r', encoding='utf-8') as f:
            file_config_dict.update(yaml.load(f.read(), Loader=yaml_loader))
    return file_config_dict


def get_text(info_dict):
    '''
    get the title description from info_dict
    return: asin_list, text_corpus
    '''
    asin_list, text_list = [], []
    for asin, info in info_dict.items():
        info_json = json.loads(info)
        if 'description' in info_json:
            asin_list.append(asin)
            text_list.append(info_json['description'])
    
    text_list = [i.lower() for i in text_list]
    return asin_list, text_list


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


def extract_text_feature(title_corpus, asin_list, tokenizer, model):
    txt_feature_path_list = [] 
    txt_total_num = len(title_corpus)
    start_detect, each_detect = 0, 2000
    end_detect = start_detect + each_detect

    while start_detect < txt_total_num:
        end_detect = txt_total_num if end_detect > txt_total_num else end_detect
        session_task = (start_detect, end_detect)
        print('starting feature extraction session task - ', session_task)
        
        txt_feature_list_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'subtask_{}-{}_raw-features.pickle'.format(session_task[0], session_task[1]))
        asin_list_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'subtask_{}-{}_asin.pickle'.format(session_task[0], session_task[1]))

        if os.path.exists(txt_feature_list_path):
            start_detect, end_detect = start_detect + each_detect, end_detect + each_detect
            txt_feature_path_list.append(txt_feature_list_path)
            print('found {}-{} raw representation file, skip'.format(session_task[0], session_task[1]))
            continue

        inputs = tokenizer(title_corpus[session_task[0]:session_task[1]], return_tensors="pt", padding=True, truncation=True)  # Batch size 1
        attention_mask = inputs['attention_mask']
        input_ids = inputs.input_ids
        encoded_sentence = model(input_ids, attention_mask)
        last_hidden_states = encoded_sentence.last_hidden_state
        sentence_embedding = last_hidden_states.mean(dim=1)

        pickle.dump(sentence_embedding, open(txt_feature_list_path, 'wb'))
        pickle.dump(asin_list[session_task[0]:session_task[1]], open(asin_list_path, 'wb'))
        txt_feature_path_list.append(txt_feature_list_path)
        print('output {}-{} raw representation'.format(session_task[0], session_task[1]))
        start_detect, end_detect = start_detect + each_detect, end_detect + each_detect

    return txt_feature_path_list


def train_pca_by_file(model, batch_size, feature_path, checkpoint_path, checkpoint=True):
    text_features = pickle.load(open(feature_path, 'rb'))
    
    cur, end_point = 0, len(text_features)
    for idx in tqdm(range((end_point//batch_size)+1)):
        if cur >= end_point:
            break
        nxt_cur = cur+batch_size if cur+batch_size<end_point else end_point
        model.fit(text_features[cur:nxt_cur].detach().numpy())
        cur = nxt_cur
    if checkpoint:
        pickle.dump(model, open(checkpoint_path, 'wb'))
    return model


def get_pca_features(features_array, config, batch_size = 512):
    text_feature_list_path = os.path.join(config['data_root'], 'preprocessed','text', config['text_extractor'], 'pca-feature{}.npy'.format(config['text_dimension']))
    if os.path.exists(text_feature_list_path):
        print('skip pca transform.')
        return np.load(text_feature_list_path)
    # pca training
    print('starting training pca...')
    # 创建增量PCA对象，并指定batch_size和n_components参数
    ipca = IncrementalPCA(n_components=config['text_dimension'], batch_size=batch_size)

    # 逐批次对特征数组进行降维
    for epoch in range(3):
        for i in range(0, len(features_array), batch_size):
            batch = features_array[i:i+batch_size]
            ipca.partial_fit(batch)

    # 对所有特征进行降维
    features_reduced = ipca.transform(features_array)
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
    cluster based on txt feature, and save it into asin_text_{}_c{}_k{}_dict.pickle [kmeans/cluster_num/link-k]
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
    
    if cluster_method == 'kmeans':
        kmeans = MiniBatchKMeans(n_clusters=cluster_num, init_size=512, batch_size=1024, random_state=2023)
        print('txt_feature_embedding,',txt_feature_embedding.shape)
        kmeans.fit(txt_feature_embedding)
        centers = kmeans.cluster_centers_ # [C * D]
        # clusters = kmeans.predict(txt_feature_embedding)
    
    elif cluster_method == 'gmm':
        gmm = GaussianMixture(n_components=cluster_num, random_state=2023)
        gmm.fit(txt_feature_embedding)
        centers = gmm.means_ # [C * D]
        # membership_probs = gmm.predict_proba(X)
    
    text_center_feature = {}
    for c_id in range(len(centers)):
        arr = centers[c_id]
        text_center_feature[c_id] = arr
    # pickle.dump(text_center_feature, open(text_center_feature_path, 'wb'))
    np.save(text_center_feature_path, text_center_feature)
    print('write ', text_center_feature_path)

    asin_text_dict = {}
    cluster_set_list = []
    for each_embedding in txt_feature_embedding:
        distances = [np.linalg.norm(each_embedding - arr) for arr in centers]
        cluster_set_list.append(list(np.argsort(distances)[:k_num]))
    for asin, c_set in zip(asin_list, cluster_set_list):
        asin_text_dict[asin] = c_set
    # pickle.dump(asin_text_dict, open(asin_text_dict_path, 'wb'))
    np.save(asin_text_dict_path, asin_text_dict)
    print('write ', asin_text_dict_path)

    return asin_text_dict


if __name__=='__main__':
    config = build_config('../../config/preprocess.yaml')
    config['data_root'] = '/ssd1/holdenhu/ML_dataset/ml-1m/'
    info_dict = pickle.load(open('/ssd1/holdenhu/ML_dataset/ml-1m/raw/info_dict.txt', 'rb'))
    
    # define T5 Model
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5EncoderModel.from_pretrained("t5-small")

    # get text info
    asin_list, text_corpus = get_text(info_dict)  # asin here means movie_id
    text_corpus = preprocess_text(tokenizer, text_corpus)

    # encode sentences
    txt_feature_path_list = extract_text_feature(text_corpus, asin_list, tokenizer, model)

    features_array = pickle.load(open(txt_feature_path_list[0], 'rb')).detach().numpy()
    for pca_feature_path in txt_feature_path_list[1:]:
        _txt_feature_embedding = pickle.load(open(pca_feature_path, 'rb')).detach().numpy()
        features_array = np.concatenate((features_array, _txt_feature_embedding), axis=0)
    print("final overall raw txt feature embedding: ", features_array.shape)

    features_reduced = get_pca_features(features_array, config, batch_size = 512)

    # alignment, to asin dict
    save_features_dict(asin_list, features_array, feature_prefix='raw', feature_dim=512)
    save_features_dict(asin_list, features_reduced, feature_prefix='pca', feature_dim=config['text_dimension'])

    # Do the Clustering, default for pca features
    cluster_and_save(asin_list, features_array, config, source='raw')
    cluster_and_save(asin_list, features_reduced, config, source='pca')
    
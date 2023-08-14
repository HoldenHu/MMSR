import sys
import os
import re
import yaml
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from utils import build_config


def print_basic_info(rating_df):
    print('Original #inter:',len(rating_df))
    print('Original #item:',len(set(rating_df['itemid'])))
    print('Original #user:',len(set(rating_df['userid'])))

def get_df(config):
    rating_df = pd.read_csv(config['ratingdf_path'], header=None)
    rating_df.columns = ['userid', 'itemid','rating','timestamp']
    print_basic_info(rating_df)
    return rating_df

def get_dataset_folder(config): # TODO
    dataset_folder = None
    if config['dataset_type'] == 'dataset_cf':
        pass
    elif config['dataset_type'] == 'dataset_sr':
        dataset_folder = 'thre-{}_filt-{}_core-{}_split-[{}]_ratio-{}'.format(
            config['rating_threshold'],
            config['filter_strategy'],
            config['core'],
            config['train_test_strategy'],
            config['split_ratio'],
        )
        dataset_folder = os.path.join(config['data_root'], config['dataset_type'], dataset_folder)
    elif config['dataset_type'] == 'dataset_mmsr':
        dataset_folder = 'thre-{}_filt-{}_core-{}_split-[{}]_ratio-{}__imagec-{}_textc-{}_k-{}-{}-{}'.format(
            config['rating_threshold'],
            config['filter_strategy'],
            config['core'],
            config['train_test_strategy'],
            config['split_ratio'],
            config['image_cluster_num'],
            config['text_cluster_num'],
            config['link_k'],
            config['image_extractor'],
            config['text_extractor'],
        )
        dataset_folder = os.path.join(config['data_root'], config['dataset_type'], dataset_folder)
    return dataset_folder


def get_user_his_from_df(itemid_dict, userid_dict, rating_df):
    userid_his = list(rating_df.sort_values(['userid','timestamp'],ascending=True).groupby('userid'))
    user_hist_dict = {}
    for user_record in userid_his:
        userid = user_record[0]
        record = user_record[1]
        item_list = [itemid_dict[i] for i in list(record['itemid'])]
        t_lit = list(record['timestamp'])
        user_hist_dict[userid_dict[userid]] = [(i,t) for i,t in zip(item_list, t_lit)]
    return user_hist_dict


def basic_filter(rating_df, config):
    '''
    (1) filter by rating threshold
    (2) filter by user/item frequency
    '''
    # filter by rating threshold
    rating_df = rating_df[~(rating_df['rating']<config['rating_threshold'])]

    # filter by frequency
    if config['filter_strategy'] == 'core':
        del_item_num, del_user_num = 100,100
        while del_item_num>0 or del_user_num>0:
            ## drop items
            item_count = pd.DataFrame(rating_df['itemid'].value_counts())
            item_count.columns = ['nums']
            item_count = item_count[item_count['nums'] < config['core']]
            item_delindexs = item_count.index
            del_item_num = len(item_delindexs)
            ## drop users
            user_count = pd.DataFrame(rating_df['userid'].value_counts())
            user_count.columns = ['nums']
            user_count = user_count[user_count['nums'] < config['core']]
            user_delindexs = user_count.index
            del_user_num = len(user_delindexs)
            if del_item_num==0 and del_user_num==0:
                print("Finish droping, keeping core users-{} and items-{}.".format(config['core'], config['core']))
                break
            rating_df = rating_df[~rating_df['itemid'].isin(item_delindexs)]
            rating_df = rating_df[~rating_df['userid'].isin(user_delindexs)]
    elif config['filter_strategy'] == 'user':
        user_count = pd.DataFrame(rating_df['userid'].value_counts())
        user_count.columns = ['nums']
        user_count = user_count[user_count['nums'] < config['core']]
        user_delindexs = user_count.index
        rating_df = rating_df[~rating_df['userid'].isin(user_delindexs)]
        print("Finish droping users with less then {} interactions.".format(config['core']))
    else:
        print("no any filtering based on frequency")

    # re-id user and item, start from 1
    itemid_dict = {} # original item id => new id
    userid_dict = {} # to save
    all_item = list(set(rating_df['itemid']))
    all_user = list(set(rating_df['userid']))
    for i in all_item:
        itemid_dict[i]=len(itemid_dict)+1
    for u in all_user:
        userid_dict[u]=len(userid_dict)+1

    return itemid_dict, userid_dict, rating_df

'''
Build dataset dir, and save the files into the folder.
dataset_cf: train.txt, test.txt
dataset_sr: all_train_seq.txt, all_test_seq.txt, train.txt, test.txt
    all_train_seq[i] = (u, [i0,i1,i2]) as (user, train_seq)
    all_test_seq[i] = (u, [i0,i1,i2], [i3,i4]) as (user ,train_seq, test_seq)
    train[0] = [1,2,3] as input
        train[1] = 4 as label
    test[0] = [1,2,3] as input
        test[1] = [4] as label
dataset_mmsr: all_train_seq.txt, all_test_seq.txt, train.txt, test.txt
    all_train_seq[i] = (u, [i0,i1,i2]) as (user, train_seq)
    all_test_seq[i] = (u, [i0,i1,i2], [i3,i4]) as (user ,train_seq, test_seq)
    train[0] = [1,2,3] as input
        train[1] = [[4,4], [5,5], [6,6]] as input of image
        train[2] = [[7,7], [8,8], [9,9]] as input of image
        train[3] = 10 as label
    test[0] = [1,2,3] as input
        test[1] = [[4,4], [5,5], [6,6]] as input of image
        test[2] = [[7,7], [8,8], [9,9]] as input of image
        test[3] = 10 as label
'''
def build_cf_dataset(config):
    rating_df = get_df(config)
    itemid_dict, userid_dict, rating_df = basic_filter(rating_df, config)
    # TODO: keeping re-id user-item pair


def build_sr_dataset(config, save=True):
    rating_df = get_df(config)
    itemid_dict, userid_dict, rating_df = basic_filter(rating_df, config)
    print_basic_info(rating_df)

    user_hist_dict = get_user_his_from_df(itemid_dict, userid_dict, rating_df) # each: 30364: [(12197, 1409011200), (7894, 1430524800), (11612, 1430524800)]

    # split train-test
    split_ratio = config['split_ratio']
    if config['train_test_strategy'] == 'time-based':   # time-based/ratio
        t_list = sorted(rating_df['timestamp'])
        split_idx = len(t_list) - int(len(t_list)*split_ratio)
        split_time = t_list[split_idx]

        all_train_seq, all_test_seq = [],[] # (u, [i0,i1,i2])
        for u, record in user_hist_dict.items():
            train_data, test_data = [], []
            for i,t in record:
                if t<= split_time:
                    train_data.append(i)
                else:
                    test_data.append(i)
            if len(train_data)>0:
                all_train_seq.append((u,train_data))
            if len(test_data)>0:
                all_test_seq.append((u,train_data,test_data))
    elif config['train_test_strategy'] == 'ratio':
        all_train_seq, all_test_seq = [],[] # (u, [i0,i1,i2])
        for u, record in user_hist_dict.items():
            split_idx = len(record) - int(len(record)*split_ratio)
            train_data = [i for i,t in record[:split_idx]]
            test_data = [i for i,t in record[split_idx:]]
            if len(train_data)>0:
                all_train_seq.append((u,train_data))
            if len(test_data)>0:
                all_test_seq.append((u,train_data,test_data))
    
    # use data_augmentation, and save as to-be-used method
    tr_seqs, tr_labs = [], []
    for u,seq in all_train_seq: # TO: I dont use the user ID for now
        for i in range(1, len(seq)):
            tar = seq[-i]
            tr_labs += [tar]
            tr_seqs += [seq[:-i][-config['max_length']:]]

    te_seqs, te_labs = [], []
    for u, train_seq, seq in all_test_seq: # TO: I dont use the user ID for now
        for i in range(1, len(seq)+1):
            tar = seq[-i]
            te_labs += [tar]
            tes_s = train_seq+seq[:-i]
            te_seqs += [tes_s[-config['max_length']:]]
    
    train = (tr_seqs, tr_labs)
    test = (te_seqs, te_labs)

    # to save the file
    if save:
        dataset_folder = get_dataset_folder(config)
        pickle.dump(itemid_dict, open(os.path.join(dataset_folder, 'itemid_dict.txt'),'wb'))
        pickle.dump(userid_dict, open(os.path.join(dataset_folder, 'userid_dict.txt'),'wb'))
        pickle.dump(all_test_seq, open(os.path.join(dataset_folder, 'all_test_seq.txt'),'wb'))
        pickle.dump(all_train_seq, open(os.path.join(dataset_folder, 'all_train_seq.txt'),'wb'))
        
        pickle.dump(train, open(os.path.join(dataset_folder, 'train.txt'),'wb'))
        pickle.dump(test, open(os.path.join(dataset_folder, 'test.txt'),'wb'))
        print("saved all files")
    
    return itemid_dict, userid_dict, all_test_seq, all_train_seq, train, test


def enrich_data_by_vl(data, node_num, image_num, text_num, iid_dict, asin_image_dict, asin_text_dict):
    # given train/test, adding image and text information to the data
    seqs = data[0]
    text_seqs, image_seqs = [],[]
    for seq in seqs:
        image_seq = []
        text_seq = []
        for i in seq:
            itemid = iid_dict[i]
            if itemid in asin_image_dict:
                if type(asin_image_dict[itemid]) == list:
                    image_c = [ c+node_num for c in asin_image_dict[itemid] ]
                else:
                    image_c = [asin_image_dict[itemid]+node_num] # asin_image_dict[itemid] is an int
            else:
                image_c = []
            image_seq.append(image_c)
            
            if itemid in asin_text_dict:
                if type(asin_text_dict[itemid]) == list:
                    text_c = [ c+node_num+image_num for c in asin_text_dict[itemid] ]
                else:
                    text_c = [asin_text_dict[itemid]+node_num+image_num]
            else:
                text_c = []
            text_seq.append(text_c)
        text_seqs.append(text_seq)
        image_seqs.append(image_seq)

    return text_seqs, image_seqs
    

def get_image_modality_list(node_num, image_num, text_num, iid_dict, asin_image_dict, asin_text_dict, link_k):
    overll_item_img_cluster, overll_item_txt_cluster = [], [] # [I K]
    for iid in range(node_num):
        if iid not in iid_dict:
            overll_item_img_cluster.append([0]*link_k)
            overll_item_txt_cluster.append([0]*link_k)
        else:
            asin = iid_dict[iid]
            if asin in asin_image_dict:
                overll_item_img_cluster.append([c+node_num for c in asin_image_dict[asin]])
            else:
                overll_item_img_cluster.append([0]*link_k)
            
            if asin in asin_text_dict:
                overll_item_txt_cluster.append([c+node_num+image_num for c in asin_text_dict[asin]])
            else:
                overll_item_txt_cluster.append([0]*link_k)
    return overll_item_img_cluster, overll_item_txt_cluster


def get_feature_matrix(itemid_dict, node_num, config, f_type='image'):
    reverse_itemid_dict = {v:k for k,v in itemid_dict.items()}
    source = config['source']
    readin_feature_path = os.path.join(config['data_root'], 'preprocessed', f_type, config['{}_extractor'.format(f_type)], 'asin-{}-feature{}.npy'.format(source, config[f_type+'_dimension']))
    found_count, un_found_count = 0,0
    # asin_features = pickle.load(open(readin_feature_path, 'rb'))
    asin_features = dict(np.load(readin_feature_path, allow_pickle=True).tolist())
    return_matrix = np.zeros((node_num, config['{}_dimension'.format(f_type)]))
    for iid in range(node_num):
        if iid not in reverse_itemid_dict:
            print('cannot found {} iid'.format(iid))
            un_found_count += 1
            continue
        asin = reverse_itemid_dict[iid]
        if asin in asin_features.keys():
            return_matrix[iid] = asin_features[asin]
            found_count += 1
        else:
            un_found_count += 1
    print('For {} feature, overall {} items, {} item feature found, {} item feature unfound'.format(f_type, node_num, found_count, un_found_count))
    return return_matrix


def get_cluster_feature_matrix(config, f_type='image'):
    cluster_num = config[f_type+'_cluster_num']
    readin_feature_path = os.path.join(config['data_root'], 'preprocessed', f_type, config['{}_extractor'.format(f_type)], '{}-center{}-feature{}.npy'.format(f_type, config[f_type+'_cluster_num'], config[f_type+'_dimension']))
    # center_features = pickle.load(open(readin_feature_path, 'rb'))
    center_features = dict(np.load(readin_feature_path, allow_pickle=True).tolist())
    return_matrix = np.zeros((cluster_num, config['{}_dimension'.format(f_type)]))
    for c, arr in center_features.items():
        return_matrix[c] = arr
    return return_matrix


def build_mmsr_dataset(config, save=True):
    itemid_dict, userid_dict, all_test_seq, all_train_seq, train, test = build_sr_dataset(config, save=False)
    reverse_iid_dict = {v: k for k, v in itemid_dict.items()} # iid => asin

    # cluster_num=20
    cluster_method = config['cluster_method']  # kmeans/gmm
    image_cluster_num = config['image_cluster_num']  # node number of images
    text_cluster_num = config['text_cluster_num']  # node number of text
    link_k = config['link_k']  # how many images/text linked with an item
    image_dimension = config['image_dimension']
    text_dimension = config['text_dimension']

    asin_image_dict_path = os.path.join(config['data_root'], 'preprocessed', 'image', config['image_extractor'], 'asin_image_feature{}_{}_c{}_k{}_dict.npy'.format(image_dimension, cluster_method, image_cluster_num, link_k))
    asin_text_dict_path = os.path.join(config['data_root'], 'preprocessed', 'text', config['text_extractor'], 'asin_text_feature{}_{}_c{}_k{}_dict.npy'.format(text_dimension, cluster_method, text_cluster_num, link_k))
    if not(os.path.exists(asin_image_dict_path) and os.path.exists(asin_text_dict_path)):
        print("ERROR. asin_image_dict_path [{}] or asin_image_dict_path [{}] not found".format(asin_image_dict_path, asin_text_dict_path))
        return
    # asin_image_dict = pickle.load(open(asin_image_dict_path, 'rb'))
    # asin_text_dict = pickle.load(open(asin_text_dict_path, 'rb'))
    asin_image_dict = np.load(asin_image_dict_path, allow_pickle=True)
    asin_image_dict = dict(asin_image_dict.tolist())
    asin_text_dict = np.load(asin_text_dict_path, allow_pickle=True)
    asin_text_dict = dict(asin_text_dict.tolist())
    
    node_num = len(itemid_dict.values())+1 # and 0<pad>
    image_num = 0  # get the max id from asin_image_dict
    for s in asin_image_dict.values():
        for i in s:
            if i > image_num:
                image_num = i
    image_num += 1

    text_num = 0  # get the max id from asin_image_dict
    for s in asin_text_dict.values():
        for i in s:
            if i > text_num:
                text_num = i
    text_num += 1
    print("item number, image number, text number are: ",len(itemid_dict.values()), image_num, text_num)

    tra_text_seqs, tra_image_seqs = enrich_data_by_vl(train, node_num, image_num, text_num, reverse_iid_dict, asin_image_dict, asin_text_dict)
    tes_text_seqs, tes_image_seqs = enrich_data_by_vl(test, node_num, image_num, text_num, reverse_iid_dict, asin_image_dict, asin_text_dict)

    overll_item_img_cluster, overll_item_txt_cluster = get_image_modality_list(node_num, image_num, text_num, reverse_iid_dict, asin_image_dict, asin_text_dict, link_k)

    itemid_image_feature = get_feature_matrix(itemid_dict, node_num, config, 'image')
    itemid_text_feature = get_feature_matrix(itemid_dict, node_num, config, 'text')

    image_cluster_feature = get_cluster_feature_matrix(config, 'image')
    text_cluster_feature = get_cluster_feature_matrix(config, 'text')

    if save:
        dataset_folder = get_dataset_folder(config)
        pickle.dump(itemid_dict, open(os.path.join(dataset_folder, 'itemid_dict.txt'),'wb'))
        pickle.dump(userid_dict, open(os.path.join(dataset_folder, 'userid_dict.txt'),'wb'))

        pickle.dump(all_test_seq, open(os.path.join(dataset_folder, 'all_test_seq.txt'),'wb'))
        pickle.dump(all_train_seq, open(os.path.join(dataset_folder, 'all_train_seq.txt'),'wb'))
        
        pickle.dump([train[0], tra_text_seqs, tra_image_seqs, train[1]], open(os.path.join(dataset_folder, 'train.txt'),'wb'))
        pickle.dump([test[0], tes_text_seqs, tes_image_seqs, test[1]], open(os.path.join(dataset_folder, 'test.txt'),'wb'))

        pickle.dump(overll_item_img_cluster, open(os.path.join(dataset_folder, 'itemid_imageid_list_cluster{}_k{}_feature{}.pickle'.format(image_cluster_num, link_k, config['image_dimension'])),'wb'))
        pickle.dump(overll_item_txt_cluster, open(os.path.join(dataset_folder, 'itemid_textid_list_cluster{}_k{}_feature{}.pickle'.format(text_cluster_num, link_k, config['text_dimension'])),'wb'))
        
        pickle.dump(itemid_image_feature, open(os.path.join(dataset_folder, 'itemid_image_feature{}.pickle'.format(config['image_dimension'])),'wb'))
        pickle.dump(itemid_text_feature, open(os.path.join(dataset_folder, 'itemid_text_feature{}.pickle'.format(config['text_dimension'])),'wb'))

        pickle.dump(image_cluster_feature, open(os.path.join(dataset_folder, 'image_cluster{}_feature{}.pickle'.format(config['image_cluster_num'], config['image_dimension'])),'wb'))
        pickle.dump(text_cluster_feature, open(os.path.join(dataset_folder, 'text_cluster{}_feature{}.pickle'.format(config['text_cluster_num'], config['text_dimension'])),'wb'))

    return

if __name__=='__main__':
    config = build_config('../../config/preprocess.yaml')
    dataset_type = config['dataset_type']
    config['data_root'] = '/ssd1/holdenhu/Amazon2014_dataset/Amazon_Beauty'
    config['ratingdf_path'] = '/ssd1/holdenhu/Amazon2014_dataset/Amazon_Beauty/raw/ratings_Beauty.csv'
    working_folder = os.path.join(config['data_root'], dataset_type)

    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    dataset_type = config['dataset_type']
    dataset_folder = get_dataset_folder(config)
    if (not config['rewrite']) and os.path.exists(dataset_folder):
        print("dataset {} exsit already.".format(dataset_folder))
    else:
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        if dataset_type == 'dataset_sr':
            build_sr_dataset(config)
        elif dataset_type == 'dataset_mmsr':
            build_mmsr_dataset(config)
        else: # dataset_cf 
            build_cf_dataset(config)

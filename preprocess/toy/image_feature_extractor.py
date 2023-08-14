import os
import torch
import pickle
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("../")
from utils import build_config, extract_raw_image_feature, decompose_feature_by_pca, decompose_feature_by_ae, cluster_features


def extract_raw_feature(config, device='cuda', batch_size=2000, image_dir=''):
    image_feature_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'raw-feature1000.npy')
    image_path_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'raw-feature-image-path.npy')
    if os.path.exists(image_feature_list_path) and os.path.exists(image_path_list_path):
        print('skip resnet extractor')
        return np.load(image_feature_list_path), np.load(image_path_list_path) 
    features_array, image_paths = extract_raw_image_feature(device, batch_size, image_dir) # from utils.py

    # 将特征数组保存到文件中
    np.save(image_feature_list_path, features_array)
    np.save(image_path_list_path, np.array(image_paths))
    print('saved ', image_feature_list_path, ' and ', image_path_list_path)
    return (features_array, image_paths)


def get_dense_features(features_array, config, batch_size = 512, method='ae'):
    image_feature_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], '{}-feature{}.npy'.format(method, config['image_dimension']))
    if os.path.exists(image_feature_list_path):
        print('skip decomposition.')
        return np.load(image_feature_list_path)
    if method == 'pca':
        features_reduced = decompose_feature_by_pca(features_array, epoch_num=5, encoding_dimension=config['image_dimension'], batch_size = 512) # from utils.py
    elif method == 'ae':
        features_reduced = decompose_feature_by_ae(features_array, epoch_num=500, encoding_dimension=config['image_dimension'], batch_size = 256) # from utils.py
    
    np.save(image_feature_list_path, features_reduced)
    print('saved ', image_feature_list_path)
    return features_reduced


def save_features_dict(asin_list, features_array, feature_prefix='raw', feature_dim=1000):
    asin_features_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'asin-{}-feature{}.npy'.format(feature_prefix, feature_dim))
    if os.path.exists(asin_features_path):
        print('found saved asin_features_dict, skip')
        return
    asin_feature_dict = {}
    for idx, asin in enumerate(asin_list):
        feature = features_array[idx]
        asin_feature_dict[asin] = feature
    np.save(asin_features_path, asin_feature_dict)
    print('saved ', asin_features_path)
    return


def cluster_and_save(asin_list, img_feature_embedding, config, source='raw'):
    '''
    cluster based on img feature, and save it into asin_image_{}_c{}_k{}_dict.pickle [kmeans/cluster_num/link-k]
    '''
    cluster_num = config['image_cluster_num']
    k_num = config['link_k']
    cluster_method = config['cluster_method']
    feat_dim = 1000 if source=='raw' else config['image_dimension']
    asin_image_cluster_dict_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'asin_image_feature{}_{}_c{}_k{}_dict.npy'.format(feat_dim, cluster_method, cluster_num, k_num))
    image_center_feature_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'image-center{}-feature{}.npy'.format(config['image_cluster_num'], feat_dim))
    if os.path.exists(asin_image_cluster_dict_path) and os.path.exists(image_center_feature_path):
        print('found the saved asin-cluster, and cluster-center feature, end.')
        return
    
    image_center_feature, asin_image_cluster_dict = cluster_features(cluster_num, cluster_method, asin_list, img_feature_embedding, k_num) # from utils.py

    np.save(image_center_feature_path, image_center_feature)
    print('write ', image_center_feature_path)
    np.save(asin_image_cluster_dict_path, asin_image_cluster_dict)
    print('write ', asin_image_cluster_dict_path)


if __name__=='__main__':
    config = build_config('../../config/preprocess.yaml')
    config['data_root'] = '/ssd1/holdenhu/Amazon2014_dataset/Amazon_Toy/'
    working_folder = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'])
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features_array, image_path_list = extract_raw_feature(config, device, batch_size=1000, image_dir=config['data_root']+'image/')
    asin_list = [i.split('/')[-1][:-4] for i in image_path_list]

    save_features_dict(asin_list, features_array, feature_prefix='raw', feature_dim=1000)
    cluster_and_save(asin_list, features_array, config, source='raw')

    decomposite_methods = ['ae']
    for dec_method in decomposite_methods:
        features_reduced = get_dense_features(features_array, config, batch_size = 512, method=dec_method)
        save_features_dict(asin_list, features_reduced, feature_prefix=dec_method, feature_dim=config['image_dimension'])
        cluster_and_save(asin_list, features_reduced, config, source=dec_method)
    
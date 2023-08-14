import os
import re
import yaml
import torch
import pickle
from tqdm import tqdm

import numpy as np

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile

from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture


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


def extract_raw_feature(config, device='cuda', batch_size=2000, image_dir=''):
    image_feature_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'raw-feature1000.npy')
    image_path_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'raw-feature-image-path.npy')
    if os.path.exists(image_feature_list_path) and os.path.exists(image_path_list_path):
        print('skip resnet-50 extractor')
        return np.load(image_feature_list_path), np.load(image_path_list_path) 
    # 加载ResNet-50模型，将其设为评估模式
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # 创建预处理函数
    preprocess = transforms.Compose([
        transforms.Resize(256), # 将图片缩放到256x256大小
        transforms.CenterCrop(224), # 从中心裁剪出224x224大小的图片
        transforms.ToTensor(), # 将图片转换为Tensor格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 对图片进行归一化
    ])

    # 指定文件夹路径
    folder_path = image_dir

    # 读取文件夹中的所有图片，保存为一个列表
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')
        images.append(img)
        image_paths.append(img_path)


    # 将图片列表拆分成若干个batch，每个batch包含batch_size张图片
    batch_size = 1000
    n_batches = len(images) // batch_size
    if len(images) % batch_size != 0:
        n_batches += 1

    # 对每个batch中的所有图片进行特征提取，并保存到一个数组中
    features_list = []
    for i in tqdm(range(n_batches)):
        batch_images = images[i*batch_size:(i+1)*batch_size]

        # 对batch中的所有图片进行预处理，并将其转换为4D张量（batch_size=batch_size）
        batch_tensor = torch.zeros((batch_size, 3, 224, 224)).to(device)
        for j, img in enumerate(batch_images):
            img_tensor = preprocess(img).cuda()
            batch_tensor[j] = img_tensor

        # 使用ResNet-50模型提取图片特征
        with torch.no_grad():
            features = model(batch_tensor)

        # 将特征向量转换为numpy数组，并将其添加到特征列表中
        if device == 'cpu':
            features_np = features.squeeze().numpy()
        else:
            features_np = features.squeeze().cpu().numpy()
        features_list.append(features_np)

    # 将特征数组保存到文件中
    features_array = np.concatenate(features_list, axis=0)
    np.save(image_feature_list_path, features_array)
    np.save(image_path_list_path, np.array(image_paths))
    print('saved ', image_feature_list_path, ' and ', image_path_list_path)
    return (features_array, image_paths)


def get_pca_features(features_array, config, batch_size = 512):
    image_feature_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'pca-feature{}.npy'.format(config['image_dimension']))
    if os.path.exists(image_feature_list_path):
        print('skip pca transform.')
        return np.load(image_feature_list_path)
    # pca training
    print('starting training pca...')
    # 创建增量PCA对象，并指定batch_size和n_components参数
    ipca = IncrementalPCA(n_components=config['image_dimension'], batch_size=batch_size)

    # 逐批次对特征数组进行降维
    for epoch in range(3):
        for i in range(0, len(features_array), batch_size):
            batch = features_array[i:i+batch_size]
            ipca.partial_fit(batch)

    # 对所有特征进行降维
    features_reduced = ipca.transform(features_array)
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
    asin_image_dict_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'asin_image_feature{}_{}_c{}_k{}_dict.npy'.format(feat_dim, cluster_method, cluster_num, k_num))
    image_center_feature_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'image-center{}-feature{}.npy'.format(config['image_cluster_num'], feat_dim))
    if os.path.exists(asin_image_dict_path) and os.path.exists(image_center_feature_path):
        print('found the saved asin-cluster, and cluster-center feature, end.')
        return
    
    if cluster_method == 'kmeans':
        kmeans = MiniBatchKMeans(n_clusters=cluster_num, init_size=512, batch_size=1024, random_state=2023)
        print('img_feature_embedding,',img_feature_embedding.shape)
        kmeans.fit(img_feature_embedding)
        centers = kmeans.cluster_centers_ # [C * D]
        # clusters = kmeans.predict(img_feature_embedding)
    
    elif cluster_method == 'gmm':
        gmm = GaussianMixture(n_components=cluster_num, random_state=2023)
        gmm.fit(img_feature_embedding)
        centers = gmm.means_ # [C * D]
        # membership_probs = gmm.predict_proba(X)
    
    image_center_feature = {}
    for c_id in range(len(centers)):
        arr = centers[c_id]
        image_center_feature[c_id] = arr
    # pickle.dump(image_center_feature, open(image_center_feature_path, 'wb'))
    np.save(image_center_feature_path, image_center_feature)
    print('write ', image_center_feature_path)

    asin_image_dict = {}
    cluster_set_list = []
    for each_embedding in img_feature_embedding:
        distances = [np.linalg.norm(each_embedding - arr) for arr in centers]
        cluster_set_list.append(list(np.argsort(distances)[:k_num]))
    for asin, c_set in zip(asin_list, cluster_set_list):
        asin_image_dict[asin] = c_set
    # pickle.dump(asin_image_dict, open(asin_image_dict_path, 'wb'))
    np.save(asin_image_dict_path, asin_image_dict)
    print('write ', asin_image_dict_path)

    return asin_image_dict


if __name__=='__main__':
    config = build_config('../../config/preprocess.yaml')
    config['data_root'] = '/ssd1/holdenhu/ML_dataset/ml-1m/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features_array, image_path_list = extract_raw_feature(config, device, batch_size=1000, image_dir=config['data_root']+'image/')
    features_reduced = get_pca_features(features_array, config, batch_size = 512)

    # alignment, to asin dict
    asin_list = [i.split('/')[-1][:-4] for i in image_path_list]
    save_features_dict(asin_list, features_array, feature_prefix='raw', feature_dim=1000)
    save_features_dict(asin_list, features_reduced, feature_prefix='pca', feature_dim=config['image_dimension'])

    # Do the Clustering, default for pca features
    cluster_and_save(asin_list, features_array, config, source='raw')
    cluster_and_save(asin_list, features_reduced, config, source='pca')
    
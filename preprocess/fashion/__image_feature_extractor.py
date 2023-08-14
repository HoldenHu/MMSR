import os
import re
import yaml
import torch
import pickle
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


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

class Resnet50FeatureExtractor(nn.Module):
    def __init__(self, device):
        super(Resnet50FeatureExtractor, self).__init__()

        model = resnet50(pretrained=True).to(device=device)
        model.train(mode=False)
        train_nodes, eval_nodes = get_graph_node_names(model)

        return_nodes = {
            'layer4.2.relu_2': 'layer4',
        }
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, detect_range, transform=None):
        super(CustomImageDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        if detect_range != None:
            self.images = self.images[detect_range[0]:detect_range[1]]
        print('image list length:', len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.images[item])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        data = {
            'image': image,
            'path': image_path
        }

        return data


def image_feature_extract(image_height, image_width, batch_size, device, image_dir, detect_range):
    image_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
    ])

    custom_image_dataset = CustomImageDataset(
        image_dir,
        detect_range=detect_range,
        transform=image_transform,
    )
    
    image_loader = DataLoader(
        custom_image_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True
    )

    model = Resnet50FeatureExtractor(device=device)

    image_path_list = []
    image_feature_list = []

    for batch_idx, (data) in tqdm(enumerate(image_loader)):
        image_path = data['path']
        image_path_list.append(image_path)

        img = data['image']
        extracted_feature = model(img.to(device=device))

        for i in extracted_feature.keys():
            img_feature = extracted_feature[i].flatten().cpu().detach().numpy()
            image_feature_list.append(img_feature)
            # print(img_feature.shape)

    print(len(image_feature_list))
    print('Done feature extraction.')

    return (image_feature_list, image_path_list)

'''
extracting raw feature session by session
return two list store the path of session task out
'''
def extract_raw_feature(config, device):
    image_feature_list_path_list = [] 
    image_path_list_path_list = []
    image_height, image_width = 512, 512
    batch_size = 1
    image_dir = os.path.join(config['data_root'],'image')
    image_total_num = len(os.listdir(image_dir))
    print('there are overall {} images in {}'.format(image_total_num, image_dir))

    start_detect, each_detect = 0, 20000
    end_detect = start_detect + each_detect

    while start_detect < image_total_num:
        end_detect = image_total_num if end_detect > image_total_num else end_detect
        session_task = (start_detect, end_detect)
        
        image_feature_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'subtask_{}-{}_raw-features.pickle'.format(session_task[0], session_task[1]))
        image_path_list_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'subtask_{}-{}_image-path.pickle'.format(session_task[0], session_task[1]))
        
        if os.path.exists(image_feature_list_path) and os.path.exists(image_path_list_path):
            start_detect, end_detect = start_detect + each_detect, end_detect + each_detect
            image_feature_list_path_list.append(image_feature_list_path)
            image_path_list_path_list.append(image_path_list_path)
            print('found {}-{} raw representation file, skip'.format(session_task[0], session_task[1]))
            continue

        print('starting representing {}-{} images by ResNet'.format(session_task[0],session_task[1]))
        image_feature_list, image_path_list = image_feature_extract(image_height=image_height, image_width=image_width, batch_size=batch_size, device=device, image_dir=image_dir, detect_range=session_task)
        
        pickle.dump(image_feature_list, open(image_feature_list_path, 'wb'))
        pickle.dump(image_path_list, open(image_path_list_path, 'wb'))
        image_feature_list_path_list.append(image_feature_list_path)
        image_path_list_path_list.append(image_path_list_path)
        print('output {}-{} raw representation'.format(session_task[0], session_task[1]))

        start_detect, end_detect = start_detect + each_detect, end_detect + each_detect

    return image_feature_list_path_list, image_path_list_path_list


def train_pca_by_file(model, batch_size, feature_path, checkpoint_path, checkpoint=True):
    image_features = pickle.load(open(feature_path, 'rb'))
    
    cur, end_point = 0, len(image_features)
    for idx in tqdm(range((end_point//batch_size)+1)):
        if cur >= end_point:
            break
        nxt_cur = cur+batch_size if cur+batch_size<end_point else end_point
        model.fit(image_features[cur:nxt_cur])
        cur = nxt_cur
    if checkpoint:
        pickle.dump(model, open(checkpoint_path, 'wb'))
    return model


def train_pca(pca, feature_path_list, batch_size, config):
    pca_checkpoint_path_list = []
    for i,feature_path in enumerate(feature_path_list):
        checkpoint_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'pca_model_dim{}_checkpoint_{}.pkl'.format(config['image_dimension'], i))
        pca_checkpoint_path_list.append(checkpoint_path)
        if os.path.exists(checkpoint_path):
            pca = pickle.load(open(checkpoint_path,'rb'))
            print('found checkpoint {}'.format(i))
            continue
        print('start training pca on {}th file'.format(i))
        pca = train_pca_by_file(pca, batch_size, feature_path, checkpoint_path)
    return pca, pca_checkpoint_path_list

def transform_feature(pca, raw_feature_path_list):
    pca_feature_path_list = []
    for i,feature_path in enumerate(raw_feature_path_list):
        new_feature_path = feature_path.replace('raw-features','pca{}-features'.format(config['image_dimension']))
        # if os.path.exists(new_feature_path):
        #     pca_feature_path_list.append(new_feature_path)
        #     continue
        image_features = pickle.load(open(feature_path, 'rb'))
        cur, end_point = 0, len(image_features)
        img_feature_embedding = []
        for idx in tqdm(range((end_point//batch_size)+1)):
            if cur >= end_point:
                break
            nxt_cur = cur+batch_size if cur+batch_size<end_point else end_point
            _img_feature_embedding = pca.transform(image_features[cur:nxt_cur])
            img_feature_embedding.append(_img_feature_embedding)
            cur = nxt_cur
        img_feature_embedding = np.concatenate(img_feature_embedding, axis=0)
        print(img_feature_embedding.shape)

        pickle.dump(img_feature_embedding, open(new_feature_path, 'wb'))
        pca_feature_path_list.append(new_feature_path)
    return pca_feature_path_list


def save_and_clean(asin_list, feature_embedding, save=True, clean=False):
    print("saving ({}) and cleaning ({}) data".format(save, clean))
    asin_feature_dict = {}
    for idx in range(len(asin_list)):
        asin_feature_dict[asin_list[idx]] = feature_embedding[idx]
    if save:
        saving_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'asin-image-features.pickle')
        pickle.dump(asin_feature_dict, open(saving_path, 'wb'))
    if clean:
        pass # TODO: delete the files
    return


def cluster_and_save(asin_list, img_feature_embedding, config):
    '''
    cluster based on img feature, and save it into asin_image_{}_c{}_k{}_dict.pickle [kmeans/cluster_num/link-k]
    '''
    cluster_num = config['image_cluster_num']
    k_num = config['link_k']
    cluster_method = config['cluster_method']
    asin_image_dict_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'asin_image_{}_c{}_k{}_dict.pickle'.format(cluster_method, cluster_num, k_num))
    image_center_feature_path = os.path.join(config['data_root'], 'preprocessed','image', config['image_extractor'], 'image-center{}-feature{}.pickle'.format(config['image_cluster_num'], config['image_dimension']))
    if os.path.exists(asin_image_dict_path) and os.path.exists(image_center_feature_path):
        print('found the saved asin-cluster, and cluster-center feature, end.')
        return
    
    if cluster_method == 'kmeans':
        kmeans = MiniBatchKMeans(n_clusters=cluster_num, init_size=512, batch_size=1024, random_state=2023)
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
    pickle.dump(image_center_feature, open(image_center_feature_path, 'wb'))
    print('write ', image_center_feature_path)

    asin_image_dict = {}
    cluster_set_list = []
    for each_embedding in img_feature_embedding:
        distances = [np.linalg.norm(each_embedding - arr) for arr in centers]
        cluster_set_list.append(list(np.argsort(distances)[:k_num]))
    for asin, c_set in zip(asin_list, cluster_set_list):
        asin_image_dict[asin] = c_set
    pickle.dump(asin_image_dict, open(asin_image_dict_path, 'wb'))
    print('write ', asin_image_dict_path)

    return asin_image_dict


if __name__=='__main__':
    config = build_config('../../config/preprocess.yaml')
    config['data_root'] = '/ssd1/holdenhu/Amazon_dataset/Amazon_Fashion/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_path_list, combined_image_path_list = extract_raw_feature(config, device)

    # pca training
    pca = PCA(n_components=config['image_dimension'])
    batch_size = 2000
    pca, pca_checkpoint_path_list = train_pca(pca, feature_path_list, batch_size, config)

    # pca transform
    pca_feature_path_list = transform_feature(pca, feature_path_list)

    # alignment
    img_feature_embedding = pickle.load(open(pca_feature_path_list[0], 'rb'))
    image_path_list = pickle.load(open(combined_image_path_list[0], 'rb'))
    for pca_feature_path, image_path in zip(pca_feature_path_list[1:], combined_image_path_list[1:]):
        _img_feature_embedding = pickle.load(open(pca_feature_path, 'rb'))
        img_feature_embedding = np.concatenate((img_feature_embedding, _img_feature_embedding), axis=0)
        _image_path_list = pickle.load(open(image_path, 'rb'))
        image_path_list += _image_path_list
        print('align {} and {}'.format(pca_feature_path, image_path))
    print("final overall img feature embedding: ", img_feature_embedding.shape)

    asin_list = [i[0].split('/')[-1][:-4] for i in image_path_list]

    save_and_clean(asin_list, img_feature_embedding, save=True, clean=False) # for MMGCN, MGAT

    # Do the Clustering
    cluster_and_save(asin_list, img_feature_embedding, config)
    
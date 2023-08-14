import os
import re
import yaml
import torch
import math 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

def extract_raw_image_feature(device='cuda', batch_size=2000, image_dir=''):
    print("utils.py: extracting image feature from {}".format(image_dir))
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

    # 读取文件夹中的所有图片，保存为一个列表
    images = []
    image_paths = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
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
        batch_tensor = torch.zeros((len(batch_images), 3, 224, 224)).to(device)
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
    return features_array, image_paths


def extract_raw_text_feature(title_corpus, asin_list, tokenizer, model, batch_size=512):
    batch_num = math.ceil(len(title_corpus)/batch_size)

    features_array = []
    for batch_idx in tqdm(range(batch_num)):
        processing_text = title_corpus[batch_idx*batch_size: (batch_idx+1)*batch_size]
        inputs = tokenizer(processing_text, return_tensors="pt", padding=True, truncation=True)  # Batch size 1
        attention_mask = inputs['attention_mask']
        input_ids = inputs.input_ids
        encoded_sentence = model(input_ids, attention_mask)
        last_hidden_states = encoded_sentence.last_hidden_state
        sentence_embedding = last_hidden_states.mean(dim=1)

        if len(features_array) == 0:
            features_array = sentence_embedding.detach().numpy()
        else:
            features_array = np.concatenate((features_array, sentence_embedding.detach().numpy()), axis=0)

    print("final overall raw txt feature embedding: ", features_array.shape)
    assert features_array.shape[0] == len(asin_list)
    return features_array


# @_@: feautre decomposition
def decompose_feature_by_pca(features_array, epoch_num, encoding_dimension=64, batch_size = 512):
    # ae training
    print('utils.py: starting training pca...')
    ipca = IncrementalPCA(n_components=encoding_dimension, batch_size=batch_size)
    # 逐批次对特征数组进行降维
    for epoch in range(epoch_num):
        for i in range(0, len(features_array), batch_size):
            batch = features_array[i:i+batch_size]
            ipca.partial_fit(batch)
    # 对所有特征进行降维
    features_reduced = ipca.transform(features_array)
    return features_reduced


class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def decompose_feature_by_ae(features_array, epoch_num, encoding_dimension=64, batch_size = 512):
    print('utils.py: starting training autoencoder...')
    # 指定输入和编码维度
    input_dim = features_array.shape[1]
    # 定义自编码器模型并初始化
    autoencoder = LinearAutoencoder(input_dim, encoding_dimension).to('cuda')
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # 训练自编码器模型
    echo_num = 10
    best_state = None
    smallest_loss, bad_epoch, bad_epoch_top = math.inf, 0, 4 # when the loss is bigger than the last for n times, stop
    for epoch in range(epoch_num):
        loss_sum = 0
        for i in range(0, len(features_array), batch_size):
            batch = torch.tensor(features_array[i:i+batch_size], dtype=torch.float).to('cuda')
            # 正向传播
            output = autoencoder(batch)
            loss = criterion(output, batch)
            loss_sum += loss
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%echo_num == 0:
            print(loss_sum)
            if loss_sum >= smallest_loss:
                bad_epoch += 1
            else:
                bad_epoch = 0
                smallest_loss = loss_sum
                best_state = autoencoder.state_dict()
            if bad_epoch == bad_epoch_top:
                print('early stop at epoch {}'.format(epoch))
                break

    autoencoder.load_state_dict(best_state)
    # 对所有特征进行降维
    features_tensor = torch.tensor(features_array, dtype=torch.float).cuda()
    features_reduced = autoencoder.encoder(features_tensor).cpu().detach().numpy()
    return features_reduced


# @_@: cluster by KMeans or GMM
def cluster_features(cluster_num, cluster_method, asin_list, feature_embedding, k_num):
    print('utils.py: starting clustering by {}.'.format(cluster_method))

    if cluster_method == 'kmeans':
        kmeans = MiniBatchKMeans(n_clusters=cluster_num, init_size=512, batch_size=1024, random_state=2023)
        print('feature_embedding,', feature_embedding.shape)
        kmeans.fit(feature_embedding)
        centers = kmeans.cluster_centers_ # [C * D]
    elif cluster_method == 'gmm':
        gmm = GaussianMixture(n_components=cluster_num, random_state=2023)
        gmm.fit(feature_embedding)
        centers = gmm.means_ # [C * D]
    
    center_feature = {}
    for c_id in range(len(centers)):
        arr = centers[c_id]
        center_feature[c_id] = arr

    asin_cluster_dict = {}
    cluster_set_list = []
    for each_embedding in feature_embedding:
        distances = [np.linalg.norm(each_embedding - arr) for arr in centers]
        cluster_set_list.append(list(np.argsort(distances)[:k_num]))
    for asin, c_set in zip(asin_list, cluster_set_list):
        asin_cluster_dict[asin] = c_set

    return center_feature, asin_cluster_dict


# @_@: other utils
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
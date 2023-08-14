import os
import sys
import torch
import argparse
import pickle
from logging import getLogger

from data_utils import split_validation, Data, get_item_num
from utils import Config, init_seed, init_logger
from utils import trans_to_cuda, get_dataset_folder

from model import VLGraph, train_and_test

# python main.py VLGraph sample
parser = argparse.ArgumentParser()
parser.add_argument('dataset', default='sample', help='data name')
inni_opt = parser.parse_args()

# model_name = inni_opt.model_name
model_name = 'VLGraph'
dataset = inni_opt.dataset

def main(para_key, para_value):
    config = Config(model=model_name, dataset=dataset, config_file_list=['config/dataset.yaml', 'config/{}::4::{}.yaml'.format(model_name, dataset)])
    init_seed(config['seed'], config['reproducibility'], config['cuda_id'])
    config = init_logger(config)
    logger = getLogger()
    if config['tune_parameters']:
        for k,v in zip(para_key, para_value):
            logger.info("TUNING PARAMETERS: {} - {}".format(str(k), str(v)))
            config[k] = v
    else:
        config['noise_type'] = []
        config['noise_level'] = 0
    logger.info(config.parameters)

    data_folder = get_dataset_folder(config, dataset)
    logger.info("Used data_folder: "+data_folder)

    train_data = pickle.load(open(os.path.join(data_folder, 'train.txt'), 'rb'))
    test_data = pickle.load(open(os.path.join(data_folder, 'test.txt'), 'rb'))
    num_item = get_item_num(train_data, test_data)
    config['num_node'][dataset] = num_item
    
    train_data = Data(config, train_data, config['link_k'])
    test_data = Data(config, test_data, config['link_k'])

    image_cluster_feature = pickle.load(open(os.path.join(data_folder, 'image_cluster{}_feature{}.pickle'.format(config['image_cluster_num'], config['embedding_size'])) , 'rb')) 
    text_cluster_feature = pickle.load(open(os.path.join(data_folder, 'text_cluster{}_feature{}.pickle'.format(config['text_cluster_num'], config['embedding_size'])) , 'rb'))  

    item_image_list = pickle.load(open(os.path.join(data_folder, 'itemid_imageid_list_cluster{}_k{}_feature{}.pickle'.format(config['image_cluster_num'], config['link_k'], config['embedding_size'])) , 'rb'))   # [I K]
    item_text_list = pickle.load(open(os.path.join(data_folder, 'itemid_textid_list_cluster{}_k{}_feature{}.pickle'.format(config['text_cluster_num'], config['link_k'], config['embedding_size'])) , 'rb'))  

    model = trans_to_cuda(VLGraph(config, image_cluster_feature, text_cluster_feature, item_image_list, item_text_list))
    logger.info(model)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=8, batch_size=model.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size, shuffle=False, pin_memory=True)

    best_hit, best_hit_epoch = [0, 0], [0, 0]
    best_mrr, best_mrr_epoch = [0, 0], [0, 0]
    best_ndcg, best_ndcg_epoch = [0, 0], [0, 0]
    bad_counter = 0

    last_flag = 0 # whether is the best result
    for epoch in range(config['epoch']):
        logger.info('-------------------------------------------------------')
        logger.info('epoch: '+ str(epoch))
        
        out_scores = train_and_test(model, train_loader, test_loader, config['topk'], logger)
        current_flag = 0 # whether there is any better result outcome
        for k_idx, out_score in enumerate(out_scores):
            k = config['topk'][k_idx]
            hit, mrr, ndcg = out_score
            logger.info('Current Result: Recall@%d: %.4f \t MMR@%d: %.4f \t NDCG@%d: %.4f' % (k, hit, k, mrr, k, ndcg))
            if hit > best_hit[k_idx]:
                best_hit[k_idx] = hit
                best_hit_epoch[k_idx] = epoch
                current_flag = 1
            if mrr > best_mrr[k_idx]:
                best_mrr[k_idx] = mrr
                best_mrr_epoch[k_idx] = epoch
                current_flag = 1
            if ndcg > best_ndcg[k_idx]:
                best_ndcg[k_idx] = ndcg
                best_ndcg_epoch[k_idx] = epoch
                current_flag = 1
        for k_idx, k in enumerate(config['topk']):
            logger.info('>> Best Result: Recall@%d: %.4f \t MMR@%d: %.4f \t NDCG@%d: %.4f   (Epoch: \t %d, \t %d, \t %d)' 
                        % (k, best_hit[k_idx], k, best_mrr[k_idx], k, best_ndcg[k_idx], best_hit_epoch[k_idx], best_mrr_epoch[k_idx], best_ndcg_epoch[k_idx]))
        
        bad_counter = bad_counter+1 if (last_flag==0 and current_flag==0) else 0
        last_flag = current_flag
        if bad_counter >= config['patience']:
            logger.info('Early stoped.')
            break
    logger.info('========================================================')
    
    new_log_path = os.path.join(config["log_dir"],str(best_mrr[1])+".log")
    os.rename(os.path.join(config["log_dir"],"logging.log"), new_log_path)
    logger.info(new_log_path)

    if not config['tune_parameters']:
        sys.exit(0)
    else:
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)


if __name__ == '__main__':
    # para_dict = {
    #     # 'embedding_size': [64, 128],
    #     # 'alpha': [0.1, 0.2],
    #     'dropout_local': [0.2, 0.4],
    #     'batch_size': [32, 128, 512],
    #     'lr': [0.001, 0.005]
    # }
    # para_dict = {
    #     'noise_type': [['image'], ['text'],['image','text']],
    #     'noise_level': [0.1, 0.3, 0.5, 0.7, 0.9]
    # }
    para_dict = {
        'noise_type': [[]],
        'noise_level': [0],
        'auxiliary_info': [['node_type','pos'],['node_type'],['pos']],
    }
    paraname, para_list = list(para_dict.keys()), list(para_dict.values())
    from itertools import product
    para_list = list(product(*para_list))

    for paras in para_list:
        main(paraname, paras)

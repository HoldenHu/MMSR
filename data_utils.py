import random
import numpy as np
import torch
from torch.utils.data import Dataset


def split_validation(train_set, valid_portion):
    train_set_x, train_set_image, train_set_text, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    valid_set_image = [train_set_image[s] for s in sidx[n_train:]]
    train_set_image = [train_set_image[s] for s in sidx[:n_train]]
    valid_set_text = [train_set_text[s] for s in sidx[n_train:]]
    train_set_text = [train_set_text[s] for s in sidx[:n_train]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_image, train_set_text, train_set_y), (valid_set_x, valid_set_image, valid_set_text, valid_set_y)


def get_item_num(train, test):
    i_max = max(max(train[-1]), max(test[-1]))
    for s in train[0]:
        for i in s:
            if i>i_max:
                i_max=i
    for s in test[0]:
        for i in s:
            if i>i_max:
                i_max=i
    return i_max+1

class Data(Dataset):
    def __init__(self, config, data, link_k, train_len=None):
        self.inverse_seq = False # TODO
        self.data_length = len(data[0])

        self.inputs = data[0]
        self.image_inputs = data[1]
        self.text_inputs = data[2]
        self.targets = data[3]
        self.inputs, max_len = self._handle_data(data[0], train_len, self.inverse_seq)

        self.k = link_k
        # self.mask = np.asarray(mask)
        self.max_len = max_len

        ## For Experiemnt
        self.noise_type = config['noise_type']
        self.noise_level = config['noise_level']
    
    def _get_max_node_number(self):
        max_node_number = 0
        for idx in range(self.data_length):
            u = self.inputs[idx]
            i = self.image_inputs[idx]
            t = self.text_inputs[idx]
            node_number = len(np.unique(u)) + len(np.unique(i)) + len(np.unique(t))
            if node_number > max_node_number:
                max_node_number = node_number
        return max_node_number
            

    def _handle_data(self, inputData, train_len=None, reverse=False):
        len_data = [len(nowData) for nowData in inputData]
        if train_len is None:
            max_len = max(len_data)
            max_node_number = self._get_max_node_number()
            if max_node_number>max_len:
                max_len = max_node_number
        else:
            max_len = train_len

        # reverse the sequence
        us_pois = []
        for upois, le in zip(inputData, len_data):
            if reverse:
                _ = list(reversed(upois)) if le < max_len else list(reversed(upois[-max_len:]))
            else:
                _ = list(upois) if le < max_len else list(upois[:max_len])
            us_pois.append(_)

        # us_msks = [[1] * le if le < max_len else [1] * max_len for le in len_data]
        return us_pois, max_len

    def replace_with_none(self, m_input, noise_level):
        for i in range(len(m_input)):
            if random.random() < noise_level:
                m_input[i] = []
        return m_input

    def __getitem__(self, index):
        u_input, image_input, text_input, target = self.inputs[index], self.image_inputs[index], self.text_inputs[index], self.targets[index]
        # insert noise into the modality input
        if 'image' in self.noise_type:
            image_input = self.replace_with_none(image_input, self.noise_level)
        if 'text' in self.noise_type:
            text_input = self.replace_with_none(text_input, self.noise_level)

        le = len(u_input) # real length of the inputs
        # relation type # u-i, u-t, i-t, in, out, self-loop, bi-direction
        
        ## To Build Graph
        u_nodes = np.unique(u_input).tolist()
        i_nodes = np.unique([y for x in image_input for y in x]).tolist()
        t_nodes = np.unique([y for x in text_input for y in x]).tolist()
        nodes = u_nodes + i_nodes + t_nodes
        nodes = np.asarray(nodes + (self.max_len - len(nodes)) * [0]) # return

        u_node_num, i_node_num, t_node_num = len(u_nodes), len(i_nodes), len(t_nodes)
        node_type_mask = [1]*u_node_num + [2]*i_node_num + [3]*t_node_num
        node_type_mask = node_type_mask + (self.max_len - len(node_type_mask)) * [0] # return

        adj = np.zeros((self.max_len, self.max_len))

        # relation type # self-loop (1), out (2), in (3), bi-direction (4)
        # relation type # u-i (5), i-u (6), i-t (7), t-i (8), i-t (9), t-i (10)
        for i in np.arange(le):
        	# item assignment
            item = u_input[i]
            item_idx = np.where(nodes == item)[0][0] # idx in nodes set
            adj[item_idx][item_idx] = 1 # self-loop
            # image assignment
            image_bundle = image_input[i] # e.g. (12,13,14)
            for img in image_bundle:
                img_idx = np.where(nodes == img)[0][0]
                adj[img_idx][img_idx] = 1
                adj[item_idx][img_idx] = 5
                adj[img_idx][item_idx] = 6
            # text assignment
            text_bundle = text_input[i]
            for txt in text_bundle:
                text_idx = np.where(nodes == txt)[0][0]
                adj[text_idx][text_idx] = 1
                adj[item_idx][text_idx] = 7
                adj[text_idx][item_idx] = 8
        	# image-text matching relation
            for img in image_bundle:
                for txt in text_bundle:
                    img_idx = np.where(nodes == img)[0][0]
                    text_idx = np.where(nodes == txt)[0][0]
                    adj[img_idx][text_idx] = 9
                    adj[text_idx][img_idx] = 10

        for i in np.arange(le - 1):
            # item relation assignment
            prev_item = u_input[i]
            next_item = u_input[i+1]
            u = np.where(nodes == prev_item)[0][0]
            v = np.where(nodes == next_item)[0][0]
            if u == v or adj[u][v] == 4:
                continue
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
            # img relation assignment
            prev_image_bundle = image_input[i] # e.g. (12,13,14)
            next_image_bundle = image_input[i+1]
            for prev_img in prev_image_bundle:
                for next_img in next_image_bundle:
                    u = np.where(nodes == prev_img)[0][0]
                    v = np.where(nodes == next_img)[0][0]
                    if u == v or adj[u][v] == 4:
                        continue
                    if adj[v][u] == 2:
                        adj[u][v] = 4
                        adj[v][u] = 4
                    else:
                        adj[u][v] = 2
                        adj[v][u] = 3
		    # text relation assignment
            prev_text_bundle = text_input[i] # e.g. (12,13,14)
            next_text_bundle = text_input[i+1]
            for prev_txt in prev_text_bundle:
                for next_txt in next_text_bundle:
                    u = np.where(nodes == prev_txt)[0][0]
                    v = np.where(nodes == next_txt)[0][0]
                    if u == v or adj[u][v] == 4:
                        continue
                    if adj[v][u] == 2:
                        adj[u][v] = 4
                        adj[v][u] = 4
                    else:
                        adj[u][v] = 2
                        adj[v][u] = 3

        alias_inputs =[] 
        for item in u_input:
            item_idx = np.where(nodes == item)[0][0]
            alias_inputs.append(item_idx)
        
        # alias_img_inputs: [B L K]
        alias_img_inputs = [[0]*self.k for i in range(self.max_len)]
        for i, img_bundle in enumerate(image_input):
            for j,img in enumerate(img_bundle):
                img_idx = np.where(nodes == img)[0][0]
                alias_img_inputs[i][j] = img_idx

        alias_txt_inputs = [[0]*self.k for i in range(self.max_len)]
        for i, txt_bundle in enumerate(text_input):
            for j,txt in enumerate(txt_bundle):
                txt_idx = np.where(nodes == txt)[0][0]
                alias_txt_inputs[i][j] = txt_idx

        alias_inputs = alias_inputs + [0] * (self.max_len-le)
        u_input = u_input + [0] * (self.max_len-le)
        us_msks = [1] * le + [0] * (self.max_len-le) if le < self.max_len else [1] * self.max_len        

        node_pos_matrix = np.zeros((self.max_len, self.max_len))
        n_idx = 0
        for item in u_nodes:
            pos_idx = [index for index, search_idx in enumerate(u_input) if item==search_idx]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1
        for image in i_nodes:
            pos_idx = [index for index, sublist in enumerate(image_input) if image in sublist]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1
        for text in t_nodes:
            pos_idx = [index for index, sublist in enumerate(text_input) if text in sublist]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1

        return [torch.tensor(adj), torch.tensor(nodes), torch.tensor(node_type_mask), torch.tensor(node_pos_matrix),
        		torch.tensor(us_msks), torch.tensor(target), torch.tensor(u_input), 
                torch.tensor(alias_inputs), torch.tensor(alias_img_inputs), torch.tensor(alias_txt_inputs)]

    def __len__(self):
        return self.data_length

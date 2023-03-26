import numpy as np
import random

import pdb
import torch
from torch.utils.data import Dataset


class MyDataset_train(Dataset):
    def __init__(self, dataset, trust_dict, user_visited_dict, user_set, item_set):
        self.data = dataset
        self.trust_dict = trust_dict
        self.user_visited_dict = user_visited_dict
        self.item_set = item_set
        self.user_set = user_set
        self.user_list = list(user_set)
        self.item_list = list(item_set)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item: int):
        uid = self.data[item, 0]
        iid = self.data[item, 1]

        return uid, iid

    def positive_sampling_user(self, input_user_list, update_trust_dict):
        pos_list = []
        for i in range(len(input_user_list)):
            uid = input_user_list[i]
            if uid not in update_trust_dict:
                uid_pos = uid
            else:
                trust_list = update_trust_dict[uid]
                uid_pos = random.sample(trust_list, 1)[0]
            pos_list.append(uid_pos)
        return pos_list


    def negative_sampling_user(self, input_user_list, update_trust_dict):
        neg_list = np.random.choice(self.user_list, len(input_user_list))
        for i in range(len(input_user_list)):
            uid = input_user_list[i]
            if uid in update_trust_dict:
                trust_set = update_trust_dict[uid]
                if neg_list[i] in trust_set:
                    uid_neg = random.sample(self.user_set, 1)[0]
                    while uid_neg in trust_set:
                        uid_neg = random.sample(self.user_set, 1)[0]
                    neg_list[i] = uid_neg
        return neg_list

    def negative_sampling_item(self, input_user_list):
        neg_list = np.random.choice(self.item_list, len(input_user_list))
        output_sampling_result = []
        for i in range(len(input_user_list)):
            visited_set = self.user_visited_dict[input_user_list[i]]
            if neg_list[i] in visited_set:
                iid_neg = random.sample(self.item_set, 1)[0]
                while iid_neg in visited_set:
                    iid_neg = random.sample(self.item_set, 1)[0]
                output_sampling_result.append(iid_neg)
            else:
                output_sampling_result.append(neg_list[i])
        return output_sampling_result




class MyDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item: int):
        uid = self.data[item, 0]
        iid = self.data[item, 1]
        return [uid, iid]



def get_collator_train():
    def collator(data_points):
        packed_feat = []
        packed_trust_list = []
        for feat, trust_list in data_points:
            packed_feat.append(torch.LongTensor(np.reshape(np.array(feat), [1, -1])))
            packed_trust_list.append(trust_list)
        packed_feat = torch.cat(packed_feat, dim=0)
        return packed_feat, packed_trust_list
    return collator


def get_collator():
    def collator(data_points):
        packed_feat = []
        for feat in data_points:
            packed_feat.append(torch.LongTensor(np.reshape(np.array(feat), [1, -1])))
        packed_feat = torch.cat(packed_feat, dim=0)
        return packed_feat
    return collator

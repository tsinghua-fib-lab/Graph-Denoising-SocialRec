import pickle
import os
import numpy as np


filepath = './data/yelp'
num_user = 0
max_length = 30


def generate_visited_dict(data):
    data_dict = {}
    for i in range(data.shape[0]):
        uid = data[i,0]
        iid = data[i,1]
        if uid in data_dict:
            data_dict[uid].append(iid)
        else:
            data_dict[uid] = [iid]
    return data_dict

def generate_pop_dict(data):
    data_dict = {}
    for i in range(data.shape[0]):
        iid = data[i,1]
        if iid in data_dict:
            data_dict[iid] += 1
        else:
            data_dict[iid] = 1
    return data_dict


def generate_visited_matrix(data_dict, pop_dict, num_user, max_length):
    visited_matrix = np.zeros((num_user, max_length))
    mask_matrix = np.zeros((num_user, max_length))
    for k, v in data_dict.items():
        pop_list = []
        for each in v:
            pop_list.append([each, pop_dict[each]])
        pop_list = sorted(pop_list, key=lambda x:x[1], reverse=True)

        item_list = []
        for i in range(min(max_length, len(v))):
            item_list.append(pop_list[i][0])

        for i in range(max_length):
            if i < len(item_list):
                visited_matrix[k, i] = item_list[i]
            else:
                mask_matrix[k, i] = 1

    return visited_matrix, mask_matrix


data_all = pickle.load(open(os.path.join(filepath, 'data_all.pkl'), 'rb'))
data_dict = generate_visited_dict(data_all[0])
pop_dict = generate_pop_dict(data_all[0])

visited_matrix, mask_matrix = generate_visited_matrix(data_dict, pop_dict, num_user, max_length)

visited_and_mask_matrix = (visited_matrix, mask_matrix)
pickle.dump(visited_and_mask_matrix, open(os.path.join(filepath, 'visited_and_mask_matrix_%d.pkl' % max_length), 'wb'))

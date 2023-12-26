import os
import pdb
import numpy as np
import random
import pickle

filepath = '***'
ratingsData = np.loadtxt(os.path.join(filepath, 'ratings.txt'))
trustData = np.loadtxt(os.path.join(filepath, 'trusts.txt'))


# filter
data = []
for i in range(ratingsData.shape[0]):
    if ratingsData[i,2]>3:
        uid = ratingsData[i, 0]
        iid = ratingsData[i, 1]
        data.append([uid, iid])


user_count = {}
item_count = {}
for i in range(len(data)):
    uid = data[i][0]
    iid = data[i][1]
    if uid not in user_count:
        user_count[uid] = 0
    if iid not in item_count:
        item_count[iid] = 0
    user_count[uid] += 1
    item_count[iid] += 1

user_set = set()
item_set = set()
for k, v in user_count.items():
    if v > 2:
        user_set.add(k)
for k, v in item_count.items():
    if v > 2:
        item_set.add(k)


user_reid_dict = dict(zip(list(user_set), list(range(len(user_set)))))
item_reid_dict = dict(zip(list(item_set), list(range(len(item_set)))))
user_set = set(user_reid_dict.values())
item_set = set(user_reid_dict.values())


trust_data = []
data_all = []
for i in range(len(data)):
    uid = data[i][0]
    iid = data[i][1]
    if uid in user_reid_dict and iid in item_reid_dict:
        data_all.append([user_reid_dict[uid], item_reid_dict[iid]])


data_all = np.array(data_all)
np.random.shuffle(data_all)

for i in range(len(trustData)):
    uid1 = trustData[i,0]
    uid2 = trustData[i,1]
    if uid1 in user_reid_dict and uid2 in user_reid_dict:
        trust_data.append([user_reid_dict[uid1], user_reid_dict[uid2]])

trust_data = np.array(trust_data)
print(len(user_set), len(item_set), len(data_all), trust_data.shape[0])


user_visited_dict = {}
for i in range(data_all.shape[0]):
    uid = data_all[i,0]
    iid = data_all[i,1]
    if uid not in user_visited_dict:
        user_visited_dict[uid] = set()
    user_visited_dict[uid].add(iid)


trust_dict = {}
for i in range(trust_data.shape[0]):
    uid1 = trust_data[i, 0]
    uid2 = trust_data[i, 1]
    if uid1 not in trust_dict:
        trust_dict[uid1] = []
    if uid2 not in trust_dict:
        trust_dict[uid2] = []
    trust_dict[uid1].append(uid2)
    trust_dict[uid2].append(uid1)


train = data_all[:int(0.7*data_all.shape[0]), :]
valid = data_all[int(0.7*data_all.shape[0]):int(0.8*data_all.shape[0]), :]
test = data_all[int(0.8*data_all.shape[0]):, :]

neg_data_valid = []
for each_user in set(valid[:, 0].tolist()):
    candidate_set = item_set-user_visited_dict[each_user]
    neg_items = random.sample(candidate_set, 100)
    for each in neg_items:
        neg_data_valid.append([each_user, each])

neg_data_test = []
for each_user in set(test[:, 0].tolist()):
    candidate_set = item_set-user_visited_dict[each_user]
    neg_items = random.sample(candidate_set, 100)
    for each in neg_items:
        neg_data_test.append([each_user, each])

user_item_dict_valid = {}
for i in range(valid.shape[0]):
    uid = valid[i, 0]
    iid = valid[i, 1]
    if uid not in user_item_dict_valid:
        user_item_dict_valid[uid] = []
    user_item_dict_valid[uid].append(i)

user_item_dict_test = {}
for i in range(test.shape[0]):
    uid = test[i, 0]
    iid = test[i, 1]
    if uid not in user_item_dict_test:
        user_item_dict_test[uid] = []
    user_item_dict_test[uid].append(i)

user_item_dict = (user_item_dict_valid, user_item_dict_test)
neg_data = (neg_data_valid, neg_data_test)
data_all = (train, valid, test)
trust_data = np.concatenate([trust_data, trust_data[:, [1, 0]]], axis=0)


pickle.dump(data_all, open(os.path.join(filepath, 'data_all.pkl'),'wb'))
pickle.dump(neg_data, open(os.path.join(filepath, 'neg_data.pkl'),'wb'))
pickle.dump(user_item_dict, open(os.path.join(filepath, 'user_item_dict.pkl'),'wb'))
pickle.dump(trust_dict, open(os.path.join(filepath, 'trust_dict.pkl'),'wb'))
pickle.dump(trust_data, open(os.path.join(filepath, 'trust_data_direct.pkl'),'wb'))
pickle.dump(user_visited_dict, open(os.path.join(filepath, 'user_visited_dict.pkl'),'wb'))







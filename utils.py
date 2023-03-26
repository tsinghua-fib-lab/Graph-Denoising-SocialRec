import logging
import math
from datetime import datetime
import numpy as np
import os
import pdb
import torch


def init_log(args):
    if not os.path.exists(os.path.join(args.log_dir, args.dataset_name)):
        os.mkdir(os.path.join(args.log_dir, args.dataset_name))
    log_time = datetime.now().strftime('%b%d_%H-%M-%S')

    log_dir_exp = os.path.join(args.log_dir, args.dataset_name,
                               'log-%s-%s-lr%s-reg%.4f-alpha%s-emb%d-gcn%d-beta%s-droprate%s_%s-%s' %
                               (args.model, log_time, str(args.lr), args.l2_reg, str(args.alpha),
                                args.embedding_size, args.num_gcn, str(args.beta), str(args.R), str(args.gamma),
                                args.trial))
    print(log_dir_exp)
    if not os.path.exists(log_dir_exp):
        os.mkdir(log_dir_exp)
    logging.basicConfig(filename=os.path.join(log_dir_exp, 'log'), level=logging.INFO, filemode="w")
    return log_dir_exp


def recursive_to(iterable, device):
    if isinstance(iterable, torch.Tensor):
        iterable.data = iterable.data.to(device)
    elif isinstance(iterable, (list, tuple)):
        for v in iterable:
            recursive_to(v, device)


def cal_recall_ndcg_with_neg(predict_prob, neg_score, user_item_dict, topk, user_id_to_idx):
    ndcg_all_user = []
    recall_all_user = []

    for uid in user_item_dict.keys():
        neg_sample_index_list = user_item_dict[uid]
        pos_score_one_user = []
        for each in neg_sample_index_list:
            pos_score_one_user.append(predict_prob[each])
        neg_score_one_user = neg_score[user_id_to_idx[uid], :].tolist()

        ndcg_one_user = []
        recall_one_user = []
        score_all = pos_score_one_user + neg_score_one_user
        for i in range(len(pos_score_one_user)):
            target_score = pos_score_one_user[i]
            rank = 0
            for j in range(len(score_all)):
                if score_all[j] >= target_score and i != j:
                    rank += 1
                if rank >= topk[-1]:
                    break
            ndcg = []
            recall = []
            for topk_each in topk:
                if rank < topk_each:
                    ndcg.append(math.log(2) / math.log(rank + 2))
                    recall.append(1)
                else:
                    ndcg.append(0)
                    recall.append(0)
            ndcg_one_user.append(ndcg)
            recall_one_user.append(recall)

        ndcg_one_user = np.sum(np.array(ndcg_one_user), axis=0)
        recall_one_user = np.sum(np.array(recall_one_user), axis=0)

        for i in range(len(topk)):
            recall_one_user[i] = recall_one_user[i] / min(topk[i], len(pos_score_one_user))
            dcg_max = 0
            for j in range(min(topk[i], len(pos_score_one_user))):
                dcg_max += math.log(2) / math.log(j + 2)
            ndcg_one_user[i] = ndcg_one_user[i] / dcg_max
            if ndcg_one_user[i] > 1.0001:
                print('Wrong NDCG value')
                pdb.set_trace()
        ndcg_all_user.append(ndcg_one_user)
        recall_all_user.append(recall_one_user)
    ndcg_all_user = list(np.mean(np.array(ndcg_all_user), axis=0))
    recall_all_user = list(np.mean(np.array(recall_all_user), axis=0))

    return recall_all_user, ndcg_all_user


def get_performance(predict_prob, neg_score, user_item_dict, user_list_neg, topk):
    neg_score = np.reshape(neg_score, [-1, 100])
    assert neg_score.shape[0] == len(user_list_neg)
    user_id_to_idx = dict(zip(user_list_neg, list(range(len(user_list_neg)))))
    recall, ndcg = cal_recall_ndcg_with_neg(predict_prob, neg_score, user_item_dict, topk, user_id_to_idx)

    return [recall, ndcg]


def get_log_text(metric):
    recall, ndcg = metric
    log_text = 'RECALL=[%.4f, %.4f], NDCG = [%.4f, %.4f]' % (recall[0], recall[1], ndcg[0], ndcg[1])

    print(log_text)
    logging.info(log_text)

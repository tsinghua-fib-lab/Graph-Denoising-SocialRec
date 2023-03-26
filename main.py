
import argparse
import logging
import math
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import setproctitle
from time import time
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from utils import *
from models import *
from datasets import *



def train_epoch(epoch, train_loader, relation_loader, train_dataset, interaction_GCN_model, optimizer, edge_weight,
                trust_dict, gpu_id, args):
    interaction_GCN_model.train()
    loss_sum = []
    reg_loss_sum = []
    social_loss_sum = []

    for input in tqdm(train_loader):
        recursive_to(input, 'cuda:%d' % gpu_id)
        recursive_to(edge_weight, 'cuda:%d' % gpu_id)
        optimizer.zero_grad()

        h_user, h_item = interaction_GCN_model(edge_weight)

        item_list_neg = train_dataset.negative_sampling_item(input[:, 0].cpu().numpy().tolist())
        item_list_neg = torch.tensor(item_list_neg, device='cuda:%d' % gpu_id).long()

        user_emb = h_user[input[:, 0]]
        item_emb = h_item[input[:, 1]]
        item_emb_neg = h_item[item_list_neg]

        score_pos = torch.reshape(torch.sum(user_emb * item_emb, dim=1), [-1])
        score_neg = torch.reshape(torch.sum(user_emb * item_emb_neg, dim=1), [-1])
        loss = torch.mean(torch.log(1 + torch.exp(score_neg - score_pos - 1e-8)))
        reg_loss = (torch.sum(user_emb*user_emb)+torch.sum(item_emb*item_emb))/input.shape[0]

        if args.alpha != 0:
            user_list_neg = train_dataset.negative_sampling_user(input[:, 0].cpu().numpy().tolist(), trust_dict)
            user_list_neg = torch.tensor(user_list_neg, device='cuda:%d' % gpu_id).long()

            user_list_pos = train_dataset.positive_sampling_user(input[:, 0].cpu().numpy().tolist(), trust_dict)
            user_list_pos = torch.tensor(user_list_pos, device='cuda:%d' % gpu_id).long()

            relation_score_pos = interaction_GCN_model.get_user_agg_embedding(input[:, 0], user_list_pos)
            relation_score_neg = interaction_GCN_model.get_user_agg_embedding(input[:, 0], user_list_neg)

            label_pos = torch.ones(relation_score_pos.shape[0], device='cuda:%d' % args.gpu_id)
            label_neg = torch.zeros(relation_score_neg.shape[0], device='cuda:%d' % args.gpu_id)

            relation_score = torch.cat([relation_score_pos, relation_score_neg], dim=0)
            label = torch.cat([label_pos, label_neg], dim=0)

            relation_loss = interaction_GCN_model.BCEloss(relation_score, label)

            loss = (1-args.alpha)*loss + args.l2_reg * reg_loss + args.alpha*relation_loss
            social_loss_sum.append(float(relation_loss.cpu()))
            if torch.isnan(relation_loss):
                print('Relation_loss was NaN')
                logging.info('Relation_loss was NaN')
                exit()
        else:
            loss = loss + args.l2_reg * reg_loss
            social_loss_sum.append(0)

        if torch.isnan(reg_loss):
            print('RegLoss was NaN')
            logging.info('RegLoss was NaN')
            exit()
        if torch.isnan(loss):
            print('Loss was NaN')
            logging.info('Loss was NaN')
            exit()
        loss.backward(retain_graph=True)
        loss_sum.append(float(loss.cpu()))
        reg_loss_sum.append(float(reg_loss.cpu()))

        optimizer.step()

    relation_score_all = []
    for relation_input in relation_loader:
        relation_score = interaction_GCN_model.get_user_agg_embedding(relation_input[:, 0], relation_input[:,1])
        relation_score_all.append(relation_score.detach().cpu().numpy())
    relation_score_all = np.concatenate(relation_score_all, axis=0)

    return sum(loss_sum) / len(train_loader), relation_score_all


def get_predict(data_loader, interaction_GCN_model, edge_weight, gpu_id):
    interaction_GCN_model.eval()
    with torch.autograd.no_grad():
        output_all = []
        for input in data_loader:
            recursive_to(input, 'cuda:%d' % gpu_id)

            h_user, h_item = interaction_GCN_model(edge_weight)
            user_emb = h_user[input[:, 0]]
            item_emb = h_item[input[:, 1]]
            output = torch.reshape(torch.sum(user_emb * item_emb, dim=1), [-1,1])
            output_all += [np.array(output.cpu())]
        output_all = np.reshape(np.concatenate(output_all, axis=0), [-1, 1])
        if np.sum(np.isnan(output_all)) != 0:
            print('Nan predict score')
            logging.info('Nan predict score')
            exit()

        return output_all


def delete_uu_edge(relation_score, edge_weight, trust_index_dict, epsilon, args):
    for k, v in trust_index_dict.items():
        if len(v) < epsilon:
            continue
        else:
            drop_num = math.pow(int(math.log2(len(v))), args.gamma)*args.R

        score_one_user = []
        for each in v:
            score_one_user.append([each, relation_score[each]])
        score_one_user = sorted(score_one_user, key=lambda x: x[1])
        for i in range(drop_num):
            edge_weight[score_one_user[i][0]] = 0
    return edge_weight


def generate_trust_dict(trust_data, edge_weight):
    trust_dict = {}
    for i in range(trust_data.shape[0]):
        if edge_weight[i] == 1:
            uid1 = trust_data[i, 0]
            uid2 = trust_data[i, 1]
            if uid2 not in trust_dict:
                trust_dict[uid2] = []
            if uid1 not in trust_dict:
                trust_dict[uid1] = []
            trust_dict[uid1].append(uid2)
            trust_dict[uid2].append(uid1)
    return trust_dict


def train_model(data_loader, train_dataset, user_item_dict, user_list_neg, trust_data, epsilon,
                interaction_GCN_model, log_dir_exp, args):
    train_loader, val_loader, test_loader, neg_data_valid_loader, neg_data_test_loader, relation_loader = data_loader
    best_metric = -1
    best_epoch_metric = [-1, -1]
    early_stop_count = 0
    topk = [1, 3]
    interaction_GCN_model.cuda()
    optimizer = Adam([{"params": interaction_GCN_model.parameters()}], lr=args.lr)
    relation_score_all_epoch = None

    edge_weight = torch.Tensor([1] * trust_data.shape[0])
    trust_dict = generate_trust_dict(trust_data, edge_weight)

    trust_index_dict = {}
    for i in range(trust_data.shape[0]):
        uid = trust_data[i, 0]
        if uid not in trust_index_dict:
            trust_index_dict[uid] = []
        trust_index_dict[uid].append(i)

    for epoch in range(args.epochs):
        train_loss, relation_score_one_epoch = \
             train_epoch(epoch, train_loader, relation_loader, train_dataset, interaction_GCN_model, optimizer,
                         edge_weight, trust_dict, args.gpu_id, args)
        if epoch == 0:
            relation_score_all_epoch = relation_score_one_epoch
        elif epoch % args.D == 0:
            relation_score_all_epoch = args.beta*relation_score_all_epoch+(1-args.beta)*relation_score_one_epoch
            edge_weight = delete_uu_edge(relation_score_all_epoch, edge_weight, trust_index_dict, epsilon, args)
            pickle.dump([edge_weight.cpu().numpy()], open(os.path.join(log_dir_exp, 'edge_%d.pkl' % epoch), 'wb'))
            trust_dict = generate_trust_dict(trust_data, edge_weight)

        neg_score_valid = np.reshape(get_predict(neg_data_valid_loader, interaction_GCN_model, edge_weight, args.gpu_id), [-1, 100])
        neg_score_test = np.reshape(get_predict(neg_data_test_loader, interaction_GCN_model, edge_weight, args.gpu_id), [-1, 100])
        val_predict = np.reshape(get_predict(val_loader, interaction_GCN_model, edge_weight, args.gpu_id), [-1])
        test_predict = np.reshape(get_predict(test_loader, interaction_GCN_model, edge_weight, args.gpu_id), [-1])

        val_result = get_performance(val_predict, neg_score_valid, user_item_dict[0], user_list_neg[0], topk)
        test_result = get_performance(test_predict, neg_score_test, user_item_dict[1], user_list_neg[1], topk)

        log_text = '[epoch %d]: train_loss =  %.4f\nval' % (epoch, train_loss)
        print(log_text)
        logging.info(log_text)
        get_log_text(val_result)
        print('test')
        logging.info('test')
        get_log_text(test_result)

        val_metric = val_result[0][0]
        if val_metric > best_metric:
            if early_stop_count < args.early_stop:
                torch.save(interaction_GCN_model, os.path.join(log_dir_exp, 'model.ckpt'))
                best_metric = val_metric
                best_epoch_metric = test_result
                early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= args.early_stop:
            print('early stop!')
            break

    print('-----Result-----')
    logging.info('-----Result-----')
    print('best_epoch')
    logging.info('best_epoch')
    get_log_text(best_epoch_metric)


def construct_graph(interaction, relation, num_embeddings):
    graph_data_interaction = {
        ('user', 'trust', 'user'): (torch.tensor(relation[:, 0]), torch.tensor(relation[:, 1])),
        ('user', 'rate', 'item'): (torch.tensor(interaction[:, 0]), torch.tensor(interaction[:, 1])),
        ('item', 'rated', 'user'): (torch.tensor(interaction[:, 1]), torch.tensor(interaction[:, 0]))
    }
    g_interaction = dgl.heterograph(graph_data_interaction,
                                    num_nodes_dict={'user': num_embeddings[0], 'item': num_embeddings[1]})

    return g_interaction


def load_data(args):
    data = pickle.load(open(os.path.join(args.data_dir, 'data_all.pkl'), 'rb'))
    user_visited_dict = pickle.load(open(os.path.join(args.data_dir, 'user_visited_dict.pkl'), 'rb'))

    neg_data = pickle.load(open(os.path.join(args.data_dir, 'neg_data.pkl'), 'rb'))
    user_item_dict = pickle.load(open(os.path.join(args.data_dir, 'user_item_dict.pkl'), 'rb'))
    trust_dict = pickle.load(open(os.path.join(args.data_dir, 'trust_dict.pkl'), 'rb'))
    trust_data = pickle.load(open(os.path.join(args.data_dir, 'trust_data_direct.pkl'), 'rb'))
    visited_and_mask_matrix = pickle.load(open(os.path.join(args.data_dir, 'visited_and_mask_matrix_30.pkl'), 'rb'))

    data_train, data_valid, data_test = data[0], data[1], data[2]
    neg_data_valid, neg_data_test = np.array(neg_data[0]), np.array(neg_data[1])
    user_item_dict = list(user_item_dict)

    data_all = np.concatenate(list(data), axis=0)

    visited_matrix = visited_and_mask_matrix[0]
    mask_matrix = visited_and_mask_matrix[1]

    user_set = set(data_all[:, 0].tolist())
    item_set = set(data_all[:, 1].tolist())

    valid_user_list_neg = neg_data_valid[:, 0].tolist()
    valid_user_list_neg = [valid_user_list_neg[i * 100] for i in range(int(len(valid_user_list_neg) / 100))]
    test_user_list_neg = neg_data_test[:, 0].tolist()
    test_user_list_neg = [test_user_list_neg[i * 100] for i in range(int(len(test_user_list_neg) / 100))]
    user_list_neg = [valid_user_list_neg, test_user_list_neg]


    if args.dataset_name == 'yelp':
        num_embeddings = [32827, 59972]
        epsilon = 5
        user_profile_dict = pickle.load(open(os.path.join(args.data_dir, 'user_profile_dict.pkl'), 'rb'))
    elif args.dataset_name == 'ciao':
        num_embeddings = [7355, 17867]
        epsilon = 10
        user_profile_dict = None
    elif args.dataset_name == 'douban':
        num_embeddings = [2669, 15940]
        epsilon = 5
        user_profile_dict = None
    else:
        raise ValueError

    g_interaction = construct_graph(data_train, trust_data, num_embeddings)

    train_dataset = MyDataset_train(data_train, trust_dict, user_visited_dict, user_set, item_set)
    val_dataset = MyDataset(data_valid)
    test_dataset = MyDataset(data_test)
    valid_neg_dataset = MyDataset(neg_data_valid)
    test_neg_dataset = MyDataset(neg_data_test)
    relation_dataset = MyDataset(trust_data)

    collate_fn = get_collator()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.data_loader_workers,
                              collate_fn=collate_fn, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.data_loader_workers,
                            collate_fn=collate_fn, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.data_loader_workers,
                             collate_fn=collate_fn, pin_memory=False)
    neg_data_valid_loader = DataLoader(valid_neg_dataset, batch_size=args.batch_size,
                                       num_workers=args.data_loader_workers,
                                       collate_fn=collate_fn, pin_memory=False)
    neg_data_test_loader = DataLoader(test_neg_dataset, batch_size=args.batch_size,
                                      num_workers=args.data_loader_workers,
                                      collate_fn=collate_fn, pin_memory=False)
    relation_loader = DataLoader(relation_dataset, batch_size=args.batch_size,
                                      num_workers=args.data_loader_workers,
                                      collate_fn=collate_fn, pin_memory=False)

    data_loader = (train_loader, val_loader, test_loader, neg_data_valid_loader, neg_data_test_loader, relation_loader)
    return data_loader, train_dataset, visited_matrix, mask_matrix, \
           num_embeddings, epsilon, user_item_dict, user_list_neg, trust_data, g_interaction, user_profile_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Run socialMF")
    parser.add_argument('--model', type=str, default='GDMSR', )
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--embedding_size', type=int, default=8)
    parser.add_argument('--early_stop', type=int, default=10, help='early stop patience')
    parser.add_argument('--min_epoch', type=int, default=10)
    parser.add_argument('--num_gcn', type=int, default=2)
    parser.add_argument('--data_loader_workers', type=int, default=15)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--R', type=float, default=0.5)
    parser.add_argument('--D', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default='yelp')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--data_dir', type=str, default='./data/yelp')
    parser.add_argument('--trial', type=str, default='0', help='Indicate trail id with same condition')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setproctitle.setproctitle('%s_%s' % (args.model, args.dataset_name))
    torch.cuda.set_device(args.gpu_id)

    log_dir_exp = init_log(args)
    data_loader, train_dataset, visited_matrix, mask_matrix, num_embeddings, epsilon, \
        user_item_dict, user_list_neg, trust_data, g_interaction, user_profile_dict = load_data(args)

    interaction_GCN_model = interaction_GCN(g_interaction, args.embedding_size, num_embeddings,
                                            visited_matrix, mask_matrix, ['trust', 'rate', 'rated'],
                                            args.gpu_id, n_layers=args.num_gcn, user_profile_dict=user_profile_dict)

    interaction_GCN_model.g = interaction_GCN_model.g.to('cuda:%d' % args.gpu_id)

    train_model(data_loader, train_dataset, user_item_dict, user_list_neg, trust_data, epsilon,
                interaction_GCN_model, log_dir_exp, args)

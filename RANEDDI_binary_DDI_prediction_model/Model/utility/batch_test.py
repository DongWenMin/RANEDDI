import utility.metrics as metrics
from utility.parser import parse_args
import multiprocessing
import heapq
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn import ensemble

from utility.loader_raneddi import RANEDDI_loader

import threading
from threading import Lock,Thread
import time,os
import random as rd

cores = multiprocessing.cpu_count() // 2
args = parse_args()
Ks = eval(args.Ks)

data_generator = RANEDDI_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False

ITEM_NUM = data_generator.n_drugs
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

lock = Lock()
total_thread = 128

def ranklist_by_heapq(drug_pos_test, test_items, rating, Ks):#drug_pos_test:测试数据中的项，test_items:需要被测试的项，这是所有的项，除去训练数据中的项
    #rating：就是对应计算的指标
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in drug_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, drug_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in drug_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(drug_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in drug_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, drug_pos_test)#item_score:所有待测试样本的得分
    return r, auc


def get_performance(drug_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(drug_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def test_one_drug(x):
    # drug u's ratings for drug u
    rating = x[0]
    #uid
    u = x[1]
    #drug u's items in the training set
    try:
        training_items = data_generator.train_drug_dict[u]
    except Exception:
        training_items = []
    #drug u's items in the test set
    drug_pos_test = data_generator.test_drug_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(drug_pos_test, test_items, rating, Ks)#预测100个结果，并与测试数据比较，返回的r=1，即预测成功，0即预测失败。
    else:
        r, auc = ranklist_by_sorted(drug_pos_test, test_items, rating, Ks)

    return get_performance(drug_pos_test, r, auc, Ks)

def test_one_drug1(x):
    # drug u's ratings for drug u
    rating = x[0]
    #uid
    u = x[1]
    training_items = data_generator.train_drug_dict[u]

    drug_pos_test = data_generator.test_drug_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    # test_items = x[2]

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(drug_pos_test, test_items, rating, Ks)#预测100个结果，并与测试数据比较，返回的r=1，即预测成功，0即预测失败。
    else:
        r, auc = ranklist_by_sorted(drug_pos_test, test_items, rating, Ks)

    return get_performance(drug_pos_test, r, auc, Ks)
#平衡标签
# def get_label_score(adj_recovered,test_dict):
#     rd.seed(2020)
#     label_true, score_predict = [],[]
#     label_true1, score_predict1 = [],[]
#     for i in range(1710):
#         for j in range(1710):
#             if i in data_generator.train_drug_dict.keys() and j in data_generator.train_drug_dict[i]:
#                 continue
#             elif i not in test_dict or j not in test_dict[i]:
#                 label_true1.append(0)
#                 score_predict1.append(adj_recovered[i][j])
#             else:
#                 label_true.append(1)
#                 score_predict.append(adj_recovered[i][j])
#     rand_neg_index = rd.sample(range(0, len(label_true1)),len(label_true))
#     label_true1 = [label_true1[i] for i in rand_neg_index]
#     score_predict1 = [score_predict1[i] for i in rand_neg_index]
#     label_true += label_true1
#     score_predict += score_predict1
#     return np.array(label_true),np.array(score_predict)

def get_label_score(adj_recovered,test_dict,test_neg_dict):
    label_true, score_predict = [],[]
    for k,value in test_dict.items():
        for v in value:
            if k<v:
              label_true.append(1)
              score_predict.append(adj_recovered[k,v])

    for k,value in test_neg_dict.items():
        for v in value:
            if k<v:
              label_true.append(0)
              score_predict.append(adj_recovered[k,v])
    return np.array(label_true),np.array(score_predict)
#不平衡标签
def get_label_score1(adj_recovered,test_dict):
    label_true, score_predict = [],[]
    for i in range(1710):
        for j in range(1710):
            if i in data_generator.train_drug_dict.keys() and j in data_generator.train_drug_dict[i]:
                continue
            elif i not in test_dict or j not in test_dict[i]:
                label_true.append(0)
            else:
                label_true.append(1)
            score_predict.append(adj_recovered[i][j])
    return np.array(label_true),np.array(score_predict)


def test(sess, model, drugs_to_test, train_dict,test_dict,test_neg_dict,drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_drugs = drugs_to_test
    n_test_drugs = len(test_drugs)
    n_drug_batchs = n_test_drugs // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_drug_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        drug_batch = test_drugs[start: end]

        
        # item_batch = range(ITEM_NUM,2*ITEM_NUM)
        item_batch = range(ITEM_NUM)
        feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                            drug_batch=item_batch,
                                                            item_batch=item_batch,
                                                            drop_flag=drop_flag)
        _,_,rate_batch = model.eval(sess, feed_dict=feed_dict)
        rate_batch1 = rate_batch.reshape((-1, len(item_batch)))
        rate_batch = rate_batch1 + rate_batch1.T


        drug_batch_rating_uid = zip(rate_batch, item_batch,test_dict.keys())
        drug_batch_rating_uid1 = zip(rate_batch, item_batch)
                
        label, rating = get_label_score(rate_batch,test_dict,test_neg_dict)

        rating = np.array(rating)
        label = np.array(label)
        result_all = evaluate(rating, label, 2)
        print('banlance:',result_all)

        rating_sigmoid=1/(1+(np.exp((-rating))))
        threshold = 0.9
        pred_label = 1*(rating_sigmoid>threshold)
        f1_micro = f1_score(label,pred_label,average='micro')
        f1_macro = f1_score(label,pred_label,average='macro')
        print('micro f1:%f,macro f1:%f'%(f1_micro,f1_macro))

        label, rating = get_label_score1(rate_batch,test_dict)
        rating = np.array(rating)
        label = np.array(label)
        result_all1 = evaluate(rating, label, 2)
        print('no balance',result_all1)
          
        # batch_result = []
        # test_drug_count = 0
        # for item in zip(drug_batch_rating_uid):
        #     if item[0][1] not in data_generator.test_drug_dict.keys():
        #       continue
        #     else:
        #       batch_result.append(test_one_drug1(item[0]))
        #       test_drug_count += 1
        # # batch_result = pool.map(test_one_drug1,drug_batch_rating_uid1)
        # for re in batch_result:
        #     result['precision'] += re["precision"]/test_drug_count
        #     result['recall'] += re["recall"]/test_drug_count
        #     result['ndcg'] += re["ndcg"]/test_drug_count
        #     result['hit_ratio'] += re["hit_ratio"]/test_drug_count
        #     result['auc'] += re["auc"]/test_drug_count


        # #other metric
        # # batch_result = pool.map(test_one_drug1,drug_batch_rating_uid)
        # print('other metric:',result)
        # print('test over')
 

    # assert count == n_test_drugs
    pool.close()
    return result_all1['aupr']

def evaluate(pred_score, y_test, event_num):
    
    result_all = {}
    y_one_hot = label_binarize(y_test, np.arange(event_num))

    precision, recall, _ = precision_recall_curve(y_test, pred_score)
    result_all['aupr'] = auc(recall, precision)#aupr
    result_all['auc'] = roc_auc_score(y_one_hot, pred_score,average='micro')
    return result_all

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        order = np.lexsort((recall,precision))
        return auc(precision[order], recall[order])

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)

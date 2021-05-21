import utility.metrics as metrics
from utility.parser import parse_args
import multiprocessing
import heapq
import numpy as np
import pandas as pd
from scipy import interp

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from utility.loader_raneddi import RANEDDI_loader

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = RANEDDI_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False


ITEM_NUM = data_generator.n_drugs
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

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
    auc = get_auc(item_score, drug_pos_test)
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

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def caculate_auc(length,test_items,drug_pos_test,rating):
    #首先得到真实标签
    label = np.zeros(length,dtype=int)
    label[drug_pos_test] = 1
    label = label[test_items]
    rating = sigmoid(rating[test_items])
    return roc_auc_score(label, rating)

def caculate_AUC_AUPR(length,test_items,drug_pos_test,rating):
    #首先得到真实标签
    label = np.zeros(length,dtype=int)
    label[drug_pos_test] = 1
    label = label[test_items]
    rating = rating[test_items]
    precision, recall, _thresholds = precision_recall_curve(label, rating)
    area = auc(recall, precision)
    return roc_auc_score(label, rating),area

def get_x_auc(x,drug_pos_test):
    # drug u's ratings for drug u
    rating = x[0]
    #uid
    u = x[1]
    #drug u's items in the training set
    try:
        training_items = data_generator.train_dict[u]
    except Exception:
        training_items = []
    #drug u's items in the test set
    # drug_pos_test = data_generator.test_drug_dict[u]
    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))
    return caculate_AUC_AUPR(len(all_items),test_items,drug_pos_test,rating)

def get_label_and_pro(x,drug_pos_test):
    rating = x[0]
    #uid
    u = x[1]
    #drug u's items in the training set
    try:
        training_items = data_generator.train_dict[u]
    except Exception:
        training_items = []
    #drug u's items in the test set
    # drug_pos_test = data_generator.test_drug_dict[u]
    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))
    length = len(all_items)
    label = np.zeros(length,dtype=int)
    label[drug_pos_test] = 1
    label = label[test_items]
    rating = rating[test_items]
    return label,rating

def get_label_and_pro1(x,drug_pos_test,train_drugs):
    rating = x[0]
    #uid
    u = x[1]
    #drug u's items in the training set
    try:
        training_items = list(train_drugs)
    except Exception:
        training_items = []
    #drug u's items in the test set
    # drug_pos_test = data_generator.test_drug_dict[u]
    all_items = set(range(ITEM_NUM))
    test_items = list(set(training_items))
    length = len(all_items)
    label = np.zeros(length,dtype=int)
    label[drug_pos_test] = 1
    label = label[test_items]
    rating = rating[test_items]
    return label,rating

def get_label_and_pro2(x,drug_pos_test):
    rating = x[0]
    all_items = set(range(1317))
#     test_items = list(all_items - test_drug_dict.keys())
    test_items = x[2]
    length = len(all_items)
#     print(length)
    label = np.zeros(length,dtype=int)
    label[drug_pos_test] = 1
    label = label[test_items]
    rating = rating[test_items]
    return label,rating

def get_test_dict_convert(test_drug_dict):
    test_dict_convert = {}
    for k,v in test_drug_dict.items():
        for i in v:
            if i in test_dict_convert and k not in test_dict_convert[i]:
                test_dict_convert[i].append(k)
            if i not in test_dict_convert:
                test_dict_convert[i] = [k]
    return test_dict_convert


def test_one_drug(x):
    # drug u's ratings for drug u
    rating = x[0]
    #uid
    u = x[1]
    #drug u's items in the training set
    try:
        training_items = data_generator.train_dict[u]
    except Exception:
        training_items = []
    #drug u's items in the test set
    drug_pos_test = data_generator.test_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(drug_pos_test, test_items, rating, Ks)#预测100个结果，并与测试数据比较，返回的r=1，即预测成功，0即预测失败。
    else:
        r, auc = ranklist_by_sorted(drug_pos_test, test_items, rating, Ks)

    # # .......checking.......
    # try:
    #     assert len(drug_pos_test) != 0
    # except Exception:
    #     print(u)
    #     print(training_items)
    #     print(drug_pos_test)
    #     exit()
    # # .......checking.......

    return get_performance(drug_pos_test, r, auc, Ks)

def test_one_drug1(x):
    # drug u's ratings for drug u
    rating = x[0]
    #uid
    u = x[1]
    #drug u's items in the training set
    # try:
    #     training_items = test_drugs
    # except Exception:
    #     training_items = []
    #drug u's items in the test set
    drug_pos_test = data_generator.train_dict[u]

    # all_items = set(range(ITEM_NUM))

    test_items = x[2]

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(drug_pos_test, test_items, rating, Ks)#预测100个结果，并与测试数据比较，返回的r=1，即预测成功，0即预测失败。
    else:
        r, auc = ranklist_by_sorted(drug_pos_test, test_items, rating, Ks)


    return get_performance(drug_pos_test, r, auc, Ks)

def test3(sess, model, drugs_to_test, train_dict,test_dict,test_feed,drop_flag=False, batch_test_flag=False):
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

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(drug_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                                   drug_batch=drug_batch,
                                                                   item_batch=item_batch,
                                                                   drop_flag=drop_flag)
                i_rate_batch = model.eval(sess, feed_dict=feed_dict)
                i_rate_batch = i_rate_batch.reshape((-1, len(item_batch)))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # item_batch = range(ITEM_NUM,2*ITEM_NUM)
            item_batch = range(ITEM_NUM)
            feed_dict1 = data_generator.generate_test_feed_dict(model=model,
                                                               drug_batch=drug_batch,
                                                               item_batch=item_batch,
                                                               test_feed = test_feed,
                                                               drop_flag=drop_flag)
            rate_batch1 = model.eval(sess, feed_dict=feed_dict1)

            feed_dict2 = data_generator.generate_test_feed_dict(model=model,
                                                               drug_batch=drug_batch,
                                                               item_batch=item_batch,
                                                               test_feed = test_feed[-1::-1],
                                                               drop_flag=drop_flag)
            rate_batch2 = model.eval(sess, feed_dict=feed_dict2)
            rate_batch = rate_batch1 + rate_batch2
            
            rate_batch_ravel = rate_batch.ravel('C')

        #获取label
        label = []
        for i in range(len(test_feed[0])):
            label.append(data_generator.adj_multi[test_feed[0][i]][test_feed[1][i]]-1)
        label_one_hot = np.eye(data_generator.type_num)[label].ravel('C')
        # precision, recall, _thresholds = precision_recall_curve(label_one_hot, rate_batch_ravel)
        # my_aupr = auc(recall, precision)#aupr
        pred_type = np.argmax(rate_batch, axis=1)
        result_all,result_eve = evaluate(pred_type, rate_batch, label, data_generator.type_num)

        print("result all:",result_all)
        # print("result_eve:",result_eve)
        # np.savetxt("10cv_result.txt",np.array(list(result_all.values())),fmt="%.4f",delimiter="\t")
        # np.savetxt("10cv_eachresult.txt",result_eve,fmt="%.4f",delimiter="\t")
        my_aupr = roc_aupr_score(np.eye(data_generator.type_num)[label], rate_batch, average='micro')
        my_auc = roc_auc_score(label_one_hot, rate_batch_ravel)#auc
        result['my_auc'] = my_auc
        result['my_aupr'] = my_aupr

    pool.close()
    return result

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        order = np.lexsort((recall, precision))
        precision1,recall1 = precision[order], recall[order]
        return auc(precision1, recall1)

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

def evaluate(pred_type, pred_score, y_test, event_num):
    # all_eval_type = 6
    result_all = {}
    each_eval_type = 6
    result_eve = np.zeros((event_num, 6), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, np.arange(event_num))
    result_all['accuracy'] = accuracy_score(y_test, pred_type)
    result_all['aupr'] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all['auc'] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all['f1'] = f1_score(y_test, pred_type, average='macro')
    result_all['precision'] = precision_score(y_test, pred_type, average='macro')
    result_all['recall'] = recall_score(y_test, pred_type, average='macro')

    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        try:
            result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        except Exception as ex:
            result_eve[i, 2] =-1
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return result_all,result_eve
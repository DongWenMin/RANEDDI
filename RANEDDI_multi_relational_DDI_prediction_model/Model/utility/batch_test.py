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

data_generator = RANEDDI_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False


ITEM_NUM = data_generator.n_drugs
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def test3(sess, model, drugs_to_test, train_dict,test_dict,test_feed,drop_flag=False, batch_test_flag=False):

    pool = multiprocessing.Pool(cores)

    batch_size = BATCH_SIZE * 2

    test_drugs = drugs_to_test
    n_test_drugs = len(test_drugs)
    n_drug_batchs = n_test_drugs // batch_size + 1


    for u_batch_id in range(n_drug_batchs):
        start = u_batch_id * batch_size
        end = (u_batch_id + 1) * batch_size

        drug_batch = test_drugs[start: end]

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

    pool.close()
    return result_all

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
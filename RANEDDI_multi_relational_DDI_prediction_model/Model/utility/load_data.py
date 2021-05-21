import collections
import numpy as np
import random as rd
from collections import defaultdict

class Data(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/10_fold/train0.txt'
        test_file = path + '/10_fold/test0.txt'

        npz_file = path +'/deepddi.npz'
        dataset = np.load(npz_file,allow_pickle=True)
        dataset = dataset['data'].item()

        self.adj_multi = dataset['Adj_multi'].astype(np.int32)
        self.type_num = np.max(self.adj_multi)
        # self.df_mat = dataset['interaction_feature']#1316:638
        self.n_train, self.n_test = 0, 0
        self.n_drugs, self.n_items = 0, 0

        self.train_data, self.train_dict = self._load_ratings(train_file)
        self.test_data, self.test_dict = self._load_ratings(test_file)
        self.exist_drugs = self.train_dict.keys()
        # self.train_dict,self.test_dict = self.get_train_test_dict(train_file,test_file)

        self._statistic_ratings()

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.n_entities = 0
        self.n_relations = 1
        self.n_entities =self.adj_multi.shape[0]
        # ----------print the basic info about the dataset-------------.
        self._print_data_info()
        self.test_feed = self.get_test_feed()

    def get_test_feed(self):
        drug1 = []
        drug2 = []
        for k,v in self.test_dict.items():
            for value in v:
                if k < value:
                    drug1.append(k)
                    drug2.append(value)
        return [np.array(drug1),np.array(drug2)]


    def get_train_test_dict(self,train_file,test_file):
        train_dict,test_dict = defaultdict(list),defaultdict(list)
        with open(train_file,'r') as f1,open(test_file,'r') as f2:
            line1 = f1.readline()
            while line1:
                line1 = list(map(int,line1.split(' ')[:-1]))
                train_dict[line1[0]] = line1[1:]
                line1 = f1.readline()
            line2 = f2.readline()
            while line2:
                line2 = list(map(int,line2.split(' ')[:-1]))
                test_dict[line2[0]] = line2[1:]
                line2 = f2.readline()
        return train_dict,test_dict

    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        drug_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                drug_dict[u_id] = pos_ids
        return np.array(inter_mat), drug_dict

    def _statistic_ratings(self):
        self.n_drugs = max(max(self.train_data.flatten()), max(self.test_data.flatten())) + 1
        self.n_items = self.n_drugs
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    def _print_data_info(self):
        print('[n_drug]=[%d]' % (self.n_drugs))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations]=[%d, %d]' % (self.n_entities, self.n_relations))
        print('[batch_size]=[%d]' % (self.batch_size))

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_drugs:
            drug = rd.sample(self.exist_drugs, self.batch_size)
        else:
            drug = [rd.choice(list(self.exist_drugs)) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_dict[u]
            n_pos_items = len(pos_items)
            # pos_batch = []
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            return [pos_items[pos_id]]

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]#随机采取负样本
                if neg_i_id not in self.train_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
                    break
            return neg_items

        pos_items, neg_items, relations= [],[],[]
        pos_tails, neg_tails = [],[]
        type_list = []
        drug_small = []
        for u in drug:
            pos_item = sample_pos_items_for_u(u, 1)
            type_list.append(self.adj_multi[u][pos_item]-1)
            drug_small.append(u)
            pos_items += pos_item
            neg_items += sample_neg_items_for_u(u, 1)
            relations.append(self.adj_multi[u,pos_items[-1]]-1)
            pos_tails.append(pos_items[-1] + self.n_drugs)
            neg_tails.append(neg_items[-1] + self.n_drugs)


        multi_type_pos = np.squeeze(np.eye(self.type_num)[type_list])

        return drug_small, pos_items, neg_items,multi_type_pos,relations,pos_tails, neg_tails
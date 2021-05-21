import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class RANEDDI(object):
    def __init__(self, data_config,args):
        self._parse_args(data_config, args)
        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model_phase_I()
        self._create_DNN_prediction()
        self._build_model_phase_II()
        self._build_loss_phase_II()
        self._build_total_loss()
        self._statistics_params()

    def _parse_args(self, data_config, args):
        # argument settings
        self.model_type = 'raneddi'

        self.n_drug1 = data_config['n_drugs']
        # self.n_items = data_config['n_items']
        self.n_drug2 = self.n_drug1
        self.n_relations = int(data_config['n_relations'])

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']

        # self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.dim = args.kge_size
        # self.batch_size_kg = args.batch_size_kg

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.verbose = args.verbose
        self.margin = args.margin

        #分类器
        #三层
        self.n_input1 = 400
        self.n_hidden1 = 200
        self.n_hidden2 = 100
        self.n_classes = 86

        self.all_sparse_adj_list = data_config['sparse_adj_list']#1317*1317
        self.sparse_adj_list = self._sparse_adj_list_process()

    def _sparse_adj_list_process(self):
        sparse_adj_list = []
        for adj in self.all_sparse_adj_list:
            convert_adj = self._convert_sp_mat_to_sp_tensor(adj)#转tensor
            sparse_adj_list.append(convert_adj)
        return sparse_adj_list

    def _build_inputs(self):
        # placeholder definition
        self.drugs = tf.placeholder(tf.int32, shape=(None,))
        self.pos_drugs = tf.placeholder(tf.int32, shape=(None,))
        self.neg_drugs = tf.placeholder(tf.int32, shape=(None,))
        # self.all_pro_user = tf.placeholder(tf.int32, shape=(None,))

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        # message dropout (adopted on the convolution operations).
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        self.type = tf.placeholder(tf.int32, shape=(None,self.n_classes))#one-hot

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['re_drug1_embed'] = tf.Variable(initializer([self.n_drug1, self.emb_dim]), name='re_drug1_embed')
        all_weights['re_drug2_embed'] = tf.Variable(initializer([self.n_drug2, self.emb_dim]), name='re_drug2_embed')
        all_weights['im_drug1_embed'] = tf.Variable(initializer([self.n_drug1, self.emb_dim]), name='im_drug1_embed')
        all_weights['im_drug2_embed'] = tf.Variable(initializer([self.n_drug2, self.emb_dim]), name='im_drug2_embed')
        print('using xavier initialization')

        all_weights['re_relation_embed'] = tf.Variable(initializer([self.n_relations, self.dim]),
                                                    name='re_relation_embed')
        all_weights['im_relation_embed'] = tf.Variable(initializer([self.n_relations, self.dim]),
                                                    name='im_relation_embed')
        all_weights['relation_mapping'] = tf.Variable(initializer([self.n_relations,2*self.dim]),
                                                    name='relation_mapping')
        all_weights['relation_matrix'] = tf.Variable(initializer([30,2*self.dim,2*self.dim]),
                                                    name='relation_matrix')
        relation_initial = np.random.randn(self.n_relations, 30).astype(np.float32)

        all_weights['alpha'] = tf.Variable(relation_initial,name = 'arb')
        all_weights['wh'] = tf.Variable(initializer([2*self.dim,1]),
                                                    name='wh')
        all_weights['wt'] = tf.Variable(initializer([2*self.dim,1]),
                                                    name='wt')
        all_weights['drug1_trans'] = tf.Variable(initializer([2*self.dim, 2*self.dim]),
                                                    name='drug1_trans')
        all_weights['drug2_trans'] = tf.Variable(initializer([2*self.dim, 2*self.dim]),
                                                    name='drug2_trans')
        relation_1_att = np.random.randn(self.n_relations,self.n_drug1,1).astype(np.float32)
        relation_2_att = np.random.randn(self.n_relations,self.n_drug2,1).astype(np.float32)
        all_weights['relation_1_att'] = tf.Variable(relation_1_att,name='relation_1_att')
        all_weights['relation_2_att'] = tf.Variable(relation_2_att,name='relation_2_att')

        self.weight_size_list = [self.emb_dim*2] + self.weight_size

        all_weights['W_mlp_0'] = tf.Variable(
            initializer([2 * self.weight_size_list[0], self.weight_size_list[1]]), name='W_mlp_0')
        all_weights['b_mlp_0'] = tf.Variable(
            initializer([1, self.weight_size_list[1]]), name='b_mlp_0')

        all_weights['nn_w1'] = tf.Variable(initializer([self.n_input1, self.n_hidden1]), name='nn_w1')
        all_weights['nn_b1'] = tf.Variable(initializer([1, self.n_hidden1]), name='nn_b1')
        all_weights['nn_w2'] = tf.Variable(initializer([self.n_hidden1, self.n_hidden2]), name='nn_w2')
        all_weights['nn_b2'] = tf.Variable(initializer([1, self.n_hidden2]), name='nn_b2')
        all_weights['nn_w3'] = tf.Variable(initializer([self.n_hidden2, self.n_classes]), name='nn_w3')
        all_weights['nn_b3'] = tf.Variable(initializer([1, self.n_classes]), name='nn_b3')

        return all_weights

    def _build_model_phase_I(self):
        self.d1_embeddings, self.d2_embeddings = self._create_bi_interaction_embed()

        self.d1_e = tf.nn.embedding_lookup(self.d1_embeddings, self.drugs)
        self.pos_d2_e = tf.nn.embedding_lookup(self.d2_embeddings, self.pos_drugs)
        self.neg_d2_e = tf.nn.embedding_lookup(self.d2_embeddings, self.neg_drugs)

    def _build_model_phase_II(self):
        self.re_h_e, self.re_pos_t_e, self.re_neg_t_e, self.im_h_e,self.im_pos_t_e, self.im_neg_t_e, self.re_r_e, self.im_r_e = self._get_kg_inference_rotate(self.h, self.r, self.pos_t, self.neg_t)
        
    def _get_kg_inference_rotate(self, h, r, pos_t, neg_t):
        pi = 3.14159265358979323846
        re_embeddings = tf.concat([self.weights['re_drug1_embed'], self.weights['re_drug2_embed']], axis=0)
        re_embeddings = tf.expand_dims(re_embeddings, 1)
        im_embeddings = tf.concat([self.weights['im_drug1_embed'], self.weights['im_drug2_embed']], axis=0)
        im_embeddings = tf.expand_dims(im_embeddings, 1)

        re_h_e = tf.nn.embedding_lookup(re_embeddings, h)
        re_pos_t_e = tf.nn.embedding_lookup(re_embeddings, pos_t)
        re_neg_t_e = tf.nn.embedding_lookup(re_embeddings, neg_t)
        im_h_e = tf.nn.embedding_lookup(im_embeddings, h)
        im_pos_t_e = tf.nn.embedding_lookup(im_embeddings, pos_t)
        im_neg_t_e = tf.nn.embedding_lookup(im_embeddings, neg_t)

        re_h_e = tf.reshape(re_h_e, [-1, self.dim])
        re_pos_t_e = tf.reshape(re_pos_t_e, [-1, self.dim])
        re_neg_t_e = tf.reshape(re_neg_t_e, [-1, self.dim])
        im_h_e = tf.reshape(im_h_e, [-1, self.dim])
        im_pos_t_e = tf.reshape(im_pos_t_e, [-1, self.dim])
        im_neg_t_e = tf.reshape(im_neg_t_e, [-1, self.dim])

        relation = self.weights['re_relation_embed']
        relation = (tf.nn.l2_normalize(relation, dim=1)-0.5)*pi

        re_r_e = tf.nn.embedding_lookup(relation, r)
        re_r_e = tf.cos(re_r_e)
        im_r_e = tf.sin(re_r_e)

        return re_h_e, re_pos_t_e, re_neg_t_e, im_h_e, im_pos_t_e, im_neg_t_e, re_r_e, im_r_e

    def _build_loss_phase_II(self):
        def _get_kg_score(re_h_e,re_pos_t_e,im_h_e,im_pos_t_e, re_r_e, im_r_e):
            re_score = tf.multiply(re_h_e,re_r_e) - tf.multiply(im_h_e,im_r_e)
            im_score = tf.multiply(re_h_e,im_r_e) + tf.multiply(im_h_e,re_r_e)

            re_score = re_score - re_pos_t_e
            im_score = im_score - im_pos_t_e
            kg_score = tf.concat([re_score,im_score], axis=1)
            kg_score = tf.reduce_sum(tf.square((kg_score)), 1, keep_dims=True)
            kg_score = tf.negative(kg_score)

            return kg_score

        pos_kg_score = _get_kg_score(self.re_h_e, self.re_pos_t_e,  self.im_h_e,self.im_pos_t_e,  self.re_r_e, self.im_r_e )
        neg_kg_score = _get_kg_score(self.re_h_e, self.re_neg_t_e,  self.im_h_e,self.im_neg_t_e,  self.re_r_e, self.im_r_e )
        #损失1 88.73
        # margin = 2.0
        maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_kg_score - neg_kg_score - self.margin),1e-8,1.0))
        kg_loss = tf.negative(tf.reduce_mean(maxi))

        kg_reg_loss = tf.nn.l2_loss(self.re_h_e) + tf.nn.l2_loss(self.re_pos_t_e) + \
                      tf.nn.l2_loss(self.im_h_e) + tf.nn.l2_loss(self.im_pos_t_e) + \
                      tf.nn.l2_loss(self.re_r_e) + tf.nn.l2_loss(self.im_r_e) + \
                      tf.nn.l2_loss(self.re_neg_t_e) + tf.nn.l2_loss(self.im_neg_t_e)
        kg_reg_loss = kg_reg_loss / self.batch_size

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.regs[1] * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # Optimization process.
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)
    
    def _build_total_loss(self):
        self.total_loss = self.loss + self.loss2
        self.total_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

    def _create_DNN_prediction(self):
        regularizer = tf.nn.l2_loss(self.d1_e) + tf.nn.l2_loss(self.pos_d2_e)
        regularizer = regularizer / self.batch_size

        # pred_embeddings = tf.concat([self.d1_e,self.pos_d2_e], 1)
        pred_embeddings = tf.concat([self.d1_e,self.pos_d2_e],1)

        #三层nn
        pred_embeddings = tf.nn.relu(tf.matmul(pred_embeddings, self.weights['nn_w1']) + self.weights['nn_b1'])
        pred_embeddings = tf.nn.dropout(pred_embeddings, 1 - self.mess_dropout[0])
        pred_embeddings = tf.nn.l2_normalize(pred_embeddings, dim=1)

        pred_embeddings = tf.nn.relu(tf.matmul(pred_embeddings, self.weights['nn_w2']) + self.weights['nn_b2'])
        pred_embeddings = tf.nn.dropout(pred_embeddings, 1 - self.mess_dropout[0])
        pred_embeddings = tf.nn.l2_normalize(pred_embeddings, dim=1)

        pred_embeddings = tf.matmul(pred_embeddings, self.weights['nn_w3']) + self.weights['nn_b3']

        self.pred_embeddings = pred_embeddings
        self.prediction_type = tf.nn.softmax(self.pred_embeddings)
        
        base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_embeddings, labels=self.type))
        self.base_loss = base_loss
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = self.regs[0] * regularizer
        self.loss = self.base_loss + self.kge_loss + self.reg_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _create_bi_interaction_embed5(self):
        d1_embedding = tf.concat([self.weights['re_drug1_embed'],self.weights['im_drug1_embed']], axis=1)#1317*200
        d2_embedding = tf.concat([self.weights['re_drug2_embed'],self.weights['im_drug2_embed']], axis=1)#1952*200

        ego_embeddings = tf.concat([d1_embedding, d2_embedding], axis=0)
        all_embeddings = [ego_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        d1_embeddings, d2_embeddings = tf.split(all_embeddings, [self.n_drug1, self.n_drug2], 0)
        return d1_embeddings, d2_embeddings

    def _create_bi_interaction_embed(self):
        pi = 3.14159265358979323846
        d1_embedding = tf.concat([self.weights['re_drug1_embed'],self.weights['im_drug1_embed']], axis=1)#1317*200
        d2_embedding = tf.concat([self.weights['re_drug2_embed'],self.weights['im_drug2_embed']], axis=1)#1952*200

        #使用R-GCN方式定义关系矩阵
        relation_embedding = []
        for i in range(self.n_relations):
          weights = tf.reshape(self.weights['alpha'][i],[-1,1,1])
          relation_matrix_temp = self.weights['relation_matrix'] * weights
          relation_matrix_temp = tf.reduce_sum(relation_matrix_temp,axis=0)
          relation_embedding.append(relation_matrix_temp)
       
        d1_neigh,d2_neigh = [],[]
        for i in range(self.n_relations):
          # print(i)
          r_d2_embedding = d2_embedding @ relation_embedding[i]
          weight_d2_embedding = r_d2_embedding * self.weights['relation_2_att'][i]
          weight_d2_embedding = weight_d2_embedding + d2_embedding
          # weight_d2_embedding = d2_embedding
          relation_1_neigh = tf.sparse_tensor_dense_matmul(self.sparse_adj_list[i],weight_d2_embedding)

          r_d1_embedding = d1_embedding @ relation_embedding[i]
          weight_d1_embedding = r_d1_embedding * self.weights['relation_1_att'][i]
          weight_d1_embedding = weight_d1_embedding + d1_embedding
          # weight_d1_embedding =  d1_embedding
          relation_2_neigh = tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.sparse_adj_list[i]),weight_d1_embedding)

          d1_neigh.append(relation_1_neigh)
          d2_neigh.append(relation_2_neigh)
          
        d1_neigh = tf.reduce_sum(d1_neigh,0)
        d2_neigh = tf.reduce_sum(d2_neigh,0)
        neigh_embed = tf.concat([d1_neigh, d2_neigh], axis=0)

        ego_embeddings = tf.concat([d1_embedding, d2_embedding], axis=0)
        all_embeddings = []

        # side_embeddings = tf.nn.l2_normalize(neigh_embed, dim=1)
        side_embeddings = neigh_embed

        side_embeddings = tf.concat([ego_embeddings, side_embeddings], 1)
        pre_embeddings = tf.nn.relu(
            tf.matmul(side_embeddings, self.weights['W_mlp_0']) + self.weights['b_mlp_0'])

        pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[0])
        all_embeddings += [pre_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        d1_embeddings, d2_embeddings = tf.split(all_embeddings, [self.n_drug1, self.n_drug2], 0)
        return d1_embeddings, d2_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        if len(coo.data) == 0:
            return tf.SparseTensor([[1,2]], [0.], coo.shape)#生成空的稀疏矩阵
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([self.total_opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss,\
                        self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.prediction_type, feed_dict)
        return batch_predictions

import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class RANEDDI(object):
    def __init__(self, data_config, args):
        self._parse_args(data_config, args)
        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model_phase_I()
        self._build_loss_phase_I()
        self._build_model_phase_II()
        self._build_loss_phase_II()
        self._build_total_loss()
        self._statistics_params()

    def _parse_args(self, data_config, args):
        # argument settings
        self.model_type = 'raneddi'

        self.n_drugs = data_config['n_drugs']
        # self.n_items = data_config['n_items']
        self.n_entities = data_config['n_drugs']
        self.n_relations = data_config['n_relations']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size
        self.batch_size_kg = args.batch_size_kg

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.verbose = args.verbose
        self.margin = args.margin
        self.B = args.B

        #这里更新一下sparse_adj_list
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

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')
        # self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['re_drug_embed'] = tf.Variable(initializer([self.n_drugs, self.emb_dim]), name='re_drug_embed')
        all_weights['re_entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]), name='re_entity_embed')
        all_weights['im_drug_embed'] = tf.Variable(initializer([self.n_drugs, self.emb_dim]), name='im_drug_embed')
        all_weights['im_entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]), name='im_entity_embed')
        print('using xavier initialization')

        all_weights['re_relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                    name='re_relation_embed')
        all_weights['im_relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                    name='im_relation_embed')
        all_weights['relation_mapping'] = tf.Variable(initializer([self.n_relations,2*self.kge_dim]),
                                                    name='relation_mapping')
        all_weights['relation_matrix'] = tf.Variable(initializer([self.B,2*self.kge_dim,2*self.kge_dim]),
                                                    name='relation_matrix')
        relation_initial = np.random.randn(self.n_relations, self.B).astype(np.float32)

        all_weights['alpha'] = tf.Variable(relation_initial,name = 'arb')

        # all_weights['relation_trans'] = tf.Variable(initializer([2*self.kge_dim, 2*self.kge_dim]),
        #                                             name='relation_trans')
        all_weights['wh'] = tf.Variable(initializer([2*self.kge_dim,1]),
                                                    name='wh')
        all_weights['wt'] = tf.Variable(initializer([2*self.kge_dim,1]),
                                                    name='wt')
        all_weights['drug_trans'] = tf.Variable(initializer([2*self.kge_dim, 2*self.kge_dim]),
                                                    name='drug_trans')
        all_weights['entity_trans'] = tf.Variable(initializer([2*self.kge_dim, 2*self.kge_dim]),
                                                    name='entity_trans')
        relation_d_att = np.random.randn(self.n_relations,self.n_drugs,1).astype(np.float32)
        relation_e_att = np.random.randn(self.n_relations,self.n_entities,1).astype(np.float32)
        all_weights['relation_d_att'] = tf.Variable(relation_d_att,name='relation_d_att')
        all_weights['relation_e_att'] = tf.Variable(relation_e_att,name='relation_e_att')

        self.weight_size_list = [self.emb_dim*2] + self.weight_size
        all_weights['W_mlp_0'] = tf.Variable(
            initializer([2 * self.weight_size_list[0], self.weight_size_list[1]]), name='W_mlp_0')
        all_weights['b_mlp_0'] = tf.Variable(
            initializer([1, self.weight_size_list[1]]), name='b_mlp_0')

        return all_weights


    def _build_model_phase_I(self):
        self.da_embeddings, self.ea_embeddings = self._create_bi_interaction_embed()

        self.d_e = tf.nn.embedding_lookup(self.da_embeddings, self.drugs)
        self.pos_e = tf.nn.embedding_lookup(self.ea_embeddings, self.pos_drugs)
        self.neg_e = tf.nn.embedding_lookup(self.ea_embeddings, self.neg_drugs)

        #所有pro drug的嵌入表示
        # self.all_pro_drug_embdding = tf.nn.embedding_lookup(self.ea_embeddings, self.all_pro_drug)

        self.batch_predictions = tf.matmul(self.d_e, self.pos_e, transpose_a=False, transpose_b=True)

    def _build_model_phase_II(self):
        self.re_h_e, self.re_pos_t_e, self.re_neg_t_e, self.im_h_e,self.im_pos_t_e, self.im_neg_t_e, self.re_r_e, self.im_r_e = self._get_kg_inference_rotate(self.h, self.r, self.pos_t, self.neg_t)

    def _get_kg_inference_rotate(self, h, r, pos_t, neg_t):
        pi = 3.14159265358979323846
        re_embeddings = tf.concat([self.weights['re_drug_embed'], self.weights['re_entity_embed']], axis=0)
        re_embeddings = tf.expand_dims(re_embeddings, 1)
        im_embeddings = tf.concat([self.weights['im_drug_embed'], self.weights['im_entity_embed']], axis=0)
        im_embeddings = tf.expand_dims(im_embeddings, 1)

        re_h_e = tf.nn.embedding_lookup(re_embeddings, h)
        re_pos_t_e = tf.nn.embedding_lookup(re_embeddings, pos_t)
        re_neg_t_e = tf.nn.embedding_lookup(re_embeddings, neg_t)
        im_h_e = tf.nn.embedding_lookup(im_embeddings, h)
        im_pos_t_e = tf.nn.embedding_lookup(im_embeddings, pos_t)
        im_neg_t_e = tf.nn.embedding_lookup(im_embeddings, neg_t)

        re_h_e = tf.reshape(re_h_e, [-1, self.kge_dim])
        re_pos_t_e = tf.reshape(re_pos_t_e, [-1, self.kge_dim])
        re_neg_t_e = tf.reshape(re_neg_t_e, [-1, self.kge_dim])
        im_h_e = tf.reshape(im_h_e, [-1, self.kge_dim])
        im_pos_t_e = tf.reshape(im_pos_t_e, [-1, self.kge_dim])
        im_neg_t_e = tf.reshape(im_neg_t_e, [-1, self.kge_dim])

        relation = self.weights['re_relation_embed']
        relation = (tf.nn.l2_normalize(relation, dim=1)-0.5)*pi

        re_r_e = tf.nn.embedding_lookup(relation, r)
        re_r_e = tf.cos(re_r_e)
        im_r_e = tf.sin(re_r_e)

        return re_h_e, re_pos_t_e, re_neg_t_e, im_h_e, im_pos_t_e, im_neg_t_e, re_r_e, im_r_e


    def _build_loss_phase_I(self):
        pos_scores = tf.reduce_sum(tf.multiply(self.d_e, self.pos_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.d_e, self.neg_e), axis=1)

        regularizer = tf.nn.l2_loss(self.d_e) + tf.nn.l2_loss(self.pos_e) + tf.nn.l2_loss(self.neg_e) + \
                tf.nn.l2_loss(self.weights['relation_matrix']) +tf.nn.l2_loss(self.weights['wh']) + \
                tf.nn.l2_loss(self.weights['wt'])
        regularizer = regularizer / self.batch_size

        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores -self.margin))
        #该部分损失进行修改
        margin = 1
        maxi = tf.math.log(tf.clip_by_value(tf.nn.sigmoid(margin-neg_scores),1e-8,1.0)) + tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores-margin),1e-8,1.0))
        base_loss = tf.negative(tf.reduce_mean(maxi))

        self.base_loss = base_loss
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = self.regs[0] * regularizer
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # Optimization process.RMSPropOptimizer
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _build_loss_phase_II(self):
        # def _get_kg_score(h_e, r_e, t_e):
        #     kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keep_dims=True)
        #     return kg_score
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
        # self.prediction = tf.negative(_get_kg_score(self.h_e, self.r_e, self.pos_t_e))
        neg_kg_score = _get_kg_score(self.re_h_e, self.re_neg_t_e,  self.im_h_e,self.im_neg_t_e,  self.re_r_e, self.im_r_e )

        maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_kg_score - neg_kg_score - self.margin),1e-8,1.0))
        kg_loss = tf.negative(tf.reduce_mean(maxi))
        
        #loss2 ：the rank-based hinge loss
        # maxi = tf.maximum(0.,neg_kg_score + self.margin - pos_kg_score)
        # kg_loss = tf.reduce_mean(maxi)

        #loss3 
        # maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(self.margin-neg_kg_score),1e-8,1.0)) + tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_kg_score-self.margin),1e-8,1.0))
        # kg_loss = tf.negative(tf.reduce_mean(maxi))

        #loss4 ：the logistic-based loss
        # maxi = tf.log(1+tf.exp(tf.negative(pos_kg_score))) + tf.log(1+tf.exp(pos_kg_score))
        # kg_loss = tf.reduce_mean(maxi)

        kg_reg_loss = tf.nn.l2_loss(self.re_h_e) + tf.nn.l2_loss(self.re_pos_t_e) + \
                      tf.nn.l2_loss(self.im_h_e) + tf.nn.l2_loss(self.im_pos_t_e) + \
                      tf.nn.l2_loss(self.re_r_e) + tf.nn.l2_loss(self.im_r_e) + \
                      tf.nn.l2_loss(self.re_neg_t_e) + tf.nn.l2_loss(self.im_neg_t_e)
        kg_reg_loss = kg_reg_loss / self.batch_size

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.regs[1] * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # Optimization process.
        self.opt2 = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def _build_total_loss(self):
        self.total_loss = self.loss + self.loss2
        # self.total_loss = self.loss
        self.total_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

    def _create_bi_interaction_embed5(self):
        drug_embedding = tf.concat([self.weights['re_drug_embed'],self.weights['im_drug_embed']], axis=1)#1317*200
        entity_embedding = tf.concat([self.weights['re_entity_embed'],self.weights['im_entity_embed']], axis=1)#1952*200

        ego_embeddings = tf.concat([drug_embedding, entity_embedding], axis=0)
        all_embeddings = [ego_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        da_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_drugs, self.n_entities], 0)
        return da_embeddings, ea_embeddings

    def _create_bi_interaction_embed(self):
        pi = 3.14159265358979323846
        drug_embedding = tf.concat([self.weights['re_drug_embed'],self.weights['im_drug_embed']], axis=1)#1317*200
        entity_embedding = tf.concat([self.weights['re_entity_embed'],self.weights['im_entity_embed']], axis=1)#1952*200

        #使用R-GCN方式定义关系矩阵
        relation_embedding = []
        for i in range(self.n_relations):
          weights = tf.reshape(self.weights['alpha'][i],[-1,1,1])
          relation_matrix_temp = self.weights['relation_matrix'] * weights
          relation_matrix_temp = tf.reduce_sum(relation_matrix_temp,axis=0)
          relation_embedding.append(relation_matrix_temp)
       
        drug_neigh,entity_neigh = [],[]
        for i in range(self.n_relations):
          # print(i)
          r_entity_embedding = entity_embedding @ relation_embedding[i]
          weight_entity_embedding = r_entity_embedding * self.weights['relation_e_att'][i]
          weight_entity_embedding = weight_entity_embedding + entity_embedding
          # weight_entity_embedding = entity_embedding
          # weight_entity_embedding = weight_entity_embedding
          relation_u_neigh = tf.sparse.sparse_dense_matmul(self.sparse_adj_list[i],weight_entity_embedding)

          r_drug_embedding = drug_embedding @ relation_embedding[i]
          weight_drug_embedding = r_drug_embedding * self.weights['relation_d_att'][i]
          # weight_drug_embedding = weight_drug_embedding + drug_embedding
          # weight_drug_embedding =  drug_embedding
          weight_drug_embedding = weight_drug_embedding
          relation_e_neigh = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(self.sparse_adj_list[i]),weight_drug_embedding)

          drug_neigh.append(relation_u_neigh)
          entity_neigh.append(relation_e_neigh)
          
        drug_neigh = tf.reduce_sum(drug_neigh,0)
        entity_neigh = tf.reduce_sum(entity_neigh,0)
        neigh_embed = tf.concat([drug_neigh, entity_neigh], axis=0)

        ego_embeddings = tf.concat([drug_embedding, entity_embedding], axis=0)
        # ego_embeddings = tf.nn.l2_normalize(ego_embeddings, dim=1)
        # all_embeddings = [ego_embeddings]
        all_embeddings = []

        # side_embeddings = tf.nn.l2_normalize(neigh_embed, dim=1)
        side_embeddings = neigh_embed

        side_embeddings = tf.concat([ego_embeddings, side_embeddings], 1)
        pre_embeddings = tf.nn.relu(
            tf.matmul(side_embeddings, self.weights['W_mlp_0']) + self.weights['b_mlp_0'])

        pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[0])

        # normalize the distribution of embeddings.
        # norm_embeddings = tf.nn.l2_normalize(pre_embeddings, dim=1)
        all_embeddings += [pre_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        da_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_drugs, self.n_entities], 0)
        return da_embeddings, ea_embeddings


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        if len(coo.data) == 0:
            return tf.SparseTensor([[1,2]], [0.], coo.shape)#生成空的稀疏矩阵
        indices = np.mat([coo.row, coo.col]).transpose()
        sparse_result = tf.SparseTensor(indices, coo.data, coo.shape)
        return tf.sparse_reorder(sparse_result)#重新排序，不然会报--稀疏矩阵乱序错误

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

    def train_A(self, sess, feed_dict):
        return sess.run([self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        d_e = sess.run(self.d_e,feed_dict)
        pos_e = sess.run(self.pos_e,feed_dict)
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return d_e,pos_e,batch_predictions
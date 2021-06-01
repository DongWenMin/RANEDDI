import tensorflow as tf
from utility.helper import *
from utility.batch_test import *
from time import time
from RANEDDI import RANEDDI
from collections import defaultdict

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if __name__ == "__main__":
    args = parse_args()
    max_aupr = 0.
    train_dict= data_generator.train_dict
    test_dict = data_generator.test_dict
    test_neg_dict = data_generator.test_neg_dict
    print('检索test，train数据完毕。')
    tf.set_random_seed(2021)
    np.random.seed(2021)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_drugs'] = data_generator.n_drugs
    config['n_relations'] = data_generator.n_relations
    # config['w_sp_matrix'] = data_generator.w_sp_matrix

    "Load the KG triplets."
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['sparse_adj_list'] = data_generator.sparse_adj_list

    t0 = time()

    model = RANEDDI(data_config=config, args=args)

    saver = tf.compat.v1.train.Saver()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    """
    *********************************************************
    Reload the model parameters to fine tune.
    """
    
    sess.run(tf.compat.v1.global_variables_initializer())
    print('without pretraining.')

    """
    *********************************************************
    Train.
    """
    print('current margin:',model.margin)

    for epoch in range(args.epoch):
          t1 = time()
          loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
          n_batch = data_generator.n_train // args.batch_size + 1
          for idx in range(n_batch):
              btime= time()

              batch_data = data_generator.generate_train_batch()
              feed_dict = data_generator.generate_train_feed_dict(model, batch_data)

              _, batch_loss1, batch_base_loss, batch_kge_loss1, batch_reg_loss1,\
                  batch_loss2, batch_kge_loss2, batch_reg_loss2 = model.train(sess, feed_dict=feed_dict)

              loss = loss + batch_loss1 + batch_loss2
              base_loss += batch_base_loss
              kge_loss = batch_kge_loss1 + batch_kge_loss2 + kge_loss
              reg_loss = reg_loss + batch_reg_loss1 + batch_reg_loss2
              break
          perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                      epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
          print(perf_str)
          if epoch%3 == 0:
              t2 = time()
              drugs_to_test = list(data_generator.test_drug_dict.keys())
              ret = test(sess, model, drugs_to_test,train_dict,test_dict,test_neg_dict, drop_flag=False, batch_test_flag=batch_test_flag)
              t3 = time()

              if args.verbose > 0:
                  print('time:',t3-t1,'performance : ',ret)
                  if ret > max_aupr:
                      max_aupr = ret
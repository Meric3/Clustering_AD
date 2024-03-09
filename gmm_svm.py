from dataset import *

from cluster import *
from classifier import *
from preprocessing import *
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

import logging
import time
from datetime import datetime 
from sklearn import metrics
from sklearn import svm
from matplotlib import pyplot as plt

log = logging.getLogger('logloglgo')
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')

now = datetime.now()

today = '%s-%s-%s'%(now.year, now.month, now.day)
second = ' %sh%sm%ss'%(now.hour, now.minute, now.second)

folder_path = './log/' + today

if os.path.exists(folder_path) == False:
    os.makedirs(folder_path)
    
data_name = 'swat_and_wadi_'
cluster_name = 'GMM_svm_0,1other'
fileHandler = logging.FileHandler(os.path.join(folder_path, data_name + cluster_name  + '.txt'))

fileHandler.setFormatter(formatter)
log.addHandler(fileHandler)
log.info("-"*99)    
wadi_dic = dict()
wadi_dic['data_path'] = './wadi_data/'

swat_dic = dict()
swat_dic['data_path'] = './swat_data/'


train_x_swat, train_y_swat, val_x_swat, val_y_swat, test_x_swat, test_y_swat = SWaT_dataset(data_path = swat_dic['data_path'])  
train_x_wadi, train_y_wadi, val_x_wadi, val_y_wadi, test_x_wadi, test_y_wadi = Wadi_dataset(data_path = wadi_dic['data_path'])



def PCA_preprocessing_modify(**kwargs):
    if kwargs['scaler'] == 'standard':
        transformer = sklearn.preprocessing.StandardScaler()
        transformer.fit(kwargs['train_x'])
    elif kwargs['scaler'] == 'normalizer':
        transformer = sklearn.preprocessing.Normalizer()
        transformer.fit(kwargs['train_x'])

    train_x = transformer.transform(kwargs['train_x'])
    val_x = transformer.transform(kwargs['val_x'])
    test_x = transformer.transform(kwargs['test_x'])

    pca = PCA(n_components = np.shape(train_x)[1])
    pca.fit(train_x)

    train_x = pca.transform(train_x)[:,kwargs['selected_dim']]
    val_x = pca.transform(val_x)[:, kwargs['selected_dim']]
    test_x = pca.transform(test_x)[:, kwargs['selected_dim']]

    return train_x, val_x, test_x


train_x_modify_swat, val_x_modify_swat, test_x_modify_swat = PCA_preprocessing_modify(scaler = 'standard', train_x = train_x_swat,                                                val_x = val_x_swat, test_x= test_x_swat
                                                                       , n_neighbors =4, n_components =2,\
                                         feature_num =2, selected_dim= [0,1,39,40])

train_x_modify_wadi, val_x_modify_wadi, test_x_modify_wadi = PCA_preprocessing_modify(scaler = 'standard', train_x = train_x_wadi,                                                val_x = val_x_wadi, test_x= test_x_wadi
                                                                       , n_neighbors =4, n_components =2,\
                                         feature_num =2, selected_dim= [0,1,83,84])

#  [0,1,39,40]
# [0,1,83,84]


log.info('%s:%s:%s'%(datetime.now().hour, datetime.now().minute, datetime.now().second))

for n_clusters in range(4,13,2):
    """
    
    
    GaussianMixture_clustering 을 다른 부분으로 수정하면 됨.



    """
    train_label_gmm_swat, val_label_gmm_swat, test_label_gmm_swat, cluster_model_gmm_swat = GaussianMixture_clustering(n_clusters = n_clusters,                                                train_x = train_x_modify_swat, val_x= val_x_modify_swat, test_x = test_x_modify_swat) 


    train_label_gmm_wadi, val_label_gmm_wadi, test_label_gmm_wadi, cluster_model_gmm_wadi = GaussianMixture_clustering(n_clusters = n_clusters,                                                train_x = train_x_modify_wadi, val_x= val_x_modify_wadi, test_x = test_x_modify_wadi) 

    train_label_swat = train_label_gmm_swat
    val_label_swat = val_label_gmm_swat
    test_label_swat = test_label_gmm_swat
    
    train_label_wadi = train_label_gmm_wadi
    val_label_wadi = val_label_gmm_wadi
    test_label_wadi = test_label_gmm_wadi    
    
    log.info("-"*30)
    log.info("-"*30)
    log.info("Whole cluster Count {}".format(n_clusters))
    log.info("-"*30)
    
    each_predict_swat = 0
    each_predict_wadi = 0
    for cluster_num in range(np.unique(train_label_swat)[-1] + 1):
        label_for_train_swat = (train_label_swat == cluster_num).astype(int)
        label_for_var_swat = (val_label_swat == cluster_num).astype(int)
        label_for_test_swat = (test_label_swat == cluster_num).astype(int)
        
        label_for_train_wadi = (train_label_wadi == cluster_num).astype(int)
        label_for_var_wadi = (val_label_wadi == cluster_num).astype(int)
        label_for_test_wadi = (test_label_wadi == cluster_num).astype(int)
        
        """
    
    
        svm.SVC 을 다른 부분으로 수정하면 됨.



        """

#         KNeighborsClassifier(n_neighbors=3)
# svm.SVC(gamma='auto')
        classifier_swat = svm.SVC(gamma='auto')
        classifier_swat.fit(train_x_modify_swat, label_for_train_swat)
        classifier_wadi = svm.SVC(gamma='auto')
        classifier_wadi.fit(train_x_modify_wadi, label_for_train_wadi)
        
        pred_train_swat = classifier_swat.predict(train_x_modify_swat)
        pred_test_swat = classifier_swat.predict(test_x_modify_swat)
        pred_train_wadi = classifier_wadi.predict(train_x_modify_wadi)
        pred_test_wadi = classifier_wadi.predict(test_x_modify_wadi)

        each_predict_swat += pred_test_swat
        each_predict_wadi += pred_test_wadi
        
        log.info("-"*30)
        log.info("CLUSTER num {}".format(cluster_num))
        log.info("SWAT TRAIN f1 {:.4f}, acc {:.4f}".format(metrics.f1_score(label_for_train_swat, pred_train_swat, pos_label = 0),                                           metrics.accuracy_score(label_for_train_swat, pred_train_swat)))
        log.info("SWAT TRAIN f1 {:.4f}, acc {:.4f}".format(metrics.f1_score(label_for_train_swat, pred_train_swat),                                           metrics.accuracy_score(label_for_train_swat, pred_train_swat)))

        log.info("SWAT TEST f1 {:.4f}, acc {:.4f}".format(metrics.f1_score(label_for_test_swat, pred_test_swat),                                           metrics.accuracy_score(label_for_test_swat, pred_test_swat)))
    
        log.info("wadi TRAIN f1 {:.4f}, acc {:.4f}".format(metrics.f1_score(label_for_train_wadi, pred_train_wadi, pos_label = 0),                                           metrics.accuracy_score(label_for_train_wadi, pred_train_wadi)))
        log.info("wadi TRAIN f1 {:.4f}, acc {:.4f}".format(metrics.f1_score(label_for_train_wadi, pred_train_wadi),                                           metrics.accuracy_score(label_for_train_wadi, pred_train_wadi)))

        log.info("wadi TEST f1 {:.4f}, acc {:.4f}".format(metrics.f1_score(label_for_test_wadi, pred_test_wadi),                                           metrics.accuracy_score(label_for_test_wadi, pred_test_wadi)))        


        log.info("SWAT Cluster measure train {:.4f}, {:.4f}".format(metrics.calinski_harabasz_score(train_x_modify_swat, label_for_train_swat), metrics.davies_bouldin_score(train_x_modify_swat, label_for_train_swat)))

        log.info("SWAT Cluster measure test {:.4f}, {:.4f}".format(metrics.calinski_harabasz_score(test_x_modify_swat, label_for_test_swat), metrics.davies_bouldin_score(test_x_modify_swat, label_for_test_swat)))
        log.info("wadi Cluster measure train {:.4f}, {:.4f}".format(metrics.calinski_harabasz_score(train_x_modify_wadi, label_for_train_wadi), metrics.davies_bouldin_score(train_x_modify_wadi, label_for_train_wadi)))

        log.info("wadi Cluster measure test {:.4f}, {:.4f}".format(metrics.calinski_harabasz_score(test_x_modify_wadi, label_for_test_wadi), metrics.davies_bouldin_score(test_x_modify_wadi, label_for_test_wadi)))

        log.info("-"*30)
        for j in range(cluster_num):
            log.info('swat ensemble predict j {} f1 : {:.4f} np.unique: {}'.format(j, f1_score(test_y_swat, (each_predict_swat < j).astype(int)),                                      np.unique(each_predict_swat, return_counts = True)[1]))
        for j in range(cluster_num):
            log.info('wadi ensemble predict j {} f1 : {:.4f} np.unique: {}'.format(j, f1_score(test_y_wadi, (each_predict_wadi < j).astype(int)),                                      np.unique(each_predict_wadi, return_counts = True)[1]))
        log.info("-"*30)
    log.info("-"*30)
    log.info("-"*30)
log.info('%s:%s:%s'%(datetime.now().hour, datetime.now().minute, datetime.now().second))






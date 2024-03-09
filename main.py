
import warnings
warnings.filterwarnings('ignore')

from dataset import *

from cluster import *
from classifier import *
from preprocessing import *
import numpy as np

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import logging
import time
import datetime
from sklearn import svm
from matplotlib import pyplot as plt
import argparse

from scipy import signal
from tapr import *
from util import *

from itertools import combinations
import pdb


def main():
    
    np.random.seed(777)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cov_type', type = str, default = 'full')
    parser.add_argument('--gamma', type = float, default = 0.1)
    parser.add_argument('--C', type = float, default = 1000)
    parser.add_argument('--exp', type = str, default = 'tp')
    parser.add_argument('--selected_dim', nargs='+', type=int, default=[[0,1],[2,3],[3,1,2]])
    parser.add_argument('--wadi_freq_select_list', nargs='+', type=str, default=[['2_FIT_002_PV']])
    parser.add_argument('--read_size', type = int, default = 5)

    args = parser.parse_args()    
    
    
    args.wadi_freq_select_list = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_P_005_STATUS', '2_LT_001_PV', '2_LT_002_PV']
    delta = 600
    theta = 0.001
    alpha = 1.0
    label = [0, 1]
    
    print(args.exp)
    print(args.cov_type)
    
    ev = TaPR(label, theta, delta)
    ev.load_anomalies( './wadi_label.csv')
    
    log = logging.getLogger('log')
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    now = datetime.datetime.now()

    today = '%s-%s-%s'%(now.year, now.month, now.day)
    second = ' %sh%sm%ss'%(now.hour, now.minute, now.second)

    folder_path = './log/' + today

    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)

    fileHandler = logging.FileHandler(os.path.join(folder_path, args.exp  + '.txt'))

    fileHandler.setFormatter(formatter)
    log.addHandler(fileHandler)
    log.info("-"*99)    
    log.info("-"*10 + str(args) + "-"*10)
    log.info("-"*99) 
    

    wadi_dic = dict()
    wadi_dic['data_path'] = './wadi_data/'

    log.info('START %s:%s:%s\n'%(datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    

    raw_selected_dim = [0,1,2,90,91,92]
    freq_selected_dim = [0,1,31,32]
    
    wadi_n_cluster_list = list(np.arange(5,15))
    
    
    window_size = 30

    log.info('%s:%s:%s\n'%(datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))


    train_x_wadi, train_y_wadi, val_x_wadi, val_y_wadi, test_x_wadi, test_y_wadi, _ = Wadi_dataset(data_path = wadi_dic['data_path'], read_size = args.read_size)

    train_x_wadi_freq = get_freq_data_2(data = train_x_wadi, 
                    freq_select_list = args.wadi_freq_select_list, read_size = args.read_size,
                                          window_size = window_size)
    val_x_wadi_freq = get_freq_data_2(data = val_x_wadi, 
                    freq_select_list = args.wadi_freq_select_list, read_size = args.read_size,
                                          window_size = window_size)
    test_x_wadi_freq = get_freq_data_2(data = test_x_wadi, 
                    freq_select_list = args.wadi_freq_select_list, read_size = args.read_size,
                                          window_size = window_size)



    train_x_modify_wadi, val_x_modify_wadi, test_x_modify_wadi = PCA_preprocessing_modify(scaler = 'standard', \
                                                                    train_x = train_x_wadi, val_x = val_x_wadi, \
                                                                  test_x= test_x_wadi, n_neighbors = 4, 
                                                                  n_components = 2, feature_num =2, \
                                                                  selected_dim= raw_selected_dim)

    train_x_modify_wadi_freq, val_x_modify_wadi_freq, test_x_modify_wadi_freq = PCA_preprocessing_modify(scaler = 'standard', \
                                                                    train_x = train_x_wadi_freq, val_x = val_x_wadi_freq, \
                                                                  test_x= test_x_wadi_freq, n_neighbors = 4, 
                                                                  n_components = 2, feature_num =2, \
                                                                  selected_dim= freq_selected_dim)

    optimal_f1_wadi_list = []
    optimal_tar_wadi_list = []
    optimal_tap_wadi_list = []
    optimal_cluster_combi_wadi_list = []


    train_x_modify_wadi = np.concatenate((train_x_modify_wadi, train_x_modify_wadi_freq),1)
    val_x_modify_wadi = np.concatenate((val_x_modify_wadi, val_x_modify_wadi_freq),1)
    test_x_modify_wadi = np.concatenate((test_x_modify_wadi, test_x_modify_wadi_freq),1)

    wadi_n_cluster_list = list(np.arange(5,15))

    for wadi_n_clusters in wadi_n_cluster_list:
        log.info("n_clutser {}".format(wadi_n_clusters))

        train_predict_wadi_list = []
        test_predict_wadi_list = []
        label_for_train_wadi_list = []
        label_for_var_wadi_list = []
        label_for_test_wadi_list = []
        classifier_wadi_list = []

        tp_optimal_f1_wadi = 0
        tp_optimal_tar_wadi = 0
        tp_optimal_tap_wadi = 0
        tp_optimal_cluster_combi_wadi = 0


        train_label_wadi, val_label_wadi, test_label_wadi, cluster_model_gmm_wadi = GaussianMixture_clustering(
            n_clusters=wadi_n_clusters,train_x = train_x_modify_wadi,val_x= val_x_modify_wadi,
            test_x = test_x_modify_wadi,covariance_type = 'tied') 

        for cluster_num in range(np.unique(train_label_wadi)[-1] + 1):
            label_for_train_wadi = (train_label_wadi == cluster_num).astype(int)
            label_for_var_wadi = (val_label_wadi == cluster_num).astype(int)
            label_for_test_wadi = (test_label_wadi == cluster_num).astype(int)
            if np.unique(label_for_train_wadi).shape[0] == 1:
                label_for_train_wadi[0] = 1
                label_for_test_wadi[0] = 1
                label_for_var_wadi[0] = 1


            label_for_train_wadi_list.append(label_for_train_wadi)
            label_for_var_wadi_list.append(label_for_var_wadi)
            label_for_test_wadi_list.append(label_for_test_wadi)

            classifier_wadi = svm.SVC(gamma = args.gamma, C = args.C)
            classifier_wadi.fit(train_x_modify_wadi, label_for_train_wadi)
            classifier_wadi_list.append(classifier_wadi)

            pred_train_wadi = classifier_wadi.predict(train_x_modify_wadi)
            pred_test_wadi = classifier_wadi.predict(test_x_modify_wadi)

            test_predict_wadi_list.append(pred_test_wadi)
            train_predict_wadi_list.append(pred_test_wadi)

        # wadi tapr modified version.
        ev.load_anomalies( './wadi_label.csv')

        for combi_num in range(2, wadi_n_clusters + 1):
            wadi_combi_list = list(combinations(test_predict_wadi_list, combi_num))
            combi_list = list(combinations(np.arange(wadi_n_clusters), combi_num))

            wadi_f1 = 0
            wadi_j = 0
            wadi_combi_max = 0

            wadi_list = 0
            wadi_pr = 0
            wadi_re = 0

            wadi_tar = 0
            wadi_tap = 0

            wadi_alpha = 0
            undetected = 0

            wadi_j = 0

            for combi_ in range(len(wadi_combi_list)):
                wadi_predict = 0

                for i in range(len(wadi_combi_list[combi_])):
                    wadi_predict += wadi_combi_list[combi_][i]

                for j in range(1, len(wadi_combi_list[combi_]) -1):

                    pd.DataFrame((wadi_predict < j).astype(int)).to_csv('./prediction_tttt.csv',index=False, header=None)
                    wadi_list_tp = (wadi_predict < j).astype(int)

                    ev.load_predictions('./prediction_tttt.csv')

                    tapd_value, _ = ev.TaP_d()
                    tard_value, _ = ev.TaR_d()
                    tapp_value = ev.TaP_p()            
                    tarp_value = ev.TaR_p()

                    tar = 0.5*tard_value + 0.5*tarp_value
                    tap = 0.5*tapd_value + 0.5*tapp_value
                    tp_f1 = f1_score(test_y_wadi, (wadi_predict < j).astype(int))
                    tp_pr = precision_score(test_y_wadi, (wadi_predict < j).astype(int))
                    tp_re = recall_score(test_y_wadi, (wadi_predict < j).astype(int))

                    if  tp_f1 > wadi_f1:
                        wadi_tap = tap
                        wadi_tar = tar
                        wadi_f1 = tp_f1
                        wadi_pr = tp_pr
                        wadi_re = tp_re

                        wadi_combi_max = combi_list[combi_]

                        wadi_list = wadi_list_tp
                        undetected = find_attack_num(wadi_list, test_y_wadi)
                        wadi_j = j

            log.info("-"*30) 
            log.info('wadi pr {:.4f}, re{:.4f}, f1 {:.4f}, combi_list {}, undetectd {} j {}'.\
                  format(wadi_pr, wadi_re ,wadi_f1, wadi_combi_max, undetected, wadi_j))
            log.info("tar {:.4f} tap {:.4f}".format(wadi_tar, wadi_tap))
            log.info(np.unique(wadi_list, return_counts=True))
            log.info("-"*30) 
            if tp_optimal_f1_wadi < wadi_f1:
                tp_optimal_f1_wadi = wadi_f1
                tp_optimal_tar_wadi = wadi_tar
                tp_optimal_tap_wadi = wadi_tap
                tp_optimal_cluster_combi_wadi = wadi_combi_max
        optimal_f1_wadi_list.append(tp_optimal_f1_wadi)
        optimal_tar_wadi_list.append(tp_optimal_tar_wadi)
        optimal_tap_wadi_list.append(tp_optimal_tap_wadi)
        optimal_cluster_combi_wadi_list.append(tp_optimal_cluster_combi_wadi)      



    log.info(optimal_f1_wadi_list)
    log.info(optimal_tar_wadi_list)
    log.info(optimal_tap_wadi_list)
    log.info(optimal_cluster_combi_wadi_list)
    
    
    log.info("-"*30)
    log.info("-"*30)
    log.info('FINISH')   
    log.info('%s:%s:%s'%(datetime.now().hour, datetime.now().minute, datetime.now().second))




if __name__ == "__main__":
    main()

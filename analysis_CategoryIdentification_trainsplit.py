'''
Object category identification

This file is a part of GenericDecoding_demo.
'''


from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd

import bdpy
from bdpy.stats import corrmat

import god_config as config


# Main #################################################################

def main():
    results_dir = config.results_dir
    output_file = "results/GenericObjectDecoding/analysis_FeaturePrediction_trainsplit2.py-Subject1-V4-cnn5.pkl"

    image_feature_file = config.image_feature_file

    # Load results -----------------------------------------------------
    print('Loading %s' % output_file)
    with open(output_file, 'rb') as f:
        results = pickle.load(f)

    data_feature = bdpy.BData(image_feature_file)
    print(data_feature.show_metadata())

    # Category identification ------------------------------------------
    print('Running pair-wise category identification')
    print(results)

    feature_list = results['feature']
    pred_percept = results['predicted_feature_averaged']
    cat_label_percept = results['category_label_set']
    cat_feature_percept = results['category_feature_averaged']
    print("cat_feature_percept: ",cat_feature_percept)
    print("feature_list:", feature_list)
    print("pred_percept:", pred_percept)
    print("cat_label_percept:", cat_label_percept)

    #ind_cat_other = test_y_label = np.array([(i % 8 == 0 and i < 1200) for i in range(16622)])
    ind_cat_other = (data_feature.select('FeatureType') == 4).flatten()
    print(data_feature.select('FeatureType'))
    print("ind_cat_other: ", ind_cat_other)
    print("ind_cat_other: ", ind_cat_other.shape)

    pwident_cr_pt = []  # Prop correct in pair-wise identification (perception)

    for f, fpt, pred_pt in zip(feature_list, cat_feature_percept,
                                             pred_percept):
        feat_other = data_feature.select(f)[ind_cat_other, :]

        n_unit = fpt.shape[1]
        feat_other = feat_other[:, :n_unit]

        feat_candidate_pt = fpt#np.vstack([fpt, feat_other])
        #feat_candidate_pt = fpt#np.array([fpt[i] for i in range(len(pred_pt)) if i % 8 == 0])#fpt

        print(feat_candidate_pt)
        print(feat_candidate_pt.shape)
        print(pred_pt)
        print(pred_pt.shape)

        simmat_pt = corrmat(pred_pt, feat_candidate_pt)

        print(simmat_pt)
        print(simmat_pt.shape)

        cr_pt = get_pwident_correctrate(simmat_pt)

        pwident_cr_pt.append(np.mean(cr_pt))

    results['catident_correct_rate_percept'] = pwident_cr_pt

    # Save the merged dataframe ----------------------------------------
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print('Saved %s' % output_file)

    # Show results -----------------------------------------------------
    tb_pt = pd.pivot_table(results, index=['roi'], columns=['feature'],
                           values=['catident_correct_rate_percept'], aggfunc=np.mean)

    print(tb_pt)


# Functions ############################################################
def softmax(x):

    y = np.exp(x)
    f_x = y / np.sum(np.exp(x))
    return f_x

def get_pwident_correctrate(simmat):
    '''
    Returns correct rate in pairwise identification

    Parameters
    ----------
    simmat : numpy array [num_prediction * num_category]
        Similarity matrix

    Returns
    -------
    correct_rate : correct rate of pair-wise identification
    '''

    num_pred = simmat.shape[0]
    labels = range(num_pred)

    correct_rate = []
    for i in range(num_pred):
        print(i)
        print(labels[i])
        pred_feat = simmat[i, :]
        #print("pred_feat: ", pred_feat)
        print("pred_feat: ", pred_feat.shape)
        #print("pred_feat: ", softmax(pred_feat*30))
        print("pred_feat: ", np.max(softmax(pred_feat*50)))
        print("pred_feat: ", np.argmax(softmax(pred_feat*50)))

        correct_feat = pred_feat[labels[i]]

        print(correct_feat)
        pred_num = len(pred_feat) - 1
        cor_rat = (pred_num - np.sum(pred_feat > correct_feat)) / float(pred_num)
        print(cor_rat)
        correct_rate.append(cor_rat)

    return correct_rate


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()

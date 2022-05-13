'''Generic Object Decoding: Feature prediction

Analysis summary
----------------

- Learning method:   Sparse linear regression
- Preprocessing:     Normalization and voxel selection
- Data:              GenericDecoding_demo
- Results format:    Pandas dataframe
'''


from __future__ import print_function

import os
import sys
import pickle
from itertools import product
from time import time

import numpy as np
import pandas as pd
from scipy import stats

#from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression  # For quick demo

import bdpy
from bdpy.bdata import concat_dataset
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.dataform import append_dataframe
from bdpy.distcomp import DistComp

import god_config as config


# Main #################################################################

def main():
    # Settings ---------------------------------------------------------

    # Data settings
    subjects = config.subjects
    rois = config.rois
    num_voxel = config.num_voxel

    image_feature = config.image_feature_file
    features = config.features

    n_iter = 200

    results_dir = config.results_dir

    # Misc settings
    analysis_basename = os.path.basename(__file__)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)

    print("DATA ALL:", data_all)
    data_feature = bdpy.BData(image_feature)
    data_feature.show_metadata()

    # Add any additional processing to data here     

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for sbj, roi, feat in product(subjects, rois, features):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])
        print('Feature:    %s' % feat)

        # Distributed computation
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        if os.path.exists(results_file):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        dist = DistComp(lockdir='tmp', comp_id=analysis_id)
        if dist.islocked():
            print('%s is already running. Skipped.' % analysis_id)
            continue

        dist.lock()

        # Prepare data
        print('Preparing data')
        dat = data_all[sbj]

        x = dat.select(rois[roi])           # Brain data
        datatype = dat.select('DataType')   # Data type
        labels = dat.select('stimulus_id')  # Image labels in brain data
        print("Labels:", labels) 
        print("Labels:", labels.shape) #(3450,1)

        y = data_feature.select(feat)             # Image features
        y_label = data_feature.select('ImageID')  # Image labels
        print("ImID:", y_label)
        print("ImID:", y_label.shape) #(16622,1)

        y = y[:, :100]

        print(x)
        print(x.shape) #(3450,860)
        print(y)
        print(y.shape) #(16622,1000)
        print(y_label)
        print(y_label.shape) #(16622,1)

        y_label_not_none = np.isnan(y_label)        
        print(y_label[y_label_not_none==False]) #(16622,1)
        print(y_label[y_label_not_none==False].shape) #(16622,1)

        test_y = np.array([y[i] for i in range(len(y_label)) if i % 8 == 0])
        train_y = np.array([y[i] for i in range(len(y_label)) if i % 8 != 0])

        test_y_label = np.array([y_label[i] for i in range(len(y_label)) if i % 8 == 0])
        train_y_label = np.array([y_label[i] for i in range(len(y_label)) if i % 8 != 0])

        print(train_y)
        print(train_y_label)

        print(test_y.shape) #(2078,1000) -> nan padding
        print(train_y.shape) #(14544,1000)
        print(test_y_label.shape) #(2078,1) -> nan padding
        print(train_y_label.shape) #(14544,1)

        # For quick demo, reduce the number of units from 1000 to 100

        y_sorted, labels_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

        print("y_sorted: ", y_sorted)
        print("y_sorted: ", y_sorted.shape)
        print("labels_sorted: ", labels_sorted)
        print("labels_sorted: ", labels_sorted.shape)

        # Get training and test dataset
        i_train = (datatype == 1).flatten()    # Index for training
        i_test_pt = (datatype == 2).flatten()  # Index for perception test
        i_test_im = (datatype == 3).flatten()  # Index for imagery test

        i_test = i_test_pt + i_test_im

        x_train_old = x[i_train, :]
        #x_test = x[i_test, :]
        
        i_train_true = np.where(i_train == True)[0]
        i_test_pt_true = np.where(i_test_pt == True)[0]

        ind_cat_other = (data_feature.select('FeatureType') == 4).flatten()
        ind_cat_other_true = np.where(ind_cat_other == True)[0]

        print(i_train_true)
        print(i_train_true.shape)
        print(i_test_pt_true)
        print(i_test_pt_true.shape)
        print(ind_cat_other_true)
        print(ind_cat_other_true.shape)

        y_train_old = y_sorted[i_train, :]
        #y_test = y_sorted[i_test, :]

        y_labels_train = labels_sorted[i_train]
        y_dict = {}
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for i in range(len(y_labels_train)):
            if y_labels_train[i] not in y_dict:
                y_dict[y_labels_train[i]] = [y_train_old[i,:]]
            else:
                y_dict[y_labels_train[i]].append(y_train_old[i,:])
            
            if len(y_dict[y_labels_train[i]]) > 7:
                x_test.append(x_train_old[i,:])
                y_test.append(y_train_old[i,:])
            else:
                x_train.append(x_train_old[i,:])
                y_train.append(y_train_old[i,:])
            
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print("x_train: ", x_train)
        print(x_train.shape)
        print("x_test: ", x_test)
        print(x_test.shape)
        print("y_train: ", y_train)
        print(y_train.shape)
        print("y_test: ", y_test)
        print(y_test.shape)

        test_label_pt = labels[i_train, :].flatten()
        catlabels_pt = np.vstack([int(n) for n in test_label_pt])
        catlabels_set_pt = np.unique(catlabels_pt)
        print("test_label_pt: ", test_label_pt)
        print("test_label_pt: ", test_label_pt.shape) #(1750, )
        print("catlabels_set_pt: ", catlabels_set_pt) # test dataset labels
        print("catlabels_set_pt: ", catlabels_set_pt.shape) # (50, )

        # Feature prediction
        length = 100
        pred_y, true_y = feature_prediction(x_train, y_train,
                                            x_test, y_test,
                                            n_voxel=num_voxel[roi],
                                            n_iter=n_iter)


        # Get averaged predicted feature
        #test_label_pt = labels[i_test_pt, :].flatten()
        #test_label_im = labels[i_test_im, :].flatten()
        test_label = labels_sorted[i_train]


        test_y_label_isnot_none = np.isnan(test_y_label)        
        test_y_label_no_none = test_y_label[test_y_label_isnot_none==False].flatten()

        print(test_y_label_no_none)
        print(test_y_label_no_none.shape)

        pred_y_av, true_y_av, test_label_set \
            = get_averaged_feature(pred_y, true_y, test_y_label_no_none[:150])

        print("pred_y: ", pred_y)
        print("pred_y: ", pred_y.shape) #(150, 1000)
        print("pred_y_av: ", pred_y_av)
        print("pred_y_av: ", pred_y_av.shape) #(150, 1000)
        print("test_label_set: ", test_label_set)
        print("test_label_set: ", test_label_set.shape) #(150, )

        # Get category averaged features
        catlabels = np.vstack([int(n) for n in test_label])  # Category labels (perception test)
        catlabels_set = np.unique(catlabels)                 # Category label set (perception test)

        y_catlabels = data_feature.select('CatID')   # Category labels in image features
        ind_catave = (data_feature.select('FeatureType') == 3).flatten()
    
        print(y_catlabels)
        print(y_catlabels.shape)# (16622, 1)
        print(ind_catave.shape)# (16622,)

        #y_catave, _ = get_refdata(test_y[:150, :], test_y_label[:150, :], test_y_label[:150])
        train_y_catave, _ = get_refdata(train_y, train_y_label, train_y_label)
        print("train_y_catave: ", train_y_catave.shape)
        y_catave_list = []
        average = 0
        for i in range(len(train_y_catave[:1050])):
            if i % 7 == 0:
                average = 0
            average += train_y_catave[i,:]
            if i % 7 == 6:
                average /= 7
                y_catave_list.append(average)
        y_catave = np.array(y_catave_list)

        print("catlabels_set: ", catlabels_set)
        print("catlabels_set: ", catlabels_set.shape)#(150,)
        print("y_catave: ", y_catave)
        print("y_catave: ", y_catave.shape)#(150,1000)

        # Prepare result dataframe
        results = pd.DataFrame({'subject' : [sbj],
                                'roi' : [roi],
                                'feature' : [feat],
                                'test_type' : ['perception'],
                                'true_feature': [true_y],
                                'predicted_feature': [pred_y],
                                'test_label' : [test_label],
                                'test_label_set' : [test_label_set],
                                'true_feature_averaged' : [true_y_av],
                                'predicted_feature_averaged' : [pred_y_av],
                                'category_label_set' : [catlabels_set],
                                'category_feature_averaged' : [y_catave]})

        # Save results
        print("Result: ", results)
        makedir_ifnot(os.path.dirname(results_file))
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print('Saved %s' % results_file)

        dist.unlock()


# Functions ############################################################

def feature_prediction(x_train, y_train, x_test, y_test, n_voxel=500, n_iter=200):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''

    n_unit = y_train.shape[1]
    print(n_unit)

    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    # Feature prediction for each unit
    print('Running feature prediction')

    y_true_list = []
    y_pred_list = []

    for i in range(n_unit):

        print('Unit %03d' % (i + 1))
        start_time = time()

        # Get unit features
        y_train_unit = y_train[:, i]
        y_test_unit =  y_test[:, i]

        # Normalize image features for training (y_train_unit)
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # Voxel selection
        corr = corrcoef(y_train_unit, x_train, var='col')

        x_train_unit, voxel_index = select_top(x_train, np.abs(corr), n_voxel, axis=1, verbose=False)
        x_test_unit = x_test[:, voxel_index]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)

        # Setup regression
        # For quick demo, use linaer regression
        model = LinearRegression()
        #model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)

        # Training and test
        try:
            model.fit(x_train_unit, y_train_unit)  # Training
            y_pred = model.predict(x_test_unit)    # Test
        except:
            # When SLiR failed, returns zero-filled array as predicted features
            y_pred = np.zeros(y_test_unit.shape)

        # Denormalize predicted features
        y_pred = y_pred * norm_scale_y + norm_mean_y

        y_true_list.append(y_test_unit)
        y_pred_list.append(y_pred)

        print('Time: %.3f sec' % (time() - start_time))

    # Create numpy arrays for return values
    y_predicted = np.vstack(y_pred_list).T
    y_true = np.vstack(y_true_list).T

    return y_predicted, y_true


def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()

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

TESTSET_source = ['n01518878_8432', 'n01639765_52902', 'n01645776_9743', 'n01664990_7133', 'n01704323_9812', 'n01726692_9090', 'n01768244_9890', 'n01770393_29944', 'n01772222_16161', 'n01784675_9652', 'n01787835_9197', 'n01833805_9089', 'n01855672_15900', 'n01877134_8213', 'n01944390_21065', 'n01963571_7996', 'n01970164_30734', 'n02005790_9371', 'n02054036_9789', 'n02055803_9893', 'n02068974_7285', 'n02084071_34874', 'n02090827_9359', 'n02131653_7902', 'n02226429_9440', 'n02233338_9484', 'n02236241_8966', 'n02317335_6776', 'n02346627_7809', 'n02374451_19966', 'n02391049_8178', 'n02432983_9121', 'n02439033_9911', 'n02445715_834', 'n02472293_5718', 'n02480855_9990', 'n02481823_8477', 'n02503517_9095', 'n02508213_7987', 'n02692877_7319', 'n02766534_64673', 'n02769748_52896', 'n02799175_9738', 'n02800213_8939', 'n02802215_9364', 'n02808440_5873', 'n02814860_39856', 'n02841315_35046', 'n02843158_9764', 'n02882647_7150', 'n02885462_9275', 'n02943871_9326', 'n02974003_9408', 'n02998003_7158', 'n03038480_7843', 'n03063599_5015', 'n03079230_8270', 'n03085013_24642', 'n03085219_8998', 'n03187595_9726', 'n03209910_8951', 'n03255030_8996', 'n03261776_40091', 'n03335030_54164', 'n03345487_8958', 'n03359137_44522', 'n03394916_47877', 'n03397947_9823', 'n03400231_8108', 'n03425413_22005', 'n03436182_9434', 'n03445777_9793', 'n03467796_862', 'n03472535_8251', 'n03483823_8564', 'n03494278_42246', 'n03496296_9005', 'n03512147_7137', 'n03541923_9436', 'n03543603_9964', 'n03544143_8694', 'n03602883_8804', 'n03607659_8657', 'n03609235_7271', 'n03612010_9076', 'n03623556_9879', 'n03642806_6122', 'n03646296_9175', 'n03649909_42614', 'n03665924_8249', 'n03721384_6760', 'n03743279_8976', 'n03746005_9272', 'n03760671_9669', 'n03790512_41520', 'n03792782_8413', 'n03793489_8932', 'n03815615_8756', 'n03837869_9127', 'n03886762_9466', 'n03918737_8191', 'n03924679_9951', 'n03950228_39297', 'n03982430_9407', 'n04009552_8947', 'n04044716_8619', 'n04070727_9914', 'n04086273_5433', 'n04090263_7624', 'n04113406_5900', 'n04123740_8706', 'n04146614_9889', 'n04154565_35063', 'n04168199_9862', 'n04180888_8500', 'n04197391_8396', 'n04225987_7933', 'n04233124_24091', 'n04254680_43', 'n04255586_878', 'n04272054_61432', 'n04273569_6437', 'n04284002_21913', 'n04313503_25193', 'n04320973_9400', 'n04373894_59446', 'n04376876_7775', 'n04398044_35327', 'n04401680_36641', 'n04409515_8358', 'n04409806_9916', 'n04412416_8901', 'n04419073_8965', 'n04442312_32591', 'n04442441_7248', 'n04477387_9242', 'n04482393_6359', 'n04497801_8739', 'n04555897_7654', 'n04587559_7411', 'n04591713_7167', 'n04612026_7911', 'n07734017_8706', 'n07734744_8647', 'n07756951_9462', 'n07758680_22370', 'n11978233_42157', 'n12582231_44661', 'n12596148_9625', 'n13111881_9170']
TESTSET = []
for item in TESTSET_source:
    test_item_front = int(item[1:].split('_')[0]) + 0.000001* int(item[1:].split('_')[1])
    TESTSET.append([test_item_front])
print(TESTSET)

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
        labels = dat.select('stimulus_id')[:1200]  # Image labels in brain data
        print("Labels:", labels) 
        print("Labels:", labels.shape) #(1200,1)

        y = data_feature.select(feat)             # Image features
        y_label = data_feature.select('ImageID')  # Image labels
        print("ImID:", y_label)
        print("ImID:", y_label.shape) #(16622,1)

        y = y[:, :1000]

        print(x)
        print(x.shape) #(3450,860)
        print(y)
        print(y.shape) #(16622,1000)
        print(y_label)
        print(y_label.shape) #(16622,1)

        train_indexes = []
        test_indexes = []
        for i, test_label in enumerate(y_label[:1200]):
            if test_label in TESTSET:
                test_indexes.append(i)
            else:
                train_indexes.append(i)

        print("train_label: ", train_indexes)
        print("train_label: ", len(train_indexes))#1050

        print("test_label: ", test_indexes)
        print("test_label: ", len(test_indexes))#150

        y_label_not_none = np.isnan(y_label)        
        print(y_label[y_label_not_none==False])
        print(y_label[y_label_not_none==False].shape) #(1250,1)

        x = x[:1200, :]
        y = y[:1200, :]

        x_train = x[train_indexes, :]
        x_test = x[test_indexes, :]

        y_train = y[train_indexes, :]
        y_test = y[test_indexes, :]

        y_label_train = y_label[train_indexes, :]
        y_label_test = y_label[test_indexes, :]

        print("x_train: ", x_train.shape)#1050,740
        print("x_test: ", x_test.shape)#150,740
        print("y_train: ", y_train.shape)#1050,
        print("y_test: ", y_test.shape)#150,
        print("y_label_train: ", y_label_train.shape)#1050,1
        print("y_label_test: ", y_label_test.shape)#150,1

        pred_y, true_y = feature_prediction(x_train, y_train,
                                            x_train, y_train,
                                            n_voxel=num_voxel[roi],
                                            n_iter=n_iter)

        print("true_y:", true_y.shape)
        print("pred_y:", pred_y.shape)

        #pred_y_av, true_y_av, test_label_set \
        #    = get_averaged_feature(pred_y, true_y, y_label_test.squeeze())

        catlabels = np.vstack([int(n) for n in y_label_test.flatten()])  # Category labels (perception test)
        catlabels_set = np.unique(catlabels)                 # Category label set (perception test)

        y_catave = true_y

        print("catlabels_set: ", catlabels_set)
        print("catlabels_set: ", catlabels_set.shape)
        print("y_catave: ", y_catave)
        print("y_catave: ", y_catave.shape)

        # Prepare result dataframe
        results = pd.DataFrame({'subject' : [sbj],
                                'roi' : [roi],
                                'feature' : [feat],
                                'test_type' : ['perception'],
                                'true_feature': [true_y],
                                'predicted_feature': [pred_y],
                                'test_label' : [y_label_test],
                                'test_label_set' : [y_label_test],
                                'true_feature_averaged' : [true_y],
                                'predicted_feature_averaged' : [pred_y],
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

import numpy as np
import time
#import os
import sys
#import time
from concon_utils import load_raw_labels_boris
#from concon_utils import load_mesh_boris
from parsimony.estimators import LogisticRegressionL1L2TV
from parsimony.algorithms.proximal import StaticCONESTA
from parsimony.algorithms.proximal import FISTA
import pandas as pd
#import YZ_matrixA_neurospin_YZmatrixA as matrixA
import YZ_matrixA_neurospin as matrixA
#import YZ_matrixA_npversion
#from scipy import sparse
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
A = matrixA.A_main()
def model_train_cv(grid):
    ## ML modle ##
    l1 = grid[0]
    l2 = grid[1]
    tv = grid[2]
    print('L1: {}, L2: {}, TV: {}'.format(l1, l2, tv))

    YZ_ML = LogisticRegressionL1L2TV(l1, l2, tv, A, algorithm=StaticCONESTA(max_iter=1000))
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=000)
    accuracy = []
    auc = []
    for train_index_i, test_index_i in cv.split(X, y):
        X_train_dataset, X_test_dataset = X[train_index_i], X[test_index_i]
        y_train_dataset, y_test_dataset = y[train_index_i], y[test_index_i]
        trained_YZ_ML = YZ_ML.fit(X_train_dataset, y_train_dataset)
        prediction = trained_YZ_ML.predict(X_test_dataset)
        accuracy.append(metrics.accuracy_score(y_test_dataset, prediction))
        predict_prob = trained_YZ_ML.predict_probability(X_test_dataset)
        auc.append(metrics.roc_auc_score(y_test_dataset, predict_prob))
    print('Best single accuracy: {}'.format(np.max(accuracy)))
    print(('4-fold-CV accuracy: %f') % (np.mean(accuracy)))
    print('Best single AUC score: {}'.format(np.max(auc)))
    print(('4-fold-CV AUC score: %f') % (np.mean(auc)))

    return np.mean(auc)
t_start = time.time()
X_train_dataset = np.load('scaled_X_train_dataset.npy')
X_test_dataset = np.load('scaled_X_test_dataset.npy')
y_train_dataset = np.load('y_train_dataset.npy')
y_test_dataset = np.load('y_test_dataset.npy')
X = np.concatenate([X_train_dataset, X_test_dataset])
y = np.concatenate([y_train_dataset, y_test_dataset])
gs_list = [[l1, l2, tv] for l1 in [0.001] for l2 in [1000] for tv in [0.001]]
#gs_list = [[l1, l2, tv] for l1 in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000] for l2 in [1000] for tv in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]]
gs_auc = [model_train_cv(grid_point) for grid_point in gs_list]
gs_optimal_point = gs_list[list.index(gs_auc, np.max(gs_auc))]
l1 = gs_optimal_point[0]
l2 = gs_optimal_point[1]
tv = gs_optimal_point[2]
print('epoch_optimal_grid_point: l1={}, l2={}, tv={}'.format(l1, l2, tv))
print('Best grid search auc: {}'.format(np.max(gs_auc)))

# L1L2TV #
gs_YZ_ML = LogisticRegressionL1L2TV(l1, l2, tv, A, algorithm=StaticCONESTA(max_iter=1000))
trained_gs_YZ_ML = gs_YZ_ML.fit(X,y)
gs_coef = trained_gs_YZ_ML.parameters()['beta'].reshape(-1,).astype(np.float32)
#print('gs_coefficients range: ({},{})'.format(gs_coef.min(), gs_coef.max()))
f_n = '{}_{}_{}_gs_coef.raw'.format(l1, l2, tv)
gs_coef.tofile(f_n)

# L1 #
penalty = float(l1)
gs_YZ_ML = LogisticRegression(penalty='l1', C=penalty, max_iter=1000)
trained_gs_YZ_ML = gs_YZ_ML.fit(X,y)
gs_coef = trained_gs_YZ_ML.coef_.astype(np.float32)
#print('l1_gs_coefficients range: ({},{})'.format(gs_coef.min(), gs_coef.max()))
f_n = 'skl1_' + str(penalty) + '_gs_coef.raw'
#gs_coef.tofile(f_n)

# L2 #
penalty = float(l2)
gs_YZ_ML = LogisticRegression(penalty='l2', C=penalty, max_iter=1000)
trained_gs_YZ_ML = gs_YZ_ML.fit(X,y)
gs_coef = trained_gs_YZ_ML.coef_.astype(np.float32)
#print('l2_gs_coefficients range: ({},{})'.format(gs_coef.min(), gs_coef.max()))
f_n = 'skl2_' + str(penalty) + '_gs_coef.raw'
#gs_coef.tofile(f_n)

t_end = time.time()
#print('gs processing time: {}s'.format(np.int((t_end - t_start))))




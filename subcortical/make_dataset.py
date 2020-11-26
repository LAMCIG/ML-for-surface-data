import numpy as np
import time
#import os
import sys
#import time
from concon_utils import load_raw_labels_boris
import pandas as pd
from sklearn.preprocessing import StandardScaler
root = sys.argv[1]
#atlas = sys.argv[2]
def subject_i(path):
    subject_loc = path
    subject_vector = []
    #for region in [atlas]:#, '11', '12', '13', '17', '18', '26', '49', '50', '51', '52', '53', '54', '58']:
    for region in ['10', '11', '12', '13', '17', '18', '26', '49', '50', '51', '52', '53', '54', '58']:
        region_id = subject_loc + '/thick_' + region + '.raw'
        #print(region_id)
        features = load_raw_labels_boris(region_id)
        #print(features, features.size)
        subject_vector= np.append(subject_vector, features, axis=0)
        #print(subject_vector, subject_vector.size)
    #print(subject_vector, subject_vector.size)
    return subject_vector

## dataset X ##
X_train = pd.read_csv('X_train_subject.csv')
X_train_dataset = []
for path in X_train['subject']:
    path = root + '/' + str(path)
    #print(path)
    X_train_dataset.append(list(subject_i(path)))
X_train_dataset = np.array(X_train_dataset)
scaler = StandardScaler()
scaled_X_train_dataset = scaler.fit(X_train_dataset).transform(X_train_dataset)
#print('X_train: {}'.format(X_train_dataset.shape))
np.save('scaled_X_train_dataset.npy', scaled_X_train_dataset)

X_test = pd.read_csv('X_test_subject.csv')
X_test_dataset = []
for path in X_test['subject']:
    path = root + '/' + str(path)
    #print(path)
    X_test_dataset.append(list(subject_i(path)))
X_test_dataset = np.array(X_test_dataset)
scaled_X_test_dataset = scaler.fit(X_test_dataset).transform(X_test_dataset)
#print('X_test: {}'.format(X_test_dataset.shape))
np.save('scaled_X_test_dataset.npy', scaled_X_test_dataset)

y_train_dataset = pd.read_csv('y_train_target.csv')
y_train_dataset = np.array(y_train_dataset)
#print('Train: {}'.format(y_train_dataset.shape))
np.save('y_train_dataset.npy', y_train_dataset)

y_test_dataset = pd.read_csv('y_test_target.csv')
y_test_dataset = np.array(y_test_dataset)
#print('Test: {}'.format(y_test_dataset.shape))
np.save('y_test_dataset.npy', y_test_dataset)


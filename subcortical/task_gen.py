import os
import sys
#import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
data_info = str(sys.argv[2])
disorder_label = str(sys.argv[3])
control_label = str(sys.argv[4])

task = pd.read_csv(data_info)
task = pd.DataFrame({'subject': task['Subject'], 'target': task['dx']})
t_disorder = task.loc[task['target'] == disorder_label]
t_control = task.loc[task['target'] == control_label]
task = pd.concat([t_disorder, t_control])
#print(task[task['target'] == 'N'])    #slect row
#print(task)
#exit(0)
enc = preprocessing.LabelEncoder()
labels = [disorder_label, control_label]
print('task: {}'.format(labels))
enc.fit(labels)
target_code = enc.transform(task['target'])
#print(target_code)
task = pd.DataFrame({'subject': task['subject'], 'target_code': target_code})
print('spreadsheet: {}'.format(len(task)))
#### Relate spread sheet to actual data folder
path = sys.argv[1]
cohort = os.listdir(path)
for subject in task['subject']:    # Note that pandas implicitly treat the file name as numbers
    if str(subject) not in cohort:      # However the bash I/O is always strings.
        task = task.drop(task[task['subject'] == subject].index, axis=0)
print('actual cohort: {}'.format(len(task)))
#print('actual distribution: PD/Control {}/{}'.format(len(task[task['target_code'] == 1]), len(task[task['target_code'] == 0])))
#### default test size: 0.25
X_train, X_test, y_train, y_test = train_test_split(task['subject'], task['target_code'])
X_train.to_csv('X_train_subject.csv', index=False, header=True)
#print('X_train: {}'.format(len(X_train)))
y_train.to_csv('y_train_target.csv', index=False, header=True)
print('Train: {}'.format(len(y_train)))
#print(y_train[y_train == 1].sum())                       # selcet row in y_train, a pd.series
#print('actual training distribution: PD/Control : {}/{}'.format(len(y_train[y_train == 1]), len(y_train[y_train == 0])))
X_test.to_csv('X_test_subject.csv', index=False, header=True)
#print('X_test: {}'.format(len(X_test)))
y_test.to_csv('y_test_target.csv', index=False, header=True)
print('Test: {}'.format(len(y_test)))
#print('actual testing distribution: PD/Control : {}/{}'.format(len(y_test[y_test == 1]), len(y_test[y_test == 0])))




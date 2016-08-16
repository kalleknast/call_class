# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 19:58:35 2016

@author: hjalmar
"""

import numpy as np
from glob import glob
from call_class_helper import *

# data_dir
data_dir = '/home/hjalmar/Dropbox/shared_calls/Vocalizations'
sub_dirs = glob(data_dir + '/*')
call_fnames = {}
N = 0
for sub_dir in glob(data_dir + '/*'):
    call_type = sub_dir.split('/')[-1]
    call_fnames[call_type] = glob(sub_dir + '/*.wav')
    N += len(call_fnames[call_type])
    

if not len(glob('call_class_features.npz')):
    X, y, codebook, reverse_codebook = get_data(call_fnames, N)
    np.savez('call_class_features.npz', X=X, y=y,
             codebook=codebook, reverse_codebook=reverse_codebook)
else:
    with np.load('call_class_features.npz') as features:
        X = features['X']
        y = features['y']
        codebook = features['codebook'].tolist()
        reverse_codebook = features['reverse_codebook'].tolist()
        
##########################
# Table 2 and Figure 2:
##########################
# Merge phee_2, phee_3 and phee_4        
drop = True
mrg_X, mrg_y, mrg_reverse_codebook, mrg_codebook = merge_calltypes(X, y, reverse_codebook, codebook,
                                              calltypes2merge=['phee_2', 'phee_3', 'phee_4'],
                                              merged_calltype='phee',
                                              drop=drop)

# Merge tsik and tsik_ek
mrg_X, mrg_y, mrg_reverse_codebook, mrg_codebook = merge_calltypes(mrg_X, mrg_y, mrg_reverse_codebook, mrg_codebook,
                                              calltypes2merge=['tsik_ek', 'tsik'],
                                              merged_calltype='tsik',
                                              drop=drop)

if not len(glob('call_class_results.npz')):
    results = get_results(mrg_X, mrg_y, mrg_reverse_codebook, n_rep=100)
    np.savez('call_class_results.npz', results=results)
else:
    with np.load('call_class_results.npz') as results:
        results = results['results'].tolist()

print_table2(results)
plot_figure2(results)

confuse_mat1 = get_confusion_matrix(mrg_X, mrg_y, mrg_reverse_codebook, n_rep=100)
print_table3(confuse_mat1, mrg_reverse_codebook)

phee_cbook = {}
phee_rcbook = {}
bix = np.zeros(y.shape[0], dtype=bool)
for call_type in reverse_codebook:
    if 'phee' in call_type:
        code = reverse_codebook[call_type]
        phee_cbook[code] = call_type
        phee_rcbook[call_type] = code
        bix = np.logical_or(bix, y == code)  # and or, not exclusive or
y_phee = y[bix]
X_phee = X[bix]
confuse_mat2 = get_confusion_matrix(X_phee, y_phee, phee_rcbook, n_rep=100)
print_table4(confuse_mat2, phee_rcbook)


tsik_cbook = {}
tsik_rcbook = {}
bix = np.zeros(y.shape[0], dtype=bool)
for call_type in reverse_codebook:
    if 'tsik' in call_type:
        code = reverse_codebook[call_type]
        tsik_cbook[code] = call_type
        tsik_rcbook[call_type] = code
        bix = np.logical_or(bix, y == code)  # and or, not exclusive or
y_tsik = y[bix]
X_tsik = X[bix]
confuse_mat3 = get_confusion_matrix(X_tsik, y_tsik, tsik_rcbook, n_rep=100)
print_table5(confuse_mat3, tsik_rcbook)



    
    

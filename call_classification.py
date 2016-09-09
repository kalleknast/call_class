# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 19:58:35 2016

@author: hjalmar
"""

import numpy as np
from glob import glob
from call_class_helper import *

fname = maybe_download()
data_dir = maybe_extract_calls(fname)

sub_dirs = glob(data_dir + '/*')
call_fnames = {}
N = 0
for sub_dir in glob(data_dir + '/*'):
    call_type = sub_dir.split('/')[-1]
    call_fnames[call_type] = glob(sub_dir + '/*.wav')
    N += len(call_fnames[call_type])
    

if not len(glob('call_class_features.npz')):
    X, y, codebook = get_data(call_fnames, N)
    np.savez('call_class_features.npz', X=X, y=y, codebook=codebook)
else:
    with np.load('call_class_features.npz') as tmp:
        X = tmp['X']
        y = tmp['y']
        codebook = tmp['codebook'].tolist()

        
##########################
# Table 2 and Figure 2:
##########################
# Merge phee_2, phee_3 and phee_4        
drop = True
X_mrg, y_mrg, codebook_mrg = merge_calltypes(X, y, codebook,
                                             calltypes2merge=['phee_2',
                                                              'phee_3',
                                                              'phee_4'],
                                             merged_calltype='phee',
                                             drop=drop)

# Merge tsik and tsik_ek
X_mrg, y_mrg, codebook_mrg = merge_calltypes(X_mrg, y_mrg, codebook_mrg,
                                             calltypes2merge=['tsik_ek',
                                                              'tsik'],
                                             merged_calltype='tsik',
                                             drop=drop)

if not len(glob('call_class_results.npz')):
    results = get_results(X_mrg, y_mrg, n_rep=100)
    np.savez('call_class_results.npz', results=results, codebook_mrg=codebook_mrg)
else:
    with np.load('call_class_results.npz') as tmp:
        results = tmp['results'].tolist()
        codebook_mrg = tmp['codebook_mrg'].tolist()

print_table2(results)
plot_figure2(results)

confuse_mat1 = get_confusion_matrix(X_mrg, y_mrg, codebook_mrg, n_rep=100)
print_table3(confuse_mat1, codebook_mrg)

phee_cbook = {}
bix = np.zeros(y.shape[0], dtype=bool)
for call_type in codebook:
    if 'phee' in call_type:
        code = codebook[call_type]
        phee_cbook[call_type] = code
        bix = np.logical_or(bix, y == code)  # and or, not exclusive or
y_phee = y[bix]
X_phee = X[bix]
confuse_mat2 = get_confusion_matrix(X_phee, y_phee, phee_cbook, n_rep=100)
print_table4(confuse_mat2, phee_cbook)


tsik_cbook = {}
bix = np.zeros(y.shape[0], dtype=bool)
for call_type in codebook:
    if 'tsik' in call_type:
        code = codebook[call_type]
        tsik_cbook[call_type] = code
        bix = np.logical_or(bix, y == code)  # and or, not exclusive or
y_tsik = y[bix]
X_tsik = X[bix]
confuse_mat3 = get_confusion_matrix(X_tsik, y_tsik, tsik_cbook, n_rep=100)
print_table5(confuse_mat3, tsik_cbook)




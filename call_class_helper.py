# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 19:54:52 2016

@author: hjalmar
"""
import timeit
import numpy as np
from scipy.io import wavfile
from libopf_py import OPF
from audiolazy import lazy_lpc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB#BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ion()
import pdb


def accuracy(y_predict, y_true, nClass=8):
    """
    Equations 4-7 in paper
    
    1st published in 
    J. P. Papa, A. X. Falc &#771;ao, and C. T. N. Suzuki,
    “Supervised pattern classification based on optimum-path forest,”
    International Journal of Imaging Systems and Technology,
    vol. 19, no. 2, pp. 120–131, 2009.
    """

    N = y_true.shape[0]
    classes = np.unique(y_true)
    e1 = np.zeros(nClass)
    e2 = np.zeros(nClass)
    
    for i, c in enumerate(classes):
        c_in_predict = (y_predict == c)
        c_in_true = (y_true == c)
        FP = np.logical_and(c_in_predict, ~ c_in_true).sum()
        FN = np.logical_and(c_in_true, ~ c_in_predict).sum()
        Nc = (y_true == c).sum()
        e1[i] = FP / (N - Nc)
        e2[i] = FN / Nc
        
    acc = 1 - (e1 + e2).sum() / nClass

    return acc

        
def get_data(call_fnames, N):
    """
    """
    order = 20
    X = np.empty((N, order))
    y = np.empty(N)
    
    codes = {}
    codebook = {}
    
    i = 0
    code = 0
    print('Extracting features...')
    for call_type in call_fnames:
        codes[call_type] = code
        for fn in call_fnames[call_type]:
            print(N-i)
            fs, signal = wavfile.read(fn)
            signal = signal.astype(float) / (2**16/2.)   # from int16 to float
            x = lazy_lpc.lpc.kautocor(signal, order)
            X[i] = x.numerator[1:]
            y[i] = codes[call_type]
            codebook[call_type] = code
            i += 1

        code += 1
    
    return X, y, codebook
    
    
def merge_calltypes(X, y, codebook,
                    calltypes2merge=['phee_2', 'phee_3', 'phee_4'],
                    merged_calltype=None, drop=True):
    """
    """

    if merged_calltype is None:
        merged_calltype = calltypes2merge[0]

    mergecode = codebook[calltypes2merge[0]]
    merged_y = y.copy()
    merged_X = X.copy()
    merged_codebook = codebook.copy()
    rev_codebook = {codebook[ct]: ct for ct in codebook}
    
    for calltype in calltypes2merge:
        code = codebook[calltype]
        merged_y[y == code] = mergecode
        merged_codebook.pop(calltype)
        merged_codebook[merged_calltype] = mergecode
        rev_codebook.pop(code)
        rev_codebook[mergecode] = merged_calltype
                     
    codes = np.unique(merged_y)
    for i, code in enumerate(codes):
        merged_y[merged_y == code] = i
        calltype = rev_codebook[int(code)]
        merged_codebook[calltype] = i
                        
    if drop:
        merged_code = merged_codebook[merged_calltype]
        merged_ix = (merged_y == merged_code).nonzero()[0]
        np.random.shuffle(merged_ix)
        keep_ix = (merged_y != merged_code).nonzero()[0]
        keep_ix = np.r_[keep_ix, merged_ix[:30]]
        merged_y = merged_y[keep_ix]
        merged_X = merged_X[keep_ix]
       
    return merged_X, merged_y, merged_codebook
    
    
def get_kNN_params(X, y):
    
    nrep = 10
    N = X.shape[0]
    
    k_range = range(1, N//2, 1)
    scores = np.empty((nrep, len(k_range)))
    
    for rep in range(nrep):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.1)
    
        for i, k in enumerate(k_range):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            scores[rep, i] = knn.score(X_cv, y_cv)
        
    scores = scores.mean(axis=0)

    return k_range[scores.argmax()]

    
def get_SVM_params(X, y, kernel='rbf'):
    
    C_range = 2. ** np.arange(-5, 16, 2)
    gamma_range = 2. ** np.arange(3, -16, -2)
    #  C_range = np.arange(2, 15, 2) Danillo strategy
    #  gamma_range = np.arange(0, 5, 2) Danillo strategy
    param_grid = dict(gamma=gamma_range, C=C_range)
        
    svr = SVC(kernel=kernel)
    cv = StratifiedKFold(y, n_folds=5)
    grid = GridSearchCV(svr, param_grid=param_grid, cv=cv)
    grid.fit(X,y)
    
    # Get the C and gamma parameters
    C, gamma = grid.best_params_['C'], grid.best_params_['gamma'] 

    return C, gamma
    
    
def get_MLP_arch(X, y):
    
    nrep = 10
    
    hidden_layers = [(4), (8), (16),
                     (4, 4), (4, 8), (4, 16),
                     (8, 4), (8, 8), (8, 16),
                     (16, 4), (16, 8), (16, 16),
                     (8, 8, 4), (8, 8, 8), (8, 16, 8)]
    
    scores = np.empty((nrep, len(hidden_layers)))
    
    for rep in range(nrep):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.1)
        for i, h_layers in enumerate(hidden_layers):
            mlp = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=h_layers)
            mlp.fit(X_train, y_train)
            scores[rep, i] = mlp.score(X_cv, y_cv)

    scores = scores.mean(axis=0)
    return hidden_layers[scores.argmax()]
    
    
def get_adaBoost_params(X, y):
    
    nrep = 10
    
    params = [(5, 0.25), (5, 0.5), (5, 1.0), (5, 2.0),
              (10, 0.25), (10, 0.5), (10, 1.0), (10, 2.0),
              (20, 0.25), (20, 0.5), (20, 1.0), (20, 2.0),
              (40, 0.25), (40, 0.5), (40, 1.0), (40, 2.0),
              (80, 0.25), (80, 0.5), (80, 1.0), (80, 2.0),
              (160, 0.25), (160, 0.5), (160, 1.0), (160, 2.0),
              (320, 0.25), (320, 0.5), (320, 1.0), (320, 2.0),]
              
    scores = np.empty((nrep, len(params)))
    
    for rep in range(nrep):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.1)
        for i, prms in enumerate(params):
            ada = AdaBoostClassifier(n_estimators=prms[0], learning_rate=prms[1])
            ada.fit(X_train, y_train)
            scores[rep, i] = ada.score(X_cv, y_cv)

    scores = scores.mean(axis=0)
    return params[scores.argmax()]              
    

def get_results(X, y, n_rep=100):
    """
    """
    ## Model hyper parameters
    #-----------------------
    # Set to None to recompute
    # SVM hyperparameters
    C = 8 # C = 32.0
    gamma = 0.5 #0.00195 #gamma = 0.03125
    # k-NN -- num nearest neighbors
    k = 1 # k = None
    # MLP architecture
    hidden_layers = (8, 16)  #hidden_layers = (8, 8)
    # AdaBoost params, n_estimators & learning_rate
    ada_params = (160, 0.5)  #
    
    if C is None or gamma is None:
        print('Getting SVM parameters...')
        C, gamma = get_SVM_params(X, y, kernel='rbf')
        print('C: %1.5f, gamma: %1.5f' % (C, gamma))
        
    if k is None:
        print('Getting k-NN parameters...')
        k = get_kNN_params(X, y)
        print('k: %d' % k)
        
    if hidden_layers is None:
        print('Searching MLP architectures...')
        hidden_layers = get_MLP_arch(X, y)
        print('hidden_layers: ', hidden_layers)
        
    if ada_params is None:
        print('Searching AdaBoost params...')
        ada_params = get_adaBoost_params(X, y)
        print('ada_params: ', ada_params)
        

    splits = np.arange(0.1, 0.91, 0.1)
    n_split = splits.shape[0]        
                
    svm = SVC(C=C, gamma=gamma)
    knn = KNeighborsClassifier(n_neighbors=k)
    bayes = GaussianNB()
    mlp = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=hidden_layers)
    ada = AdaBoostClassifier(n_estimators=ada_params[0], learning_rate=ada_params[1])
    logits = LogisticRegression(multi_class='ovr')
    opf_ec = OPF()
    opf_mh = OPF()
    opf_ca = OPF()
    opf_cs = OPF()    
    opf_bc = OPF()
         
    acc = np.recarray((n_rep, n_split), dtype=[('SVM', float),
                                               ('k-NN', float),
                                               ('Naive Bayes', float),
                                               ('MLP', float),
                                               ('AdaBoost', float),
                                               ('OPF -- Manhattan', float),
                                               ('OPF -- Euclidean', float),
                                               ('OPF -- Canberra', float),
                                               ('OPF -- Chi-Square', float),
                                               ('OPF -- Bray Curtis', float),
                                               ('Logistic Regression', float)])    
    f1 = acc.copy()
    time = {d: np.nan  for d in acc.dtype.fields}
    M_acc = {d: np.nan  for d in acc.dtype.fields}
    SE_acc = {d: np.nan  for d in acc.dtype.fields}
    M_f1 = {d: np.nan  for d in acc.dtype.fields}
    SE_f1 = {d: np.nan  for d in acc.dtype.fields}
        
    for i, split in enumerate(splits):
        for rep in range(n_rep):

            # Split data
            X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=split)

            # Train
            svm.fit(X_train, y_train)
            knn.fit(X_train, y_train)
            bayes.fit(X_train, y_train)
            mlp.fit(X_train, y_train)
            ada.fit(X_train, y_train)
            logits.fit(X_train, y_train)
            opf_ec.fit(X_train, y_train.astype(np.int32), metric="euclidian")
            opf_mh.fit(X_train, y_train.astype(np.int32), metric="manhattan")
            opf_ca.fit(X_train, y_train.astype(np.int32), metric="canberra")
            opf_cs.fit(X_train, y_train.astype(np.int32), metric="chi_square")
            opf_bc.fit(X_train, y_train.astype(np.int32), metric="bray_curtis")
            
            # Evaluate, accuracies
            acc[rep, i]['k-NN'] = accuracy(knn.predict(X_cv), y_cv)
            acc[rep, i]['SVM'] = accuracy(svm.predict(X_cv), y_cv)
            acc[rep, i]['Naive Bayes'] = accuracy(bayes.predict(X_cv), y_cv)
            acc[rep, i]['MLP'] = accuracy(mlp.predict(X_cv), y_cv)
            acc[rep, i]['AdaBoost'] = accuracy(ada.predict(X_cv), y_cv)
            acc[rep, i]['Logistic Regression'] = accuracy(logits.predict(X_cv), y_cv)
            acc[rep, i]['OPF -- Euclidean'] = accuracy(opf_ec.predict(X_cv), y_cv)
            acc[rep, i]['OPF -- Manhattan'] = accuracy(opf_mh.predict(X_cv), y_cv)
            acc[rep, i]['OPF -- Canberra'] = accuracy(opf_ca.predict(X_cv), y_cv)
            acc[rep, i]['OPF -- Chi-Square'] = accuracy(opf_cs.predict(X_cv), y_cv)
            acc[rep, i]['OPF -- Bray Curtis'] = accuracy(opf_bc.predict(X_cv), y_cv)
            
            # Evaluate, f1 scores
            f1[rep, i]['k-NN'] = f1_score(y_cv, knn.predict(X_cv), average='weighted')
            f1[rep, i]['SVM'] = f1_score(y_cv, svm.predict(X_cv), average='weighted')
            f1[rep, i]['Naive Bayes'] = f1_score(y_cv, bayes.predict(X_cv), average='weighted')
            f1[rep, i]['MLP'] = f1_score(y_cv, mlp.predict(X_cv), average='weighted')
            f1[rep, i]['AdaBoost'] = f1_score(y_cv, ada.predict(X_cv), average='weighted')
            f1[rep, i]['Logistic Regression'] = f1_score(y_cv, logits.predict(X_cv), average='weighted')
            f1[rep, i]['OPF -- Euclidean'] = f1_score(y_cv, opf_ec.predict(X_cv), average='weighted')
            f1[rep, i]['OPF -- Manhattan'] = f1_score(y_cv, opf_mh.predict(X_cv), average='weighted')
            f1[rep, i]['OPF -- Canberra'] = f1_score(y_cv, opf_ca.predict(X_cv), average='weighted')
            f1[rep, i]['OPF -- Chi-Square'] = f1_score(y_cv, opf_cs.predict(X_cv), average='weighted')
            f1[rep, i]['OPF -- Bray Curtis'] = f1_score(y_cv, opf_bc.predict(X_cv), average='weighted')            
            
    # Prediction times.
    # A wrapper to provide the proper inputs to timeit.
    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    n = 1000
    
    time['k-NN'] = 1000*timeit.timeit(wrapper(knn.predict, X[0]), number=n)
    time['SVM'] = 1000*timeit.timeit(wrapper(svm.predict, X[0]), number=n)
    time['Naive Bayes'] = 1000*timeit.timeit(wrapper(bayes.predict, X[0]), number=n)
    time['MLP'] = 1000*timeit.timeit(wrapper(mlp.predict, X[0]), number=n)
    time['AdaBoost'] = 1000*timeit.timeit(wrapper(ada.predict, X[0]), number=n)
    time['Logistic Regression'] = 1000*timeit.timeit(wrapper(logits.predict, X[0]), number=n)
    time['OPF -- Euclidean'] = 1000*timeit.timeit(wrapper(opf_ec.predict, X[0].reshape((1,20))), number=n)
    time['OPF -- Manhattan'] = 1000*timeit.timeit(wrapper(opf_mh.predict, X[0].reshape((1,20))), number=n)
    time['OPF -- Canberra'] = 1000*timeit.timeit(wrapper(opf_ca.predict, X[0].reshape((1,20))), number=n)
    time['OPF -- Chi-Square'] = 1000*timeit.timeit(wrapper(opf_cs.predict, X[0].reshape((1,20))), number=n)
    time['OPF -- Bray Curtis'] = 1000*timeit.timeit(wrapper(opf_bc.predict, X[0].reshape((1,20))), number=n)
    
    # Get averages for table 2
    for method in acc.dtype.fields:
        M_acc[method] = np.nanmean(acc[method], axis=0)
        SE_acc[method] = np.nanstd(acc[method], axis=0) / np.sqrt((~ np.isnan(acc[method])).sum(axis=0))
        M_f1[method] = np.nanmean(f1[method], axis=0)
        SE_f1[method] = np.nanstd(f1[method], axis=0) / np.sqrt((~ np.isnan(f1[method])).sum(axis=0))
                        
    results = {'acc': acc, 'M_acc': M_acc, 'SE_acc': SE_acc,
               'f1': f1, 'M_f1': M_f1, 'SE_f1': SE_f1,
               'time': time, 'splits': splits}
               
    return results
    
    
def get_confusion_matrix(X, y, codebook, n_rep=30):
    """
    """
    confuse_mat = {call_type: {} for call_type in codebook}
    opf = OPF()
    classes = np.unique(y)
    
    for rep in range(n_rep):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=0.8)
        opf.fit(X_train, y_train.astype(np.int32), metric="euclidian")
    
        for call_type in codebook:
            code = codebook[call_type]
            bix = (y_cv == code)
            y_pred = opf.predict(X_cv[bix])
            for ctype in codebook:
                cde = codebook[ctype]
                if not ctype in confuse_mat[call_type]:
                    confuse_mat[call_type][ctype] = []
                else:
                    
                    pct = 100*(y_pred==cde).sum()/(y_cv==cde).sum()
                    #if np.isnan(pct):
                     #   pdb.set_trace()
                    #else:
                    confuse_mat[call_type][ctype].append(pct)
                            
    # Normalize the confusion matrix
    for call_type in codebook:
        code = codebook[call_type]
        s = 0.0
        for ctype in codebook:
            m = np.nanmean(confuse_mat[call_type][ctype])
            s += m
            confuse_mat[call_type][ctype] = m
        # they might not sum up to 100 due to the sampling.
        # -> make sure that they do.
        for ctype in codebook:            
            confuse_mat[call_type][ctype] *= 100./s
            
    return confuse_mat
    

def print_table3(data, codebook):
    
    call_types = list(codebook.keys())
    call_types.sort()
    
    print('ROT 90 DEGREES COMPARED TO TABLE 3!!!')
    print('  Table 3')
    print('%-35s\t\t\t' % 'Classified as[%]')
    print('\t    | %11s| %11s| %11s| %11s| %11s| %11s| %11s| %11s' % tuple(call_types))
    for call_type in call_types:
        print('%11s ' % call_type, end="")
        for ct in call_types:
            print("|    %1.1f     " % data[call_type][ct], end="")
        print('')


def print_table4(data, codebook):
    
    call_types = list(codebook.keys())
    call_types.sort()
    
    print('ROT 90 DEGREES COMPARED TO TABLE 3!!!')
    print('  Table 4')
    print('%-35s\t\t\t' % 'Classified as[%]')
    print('\t    | %11s | %11s | %10s' % tuple(call_types))
    for call_type in call_types:
        print('%11s ' % call_type, end="")
        for ct in call_types:
            print("|    %1.1f     " % data[call_type][ct], end="")
        print('')


def print_table5(data, codebook):
    
    call_types = list(codebook.keys())
    call_types.sort()
    
    print('ROT 90 DEGREES COMPARED TO TABLE 3!!!')
    print('  Table 4')
    print('%-35s\t\t\t' % 'Classified as[%]')
    print('\t    | %7s     | %9s ' % tuple(call_types))
    for call_type in call_types:
        print('%11s ' % call_type, end="")
        for ct in call_types:
            print("|    %1.1f     " % data[call_type][ct], end="")
        print('')
            
            
def print_table2(results):
    
    M_acc = results['M_acc']
    SE_acc = results['SE_acc']
    M_f1 = results['M_f1']
    SE_f1 = results['SE_f1']    
    
    print('  Table 2')
    print('%20s |    F1-score   |   Accuracy    |   Time  ' % ('Method'))
    print('%20s |  Mean |  SEM  |  Mean |  SEM  |   Mean  ' % (''))
    print('   ', '-'*58)
    for method in M_acc.keys():
        print('%20s | %1.3f | %1.3f | %1.3f | %1.3f | %1.3f ' %
              (method, M_f1[method][-1], SE_f1[method][-1],
               M_acc[method][-1], SE_acc[method][-1], results['time'][method]))

    
def plot_figure2(results):
    """
    """
    splits = results['splits']
    M = results['M_acc']
    SE = results['SE_acc']

    linespec = {}
    linespec['k-NN'] = {'mfc': 'r', 'ms': 'o', 'ls': '-', 'lw':2, 'lc': 'r'}
    linespec['SVM'] = {'mfc': 'y', 'ms': 'o', 'ls': '-', 'lw':2, 'lc': 'y'}
    linespec['MLP'] = {'mfc': 'g', 'ms': 'o', 'ls': '-', 'lw':2, 'lc': 'g'}
    linespec['Logistic Regression'] = {'mfc': [246/255., 0, 1.], 'ms': 'o', 'ls': '-', 'lw':2, 'lc': [246/255., 0, 1.]}
    linespec['AdaBoost'] = {'mfc': [1., 126/255, 0.], 'ms': 'o', 'ls': '-', 'lw':2, 'lc': [1., 126/255, 0.]}
    linespec['OPF -- Euclidean'] = {'mfc': 'k', 'ms': 's', 'ls': '-', 'lw':2, 'lc': 'k'}
    linespec['OPF -- Manhattan'] = {'mfc': 'w', 'ms': 'o', 'ls': '--', 'lw':2, 'lc': 'k'}
    linespec['OPF -- Canberra'] = {'mfc': 'k', 'ms': '^', 'ls': ':', 'lw':2, 'lc': 'k'}

    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    methods = list(M.keys())
    methods.remove('OPF -- Chi-Square')
    methods.remove('OPF -- Bray Curtis')
    methods.remove('Naive Bayes')
    for i, method in enumerate(methods):
        ls = linespec[method]
        ax.errorbar(splits, M[method], SE[method], linestyle=ls['ls'],
                    marker=ls['ms'], color=ls['lc'], label=method, 
                    markerfacecolor=ls['mfc'], elinewidth=1.5, capsize=0,
                    capthick=False)
            
    ax.set_xlabel('Fraction of dataset', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.set_ylim([0.125,1])
    
    ax.set_xlim([splits.min() - 0.01, splits.max() + 0.01])
    ax.legend(loc=[0.75, 0.0], frameon=False, fontsize=11)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])            
                    
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')    
    ax.set_position([0.125, 0.125, 0.8, 0.8])
    fig.set_facecolor('w')
    fig.savefig('Fig2.png')
    fig.savefig('Fig2.svg')
    fig.savefig('Fig2.eps')
    plt.show()                    
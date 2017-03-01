# Call classification

A script to produce the results of the paper
[Machine Learning Algorithms for Automatic Classification of Marmoset Vocalizations](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0163041)
by Hjalmar K. Turesson, Sidarta Ribeiro, Danillo R. Pereira, João P. Papa & Victor Hugo C. de Albuquerque
Published in Plos One 2016.

## Paper abstract
Automatic classification of vocalization type could potentially become a useful tool for acoustic the monitoring of captive colonies of highly vocal primates. However, for classification to be useful in practice, a reliable algorithm that can be successfully trained on small datasets is necessary. In this work, we consider seven different classification algorithms with the goal of finding a robust classifier that can be successfully trained on small datasets. We found good classification performance (accuracy > 0.83 and F 1 -score > 0.84) using the Optimum Path Forest classifier. Dataset and algorithms are made publicly available.

## To run
Make sure ```numpy```, ```scipy```, ```audiolazy```, ```libopf_py```, ```sklearn``` and ```matplotlib``` are installed.

Get ```libopf_py``` from https://github.com/LibOPF/LibOPF.

From ipython run:
```python
%run call_classification.py
```

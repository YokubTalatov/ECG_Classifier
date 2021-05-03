# -*- coding: utf-8 -*-
"""
DIAGNOSTICS OF THE CARDIOVASCULAR SYSTEM BASED ON NEURAL NETWORKS
Classify ECG-Signals to CAD-Symptoms by means of artificial neural networks

Skript to design and train MLP classifier.
"""
# %% Imports and Settings
import os
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.decomposition import PCA

from Skripts.HelperFunctions import *

#%% Loading Dataset
datasets = ['mit-bih-normal-sinus-rhythm-database-1.0.0','mit-bih-arrhythmia-database-1.0.0', 'RohmineEcgDatabase']    
datasetname = datasets[1]
rootPath = 'C:/Users/Yokub/Documents/Study/IAESTE/ECGNetwork'
featurePath = os.path.join(rootPath, "Results_" + datasetname, 'features_labeled.csv')
ecg_dataset = pd.read_csv(featurePath, header = 0, index_col = 0)
ecg_dataset.reset_index(drop=True, inplace = True)

#%% Concating Datasets
datasetname1 = datasets[1]
datasetname2 = datasets[2]
rootPath = 'C:/Users/Yokub/Documents/Study/IAESTE/ECGNetwork'
featurePath1 = os.path.join(rootPath, "Results_" + datasetname1, 'features_labeled.csv')
featurePath2 = os.path.join(rootPath, "Results_" + datasetname2, 'features_labeled.csv')
dataset1 = pd.read_csv(featurePath1, header = 0, index_col = 0)
dataset1.reset_index(drop=True, inplace = True)
dataset2 = pd.read_csv(featurePath2, header = 0, index_col = 0)
dataset2.reset_index(drop=True, inplace = True)

ecg_dataset = pd.concat([dataset1, dataset2], ignore_index= True)

#%% Split to data and labels
feature_names = ["AVNN","SDNN","RMSSD" ,"SKEW" ,"KURT","MO","AMO","M0","R1","DELp","DEL0","DELn"]
X = ecg_dataset.drop(['Label'], axis = 1)
#X = ecg_dataset[feature_names]
y = ecg_dataset['Label']

# Split to Train and Test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size = 0.3, stratify = y, random_state=42)

#%% Plotting Distribution of classes in Dataset
plot_class_distribution(ecg_dataset,y_train, y_test)

#%% Creating Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = 9)), ('classifier',  MLPClassifier(solver = 'lbfgs', activation = 'relu',hidden_layer_sizes = 8,random_state = 42, max_iter = 500))])

#%% possiible parameters for grid_search
solver_param = ['lbfgs', 'sgd'] 
activation_param = ['relu', 'tanh', 'logistic']
alpha = [0.0001, 0.001, 0.01, 0.1, 1]
pca_comp = [4,5,6,7,8,9,10,11]
learning_rate_param = ['invscaling', 'adaptive']
hidden_layer_size = [4,5,6,7,8,9,10, 20, 50, 100]


#%% Optimise parameters via gridsearch
param_grid = [{'classifier__solver': ['lbfgs'], 'pca__n_components' : pca_comp, 'classifier__activation': activation_param, 'classifier__hidden_layer_sizes': hidden_layer_size, 'classifier__alpha': alpha}, {'classifier__solver': ['sgd'], 'pca__n_components' : pca_comp, 'classifier__activation': activation_param, 'classifier__hidden_layer_sizes': hidden_layer_size, 'classifier__alpha': alpha, 'classifier__learning_rate': learning_rate_param}]
kfold = StratifiedKFold(n_splits= 5, shuffle = True, random_state =0)
grid = GridSearchCV(pipe, param_grid, cv = kfold, scoring = 'f1_weighted', n_jobs = -1) # 
grid.fit(X_train, y_train)
model = grid.best_estimator_


#%% Printing Performance Evaluation
print(f'Evaluation of preferred model:')
print(f'Mean score of the grid search crossvalidation:\n\t {grid.best_score_}')
print(f'Accuracy Score on test dataset:\n\t {accuracy_score(y_test, grid.predict(X_test))}')
print(f'F1 Score on test dataset: \n\t {f1_score(y_test, grid.predict(X_test), average="weighted")}')
print('Confusion matrix:')
print(confusion_matrix(y_test,grid.predict(X_test)))
print('Classification report:')
print(classification_report(y_test,grid.predict(X_test)))

#%% Saving the Estimator
# save the model
model_filename = 'model_all_parameters'
modelPath = os.path.join(rootPath, model_filename + '.joblib')
dump(grid, modelPath)

# save the results
results = pd.DataFrame(grid.cv_results_)
results_filename = 'results_' + model_filename + '.csv'
resultsdfPath = os.path.join(rootPath, results_filename)
results.to_csv(resultsdfPath)
 

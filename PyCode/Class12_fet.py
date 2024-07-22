#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import scipy 
from scipy import io
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

import ewt
from ewt.boundaries import *
import ewt.ewt2d

import time
start_time=time.time()

import antropy as ant

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain

from scipy.signal import butter,filtfilt
import plotly.graph_objects as go  # Import the go module

from scipy.stats import skew, kurtosis, entropy, pearsonr
from scipy.io import loadmat
from scipy import signal

from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances


# # Class1

# In[3]:


mat1 = result_mat = scipy.io.loadmat("D:\\Users\\sugam patel\\Downloads\\only_slot1_all\\Class1_10_all.mat")


# In[11]:


Class1 = mat1['Class1_filtered'] 
Class1.shape


# # Standard Scalar

# In[15]:


scaler=StandardScaler()
data_std=Class1
for  epoch in range(len(Class1)):
    channel=Class1[epoch]
    scaler.fit(channel.transpose())
    data_std[epoch]=np.array(scaler.transform(channel.transpose())).transpose()


# In[17]:


data_std.shape


# In[18]:


### Range distance for RangeEn_B and RangeEn_A
def dist_range(x, y):
    return (np.max(np.abs(x - y), axis=1) - np.min(np.abs(x - y), axis=1)) / (np.max(np.abs(x - y), axis=1) + np.min(np.abs(x - y), axis=1))

def RangeEn_B(x, emb_dim=2, tolerance=.2, dist=dist_range):

    n = np.shape(x)
    n = np.max(n)

    tVecs = np.zeros((n - emb_dim, emb_dim + 1))
    for i in range(tVecs.shape[0]):
        tVecs[i, :] = x[i:i + tVecs.shape[1]]
    counts = []
    for m in [emb_dim, emb_dim + 1]:
        counts.append(0)
        # get the matrix that we need for the current m
        tVecsM = tVecs[:n - m + 1, :m]
        # successively calculate distances between each pair of template vectors
        for i in range(len(tVecsM)):
            dsts = dist(tVecsM, tVecsM[i])
            # delete self-matching
            dsts = np.delete(dsts, i, axis=0)
            # delete undefined distances coming from zero segments
            # dsts = [x for i, x in enumerate(dsts) if not np.isnan(x) and not np.isinf(x)]
            # count how many 'defined' distances are smaller than the tolerance
            # if (dsts):
            counts[-1] += np.sum(dsts < tolerance)/(n - m - 1)

    if counts[1] == 0:
        # log would be infinite => cannot determine RangeEn_B
        RangeEn_B = np.nan
    else:
        # compute log of summed probabilities
        RangeEn_B = -np.log(1.0 * counts[1] / counts[0])

    return RangeEn_B


# In[19]:


fs = 500  # Hz
result_list=np.empty([0,1326], dtype=float)

for epoch in range(len(data_std)):
    row = []
    print("Epoch: ",epoch)
    f = np.transpose(data_std[epoch])
    params = ewt.utilities.ewt_params()
    params.log = 1
    params.removeTrends = 'opening'
    params.option = 2

       
    [ewtLP, mfb, boundaries] = ewt.ewt2d.ewt2dLP(f, params)

   # parameter1 = [np.mean(ewtLP[0][1,:]), np.var(ewtLP[0][1,:]), kurtosis(ewtLP[0][1,:]), np.ptp(ewtLP[0][1,:]), np.std(ewtLP[0][1,:])]
    #print(parameter1)
   # parameter2 = [np.mean(ewtLP[1][1,:]), np.var(ewtLP[1][1,:]), kurtosis(ewtLP[1][1,:]), np.ptp(ewtLP[1][1,:]), np.std(ewtLP[1][1,:])]
    #print(parameter2)
   # k=np.hstack((parameter1,parameter2))
 
    #result_list4=np.vstack((result_list4,k))
    parameter = [            
                np.mean(ewtLP[0][1,:]),
                np.var(ewtLP[0][1,:]),
                skew(ewtLP[0][1,:]),  # Flatten the array for skew and kurtosis calculations
                kurtosis(ewtLP[0][1,:]),
                np.ptp(ewtLP[0][1,:]),
                np.sqrt(np.mean(ewtLP[0][1,:]**2)),  # RMS
                np.std(ewtLP[0][1,:]),
                ant.num_zerocross(ewtLP[0][1,:]), # Number of Zero Crossings
                #ant.hjorth_params(ewtLP[0][1,:][0]), #Hjorth Mobility
                #ant.hjorth_params(ewtLP[0][1,:][1]), 
                ant.petrosian_fd(ewtLP[0][1,:]), #Petrosian Fractal Dimension
                ant.perm_entropy(ewtLP[0][1,:], normalize=True),  
         #ewt2
                np.mean(ewtLP[1][1,:]),
                np.var(ewtLP[1][1,:]),
                skew(ewtLP[1][1,:]),  # Flatten the array for skew and kurtosis calculations
                kurtosis(ewtLP[1][1,:]),
                np.ptp(ewtLP[1][1,:]),
                np.sqrt(np.mean(ewtLP[1][1,:]**2)),  # RMS
                np.std(ewtLP[1][1,:]),
                ant.num_zerocross(ewtLP[1][1,:]),
                #ant.hjorth_params(ewtLP[1][1,:][0]),
                #ant.hjorth_params(ewtLP[1][1,:][1]),
                ant.petrosian_fd(ewtLP[1][1,:]),
                ant.perm_entropy(ewtLP[1][1,:], normalize=True)
            ]     
    row.extend(parameter)
    #print(len(row))
    frequencies, psd = signal.welch(f.flatten(), fs=fs)
    parameter=[np.sum(psd),
                   np.mean(psd),
                   np.var(psd),
                   skew(psd),
                   kurtosis(psd),
                   -np.sum(psd * np.log2(psd + 1e-10)),
                   np.median(psd),
                  ]
    row.extend(parameter)
    #print(len(row))
    result_list=np.vstack((result_list,row))


# In[ ]:


a = np.asarray(result_list)
np.savetxt("D:\\Users\\sugam patel\\Downloads\\only_slot1_all\\Class1_10_all_fet.csv", a, delimiter=",")


# In[3]:


# Create label with numpy array
a = 1326
b = 1394


# Create a list with the desired values and their repetitions
values = [1] * a + [2] * b 

# Convert the list to a NumPy array
label = np.array(values)

print(label)
label.shape


######################################


# # Classifier

# In[ ]:





# In[ ]:





# # RandomForest

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.1, random_state=42)

# initialize LabelPowerset multi-label classifier with a RandomForest
classifier = ClassifierChain(
    classifier = RandomForestClassifier(n_estimators=100, random_state=42),
    require_dense = [False, True]
)

# train
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Calculate F-Score
f_score = f1_score(y_test, y_pred.toarray(), average='micro')  # 'micro' averaging for multi-label classification
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred.toarray())

print("F-Score:", f_score)
print("Accuracy:", accuracy)


# # SelectKBest

# In[ ]:


from sklearn.feature_selection import SelectKBest, mutual_info_classif

X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.1, random_state=42)
# Feature selection using SelectKBest with mutual information scoring function
k_best_selector = SelectKBest(score_func=mutual_info_classif, k=4)  # Using all features for demonstration
X_train_k_best = k_best_selector.fit_transform(X_train, y_train)
X_test_k_best = k_best_selector.transform(X_test)

# Now, create and train a RandomForestClassifier with optimized hyperparameters
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
classifier.fit(X_train_k_best, y_train)

# Make predictions on the test set using the selected features
predictions = classifier.predict(X_test_k_best)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions,average='micro')  # You can adjust the 'average' parameter as needed

print(f"Accuracy with SelectKBest: {accuracy}")
print(f"F1 Score with SelectKBest: {f1}")


# # OneVsRestClassifier

# In[ ]:


#OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(result_list, label, random_state=42)
clf = OneVsRestClassifier(SVC(kernel="linear"))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Calculate F-Score
f_score = f1_score(y_test, y_pred, average='micro')  # 'micro' averaging for multi-label classification
accuracy = accuracy_score(y_test, y_pred)
print("F-Score:", f_score)


# #  KNN_classifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.1, random_state=42)

# Create a k-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"K-Nearest Neighbors Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# #  DecisionTreeClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Decision Tree Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# # GradientBoostingClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.2, random_state=42)

# Create a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier()

# Train the classifier
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Gradient Boosting Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# #  BaggingClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.2, random_state=42)


# Create a base Decision Tree classifier (you can replace it with any other classifier)
base_classifier = DecisionTreeClassifier()

# Create a BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the classifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Bagging Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# # ExtraTreeClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.2, random_state=42)


# Create an ExtraTreeClassifier
extra_tree_classifier = ExtraTreeClassifier(random_state=42)

# Train the classifier
extra_tree_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = extra_tree_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Extra Tree Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# #  MLPClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.2, random_state=42)

# Create an MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the classifier
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"MLP Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# # Class 2

# In[20]:


mat2 = result_mat = scipy.io.loadmat("D:\\Users\\sugam patel\\Downloads\\only_slot2_all\\Class2_10_all.mat")


# In[21]:


Class2 = mat2['Class2_filtered'] 
Class2.shape


# In[22]:


scaler=StandardScaler()
data_std1=Class2
for  epoch in range(len(Class2)):
    channel=Class2[epoch]
    scaler.fit(channel.transpose())
    data_std1[epoch]=np.array(scaler.transform(channel.transpose())).transpose()


# In[23]:


data_std1.shape


# In[ ]:


fs = 500  # Hz
result_list1=np.empty([0,1394], dtype=float)

for epoch in range(len(data_std1)):
    row = []
    print("Epoch: ",epoch)
    f = np.transpose(data_std1[epoch])
    params = ewt.utilities.ewt_params()
    params.log = 1
    params.removeTrends = 'opening'
    params.option = 2

       
    [ewtLP, mfb, boundaries] = ewt.ewt2d.ewt2dLP(f, params)

   # parameter1 = [np.mean(ewtLP[0][1,:]), np.var(ewtLP[0][1,:]), kurtosis(ewtLP[0][1,:]), np.ptp(ewtLP[0][1,:]), np.std(ewtLP[0][1,:])]
    #print(parameter1)
   # parameter2 = [np.mean(ewtLP[1][1,:]), np.var(ewtLP[1][1,:]), kurtosis(ewtLP[1][1,:]), np.ptp(ewtLP[1][1,:]), np.std(ewtLP[1][1,:])]
    #print(parameter2)
   # k=np.hstack((parameter1,parameter2))
 
    #result_list4=np.vstack((result_list4,k))
    parameter = [            
                np.mean(ewtLP[0][1,:]),
                np.var(ewtLP[0][1,:]),
                skew(ewtLP[0][1,:]),  # Flatten the array for skew and kurtosis calculations
                kurtosis(ewtLP[0][1,:]),
                np.ptp(ewtLP[0][1,:]),
                np.sqrt(np.mean(ewtLP[0][1,:]**2)),  # RMS
                np.std(ewtLP[0][1,:]),
                ant.num_zerocross(ewtLP[0][1,:]), # Number of Zero Crossings
                #ant.hjorth_params(ewtLP[0][1,:][0]), #Hjorth Mobility
                #ant.hjorth_params(ewtLP[0][1,:][1]), 
                ant.petrosian_fd(ewtLP[0][1,:]), #Petrosian Fractal Dimension
                ant.perm_entropy(ewtLP[0][1,:], normalize=True),  
         #ewt2
                np.mean(ewtLP[1][1,:]),
                np.var(ewtLP[1][1,:]),
                skew(ewtLP[1][1,:]),  # Flatten the array for skew and kurtosis calculations
                kurtosis(ewtLP[1][1,:]),
                np.ptp(ewtLP[1][1,:]),
                np.sqrt(np.mean(ewtLP[1][1,:]**2)),  # RMS
                np.std(ewtLP[1][1,:]),
                ant.num_zerocross(ewtLP[1][1,:]),
                #ant.hjorth_params(ewtLP[1][1,:][0]),
                #ant.hjorth_params(ewtLP[1][1,:][1]),
                ant.petrosian_fd(ewtLP[1][1,:]),
                ant.perm_entropy(ewtLP[1][1,:], normalize=True)
            ]     
    row.extend(parameter)
    #print(len(row))
    frequencies, psd = signal.welch(f.flatten(), fs=fs)
    parameter=[np.sum(psd),
                   np.mean(psd),
                   np.var(psd),
                   skew(psd),
                   kurtosis(psd),
                   -np.sum(psd * np.log2(psd + 1e-10)),
                   np.median(psd),
                  ]
    row.extend(parameter)
    #print(len(row))
    result_list1=np.vstack((result_list1,row))


# In[ ]:


a = np.asarray(result_list1)
np.savetxt("D:\\Users\\sugam patel\\Downloads\\only_slot1_all\\Class2_10_all_fet.csv", a, delimiter=",")


# # ML Classifier 

# #  RandomForestClassifier

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_list1, label, test_size=0.1, random_state=42)

# initialize LabelPowerset multi-label classifier with a RandomForest
classifier = ClassifierChain(
    classifier = RandomForestClassifier(n_estimators=100, random_state=42),
    require_dense = [False, True]
)

# train
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Calculate F-Score
f_score = f1_score(y_test, y_pred.toarray(), average='micro')  # 'micro' averaging for multi-label classification
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred.toarray())

print("F-Score:", f_score)
print("Accuracy:", accuracy)


# #  SelectKBest

# In[ ]:


from sklearn.feature_selection import SelectKBest, mutual_info_classif

X_train, X_test, y_train, y_test = train_test_split(result_list1, label, test_size=0.1, random_state=42)
# Feature selection using SelectKBest with mutual information scoring function
k_best_selector = SelectKBest(score_func=mutual_info_classif, k=4)  # Using all features for demonstration
X_train_k_best = k_best_selector.fit_transform(X_train, y_train)
X_test_k_best = k_best_selector.transform(X_test)

# Now, create and train a RandomForestClassifier with optimized hyperparameters
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
classifier.fit(X_train_k_best, y_train)

# Make predictions on the test set using the selected features
predictions = classifier.predict(X_test_k_best)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions,average='micro')  # You can adjust the 'average' parameter as needed

print(f"Accuracy with SelectKBest: {accuracy}")
print(f"F1 Score with SelectKBest: {f1}")


# # OneVsRestClassifier

# In[ ]:


#OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(result_list1, label, random_state=42)
clf = OneVsRestClassifier(SVC(kernel="linear"))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Calculate F-Score
f_score = f1_score(y_test, y_pred, average='micro')  # 'micro' averaging for multi-label classification
accuracy = accuracy_score(y_test, y_pred)
print("F-Score:", f_score)


# #  KNeighborsClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list1, label, test_size=0.1, random_state=42)

# Create a k-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"K-Nearest Neighbors Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# # DecisionTreeClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list1, label, test_size=0.2, random_state=42)


# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Decision Tree Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# # GradientBoostingClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list1, label, test_size=0.2, random_state=42)

# Create a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier()

# Train the classifier
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Gradient Boosting Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# #  BaggingClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list1, label, test_size=0.2, random_state=42)


# Create a base Decision Tree classifier (you can replace it with any other classifier)
base_classifier = DecisionTreeClassifier()

# Create a BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the classifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Bagging Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# # Extra_tree_classifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list, label, test_size=0.2, random_state=42)


# Create an ExtraTreeClassifier
extra_tree_classifier = ExtraTreeClassifier(random_state=42)

# Train the classifier
extra_tree_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = extra_tree_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Extra Tree Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# # MLPClassifier

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_list1, label, test_size=0.2, random_state=42)

# Create an MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the classifier
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"MLP Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# In[ ]:





# In[ ]:





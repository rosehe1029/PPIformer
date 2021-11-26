import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, recall_score, roc_curve, roc_auc_score
from sklearn.metrics.ranking import auc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
from sklearn.linear_model import LogisticRegression

data1=pd.read_csv("train10.txt",header=None )
print(data1.shape)
print(data1.head(5))

def genData(file, max_len):
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
    long_pep_counter = 0
    pep_codes = []
    labels = []
    for pep in lines:
        pep, label = pep.split(",")
        labels.append(int(label))
        if not len(pep) > max_len:
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
    print("length > 81:", long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    return data, torch.tensor(labels)


data, label = genData("train10.txt", 81)
# data,label=genData0("1czy_output.txt",81)
print(data.shape, label.shape)
# print(data1.shape,label1.shape)
from sklearn.model_selection import train_test_split
#train_data, train_label = data[:329420], label[:329420]
#test_data, test_label = data[329420:], label[329420:]
train_data, test_data, train_label, test_label=train_test_split(data,label,test_size=0.1,random_state=2021)

print(train_data.shape,test_data.shape)
t=test_label#.numpy().astype(int)
np.savetxt('LR_label.csv',t,delimiter=',')
train_dataset = Data.TensorDataset(train_data, train_label)
test_dataset = Data.TensorDataset(test_data, test_label)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#print(train_data)
ss = StandardScaler()
#train_X = ss.fit_transform(train_data)
#test_X = ss.transform(test_data)
#print(train_X)
#SVM
from sklearn.metrics import matthews_corrcoef #计算MCC

lr=LogisticRegression()
lr.fit(train_data,train_label)

p=lr.predict_proba(test_data)
pp=lr.predict(test_data)

np.savetxt('LR.csv',p,delimiter=',')
fpr, tpr, threshold = roc_curve(test_label, p[:,1])
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(test_label,pp))
print("AUC",roc_auc)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
lw=2, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression:Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

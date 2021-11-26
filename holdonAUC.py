from sklearn.metrics import matthews_corrcoef #计算MCC
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, recall_score, roc_curve, roc_auc_score
from sklearn.metrics.ranking import auc
import pandas  as pd
import matplotlib.pyplot as plt
test_label=pd.read_csv("_label.csv", header=None)
print(test_label.shape)

p=pd.read_csv("test.csv", header=None)
p=p.iloc[:,2]
print(p.shape)
pp=[i>0.5 for i in p]
fpr, tpr, threshold = roc_curve(test_label, p)
roc_auc = auc(fpr, tpr)
print("MCC: %f " %matthews_corrcoef(test_label,pp))
print("AUC",roc_auc)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='red',lw=2, label='PPIformer :ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve comparison of the models')
'''
#Data  A
Ctest_label=pd.read_csv("_label.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("test_111_2.csv", header=None)
Cp=Cp.iloc[:,2]
print(Cp.shape)
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='green',lw=2, label='without Data Augmentation :ROC curve (area = %0.2f)' % roc_auc)##假正率为横坐标，真正率为纵坐标做曲线

''' 
#RF
Ctest_label=pd.read_csv("RF_label.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("RF.csv", header=None)
Cp=Cp.iloc[:,1]
print(Cp.shape)
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='green',lw=2, label='RF:ROC curve (area = %0.2f)' % roc_auc)##假正率为横坐标，真正率为纵坐标做曲线


#textCNN
Ctest_label=pd.read_csv("CNN_label.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("CNN2.csv", header=None)
Cp=Cp.iloc[:,2]
print(Cp.shape)
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue',lw=2, label='TextCNN:ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
#svm
Ctest_label=pd.read_csv("SVM_label.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("SVM.csv", header=None)
Cp=Cp.iloc[:,0]
print(Cp.shape)
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='orange',lw=2, label='SVM:ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线




#DNN
Ctest_label=pd.read_csv("DNN_label.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("DNN2.csv", header=None)
Cp=Cp.iloc[:,2]
print(Cp.shape)
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='pink',lw=2, label='DNN:ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
#LR
Ctest_label=pd.read_csv("LR_label.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("LR.csv", header=None)
Cp=Cp.iloc[:,1]
print(Cp.shape)
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='yellow',lw=2, label='LR:ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
#lstm
Ctest_label=pd.read_csv("lstm_label.csv", header=None)
print(Ctest_label.shape)
Cp=pd.read_csv("lstm10.csv", header=None)
Cp=Cp.iloc[:,1]
print(Cp.shape)
fpr, tpr, threshold = roc_curve(Ctest_label, Cp)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='purple',lw=2, label='LSTM:ROC curve (area = 0.52)' )
plt.legend(loc=0)
plt.legend(loc="lower right")
plt.show()

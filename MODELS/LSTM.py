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
import time
import pickle
from termcolor import colored
import pandas  as pd
import os
import math
from torchnet import meter
#from valid_metrices import *
import argparse

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



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
t=test_label.numpy().astype(int)
np.savetxt('lstm_label.csv',t,delimiter=',')
train_dataset = Data.TensorDataset(train_data, train_label)
test_dataset = Data.TensorDataset(test_data, test_label)
batch_size = 128  # 256
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

class LSTM(nn.Module):
    # 多通道textcnn
    def __init__(self):
        super(LSTM, self).__init__()
        label_num = 2# 标签的个数
        vocab_size = 24
        self.embedding_dim = 128
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm= nn.LSTM(input_size=self.embedding_dim, hidden_size=20, num_layers=1)
        # 输入向量维数10, 隐藏元维度20, 2个LSTM层串联(若不写则默认为1）
        #self.linear =
        # 输入（seq_len,batch , input_size） 序列长度为5 batch为3 输入维度为10
        # print(input)
        # h_0(num_layers * num_directions, batch, hidden_size)  num_layers = 2 ，batch=3 ，hidden_size = 20
        # 同上

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        x = self.embedding(x) # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        #x=x.view(50,batch_size,self.embedding_dim)
        #print(x.shape) #[128,50,128]
        h0 = torch.zeros(1, 50, 20)
        c0 = torch.zeros(1, 50,20)
        r, _ = self.lstm(x,(h0,c0))
        #x = x.view(batch_size, -1 )
        # 全连接层
        r=r.permute(1,0,2)
        #print(r.shape)
        r=r[-1]
        #print(r.shape)
        self.linear=nn.Linear(20, 2)
        logits=self.linear(r)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True)

device = torch.device("cuda")


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        #x, y = x.to(device), y.to(device)
        outputs = net(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def testcalc_metrics(data_iter, net):
    scores = {}
    iter_dict = {}
    iter_dict['val_labels'] = []
    iter_dict['val_outputs'] = []
    iter_dict['output_prob'] = []
    yy = []
    outputss = []
    outputsp = []
    for x, y in data_iter:
        #x, y = x.to(device), y.to(device)
        outputs = net(x)
        yy.append(y)
        o = torch.argmax(outputs, dim=1)
        outputss.append(o)
        outputsp.append(outputs)
    iter_dict['val_labels'] = torch.cat(yy).cpu().numpy()
    iter_dict['val_outputs'] = torch.cat(outputss).cpu().numpy()
    iter_dict['output_prob'] = torch.cat(outputsp).cpu().numpy()[:, 1:]
    # print(type(iter_dict['val_labels']))
    # print(type(iter_dict['val_outputs']))
    # print(iter_dict['output_prob'])
    temp = confusion_matrix(iter_dict['val_labels'], iter_dict['val_outputs'])
    TN, FP, FN, TP = temp.ravel()
    # scores['misclassification_rate'] = (FP + FN) / (TN + FP + FN + TP ) # or 1-accuracy
    scores['sensitivity'] = TP / (FN + TP)  # aka sensitivity or recall
    # scores['false_pos_rate'] = FP / (TN + FP)
    scores['specificity'] = TN / (TN + FP)  # aka specificity
    precision = TP / (TP + FP)
    # scores['prevalence'] = (TP + FN) / (TN + FP + FN + TP )
    scores['f_score'] = (2 * scores['sensitivity'] * precision) / (scores['sensitivity'] + precision)

    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if mcc_denominator == 0: mcc_denominator = 1
    scores['mcc'] = mcc_numerator / math.sqrt(mcc_denominator)
    scores['auc'] = roc_auc_score(iter_dict['val_labels'], iter_dict['output_prob'])
    scores['accuracy'] = (TP + TN) / (TN + FP + FN + TP)
    scores['conf_matrix'] = temp
    return scores['sensitivity'], scores['specificity'], scores['f_score'], scores['mcc'], scores['auc']


import sys
np.set_printoptions(threshold=sys.maxsize)

def testcalc_metrics1(data_iter, net,epoch):
    scores = {}
    iter_dict = {}
    iter_dict['val_labels'] = []
    iter_dict['output_prob'] = []
    yy = []
    outputsp = []
    for x, y in data_iter:
        #x, y = x.to(device), y.to(device)
        outputs = net(x)
        yy.append(y)
        outputsp.append(outputs)
    iter_dict['val_labels'] = torch.cat(yy).cpu().numpy()
    iter_dict['output_prob'] = torch.cat(outputsp).cpu().numpy()
    num=torch.cat(outputsp).cpu().numpy()
    import pandas as pd
    epoch=epoch+1
    data1 = pd.DataFrame(num)  # header:原第一行的索引，index:原第一列的索引
    data1.to_csv('lstm'+str(epoch)+'.csv')
    #np.savetxt('train11_test'+epoch+1+'.csv',iter_dict['output_prob'], delimiter=',')
    return iter_dict

for num_model in range(10):
    net = LSTM()
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    best_auc = 0
    EPOCH = 4
    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        net.train()
        for seq1,  label in train_iter_cont:
            output1 = net(seq1)
            #print(output1.shape)
            loss = criterion(output1,  label)
            #loss2 = criterion_model(output3, label1)
            #loss3 = criterion_model(output4, label2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())

        net.eval()
        with torch.no_grad():
            train_acc = evaluate_accuracy(train_iter, net)
            test_acc = evaluate_accuracy(test_iter, net)
            #train_auc=testcalc_metrics(train_iter,net)
            #test_auc=testcalc_metrics(test_iter,net)
            train_sen, train_spec, train_fscore, train_mcc, train_auc = testcalc_metrics(train_iter, net)
            test_sen, test_spec, test_fscore, test_mcc, test_auc = testcalc_metrics(test_iter, net)
            testcalc_metrics1(test_iter, net,epoch)
        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
        results += f'\ttrain_auc: {train_auc:.4f}, test_auc: {colored(test_auc, "red")}'
        #results += f'\ttrain_sen: {train_sen:.4f}, test_sen: {colored(test_sen, "red")}'
        #results += f'\ttrain_spec: {train_spec:.4f}, test_spec: {colored(test_spec, "red")}'
        results += f'\ttrain_fscore: {train_fscore:.4f}, test_fscore: {colored(test_fscore, "red")}'
        results += f'\ttrain_mcc: {train_mcc:.4f}, test_mcc: {colored(test_mcc, "red")}'
        print(results)
        #to_log(results)
        if test_auc > best_auc:
            best_auc = test_auc
            torch.save({"best_auc": best_auc, "model": net.state_dict()}, f'./{num_model}.pl')
            print(f"best_auc: {best_auc}")



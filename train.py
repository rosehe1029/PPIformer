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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--DATA',type=str)
parser.add_argument('--BATCH_SIZE', type=int)
parser.add_argument('--MAXLEN',type=int)
parser.add_argument('--EPOCHS', type=int)
parser.add_argument('--LR',type=float)


args = parser.parse_args()
TRAIN=args.DATA
batch_size=args.BATCH_SIZE
maxlen0=args.MAXLEN
epochs=args.EPOCHS #= 16
LR=args.LR

#from valid_metrices import *
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
    #print("length > 81:", long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    return data, torch.tensor(labels)


data, label = genData(TRAIN, 81)
# data,label=genData0("1czy_output.txt",81)
print(data.shape, label.shape)
# print(data1.shape,label1.shape)
from sklearn.model_selection import train_test_split
#train_data, train_label = data[:3253], label[:3253]
#test_data, test_label = data[3253:], label[3253:]
train_data, test_data, train_label, test_label=train_test_split(data,label,test_size=0.3,random_state=2021)

print(train_data.shape,test_data.shape)

#t=test_label.numpy().astype(int)
#np.savetxt('_label.csv',t,delimiter=',')
train_dataset = Data.TensorDataset(train_data, train_label)
test_dataset = Data.TensorDataset(test_data, test_label)
batch_size = 128  # 256
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class newModel(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.hidden_dim = 25
        self.batch_size = 256  # 256
        self.emb_dim = 512  # 512

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True)#, dropout=0.2)

        self.block1 = nn.Sequential(nn.Linear(2600, 1024),  # 4050
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256),
                                    )

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        )
        # add
        self.tanh1 = nn.Tanh()
        hidden_size2 = 2600
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(self.hidden_dim * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(self.hidden_dim * 2, hidden_size2)

    def softmax(x):
        s = torch.exp(x)
        return s / torch.sum(s, dim=1, keepdim=True)

    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer_encoder(x).permute(1, 0, 2)
        output, hn = self.gru(output)
        output = output.permute(1, 0, 2)
        M = self.tanh1(output)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = output * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        # print(out.shape)
        ''' 
        output = output.permute(1, 0, 2)
        hn = hn.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        hn = hn.reshape(output.shape[0], -1)
        output = torch.cat([output, hn], 1)
        #         print(output.shape,hn.shape)

        ##add
        emb=25
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        '''
        return self.block1(out)

    def trainModel(self, x):
        with torch.no_grad():
            output = self.forward(x)
        o=self.block2(output)
        s = torch.exp(o)
        ou=s / torch.sum(s, dim=1, keepdim=True)
        return ou


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp澶规柇鐢ㄦ硶
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def  reduce_loss(loss,reduction='mean'):
    return loss.mean()  if reduction=='mean'  else loss.sum()  if reduction=='sum'  else loss

def   linear_combination(x,y,epsilon):
    return epsilon*x+(1-epsilon)*y

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,epsilon:float=0.1,reduction='mean'):
        super().__init__()
        self.epsilon=epsilon
        self.reduction=reduction

    def forward(self,preds,target):
        n=preds.size()[-1]
        log_preds=F.log_softmax(preds,dim=1)
        loss=reduce_loss(-log_preds.sum(dim=-1),self.reduction)
        nll=F.nll_loss(log_preds,target,reduction=self.reduction)
        return linear_combination(loss/n,nll,self.epsilon)


def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1 = batch[i][0], batch[i][1]
        seq2, label2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2


train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate)

device = torch.device("cuda")


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
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
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
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

def testcalc_metrics11(data_iter, net,epoch):
    scores = {}
    iter_dict = {}
    iter_dict['val_labels'] = []
    iter_dict['output_prob'] = []
    yy = []
    outputsp = []
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
        yy.append(y)
        outputsp.append(outputs)
    iter_dict['val_labels'] = torch.cat(yy).detach().cpu().numpy()
    iter_dict['output_prob'] = torch.cat(outputsp).detach().cpu().numpy()
    num=torch.cat(outputsp).detach().cpu().numpy()
    import pandas as pd
    epoch=epoch+1
    data1 = pd.DataFrame(num)  # header:原第一行的索引，index:原第一列的索引
    data1.to_csv('train'+str(epoch)+'.csv')
    #np.savetxt('train11_test'+epoch+1+'.csv',iter_dict['output_prob'], delimiter=',')
    return iter_dict

def testcalc_metrics1(data_iter, net,epoch):
    scores = {}
    iter_dict = {}
    iter_dict['val_labels'] = []
    iter_dict['output_prob'] = []
    yy = []
    outputsp = []
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
        yy.append(y)
        outputsp.append(outputs)
    iter_dict['val_labels'] = torch.cat(yy).detach().cpu().numpy()
    iter_dict['output_prob'] = torch.cat(outputsp).detach().cpu().numpy()
    num=torch.cat(outputsp).detach().cpu().numpy()
    import pandas as pd
    epoch=epoch+1
    data1 = pd.DataFrame(num)  # header:原第一行的索引，index:原第一列的索引
    data1.to_csv('test'+str(epoch)+'.csv')
    #np.savetxt('train11_test'+epoch+1+'.csv',iter_dict['output_prob'], delimiter=',')
    return iter_dict


def to_log(log):
    with open("./modelLog.log", "a+") as f:
        f.write(log + '\n')


for num_model in range(1):
    net = newModel().to(device)
    lr =LR #0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = ContrastiveLoss()
    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    criterion_F=FocalLoss()
    criterion_m=LabelSmoothingCrossEntropy()
    best_auc = 0
    epochs = 200
    for epoch in range(epochs):
        loss_ls = []
        loss1_ls = []
        loss2_3_ls = []
        t0 = time.time()
        net.train()
        for seq1, seq2, label, label1, label2 in train_iter_cont:
            output1 = net(seq1)
            output2 = net(seq2)
            output3 = net.trainModel(seq1)
            output4 = net.trainModel(seq2)
            loss1 = criterion(output1, output2, label)
            #loss2 = criterion_model(output3, label1)
            #loss3 = criterion_model(output4, label2)
            loss2 = criterion_m(output3, label1)
            loss3 = criterion_m(output4, label2)
            loss = loss1 + loss2 + loss3
            #             print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
            loss1_ls.append(loss1.item())
            loss2_3_ls.append((loss2 + loss3).item())

        net.eval()
        with torch.no_grad():
            train_acc = evaluate_accuracy(train_iter, net)
            test_acc = evaluate_accuracy(test_iter, net)
            train_auc=testcalc_metrics(train_iter,net)
            test_auc=testcalc_metrics(test_iter,net)
            train_sen, train_spec, train_fscore, train_mcc, train_auc = testcalc_metrics(train_iter, net)
            test_sen, test_spec, test_fscore, test_mcc, test_auc = testcalc_metrics(test_iter, net)
        testcalc_metrics11(train_iter, net,epoch)
        testcalc_metrics1(test_iter, net, epoch)
        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
        results += f'\ttrain_auc: {train_auc:.4f}, test_auc: {colored(test_auc, "red")}'
        #results += f'\ttrain_sen: {train_sen:.4f}, test_sen: {colored(test_sen, "red")}'
        #results += f'\ttrain_spec: {train_spec:.4f}, test_spec: {colored(test_spec, "red")}'
        results += f'\ttrain_fscore: {train_fscore:.4f}, test_fscore: {colored(test_fscore, "red")}'
        results += f'\ttrain_mcc: {train_mcc:.4f}, test_mcc: {colored(test_mcc, "red")}'
        print(results)
        to_log(results)
        if test_auc > best_auc:
            best_auc = test_auc
            torch.save({"best_auc": best_auc, "model": net.state_dict()}, f'./{num_model}.pl')
            print(f"best_auc: {best_auc}")

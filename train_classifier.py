# from matplotlib.font_manager import _Weight
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import os
from torch.utils.data import dataloader

from tqdm.std import TqdmMonitorWarning
sys.path.append("./model")
sys.path.append("./loss")
sys.path.append('./utils')
import dataset
import VGANCox
import argparse
from torch.backends import cudnn
import datetime
import time
from tqdm import tqdm
import numpy as np
import logging
from sklearn.metrics import r2_score
import summary
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import gc
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def get_confusion_matrix(trues, preds):
    # labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    conf_matrix = confusion_matrix(trues, preds)
    return conf_matrix


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss * self.alpha)
    
    
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,30,20,20,20,50,20,20,30,20,40,20,20,20,20,50,20,20], gamma=3, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark=True

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data_new', help='dataset path')
parser.add_argument('--input_g',type=int,default=100)
parser.add_argument('--mid_g',type=int,default=1024)
parser.add_argument('--mid_d',type=int,default=2048)
parser.add_argument('--out_d',type=int,default=100)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--beta1',type=float,default=0.5)
parser.add_argument('--beta2',type=float,default=0.999)
parser.add_argument('--epoch', type=int, default=0, help='epoch')
parser.add_argument('--decay_epoch', type=int, default=60, help='decay epoch')
parser.add_argument('--n_epochs', type=int, default=30, help='train epoch')
parser.add_argument('--checkpoints_interval', type=int, default=20, help='check')
parser.add_argument('--sample_interval',type=int,default=2)
parser.add_argument('--result_log',type=str,default='COX')
parser.add_argument('--model_path',type=str,default='COX')
parser.add_argument('--batch_size',type=int,default=256)
parser.add_argument('--seq_length',type=int,default=25182)
parser.add_argument('--sample_length',type=int,default=1024)
parser.add_argument('--latern_dim',type=int,default=256)
parser.add_argument('--n_critic',type=int,default=4)
parser.add_argument('--miseq_length',type=int,default=1285)
parser.add_argument('--misample_length',type=int,default=256)
parser.add_argument('--milatern_dim',type=int,default=100)
parser.add_argument('--rna_seq_dict',type=str,default='./saved_models/pretrain/g_e_200')
parser.add_argument('--mirna_seq_dict',type=str,default='./saved_models/pretrain/g_mirna_e_200')
parser.add_argument('--lasso',type=bool,default=True)
parser.add_argument('--encoder_type',type=str,default='attention')
parser.add_argument('--omics_type',type=str,default='cancer_classification')

config = parser.parse_args()


def train(config,dataloader,dataloader_val,num_epochs, batch_size, learning_rate,  measure, verbose,save_state,time_str,idx ):
    torch.cuda.empty_cache()
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    model=VGANCox.Classifier(config.seq_length,config.sample_length,config.latern_dim,config.encoder_type)
    model=model.cuda()
    entropy_loss = MultiClassFocalLossWithAlpha(gamma=2)
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_best = 0
    model.train()

    for epoch in tqdm(range(num_epochs)):
        
    
        for i,data in enumerate(dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            exp=data['exp'].cuda().to(torch.float32)
            label = data['type'].cuda().to(torch.int64)
            output,code,aux = model(exp)
            loss = entropy_loss(output,label)

            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if measure or epoch == (num_epochs - 1):

            # VAL
            sklearn_accuracy,sklearn_precision,sklearn_recall,sklearn_f1 = \
            test(epoch,model, dataloader_val,  batch_size,  verbose)

            if sklearn_accuracy > accuracy_best:
                accuracy_best = sklearn_accuracy
                torch.save(model.state_dict(), 'result/CLASS_%s/%s_%s/%d/model.pth'%(config.encoder_type,time_str,config.omics_type, idx+1))
            torch.cuda.empty_cache()
    # torch.save(model.state_dict(),'result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/model.pth')       
    return(model,sklearn_accuracy,sklearn_precision,sklearn_recall,sklearn_f1)
            
def test(epoch, model, dataloader_val,  batch_size,  verbose):

    model.eval()
    entropy_loss = MultiClassFocalLossWithAlpha(gamma=2)
    tot_loss = 0.0
    tot_acc = 0.0
    train_preds = []
    train_trues = []
    for data in dataloader_val:
        torch.cuda.empty_cache()
        exp=data['exp'].cuda().to(torch.float32)
        # mi_exp=data['mi_exp'].cuda().to(torch.float32)
        label = data['type'].cuda().to(torch.int64)
        output,code,aux_loss=model(exp)
        loss = entropy_loss(output,label)
        
        tot_loss += loss.detach()
        train_outputs = output.argmax(dim=1)

        train_preds.extend(train_outputs.detach().cpu().numpy())
        train_trues.extend(label.detach().cpu().numpy())
    
    sklearn_accuracy = accuracy_score(train_trues, train_preds) 
    sklearn_precision = precision_score(train_trues, train_preds, average='micro')
    sklearn_recall = recall_score(train_trues, train_preds, average='micro')
    sklearn_f1 = f1_score(train_trues, train_preds, average='micro')
        
    
    if verbose > 0:
        print("[sklearn_metrics] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} \
            recall:{:.4f} f1:{:.4f}".format(epoch, tot_loss, sklearn_accuracy, 
                                            sklearn_precision, sklearn_recall, sklearn_f1))
    torch.cuda.empty_cache()
    return(sklearn_accuracy,sklearn_precision,sklearn_recall,sklearn_f1)
            


def main():
    os.makedirs('result/CLASS_%s'%config.encoder_type,exist_ok=True)
    time_str  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
    # os.makedirs('result/COX/%s'%time_str,exist_ok=True)

    cudnn.benchmark = True
    # os.makedirs('saved_models/%s/' % (config.model_path), exist_ok=True)
    os.makedirs('result/CLASS_%s/%s_%s'%(config.encoder_type,time_str,config.omics_type),exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                    filename='result/CLASS_%s/%s_%s/%s.log'%(config.encoder_type,time_str,config.omics_type,config.result_log),
                    filemode='a',
                    format=
                    '[out]-%(levelname)s:%(message)s'
                    )

    # learning_rate_range = 10**np.arange(-4,-1,0.3)
    # lambda_1 = 1e-5
    for i in range(5):
        torch.cuda.empty_cache()
        os.makedirs('result/CLASS_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, i+1),exist_ok=True)
        print('running 5cv ---------num: %d'%(i+1))
        

        logging.info('input (-1,1) run folds %d'%(i+1))
        # logging.info('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        # logging.info('acc. and distance')
        ci_list=[]
        dataloader=dataset.get_loader(config.path,config.batch_size,'train',dataset_type='classify',kf=i,omics_type=config.omics_type)
        dataloader_val=dataset.get_loader(config.path,1,'test',dataset_type='classify',kf=i,omics_type=config.omics_type)

        torch.cuda.empty_cache()
        model,sklearn_accuracy,sklearn_precision,sklearn_recall,sklearn_f1=\
            train(config,dataloader,dataloader_val,config.n_epochs, config.batch_size, config.lr,   True, True, save_state=True,time_str=time_str,idx=i,
                  )
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load('result/CLASS_%s/%s_%s/%d/model.pth'%(config.encoder_type,time_str,config.omics_type, i+1)))
        model.eval()
        entropy_loss = MultiClassFocalLossWithAlpha(gamma=2)
        tot_loss = 0.0
        tot_acc = 0.0
        test_preds = []
        test_trues = []
        with torch.no_grad():
            for data in dataloader_val:
                torch.cuda.empty_cache()
                exp=data['exp'].cuda().to(torch.float32)
                # mi_exp=data['mi_exp'].cuda().to(torch.float32)
                label = data['type'].cuda().to(torch.int64)
                output,code,aux_loss=model(exp)
                loss = entropy_loss(output,label)
                
                tot_loss += loss.detach()
                train_outputs = output.argmax(dim=1)

                test_preds.extend(train_outputs.detach().cpu().numpy())
                test_trues.extend(label.detach().cpu().numpy())
                
            logging.info(classification_report(test_trues, test_preds,digits=6))
            conf_matrix = get_confusion_matrix(test_trues, test_preds)
            sklearn_accuracy = accuracy_score(test_trues, test_preds) 
            precision = precision_score(test_trues, test_preds,average='micro')
            recal = recall_score(test_trues, test_preds,average='micro')
            f1_scores = f1_score(test_trues, test_preds,average='micro')
            logging.info(conf_matrix)
            
            logging.info(sklearn_accuracy)
            logging.info(precision)
            logging.info(recal)
            logging.info(f1_scores)
            with open('result/CLASS_%s/%s_%s/%d/confusion.pickle'%(config.encoder_type,time_str,config.omics_type, i+1),'wb') as handle:
                pickle.dump(conf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)


        
    
        




if __name__=='__main__':
    main()


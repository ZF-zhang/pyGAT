from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test, SPidx,idx,labelweight= KNN()

labelweight=[10,1,1,1,0.1,1,1,1,1]
labelweight=np.array(labelweight)
labelweight=torch.from_numpy(labelweight).float()
# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx1=idx[0].cuda()
    idx2=idx[1].cuda()
    idx3=idx[2].cuda()
    idx4=idx[3].cuda()
    idx5=idx[4].cuda()
    idx6=idx[5].cuda()
    idx7=idx[6].cuda()
    idx8=idx[7].cuda()
    idx9=idx[8].cuda()

    labelweight=labelweight.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_1=F.nll_loss(output[idx1], labels[idx1])
    loss_2=F.nll_loss(output[idx2], labels[idx2])
    loss_3=F.nll_loss(output[idx3], labels[idx3])
    loss_4=F.nll_loss(output[idx4], labels[idx4])
    loss_5=F.nll_loss(output[idx5], labels[idx5])
    loss_6=F.nll_loss(output[idx6], labels[idx6])
    loss_7=F.nll_loss(output[idx7], labels[idx7])
    loss_8=F.nll_loss(output[idx8], labels[idx8])
    loss_9=F.nll_loss(output[idx9], labels[idx9])
    loss_train=loss_1*labelweight[0]+loss_2*labelweight[1]+loss_3*labelweight[2]+loss_4*labelweight[3]+loss_5*labelweight[4]+loss_6*labelweight[5]+loss_7*labelweight[6]+loss_8*labelweight[7]+loss_9*labelweight[8]    
    
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    return output


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
output=compute_test()
output=output.cpu().detach().numpy()
predis=np.argmax(output,axis=1)
labels=labels.cpu().detach().numpy()
Cacc=classacc(labels,predis)
print("Classification Accuracy:",Cacc)

args.Visualisation=False
if args.Visualisation:
    print("Broadcast Labels...")
    # output=output.cpu().detach().numpy()
    # predis=np.argmax(output,axis=1)
    xyz,outlabel,inlabel=Broadcastlabel(pointname=args.OPname, SPidx=SPidx,labels=predis)
    CM=confusionmatrix(outlabel,inlabel)
    print("Confusion Matrix:",CM)
    macro_P,macro_R,macro_F,precision,recall,Fscore,miou=F_score(CM)
    print("OAprecision=",macro_P)
    print("OArecall=",macro_R)
    print("OAFscore=",macro_F)
    print("precision=",precision)
    print("recall=",recall)
    print("Fscore=",Fscore)
    print("MIoU=",miou)
    print("Broadcast Labels Finished!")
    print("Visualisation...")
    paint(xyz,inlabel,args.INpaint)
    paint(xyz,outlabel,txtname='./Points/GAT.txt')
    print("Visualisation Finished!")
    NCM=normalize_adj(CM)
    ax= sns.heatmap(pd.DataFrame(NCM), annot=True,square=True, cmap="YlGnBu",
    xticklabels=["Barren", "Building", "Car","Grass","Powerline", "Road", "Ship","Tree","Water"],
     yticklabels=["Barren", "Building", "Car","Grass","Powerline", "Road", "Ship","Tree","Water"],fmt=".2f")
    plt.show()
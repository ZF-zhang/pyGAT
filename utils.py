from gzip import FNAME
from tokenize import PlainToken
from winsound import MB_ICONQUESTION
import numpy as np
import torch
from arguments import args
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from PIL import Image


def pointloader(txtname):
    '''
    Read a numpy array from a text file.
    '''
    f = open(txtname, mode='r')
    a = []
    for line in f:
        a.append(line[:-1].split(' '))
    return np.array(a, dtype=np.double)


def MatrixDistance(Mx):
    '''
    Calculate the distance matrix.
    '''
    DisMatrix=np.zeros((Mx.shape[0],Mx.shape[0]))
    for i in range(Mx.shape[1]):
        col=Mx[:,[i]]
        len=Mx.shape[0]
        a=col**2
        A = a.repeat(len,axis=-1)
        B=col*col.T
        c=a.T
        C=c.repeat(len,axis=0)
        D=A+C-2*B
        DisMatrix+=D
    return DisMatrix



def MultikernelMatrix(Mx,Spatialweight,sigmalist):
    '''
    Multikernel Matrix
    '''
    print('Building Multikernel Matrix')
    Spatial=Mx[:, :3]
    Spectral=Mx[:, 3:6]
    SpatialDistance=MatrixDistance(Spatial)
    SpectralDistance=MatrixDistance(Spectral)
    MultikernelMatrix=np.zeros((Mx.shape[0],Mx.shape[0],len(sigmalist)))
    for i in range (len(sigmalist)):
        SpatialGaussAdj=np.exp(-SpatialDistance/sigmalist[i])   -np.eye(SpatialDistance.shape[0])
        SpectralGaussAdj=np.exp(-SpectralDistance/sigmalist[i])   -np.eye(SpectralDistance.shape[0])
        SpatialGaussAdj=normalize_maxmin(SpatialGaussAdj)
        SpectralGaussAdj=normalize_maxmin(SpectralGaussAdj)
        ADJMatrix=SpatialGaussAdj*Spatialweight+SpectralGaussAdj*(1-Spatialweight)
        MultikernelMatrix[:,:,i]=ADJMatrix
    print('Multikernel Matrix Done')
    return MultikernelMatrix

def normalize_maxmin(Mx, axis=2):
    '''
    Normalize the matrix Mx by max-min normalization.
    axis=0: normalize each row
    axis=1: normalize each column
    axis=2: normalize the whole matrix
    '''
    if axis == 1:
        M_min = np.amin(Mx, axis=1)
        M_max = np.amax(Mx, axis=1)
        for i in range(Mx.shape[1]):
            Mx[:, i] = (Mx[:, i] - M_min) / (M_max - M_min)
    elif axis == 0:
        M_min = np.amin(Mx, axis=0)
        M_max = np.amax(Mx, axis=0)
        for i in range(Mx.shape[0]):
            Mx[i, :] = (Mx[i, :] - M_min) / (M_max - M_min)
    elif axis == 2:
        M_min = np.amin(Mx)
        M_max = np.amax(Mx)
        Mx = (Mx - M_min) / (M_max - M_min)
    else:
        print('Error')
        return None
    return Mx

def normalize_adj(adj):
    '''
    Nomalize the adjacency matrix
    '''
    sumrow=np.sum(adj,axis=1)
    normalizedadj=adj/sumrow[:,np.newaxis]
    return normalizedadj


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def KNN(txtname=args.SPname):
    print('Loading features')
    OriginData = pointloader(txtname)
    OriginData = np.random.permutation(OriginData)
    if OriginData.shape[0]<args.num_points:
        args.num_points = OriginData.shape[0]
    A=OriginData[:args.num_points,:]
    A,labelnum=predivide(A)
    SPidx=A[:,-2]
    if args.SpectralGraph:
        dis=MatrixDistance(A[:,3:6])
    else:
        dis=MatrixDistance(A[:,:3])
    # adj=takemaxinrow(adj,8)
    dis=-dis+np.max(dis)
    dis=takemaxinrow(dis,8)
    adj=(normalize_adj(dis)+normalize_adj(dis).T)/2
    featuresxyz=A[:, :3]

    featuresxyz=normalize_maxmin(featuresxyz)
    featuresxyz=torch.from_numpy(featuresxyz)
    featuresxyz=featuresxyz.float()

    featurelamda=A[:, 3:6]
    featurelamda=normalize_maxmin(featurelamda)
    featurelamda=torch.from_numpy(featurelamda)
    featurelamda=featurelamda.float()

    features=torch.cat([featuresxyz,featurelamda],1)
    features=features.float()
    features=np.concatenate((features,adj),1)
    adj=torch.from_numpy(adj).float()
    features=torch.from_numpy(features).float()
    labels = torch.LongTensor(A[:, -1])

    # idx_train = range(int(args.num_points))
    idx_train = range(int(0.6*args.num_points))
    idx_val = range(int(0.6*args.num_points),int(0.8*args.num_points))
    idx_test = range(int(0.6*args.num_points), int(args.num_points))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    idx_1=torch.LongTensor(range(int(labelnum[0])))
    idx_2=torch.LongTensor(range(int(labelnum[0]),int(labelnum[0]+labelnum[1])))
    idx_3=torch.LongTensor(range(int(labelnum[0]+labelnum[1]),int(labelnum[0]+labelnum[1]+labelnum[2])))
    idx_4=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3])))
    idx_5=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4])))
    idx_6=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5])))
    idx_7=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6])))
    idx_8=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]+labelnum[7])))
    idx_9=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]+labelnum[7]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]+labelnum[7]+labelnum[8])))
    idx=[idx_1,idx_2,idx_3,idx_4,idx_5,idx_6,idx_7,idx_8,idx_9]

    labelweight=np.zeros(len(labelnum))
    # for i in range(len(labelnum)):
    #     labelweight[i]=np.sum(labelnum)/labelnum[i]
    # labelweight[0]=0.1
    # labelweight[7]=10*labelweight[7]

    # labelweight=[0.1,1,1,1,10,1,1,1,1]
    # labelweight=np.array(labelweight)

    # labelweight=torch.from_numpy(labelweight).float()



    print('Loading features Done')
    return adj, features, labels, idx_train, idx_val, idx_test, SPidx,idx,labelweight




def load_data(txtname=args.SPname):
    '''
    load points data and creat data and creat adj, features, labels, idx_train, idx_val, idx_test
    '''
    print('Loading features')
    OriginData = pointloader(txtname)
    OriginData = np.random.permutation(OriginData)
    if OriginData.shape[0]<args.num_points:
        args.num_points = OriginData.shape[0]
    A=OriginData[:args.num_points,:]
    A,labelnum=predivide(A)
    A[:,2]=5*A[:,2]
    SPidx=A[:,-2]
    ADJ=MultikernelMatrix(A,args.Spatialweight,args.sigmalist)
    featureadj=np.zeros((ADJ.shape[0],ADJ.shape[0]))
    for i in range(args.num_kernels):
        featureadj+=ADJ[:,:,i]
    featureadj=normalize_adj(featureadj)+np.eye(featureadj.shape[0])
    featureadj=torch.from_numpy(featureadj).float()
    # for i in range(args.num_kernels):
    #     ADJ[:,:,i]=takemaxinrow(ADJ[:,:,i],args.num_neighbors)
    ADJ=torch.from_numpy(ADJ).float()
    for i in range(args.num_kernels):
        ADJ[:,:,i]=normalize(ADJ[:,:,i])+torch.eye(ADJ[:,:,i].shape[0])
    featuresxyz=A[:, :3]

    featuresxyz=normalize_maxmin(featuresxyz)
    featuresxyz=torch.from_numpy(featuresxyz)
    featuresxyz=featuresxyz.float()

    featurelamda=A[:, 3:6]
    featurelamda=normalize_maxmin(featurelamda)
    featurelamda=torch.from_numpy(featurelamda)
    featurelamda=featurelamda.float()

    features=torch.cat([featuresxyz,featurelamda,featureadj],1)
    # features=torch.cat([featuresxyz,featurelamda],1)
    features=features.float()
    print('Loading features Done')

    labels = torch.LongTensor(A[:, -1])

    idx_train = range(int(0.6*args.num_points))
    idx_val = range(int(0.6*args.num_points),int(0.8*args.num_points))
    idx_test = range(int(0.6*args.num_points), int(args.num_points))

    idx_1=torch.LongTensor(range(int(labelnum[0])))
    idx_2=torch.LongTensor(range(int(labelnum[0]),int(labelnum[0]+labelnum[1])))
    idx_3=torch.LongTensor(range(int(labelnum[0]+labelnum[1]),int(labelnum[0]+labelnum[1]+labelnum[2])))
    idx_4=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3])))
    idx_5=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4])))
    idx_6=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5])))
    idx_7=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6])))
    idx_8=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]+labelnum[7])))
    idx_9=torch.LongTensor(range(int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]+labelnum[7]),int(labelnum[0]+labelnum[1]+labelnum[2]+labelnum[3]+labelnum[4]+labelnum[5]+labelnum[6]+labelnum[7]+labelnum[8])))
    idx=[idx_1,idx_2,idx_3,idx_4,idx_5,idx_6,idx_7,idx_8,idx_9]

    # labelweight=np.zeros(len(labelnum))
    # for i in range(len(labelnum)):
    #     labelweight[i]=np.sum(labelnum)/labelnum[i]
    # labelweight=[1,2,2,1,20,1,2,0.1,2]
    # labelweight=[0.1,5,5,0.5,10,2,3,0.5,3]#=0.8071~0.8175
    # labelweight=[0.05,10,10,0.25,20,1,2,0.25,2]#=0.8042~0.8171
    # labelweight=[0.05,15,15,0.25,25,1,2,0.2,2]#H64-0.7913~0.8108 #H128-0.7950~0.8117 #H256-0.7992~8183 #H512-0.8021~8158 #H1024=0.8042~0.8171(N=8) 0.8046~0.8171(N=100) 0.8046~.8171(N=256)  #H2048=0.8038~8179 #H4096=0.8058~0.8183
    #5z,0.5ss-8050~8154
    # labelweight=[0.1,1,1,1,10,1,1,1,1]#8425[0.79351536 0.37899543 0.2797619  0.87632509 0.32 0.42152466 0.36666667 0.97292372 0.39215686]
    # labelweight=[0.1,5,8,1,50,5,8,0.1,5]#8413[0.8003413  0.38812785 0.28571429 0.8869258  0.34 0.41255605 0.425      0.96459255 0.41568627]
    # labelweight=[0.1,10,15,1,80,10,15,0.05,10]#8304~8354  [0.69795222 0.38812785 0.29166667 0.89399293 0.38 0.43497758 0.45  0.95834418 0.41764706]
    # labelweight=[0.2,30,45,1,120,20,50,0.01,20]#7908~7900
    # labelweight=np.array(labelweight)
    labelweight=[10,1,1,1,0.1,1,1,1,1]
    # labelweight=[0.1,1,1,1,10,1,1,1,1]#0.7733
    # labelweight=[2,5,10,10,22,8,2,1,1.5]#0.7458
    labelweight=np.array(labelweight)
    labelweight=torch.from_numpy(labelweight).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return ADJ, features, labels, idx_train, idx_val, idx_test, SPidx,idx,labelweight


def normalize(mx):
    """
    row-normalize matrix torch
    """
    rowsum = torch.sum(mx,1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv.mm(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def takemaxinrow(mx,n):
    """
    take max n value in each row
    """
    temp=np.sort(mx,axis=1)
    limit=temp[:,-n].T
    # zero=np.zeros((1,mx.shape[1]))
    for i in range(mx.shape[0]):
        mx[i]=np.where(mx[i]>=limit[i],mx[i],0)
    return mx


def move_to_positive(mx):
    """
    move all value to positive
    """
    if torch.min(mx)<0:
        mx=mx-torch.min(mx)
    return mx



def make_graph(ADJ,kernelweight):
  
    if args.cuda:            
        adj=torch.zeros((args.num_points,args.num_points)).cuda()
    else:
        adj=torch.zeros((args.num_points,args.num_points))


    for i in range (args.num_kernels):
        a=kernelweight[i]
        adji=ADJ[:,:,i]
        temp=a*adji
        adj=temp+adj
    if args.cuda:   
        adj=adj.cpu().numpy()
        adj=np.array(adj)
        adj=takemaxinrow(adj,args.num_neighbors)
        adj=torch.from_numpy(adj).float()
        adj=adj.cuda()
    else:
        adj=adj.numpy()
        adj=np.array(adj)
        adj=takemaxinrow(adj,args.num_neighbors)
        adj=torch.from_numpy(adj).float()    
    adj=normalize(adj)
    adj=(adj+adj.T)/2
    return adj

def paint(xyz,labels,txtname):
    '''
    paint the test data
    '''
    color=np.zeros((xyz.shape[0],3))
    # xyz=xyz.cpu().numpy()
    # labels=labels.cpu().numpy()
    for i in range(xyz.shape[0]):
        if labels[i]==1:
            color[i]=[238,125,48];"Barren"
        elif labels[i]==2:
            color[i]=[29,40,92];"Building"
        elif labels[i]==3:
            color[i]=[238,30,34];"Car"
        elif labels[i]==4:
            color[i]=[162,209,142];"Grass"
        elif labels[i]==5:
            color[i]=[132,60,12];"Powerline"
        elif labels[i]==6:
            color[i]=[70,113,184];"Road"
        elif labels[i]==7:
            color[i]=[249,93,93];"Ship"
        elif labels[i]==8:
            color[i]=[56,87,35];"Tree"
        elif labels[i]==9:
            color[i]=[39,173,227];"Water"
    paint=np.concatenate((xyz,color),1)
    np.savetxt(txtname,paint)



def Broadcastlabel(pointname,SPidx,labels):
    '''
    broadcast the label
    '''
    A=pointloader(pointname)
    xyz=A[:,:3]
    idxsp=A[:,-2]
    Reallabels=A[:,-1]
    outlabel=np.zeros((xyz.shape[0]))
    label=labels[np.argsort(SPidx)]
    for i in range(xyz.shape[0]):
        idx=idxsp[i]
        idx=int(idx)
        outlabel[i]=label[idx]
    return xyz,outlabel,Reallabels



def predivide(A):
    '''
    divide the test data
    '''
    labels=A[:,-1]
    B1=[];B2=[];B3=[];B4=[];B5=[];B6=[];B7=[];B8=[];B9=[]
    for i in range (A.shape[0]):
        if labels[i]==1:
            B1.append(A[i])
        elif labels[i]==2:
            B2.append(A[i])
        elif labels[i]==3:
            B3.append(A[i])
        elif labels[i]==4:
            B4.append(A[i])
        elif labels[i]==5:
            B5.append(A[i])
        elif labels[i]==6:
            B6.append(A[i])
        elif labels[i]==7:
            B7.append(A[i])
        elif labels[i]==8:
            B8.append(A[i])
        elif labels[i]==9:
            B9.append(A[i])
    B1=np.array(B1);B2=np.array(B2);B3=np.array(B3);B4=np.array(B4);B5=np.array(B5);B6=np.array(B6);B7=np.array(B7);B8=np.array(B8);B9=np.array(B9)
    B=[B1,B2,B3,B4,B5,B6,B7,B8,B9]
    labelnum=[]
    Train=[]
    Test=[]
    for i in range(len(B)):
        numtrain=int(B[i].shape[0]*args.percenttrain)
        labelnum.append(numtrain)
        Train.append(B[i][:numtrain,:])
        Test.append(B[i][numtrain:,:])
    Train=np.concatenate(Train,0)
    Test=np.concatenate(Test,0)
    Point=np.concatenate((Train,Test),0)
    return Point,labelnum


def classacc(inlabel,outlabel):
    '''
    calculate the accuracy of every class
    '''
    label=np.unique(inlabel)
    acc=np.zeros((label.shape[0]))
    for i in range(label.shape[0]):
        idx=np.where(inlabel==label[i])
        idx=idx[0]
        acc[i]=np.sum(outlabel[idx]==inlabel[idx])/idx.shape[0]
    return acc

        
def confusionmatrix(outlabel,inlabel):
    '''
    calculate the confusion matrix
    '''
    label=np.unique(inlabel)
    matrix=np.zeros((label.shape[0],label.shape[0]))
    for i in range(label.shape[0]):
        for j in range(label.shape[0]):
            idx=np.where(inlabel==label[i])
            idx=idx[0]
            # matrix[i,j]=np.sum(outlabel[idx]==label[j])/idx.shape[0]
            matrix[i,j]=np.sum(outlabel[idx]==label[j])
    return matrix

def F_score(confusionmatrix):
    '''
    calculate the fscore of n class
    '''
    precision = np.zeros(confusionmatrix.shape[0])
    recall = np.zeros(confusionmatrix.shape[0])
    Fscore = np.zeros(confusionmatrix.shape[0])
    miou=0
    for n in range(confusionmatrix.shape[0]):

        TP = confusionmatrix[n,n]
        FP = np.sum(confusionmatrix[n,:])-TP
        FN = np.sum(confusionmatrix[:,n])-TP
        # TN = np.sum(confusionmatrix)-TP-TN-FP

        precision[n] = TP/(TP+FP)
        recall[n] = TP/(TP+FN)
        Fscore[n] = 2*precision[n]*recall[n]/(precision[n]+recall[n])

        miou = TP/(TP+FP+FN) + miou
    miou = miou/confusionmatrix.shape[0]  

    macro_P = np.sum(precision)/confusionmatrix.shape[0]
    macro_R = np.sum(recall)/confusionmatrix.shape[0]
    macro_F = 2*macro_P*macro_R/(macro_P+macro_R)
    return macro_P,macro_R,macro_F,precision,recall,Fscore,miou









if __name__ == '__main__':
    # A=pointloader('./Points/finaladj.txt')
    # A=[[1,2],[2,4]]
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(pd.DataFrame(np.round(A, 2)), annot=True,square=True, cmap="YlGnBu")
    # ax.set_title('adj', fontsize=18)
    # plt.show()
    # A *= 255
    # im = Image.fromarray(A)
    # im = im.convert('L') 
    # im.save('adj1.png')
    # A=[[1,3,2],[2,1,5],[3,5,1]]
    # A=np.array(A)
    # A=takemaxinrow(A,2)
    # print(A)
    A=[1,2,1,2,3,2,1,4,3]
    A=np.array(A).T
    B=[1,2,3,2,1,2,1,4,3]
    # B=[1,2,1,2,3,2,1,4,3]
    B=np.array(B).T
    C=confusionmatrix(A,B)
    print(C)
    ca=classacc(B,A)
    print(ca)
    macro_P,macro_R,macro_F,precision,recall,Fscore,miou=F_score(C)
    print("OAprecision=",macro_P)
    print("OArecall=",macro_R)
    print("OAFscore=",macro_F)
    print("precision=",precision)
    print("recall=",recall)
    print("Fscore=",Fscore)
    print("MIoU=",miou)
    # ax= sns.heatmap(pd.DataFrame(C), annot=True,square=True, cmap="YlGnBu")
    ax= sns.heatmap(pd.DataFrame(C), annot=True,square=True,xticklabels=["ant", "bird", "cat","tiger"], cmap="YlGnBu")
    plt.show()
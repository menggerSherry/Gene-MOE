import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import os
import numpy as np
import umap
plt.ioff()
def plot_R2(path,x,y):

    # time_str  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
    plt.figure(figsize=(8,4))
    plt.plot(x,y,'-',linewidth=1)
    plt.legend(['test R^2 score'])
    plt.xlabel("epochs")
    plt.ylabel("score")
    # os.makedirs(os.path.join(path,time_str),exist_ok=True)
    plt.savefig(os.path.join(path,'r2_score.png'),dpi=300)
    plt.close()

def plt_kl(path,x,y):
    plt.figure(figsize=(8,4))
    plt.plot(x,y,'-',linewidth=1)
    plt.legend(['test kl_div'])
    plt.xlabel("epochs")
    plt.ylabel("div")
    plt.savefig(os.path.join(path,'kl_div.png'),dpi=300)
    plt.close()

def plt_js(path,x,y):
    plt.figure(figsize=(8,4))
    plt.plot(x,y,'-',linewidth=1)
    plt.legend(['test js_div'])
    plt.xlabel("epochs")
    plt.ylabel("div")
    plt.savefig(os.path.join(path,'js_div.png'),dpi=300)
    plt.close()

def plot_tsne(path,x,filename):
    mid=x.shape[0]//2
    # print(mid)
    # color=['r' if i==1 else 'b' for i in label]
    label_legend=['real expression','reconstruct expression' ]
    tsne=TSNE(perplexity=40,learning_rate=200,init='pca')
    Y=tsne.fit_transform(x)
    fig, ax = plt.subplots()
    ax.scatter(Y[:mid,0],Y[:mid,1],s=50,c='r',alpha=0.5, edgecolors='none',label='real expression')
    ax.scatter(Y[mid:,0],Y[mid:,1],s=50,c='b', alpha=0.5, edgecolors='none',label='reconstruct expression')
    # ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig(os.path.join(path,filename),dpi=300)
    plt.close()
    # tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index)


def plot_umap(path,x,filename):
    mid=x.shape[0]//2
    # print(mid)
    # color=['r' if i==1 else 'b' for i in label]
    label_legend=['real expression','reconstruct expression' ]
    reducer = umap.UMAP(random_state=42)

    Y=reducer.fit_transform(x)
    fig, ax = plt.subplots()
    ax.scatter(Y[:mid,0],Y[:mid,1],s=50,c='r',alpha=0.5, edgecolors='none',label='real expression')
    ax.scatter(Y[mid:,0],Y[mid:,1],s=50,c='b', alpha=0.5, edgecolors='none',label='reconstruct expression')
    # ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig(os.path.join(path,filename),dpi=300)
    plt.close()
    # tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index)


def plot_scatter(path,x,filename):
    mid=x.shape[0]//2
    label_legend=['real expression','reconstruct expression' ]
    tsne=TSNE(perplexity=40,learning_rate=200,init='pca')
    Y_tsne=tsne.fit_transform(x)
    
    reducer = umap.UMAP(random_state=42)
    Y_umap=reducer.fit_transform(x)
    plt.figure(figsize=(10, 8))
    fig, (ax0,ax1) = plt.subplots(1,2,figsize=(10, 5))
    ax0.scatter(Y_tsne[:mid,0],Y_tsne[:mid,1],s=50,c='r',alpha=0.5, edgecolors='none',label='real')
    ax0.scatter(Y_tsne[mid:,0],Y_tsne[mid:,1],s=50,c='b', alpha=0.5, edgecolors='none',label='reconstruct')
    # ax.grid('off')
    # ax0.set_title('TSNE')
    ax0.set_xticks([])
    ax0.set_yticks([])
    
    ax1.scatter(Y_umap[:mid,0],Y_umap[:mid,1],s=50,c='r',alpha=0.5, edgecolors='none')
    ax1.scatter(Y_umap[mid:,0],Y_umap[mid:,1],s=50,c='b', alpha=0.5, edgecolors='none')
    # ax.grid('off')
    # ax1.set_title('UMAP')
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.tight_layout()
    # plt.legend(loc='lower center',bbox_to_anchor=(1, 0),ncol=2)
    plt.savefig(os.path.join(path,filename),dpi=1000)
    
    plt.close()


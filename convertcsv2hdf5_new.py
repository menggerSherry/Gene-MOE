import h5py
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from sklearn.model_selection import KFold,train_test_split
import sys

def norm(scaler,input_data):
        # maximums,minimums,avgs = input_data.max(axis = 0), input_data.min(axis = 0),input_data.sum(axis = 0)/input_data.shape[0]
        # #归一化
        # norm_1=(input_data-minimums)/(maximums-minimums)
        input_data=scaler.transform(input_data)
        print('max',input_data.max())
        print('min',input_data.min())

        return (input_data-0.5)/0.5
    

# pretrain
data = pd.read_csv('./TCGA_pre/data_new/GDC_PANCANCER.htseq_fpkm-uq_final.tsv', sep='\t', index_col=0)

# bio_mart_data = pd.read_csv('./TCGA_pre/gencodev22.tsv',delimiter='\t')
# print(bio_mart_data.head(5))
# coor_frame = pd.merge(data, bio_mart_data, how = 'left', left_index=True, right_index=True)
# print(coor_frame.shape) 
data = data.dropna()
# data = data.set_index('Symbol')
print(data.head(5))
# print(data.shape)
pan_gene_id = np.array(data.index,dtype=object)
print(2132132,pan_gene_id)


# 
# sys.exit()
# 

data=data.values.astype(np.float32).T
# 9896,19760
print("pan cancer shape:",data.shape)
print(np.max(data))
print(np.min(data))
# data=norm(data)
print(np.max(data))
print(np.min(data))

# split
data_train,data_test=train_test_split(data,test_size=0.1,random_state=0)
print('len',data_test.shape[0])
print('len',data_train.shape[0])

scaler=preprocessing.MinMaxScaler()
scaler.fit(data)
data_train=norm(scaler,data_train)
data_test=norm(scaler,data_test)

# data=norm(data)
print(np.max(data_train))
print(np.min(data_test))
print('converting...')
data_file=h5py.File("./data_new/GDC_PANCANCER.htseq_fpkm-uq_finalpretrain.hdf5","w")
string_dt = h5py.special_dtype(vlen=str)
g_gene = data_file.create_group("exp_name")
g_gene.create_dataset('ENSG',data=pan_gene_id,dtype=string_dt)
# g_gene=data_file.create_group('Ensembl_id')
# g_gene.create_dataset("ENSG",data=gene_id)
length=data_file.create_group('dataset_dim')
length.create_dataset('train',data=data_train.shape)
length.create_dataset('test',data=data_test.shape)

g=data_file.create_group('pancancer_exp')
# print(data[1,:].shape)
# print(np.max(data[2,:]))
# print(data[1,:])
index=0
for i in range(data_train.shape[0]):
    g.create_dataset('train_%d'%i,data=data_train[i,:])
    index+=1

index2=0
for j in range(data_test.shape[0]):
    g.create_dataset('test_%d'%j,data=data_test[j,:])
    index2+=1

print('finished!')
print('total is',index)
print('total test is',index2)
print(len(g.keys()))
data_file.close()
print()
print()


print("convert 33 cancer type....")

class_data_train = []
class_label_train = []


class_data_test = []
class_label_test = []

class_data_total = []

for i, data_type in enumerate(["TCGA-BLCA","TCGA-BRCA","TCGA-KIRC","TCGA-HNSC","TCGA-LGG", 
  "TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-OV","TCGA-STAD","TCGA-COAD","TCGA-SARC",
  "TCGA-UCEC","TCGA-CESC","TCGA-PRAD","TCGA-SKCM", "TCGA-UCS", "TCGA-THCA","TCGA-THYM", "TCGA-TGCT","TCGA-READ",
  "TCGA-PCPG", "TCGA-PAAD", "TCGA-UVM", "TCGA-MESO", "TCGA-DLBC", "TCGA-KIRP", "TCGA-KICH", "TCGA-GBM", "TCGA-ESCA",
  "TCGA-CHOL", "TCGA-ACC", "TCGA-LAML"]):
    data=pd.read_csv(os.path.join('TCGA_pre','data_new','%s.htseq_fpkm-uq_final.tsv' % data_type),sep='\t',index_col=0)

    # bio_mart_data = pd.read_csv('./TCGA_pre/genecodev22,tsv',delimiter='\t')
    # print(bio_mart_data.head(5))
    # coor_frame = pd.merge(data, bio_mart_data, how = 'left', left_index=True, right_index=True)
    # print(coor_frame.shape) 
    data = data.dropna()
    # data = data.set_index('Symbol')
    gene_id = np.array(data.index,dtype=object)
    if((gene_id==pan_gene_id).all()):
        print('equal')
    else:
        print('not equal')
        sys.exit()
    # gene_id=np.array(data.index,dtype=object)
    print('%s:'%data_type)
    print(gene_id)
    print(gene_id.shape)
    data=data.values.astype(np.float32).T
    label = np.full((data.shape[0],1),i).astype(np.float32)
    data = norm(scaler, data)
    print(data.max())
    print(data.min())

    data = np.concatenate((data,label),axis=1)
    # data_train,data_test=train_test_split(data,test_size=0.2,random_state=0)
    class_data_total.append(data)
    # class_data_train.append(data_train)
    # class_data_test.append(data_test)
    # data_lable = np.concatenate((data,label),axis=1)
    # class_data.append(data)
    
data_total = np.concatenate(class_data_total,axis=0)



class_file = h5py.File("./data_new/GDC_PANCANCER.classification.hdf5","w")

g = class_file.create_group("exp")

for idx in range(5):

    f=g.create_group('cross_%d'%idx)
    train_data,test_data=train_test_split(data_total,test_size=0.2)
    
    # train_data=exp_data[train_index,:]
    # test_data=exp_data[test_index,:]
    # norm

    f.create_dataset('train',data=train_data)
    f.create_dataset('test',data=test_data)

    print('finished!')
    

# g.create_dataset('train',data=classify_data_train)
# g.create_dataset('test',data=classify_data_test)

class_file.close()

print("finished create!")
print()


print("convert cancer type....")

for data_type in ["TCGA-BLCA","TCGA-BRCA","TCGA-KIRC","TCGA-HNSC","TCGA-LGG","TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-OV",
                  "TCGA-STAD","TCGA-COAD","TCGA-SARC","TCGA-UCEC","TCGA-CESC","TCGA-PRAD","TCGA-SKCM"]:
    data=pd.read_csv(os.path.join('TCGA_pre','data_new','%s.htseq_fpkm-uq_finalsurviva.tsv' % data_type),sep='\t',index_col=0)
    # data.head()
    clinical_data=pd.read_csv(os.path.join('TCGA_pre','data_new','%s.survival_clean.tsv' % data_type),sep='\t',index_col=0)

    
    
    data = data.dropna()
    # data = data.set_index('Symbol')
    gene_id = np.array(data.index,dtype=object)
    if((gene_id==pan_gene_id).all()):
        print('equal')
    else:
        print('not equal')
        sys.exit()
    # gene_id=np.array(data.index,dtype=object)

    data=data.values.astype(np.float32).T

    # scaler_mrna=preprocessing.MinMaxScaler()
    # scaler_mrna.fit(data)
    data=norm(scaler,data)
    print("shape:",data.shape)
    print("end")
    print(data.max())
    print(data.min())



    
    os_event=clinical_data['OS'].values.astype(np.int32).reshape((-1,1))
    os_time=clinical_data['OS.time'].values.astype(np.int32).reshape((-1,1))
    # print('normalizing....')

    # data=norm(data)
    # print(np.max(data))
    # print(np.min(data))

    if os_event.shape[0] == os_time.shape[0] and os_event.shape[0]==data.shape[0]:
        exp_data=np.concatenate((data,os_event,os_time),axis=1)
        print(exp_data.shape)
    else:
        raise NotImplementedError("error")

    data_file=h5py.File('./data_new/%s.5_folds.hdf5'%data_type,'w')
    string_dt = h5py.special_dtype(vlen=str)
    g_gene=data_file.create_group('exp_name')
    g_gene.create_dataset('ENSG',data=gene_id,dtype=string_dt)
    g=data_file.create_group('exp')
    print('exp_data',exp_data.shape)
    # print(exp_data.max())
    # print(exp_data.min())
    # # exp_data=norm(exp_data)
    # print(exp_data.max())
    # print(exp_data.min())
    
    # data_train,data_test=train_test_split(exp_data,test_size=0.2,random_state=0)
    # g.create_dataset('train',data = data_train)
    # g.create_dataset('test', data = data_test)
    # data_file.close()
    
    
    # kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for idx in range(5):

        f=g.create_group('cross_%d'%idx)
        train_data,test_data=train_test_split(exp_data,test_size=0.2)
        
        # train_data=exp_data[train_index,:]
        # test_data=exp_data[test_index,:]
        # norm

        f.create_dataset('train',data=train_data)
        f.create_dataset('test',data=test_data)

    print('finished!')
    data_file.close()
    # print()
    # print()

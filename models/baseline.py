import sys
sys.path.append('/share/home/202220143416/project/LLM_CFD')
sys.path.append('/share/home/202220143416/project/LLM_CFD/utils')

from utils.get_data import getRawDataset, getSplitedDataSet
sys.path.append('/share/home/202220143416/project/LLM_CFD/deepod')

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder_torch import AutoEncoder
# from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lscp import LSCP
from pyod.models.ecod import ECOD
from pyod.models.gmm import GMM

from deepod.models.tabular.trans import MSE_Transformer



from deepod.models.tabular.dsvdd import DeepSVDD
from pyod.models.lunar import LUNAR



import numpy as np




def get_model(model_name,device='cuda'):
    EPOCHS = 10
    BS_NUM = 64
    if model_name == 'KNN':
        return KNN()
    elif model_name == 'LOF':
        return LOF()
    elif model_name == 'PCA':
        return PCA(n_components=5)
    elif model_name == 'IForest':
        return IForest(n_estimators=1000,n_jobs=8,verbose=1)
    elif model_name == 'OCSVM':
        return OCSVM(max_iter=1000)
    elif model_name == 'AE':
        return AutoEncoder(epochs=EPOCHS,batch_size=BS_NUM,device='cuda',preprocessing=False)
    elif model_name == 'LSCP':
        return LSCP([KNN(),PCA(n_components=5)])
    elif model_name == 'ECOD':
        return ECOD(n_jobs=8)
    elif model_name == 'GMM':
        return GMM()
    elif model_name == 'DeepSVDD':
        return DeepSVDD(epochs=EPOCHS,batch_size=BS_NUM,rep_dim=128,device=device)
    elif model_name == 'LUNAR':
        return LUNAR(n_epochs=EPOCHS)
    elif model_name == 'Transformer':
        return MSE_Transformer(device=device)
    else:
        raise ValueError("no model")
    
def trans_dataset_2_pyod(concat_dataset):
    # 如果ConcatDataset包含的是数据和标签的元组
    all_data, all_labels = [], []
    for i in range(len(concat_dataset)):
        data_point, label = concat_dataset[i]
        all_data.append(data_point.numpy())
        all_labels.append(label.numpy())

    # print(data_point.shape)
    # print(label.shape)
    all_data = np.stack(all_data,dtype=np.float32)
    all_labels = np.stack(all_labels,dtype=np.float32)
    return all_data, all_labels
    
from sklearn.preprocessing import MinMaxScaler

def get_scores_labels(model, x_train_dataset, x_test_dataset,train_on_test=0,sampling=0):
    x_test,labels = trans_dataset_2_pyod(x_test_dataset)
    if not train_on_test:
        x_train,_ = trans_dataset_2_pyod(x_train_dataset)
        if sampling:
            N_test = len(x_test)
            sampling_idx = np.random.choice(len(x_train),N_test,replace=False)
            x_train = x_train[sampling_idx]

    ms = MinMaxScaler()
    x_train = ms.fit_transform(x_train)
    x_test = ms.transform(x_test)
    
    if train_on_test:
        model.fit(x_test)
    else:
        model.fit(x_train)
    scores = model.decision_function(x_test)
    return scores, labels


def test1():
    import numpy as np
    import torch
    from tqdm import tqdm
    dataset1 = 'dh'
    # dataset1 = 'jop'
    raw_data = getRawDataset(type_=dataset1)
    test_num = raw_data.y1_num()
    test_left_num = 0

    train_data,test_data,enc_in = getSplitedDataSet(raw_dataset=raw_data,test_num=test_num,test_left_num=test_left_num).getSpliedDatasets()

    for m in USABLE_MODELS:
        model = get_model(m)
        print(m)
        TRAIN_ON_TRAIN = ['AE','VAE','DeepSVDD','GMM','ECOD']
        if m in TRAIN_ON_TRAIN:
            train_on_test = 0
        else:
            train_on_test = 1
        scores, labels = get_scores_labels(model, train_data, test_data, train_on_test)
        # 确定一个合适的最大分数阈值，例如使用scores中的最大值
        max_score_threshold = np.max(scores[scores != np.inf])
        # 替换scores中的无穷大值为max_score_threshold
        scores = np.where(np.isinf(scores), max_score_threshold, scores)
        # replace inf in scores to max
        
        print(scores.shape)
        print(labels.shape)
        print(scores[:10])


def test2():
    dataset1 = 'shu'
    # root_path = 
    raw_data = getRawDataset(type_=dataset1)
    test_num = raw_data.y1_num()
    test_left_num = 0
    train_data,test_data,enc_in = getSplitedDataSet(raw_dataset=raw_data,test_num=test_num,test_left_num=test_left_num).getSpliedDatasets()
    print(train_data.__dict__)
    print(train_data.datasets)

USABLE_MODELS = ['KNN', 'LOF', 'PCA', 'IForest', 'OCSVM', 
                 'AE', 'LSCP', 'ECOD', 'GMM',
                   'DeepSVDD', 'LUNAR']

# USABLE_MODELS = ['ECOD', 'GMM',
#                    'DeepSVDD', 'LUNAR']

    

if __name__ == '__main__':
    test1()

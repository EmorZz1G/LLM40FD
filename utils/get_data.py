import csv
import torch
from torch.utils.data.dataset import Dataset,ConcatDataset,TensorDataset
from torch.utils.data import random_split
torch.manual_seed(0)#for reproducibility
import pandas as pd
import numpy as np
import os

class getRawDataset(Dataset):
    def __init__(self, type_='credit', base_pth=r'/share/home/202220143416/anomaly_data/fraud_detection/', preprocess='none'):
        if base_pth is None:
            raise RuntimeError("NO dataset pth")

        if type_ in ['credit', 'cd']:
            pth = os.path.join(base_pth,'creditcard.csv')
            pth1 = os.path.join(base_pth,'creditcard.npz')
            if os.path.exists(pth1):
                dz = np.load(pth1)
                self.features = dz['features']
                self.labels = dz['labels']
            else:
                # data loading
                self.features = []
                self.labels = []
                csvCreditCard = open(pth)
                CreditCardData = pd.read_csv(csvCreditCard)
                self.features = CreditCardData.iloc[:, :-1].values
                self.labels = CreditCardData.iloc[:, -1].values

        elif type_ in [ 'shubhamjoshi2130of', 'shu']: 
            pth = os.path.join(base_pth,'creditcardcsvpresent.csv')
            pth1 = os.path.join(base_pth,'creditcardcsvpresent.npz')

            if os.path.exists(pth1):
                dz = np.load(pth1)
                self.features = dz['features']
                self.labels = dz['labels']
            else:
                df = pd.read_csv(pth)
                df.loc[df['Is declined'] == 'N', 'Is declined'] = 0
                df.loc[df['Is declined'] == 'Y', 'Is declined'] = 1
                df['Is declined'].unique()

                df.loc[df['isForeignTransaction'] == 'N', 'isForeignTransaction'] = 0
                df.loc[df['isForeignTransaction'] == 'Y', 'isForeignTransaction'] = 1
                df['isForeignTransaction'].unique()

                df.loc[df['isHighRiskCountry'] == 'N', 'isHighRiskCountry'] = 0
                df.loc[df['isHighRiskCountry'] == 'Y', 'isHighRiskCountry'] = 1
                df['isHighRiskCountry'].unique()

                df.loc[df['isFradulent'] == 'N', 'isFradulent'] = 0
                df.loc[df['isFradulent'] == 'Y', 'isFradulent'] = 1
                df['isFradulent'].unique()

                df.dropna(axis=1, inplace=True)

                df = df.rename(columns={'Average Amount/transaction/day': 'AverageAmount_transaction_day'})
                df = df.rename(columns={'Is declined': 'IsDeclined'})
                df = df.rename(columns={'Total Number of declines/day': 'TotalNumberOfDeclines_day'})
                df = df.rename(columns={'6-month_chbk_freq': '6_month_chbk_freq'})

                # avgf = df[df['isFradulent'] == 1].Transaction_amount / df.AverageAmount_transaction_day
                # avgl = df[df['isFradulent'] == 0].Transaction_amount / df.AverageAmount_transaction_day

                df['avgs'] = df.Transaction_amount / df.AverageAmount_transaction_day
                

                Y = df['isFradulent']
                X = df.drop(['Merchant_id', 'isFradulent'], axis=1)

                X = X.to_numpy().astype(float)
                Y = Y.to_numpy().astype(int)
                self.features = X
                self.labels = Y

        elif type_ in ['jaini_ol_pay','jop']:
            pth = os.path.join(base_pth,'onlinefraud.csv')
            pth1 = os.path.join(base_pth,'onlinefraud.npz')
            if os.path.exists(pth1):
                dz = np.load(pth1)
                self.features = dz['features']
                self.labels = dz['labels']
            else:
                data = pd.read_csv(pth, sep=",")

                dummies = pd.get_dummies(data["type"], prefix='Type_')

                data.drop("type", inplace=True, axis=1)
                data = pd.concat([data, dummies], axis='columns')

                data["NAMEORIG_LETTER"] = data.apply(lambda row: row["nameOrig"][0], axis=1)
                # data["NAMEORIG_NUM"]  = data.apply(lambda row: float(row["nameOrig"][1:]), axis=1)
                data["F_NAMEORIGC"] = data["NAMEORIG_LETTER"].map({"C": 1.0, "M": 0.0})
                # data["F_NAMEORIGM"] = data["NAMEORIG_LETTER"].map({"C":0.0, "M":1.0})
                data = data.drop("NAMEORIG_LETTER", axis=1)
                data = data.drop("nameOrig", axis=1)

                data["NAMEDEST_LETTER"] = data.apply(lambda row: row["nameDest"][0], axis=1)
                # data["NAMEDEST_NUM"]  = data.apply(lambda row: float(row["nameDest"][1:]), axis=1)
                data["F_NAMEDESTC"] = data["NAMEDEST_LETTER"].map({"C": 1.0, "M": 0.0})
                # data["F_NAMEDESTM"] = data["NAMEDEST_LETTER"].map({"C":0.0, "M":1.0})
                data = data.drop("NAMEDEST_LETTER", axis=1)
                data = data.drop("nameDest", axis=1)

                data['step'] = (data['step'] - data['step'].mean()) / data['step'].std()


                def stdz_column(df, colname):
                    df[colname] = (df[colname] - df[colname].mean()) / df[colname].std()

                labels = data["isFraud"]
                labels = labels.astype(np.int32)
                data.drop("isFraud", inplace=True, axis=1)
                # data.drop("isFlaggedFraud", inplace=True, axis=1)
                df_train = data.to_numpy().astype(np.float32)

                labels = labels.to_numpy()
                self.features = df_train
                self.labels = labels

        elif type_ in ['dhanush','dh']:
            pth = os.path.join(base_pth,'card_transdata.csv')
            pth1 = os.path.join(base_pth,'card_transdata.npz')
            if os.path.exists(pth1):
                dz = np.load(pth1)
                self.features = dz['features']
                self.labels = dz['labels']
            else:
                raw_df = pd.read_csv(pth)

                df = raw_df.copy()

                df.drop_duplicates(inplace=True)
                print("Duplicated values dropped succesfully")
                print("*" * 100)
                print(df.shape)
                y = df['fraud']
                X = df.drop('fraud', axis=1)
                self.features = X.to_numpy()
                self.labels = y.to_numpy()

        elif type_ in ['20k']:
            pth = os.path.join(base_pth,'fraud_detection_bank_dataset.csv')
            pth1 = os.path.join(base_pth,'fraud_detection_bank_dataset.npz')

            if os.path.exists(pth1):
                dz = np.load(pth1)
                self.features = dz['features']
                self.labels = dz['labels']
            else:
                nRowsRead = None  # specify 'None' if want to read whole file
                df1 = pd.read_csv(pth, delimiter=',', nrows=nRowsRead)
                nRow, nCol = df1.shape

                x = df1.iloc[:, :-1].to_numpy()
                y = df1.iloc[:, -1].to_numpy()
                self.features = x
                self.labels = y

        else:
            raise ValueError("Invalid dataset type")

        if not os.path.exists(pth1):
            data_process = {
                'features':self.features,
                'labels':self.labels,
            }
            np.savez(pth1,**data_process)
            print('Init, save precess dataset')

        from sklearn.preprocessing import MinMaxScaler
        if preprocess == 'minmax':
            ms = MinMaxScaler()
            self.features = ms.fit_transform(self.features)


        y0 = np.sum(self.labels == 0)
        y1 = np.sum(self.labels == 1)
        print(f"Number of non-fraudulent transactions: {y0}")
        print(f"Number of fraudulent transactions: {y1}")
        print(f"Ratio of fraudulent transactions: {y1 / (y0 + y1)}")
        print(f"Number of Features: {len(self.features[0])}")

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.features)

    def y1_num(self):
        return np.sum(self.labels == 1)
        
    def feat_dim(self):
        return len(self.features[0])

    def get_non_fraud(self):
        idx = self.labels==0
        tt = TensorDataset(torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx]))
        return  tt
    
    def get_fraud(self):
        idx = self.labels==1
        tt = TensorDataset(torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx]))
        return  tt
        

class getSplitedDataSet():
    def __init__(self, raw_dataset, test_num=490, test_left_num=2):
        non_fraud_data = raw_dataset.get_non_fraud()
        fraud_data = raw_dataset.get_fraud()
        self.c_dim = raw_dataset.feat_dim()
        data_point_num = len(non_fraud_data)
        train_data_point_num = data_point_num - test_num
        # print(len(non_fraud_data[0]),train_data_point_num,test_num)

        train0, test0 = random_split(non_fraud_data, [train_data_point_num, test_num])
        test1, train1 = random_split(fraud_data, [test_num, test_left_num])
        self.train = ConcatDataset([train0, train1])
        self.test = ConcatDataset([test0, test1])

    def getSpliedDatasets(self):
        return (self.train, self.test, self.c_dim)
    


        
if __name__ == "__main__":
    data = getRawDataset()
    t1,t2,c = getSplitedDataSet(data).getSpliedDatasets()
    print(c)
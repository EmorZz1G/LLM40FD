import argparse
import ast
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import precision_recall_curve
def mystr(s):
    # print(s)
    if s.startswith('['):
        # print(s)
        x = []
        for i in s:
            if i in ['[',']',"\"","\'"]:
                continue
            x.append(i)
        x = ''.join(x)
        # print(x)
        x = x.split(',')
        x=list(map(str,x))
        # print(x)
        return x
    x = s.split(',')
    x = list(map(str,x))
    return x

parser = argparse.ArgumentParser('LLM40CFD')
# basic config
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--debug', type=int, default=0, help='debug mode')
parser.add_argument('--saving_name', type=str, default='TSNE', help='saving_name of dir')
parser.add_argument('--zero_shot', type=int, default=1, help='zero shot')


# data loader
parser.add_argument('--dataset1', type=str, default='shu',choices=['20k','credit','jop','shu'], help='dataset type')
parser.add_argument('--dataset2', type=mystr, default="credit,shu", help='dataset type')
parser.add_argument('--root_path', type=str, default=r'./datasets/fraud_detection/', help='root path of the data file') 

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (NO NEED FOR THIS, AUTO-OBTAIN THE ENC_IN)')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--strides', type=eval, default=[1,3,5,11], help='strides of walking embedding')
parser.add_argument('--llm_layers', type=int, default=3)
parser.add_argument('--llm_model', type=str, default='GPT2', choices=['GPT2', 'LLAMA', 'BERT', 'None'], help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default=768,choices=[4096,768], help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--train_epochs','-ep', type=int, default=1, help='train epochs')
parser.add_argument('--few_data_per', type=float, default=0.1,choices=[0.05,0.1,0.3,0.5,1.], help='train epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=64, help='batch size of train input data')
parser.add_argument('--learning_rate','-lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='OC_CE', help='loss function')
parser.add_argument('--pos_r', type=float, default=0.8, help='pos ratio')
parser.add_argument('--neg_r', type=float, default=2, help='neg ratio')


config = parser.parse_args()

from models.LLM_CFD import LLM_CFD
from utils.get_data import getSplitedDataSet, getRawDataset
import torch
from losses.loss import DSVDDLoss,DSVDDLossCE
from tqdm import tqdm
from math import ceil
import os
import pandas as pd
from utils.metrics import get_score_list, save_scores_labels
from sklearn.preprocessing import minmax_scale

def get_loss(config):
    loss_cls = config.loss
    if loss_cls == 'OC':
        loss_cls = DSVDDLoss
    elif loss_cls == 'OC_CE':
        loss_cls = DSVDDLossCE
    else:
        raise ValueError("no loss")
    return loss_cls

def get_scores_labels_latents(model, test_loader, config):
    import numpy as np
    import torch.nn.functional as F
    device = config.device
    loss_cls = get_loss(config)

    print('Using loss of '+ loss_cls.__name__)

    with torch.no_grad():
        model.eval()
        scores = []
        labels = []
        latents = []
        kv_ch = None
        for x,y in tqdm(test_loader):
            x = x.float().to(device)
            out,lt,kv_ch = model(x,it=0)
            criteria = loss_cls(model.get_c())
            criteria.reduction = 'none'
            if config.loss == 'OC_CE':
                loss = criteria(torch.softmax(out,1))
                loss = loss.mean(1)
            else:
                loss = criteria(out)
            
            loss = loss.cpu().numpy()

            latents.append(out.cpu().numpy())
            scores.append(loss)
            labels.append(y)
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        latents = np.concatenate(latents)
    return scores, labels, latents

from transformers import GPT2LMHeadModel,GPT2Config

gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
llm_model2 = GPT2LMHeadModel.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=gpt2_config,
                )

lm_head = None
import numpy as np

def get_scores_labels_outputs(model:LLM_CFD, test_loader, config):
    import numpy as np
    import torch.nn.functional as F
    device = config.device
    loss_cls = get_loss(config)
    global lm_head
    if lm_head is None:
        lm_head = llm_model2.lm_head.to(device)

    print('Using loss of '+ loss_cls.__name__)

    with torch.no_grad():
        model.eval()
        scores = []
        labels = []
        says = []
        kv_ch = None
        for x,y in tqdm(test_loader):
            x = x.float().to(device)
            out,lt,kv_ch = model(x,it=0)
            criteria = loss_cls(model.get_c())
            criteria.reduction = 'none'
            if config.loss == 'OC_CE':
                loss = criteria(torch.softmax(out,1))
                loss = loss.mean(1)
            else:
                loss = criteria(out)


            for i in range(len(model.in_layers)):
                if i == 0:
                    outputs = model.in_layers[i](x)
                else:
                    try:
                        outputs = torch.cat((outputs, model.in_layers[i](x)), dim=1)
                    except Exception as e:
                        continue

            if model.llm_model is not None:
                latent = model.llm_model(inputs_embeds=outputs)
                kvcache = latent.past_key_values
                latent = latent.last_hidden_state

            ohead = lm_head(latent)
            say_output = model.tokenizer.batch_decode(ohead.argmax(-1))
            
            loss = loss.cpu().numpy()
            scores.append(loss)
            labels.append(y)
            says.append(say_output)

        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        says = np.concatenate(says)
    return scores, labels, says


from sklearn.manifold import TSNE

def main(config):
    debuging = config.debug
    dataset1 = config.dataset1
    root_path = config.root_path
    preprocess = 'minmax'
    raw_data = getRawDataset(type_=dataset1, base_pth=root_path, preprocess=preprocess)
    test_num = raw_data.y1_num()
    test_left_num = 0

    train_data,test_data,enc_in = getSplitedDataSet(raw_dataset=raw_data,test_num=test_num,test_left_num=test_left_num).getSpliedDatasets()

    num_workers = config.num_workers
    batch_size = config.batch_size
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    config.enc_in = enc_in
    device = config.device

    model = LLM_CFD(config)
    model = model.to(device)

    lr = config.learning_rate
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    few_data_per = config.few_data_per

    # Training
    data_bs_num = int (ceil(len(train_loader)*few_data_per))
    pos_r = config.pos_r
    neg_r = config.neg_r
    loss_cls = get_loss(config)
    epochs = config.train_epochs

    data_saving_base_pth = './imgs'
    saving_name = config.saving_name
    loss_name = config.loss

    score_pth = os.path.join(data_saving_base_pth, saving_name)
    os.makedirs(score_pth, exist_ok=True)
    print('scores will be saved in '+score_pth)

    for i in range(epochs):
        for ii,(x,y) in tqdm(enumerate(train_loader),total=data_bs_num,desc='train'):
            if ii+1 == ( data_bs_num ):
                break
            model.train()
            x = x.float().to(device)
            out,lt,ch = model(x,it=debuging)

            mu,sig = torch.mean(x,dim=0,keepdim=True),torch.std(x,dim=0,keepdim=True)

            if pos_r > 0:
                x1 = pos_r * torch.randn_like(x) * sig + mu
                out1,lt1,ch = model(x1,it=debuging)
            if neg_r > 0:
                x2 = neg_r * torch.randn_like(x) * sig + x
                out2,lt2,ch = model(x2,it=debuging)

            optimizer.zero_grad()
            criteria = loss_cls(model.get_c())
            if config.loss != 'OC':
                Lout = torch.softmax(out, dim=-1)
                if pos_r > 0: Lout1 = torch.softmax(out1, dim=-1)
                if neg_r > 0: Lout2 = torch.softmax(out2, dim=-1)
            else:
                Lout = out
                if pos_r > 0: Lout1 = out1
                if neg_r > 0: Lout2 = out2
            
            loss_c1 = criteria(Lout)
            if pos_r > 0: loss_c1p = criteria(Lout1)
            if neg_r > 0: loss_c1n = criteria(Lout2)
            
            loss = loss_c1
            if pos_r > 0: loss += loss_c1p
            if neg_r > 0: loss -= loss_c1n 

            loss.backward()
            optimizer.step()

            if ii%(max(len(train_loader)//100,10))==0:
                print(f'A{loss_c1.item():.5f}')
                if pos_r > 0: print(f'P{loss_c1p.item():.5f}')
                if neg_r > 0: print(f'N{loss_c1n.item():.5f}')

        # # TSETING D1
        # scores, labels = get_scores_labels(model, test_loader, config)
        # scores = minmax_scale(scores)
        # scores_list, scores = get_score_list(labels,scores, dataset1, dataset1, None)
        # for k,v in scores_list.items():
        #     print(f'{k}:{v:.4f}')

        # save_scores_labels(scores, labels, dataset1, dataset1, None)


    
    # TSETING D1
    scores, labels , latents = get_scores_labels_latents(model, test_loader, config)
    scores = minmax_scale(scores)
    scores_list, scores = get_score_list(labels,scores, dataset1, dataset1, None)
    for k,v in scores_list.items():
        print(f'{k}:{v:.4f}')

    name = f'hist_{loss_name}_{dataset1}_{dataset1}{scores_list["best_f1"]:.4f}.pdf'
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(latents)
    # save img
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = ['g', 'r']
    target_names = ['non-fraud', 'fraud']
    myscores = minmax_scale(scores)
    precision, recall, thresholds = precision_recall_curve(labels, myscores)
    f1_list = 2 * precision * recall / (precision + recall + 1e-6)
    f1_idx = f1_list.argmax()
    best_threshold = thresholds[f1_idx]
    for i, c, label in zip([0,1], colors, target_names):
        data_tm = myscores[labels == i]
        plt.hist(data_tm, bins=100, color=c, alpha=0.5, label=label)

    plt.axvline(best_threshold, color='b', linestyle='--', label='Threshold')
    plt.legend()
    plt.savefig(os.path.join(score_pth,name),bbox_inches='tight')
    plt.show()




    # ZERO SHOT TSETING
    dataset2_list = config.dataset2
    zere_shot_loader = None
    for dataset2 in dataset2_list:
        if dataset2 == dataset1:continue

        raw_data2 = getRawDataset(type_=dataset2, base_pth=root_path)
        test_num2 = raw_data2.y1_num()
        test_left_num2 = 0
        train_data2,test_data2,enc_in2 = getSplitedDataSet(raw_dataset=raw_data2,test_num=test_num2,test_left_num=test_left_num2).getSpliedDatasets()
        zere_shot_loader = torch.utils.data.DataLoader(test_data2, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        scores, labels, latents = get_scores_labels_latents(model, zere_shot_loader, config)
        scores = minmax_scale(scores)
        scores_list, scores = get_score_list(labels,scores,dataset1, dataset2, None)
        # save scores_list
        for k,v in scores_list.items():
            print(f'{k}:{v:.4f}')

        name = f'hist_{loss_name}_{dataset1}_{dataset2}{scores_list["best_f1"]:.4f}.pdf'
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(latents)
        # save img
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        colors = ['g', 'r']
        target_names = ['non-fraud', 'fraud']
        myscores = minmax_scale(scores)
        precision, recall, thresholds = precision_recall_curve(labels, myscores)
        f1_list = 2 * precision * recall / (precision + recall + 1e-6)
        f1_idx = f1_list.argmax()
        best_threshold = thresholds[f1_idx]
        for i, c, label in zip([0,1], colors, target_names):
            data_tm = myscores[labels == i]
            plt.hist(data_tm, bins=100, color=c, alpha=0.5, label=label)

        plt.axvline(best_threshold, color='b', linestyle='--', label='Threshold')
        plt.legend()
        plt.savefig(os.path.join(score_pth,name),bbox_inches='tight')
        plt.show()

        

    

if __name__ == '__main__':
    # CONFIG
    config.dataset2 = config.dataset2
    print(config)
    
    # print(config.strides)
    # MAIN
    main(config)

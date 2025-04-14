import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, BertModel, BertConfig, BertTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer

from .layers import WalkingEmbedding
import torch

class Block(nn.Module):

    def __init__(self, emb_dim, d_model):
        super(Block, self).__init__()
        self.llm_dim = emb_dim
        self.d_model = d_model
        self.out_layer = nn.Sequential(
            # nn.LayerNorm(self.llm_dim),
            nn.BatchNorm1d(self.llm_dim),
            nn.Linear(self.llm_dim, self.d_model),
            nn.ReLU(),
            # nn.LayerNorm(self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        self.out_layer_linear = nn.Linear(self.llm_dim, self.d_model)

    def forward(self, x):
        x1 = self.out_layer(x)
        x2 = self.out_layer_linear(x)
        return x1 + x2
    

from einops.layers.torch import Rearrange

class LLM_CFD(nn.Module):
    def __init__(self, configs, wkstps = [1, 3, 5, 11]):
        super(LLM_CFD, self).__init__()
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.llm_dim = configs.llm_dim

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = False
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = False
            self.gpt2_config.output_hidden_states = True

            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            self.tokenizer = None
            self.llm_model = None
            print('LLM model is not defined')

        if -1 not in wkstps:
            self.in_layers = nn.ModuleList([WalkingEmbedding(c_in=self.enc_in, emb_dim=self.llm_dim, walking_step=wkstp) for wkstp in wkstps] )
        else:
            self.in_layers = nn.ModuleList([Rearrange('b n -> b n 1'), nn.Linear(1, self.llm_dim)])

        out_layers = 1
        self.out_layer = nn.Sequential()
        for i in range(out_layers):
            if i == 0:
                self.out_layer.add_module('block_{}'.format(i), Block(self.llm_dim, self.d_model))
            else:
                self.out_layer.add_module('block_{}'.format(i), Block(self.d_model, self.d_model))

        # Freeze the gpt network
        if self.llm_model is not None:
            if configs.zero_shot == 1:
                for i, (name, param) in enumerate(self.llm_model.named_parameters()):
                    if 'norm' in name or 'ln' in name or 'bias' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = False
            else:
                print('Fine-tuning the LLM model')
                for i, (name, param) in enumerate(self.llm_model.named_parameters()):
                    if 'norm' in name or 'ln' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = True

        self.c = nn.Parameter(torch.randn(self.d_model))

    def forward(self, x, kvcache=None, it=0):
        B, M = x.shape
        for i in range(len(self.in_layers)):
            if i == 0:
                outputs = self.in_layers[i](x,it)
            else:
                try:
                    outputs = torch.cat((outputs, self.in_layers[i](x,it)), dim=1)
                except Exception as e:
                    continue

        if it:print(outputs.shape)
        if self.llm_model is not None:
            latent = self.llm_model(inputs_embeds=outputs,past_key_values=kvcache)
            kvcache = latent.past_key_values
            latent = latent.last_hidden_state
            latent = torch.mean(latent,1)
            # latent = latent[:,-1,:] # take the last token
            # latent = torch.max(latent,1).values
        else:
            latent = torch.mean(outputs,1)
        outputs = self.out_layer(latent) 
       
        return outputs,latent,kvcache

    def get_c(self):
        return self.c
        
# Comum libraries
import os
import random
from typing import Dict
from typing import List
import pandas as pd
import re
from argparse import Namespace
from tqdm.notebook  import trange, tqdm_notebook

# Torch
import torch
from torch.utils.data import Dataset

class CLPDDataset():
    """
    This funcition uses a pkl file, already downloaded, to load and organize a dataset.
    """
    
    def __init__(self,
                 name: str,
                 data_type: str,
                 sample_size: int = 20000,
                 val_size: float = 0.3,
                 n_negatives: int = 1,
                 max_length: int = 200,
                 seed: int=0,
                 data_path=None
                 ):
        
        """
        Parameters:
            name: (str) name of the data used
            data_path: (optional) path to pkl data on pandas format
            data_type: (str) <train|test>
            tokenizer: a hugginface based tokeinizer
            sample_size: (int) number of samples required on the dataset
                Not valid if data is testset
            val_size: (float) size of the validation set associated with the data
                Not valid if data is testset
            n_negatives: number of negatives examples for each PT|ENG sentence. n_negative in [1,5]
            max_length: (int) max number of tokens considered on a sentence.
            seed: used on random variables during the split and dataset sampling
            
        """
        
        
        # Associate data_path with respective name path
        self.name = name
        if name == 'capes' and data_path is None:
            self.data_path = '/work/datasets/capes/'
            
        elif name == 'scielo' and data_path is None:
            self.data_path = '/work/datasets/scielo/'
            
        elif name == 'books' and data_path is None:
            self.data_path = '/work/datasets/books/'
        
        elif not name in ['capes', 'scielo' , 'books']:
            raise IOError(f"{name} unkown for dataset")
            
            
            
        # Data type
        self.data_type = data_type
        
        if not self.data_type in ['train','test']:
            raise IOError(f"{name} unkown for datatype")
                  
        # Attributes for training
        if self.data_type == 'train':
            # Number of samples on dataset
            self.sample_size = sample_size

            # Lenght of Validation set
            self.val_size = val_size
            if  self.val_size < 0 or self.val_size > 1:
                raise IOError(f"val_size must be in [0,1]")

            # Number of negatives sentence for each PT | ENG sentence
            self.n_negatives = n_negatives
            if (0 > self.n_negatives or self.n_negatives > 5):
                raise IOError(f"N_NEGATIVES must be in [1,5]")
            
            # seed
            self.seed = seed
            

        # Max number of tokens in sentence
        self.max_length = max_length
        
      
        
    def get_organized_data(self,tokenizer):
    
        if self.data_type == 'train':
            if self.name == 'books':
                raise IOError ("Books does not have a train mode")
                
            return  self.organize_train(tokenizer)
        
        elif self.data_type == 'test':
            return self.organize_test(tokenizer)
            
    
    def organize_train(self,tokenizer):
        
        # Load pandas pkl
        dataset = pd.read_pickle(f"{self.data_path}/TRAINSET.pkl")
        
        # Sample Data
        if self.sample_size > len(dataset):
            self.sample_size = len(dataset)

        dataset = dataset.sample(self.sample_size,random_state=self.seed)
        
        # Assert that index is the row line
        dataset = dataset.reset_index(drop=True)
        
        # Dividing dataset in train and validation
        trainset, valset = train_test_split(dataset, test_size=self.val_size, random_state=self.seed)
        
        if len(valset) > 0:
            trainset_encoded = DataloaderCapesScielo(trainset,tokenizer, self.n_negatives,self.max_length,f"{self.name} Train")
            valset_encoded = DataloaderCapesScielo(valset,tokenizer, self.n_negatives,self.max_length,f"{self.name} Validation")
            
            return trainset_encoded, valset_encoded
        
        else:
            trainset_encoded = DataloaderCapesScielo(trainset,tokenizer, self.n_negatives,self.max_length,f"{self.name} Train")
            return trainset_encoded
        
        
    def organize_test(self,tokenizer):
        
        # Load pandas pkl
        testset = pd.read_pickle(f"{self.data_path}/TESTSET.pkl")
        
        # Assert that index is the row line
        testset = testset.reset_index(drop=True)
        
        if self.name in ['capes','scielo']:
            testset_encoded = DataloaderCapesScielo(testset,tokenizer, 5 ,self.max_length,f"{self.name} TEST")
        
        elif self.name in ['books']:
            testset_encoded = DataloaderBooks(testset, tokenizer,self.max_length,f"{self.name} TEST")
        
        return testset_encoded

        
class DataloaderCapesScielo(Dataset):
    
    def __init__(self,dataset,tokenizer,n_negatives,max_length,name):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Capes or Scielo Dataset
        """
        
        self.name = name
        self.dataset = dataset
        self.n_negatives = n_negatives
        self.max_length = max_length
        
        self.token_ids, self.attention_mask, self.token_type_ids , self.labels, self.pairs = self.encode(self.dataset, tokenizer)
        self.labels =  torch.LongTensor(self.labels)
        
    def __len__(self):
        return len(self.labels)

    def encode(self, dataframe,tokenizer):
        
        token_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        pairs = []
        # Setup name of tqdm bar
        desc = f"Processing {self.name.upper()}"
            
        for index in tqdm_notebook(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
            # Sentences that are considered 'plagiarism' ENG->PT
            pairs.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            t_id,attention,tt_ids  =  self.get_encode(row.ENG,row.PT,tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(1)
            
            # Sentences that are considered 'plagiarism' PT->ENG
            pairs.append(f"PT: {row.PT}\nENG:  {row.ENG}")
            t_id,attention,tt_ids  =  self.get_encode(row.PT,row.ENG,tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(1)
            
            # Sentences that not are considered 'plagiarism' from eng to pt
            for neg in range(self.n_negatives):
                text1 = row.ENG
                text2 = eval(f"row.top_{neg+1}_qeng_pt")
                pairs.append(f"ENG: {text1}\nNEGATIVE_{neg+1}_PT: {text2}")
                t_id,attention,tt_ids  =  self.get_encode(text1,text2,tokenizer)
                token_ids.append(t_id); attention_mask.append(attention); token_type_ids.append(tt_ids);
                labels.append(0)
                
            # Sentences that not are considered 'plagiarism' from pt to eng
            for neg in range(self.n_negatives):
                text1 = row.PT
                text2 = eval(f"row.top_{neg+1}_qpt_eng")
                pairs.append(f"PT: {text1}\nNEGATIVE_{neg+1}_ENG: {text2}")
                t_id,attention,tt_ids  =  self.get_encode(text1,text2,tokenizer)
                token_ids.append(t_id); attention_mask.append(attention); token_type_ids.append(tt_ids);
                labels.append(0)
            

        
        return  token_ids, attention_mask,token_type_ids, labels, pairs

    def get_encode(self,text1,text2,tokenizer):
        
        encode =  tokenizer.encode_plus(text=[text1,text2], max_length=self.max_length,
                                                pad_to_max_length=True,add_special_tokens=True)
        return encode['input_ids'], encode['attention_mask'], encode['token_type_ids']
        
    def __getitem__(self, idx):
                
        return torch.LongTensor(self.token_ids[idx]),\
               torch.LongTensor(self.attention_mask[idx]),\
               torch.LongTensor(self.token_type_ids[idx]),\
               self.labels[idx]
    
    

class DataloaderBooks(Dataset):
    
    def __init__(self,dataset,tokenizer,max_length,name):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Books Dataset
        """
        
        self.name = name
        self.dataset = dataset
        self.max_length = max_length
        
        self.token_ids, self.attention_mask, self.token_type_ids , self.labels, self.pairs = self.encode(self.dataset, tokenizer)
        self.labels =  torch.LongTensor(self.labels)
        
    def __len__(self):
        return len(self.labels)

    def encode(self, dataframe,tokenizer):
        
        token_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        pairs = []
        
        # Setup name of tqdm bar
        desc = f"Processing {self.name.upper()}"
            
        for index in tqdm_notebook(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
            # Sentences that are considered 'plagiarism' ENG->PT
            pairs.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            t_id,attention,tt_ids  =  self.get_encode(row.ENG,row.PT,tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(1)
            
            # Sentences that are considered 'plagiarism' ENG->PT_PARAPHRASE
            pairs.append(f"ENG: {row.ENG}\nPT_PARAPHRASE: {row['paraphrase-pt']}")
            t_id,attention,tt_ids  =  self.get_encode(row.ENG,row['paraphrase-pt'],tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(1)
            
              # Sentences that are considered 'plagiarism' PT->ENG
            pairs.append(f"PT: {row.PT}\nENG: {row.ENG}")
            t_id,attention,tt_ids  =  self.get_encode(row.PT,row.ENG,tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(1)
            
            # Sentences that are considered 'plagiarism' PT->ENG_PARAPHRASE
            pairs.append(f"PT: {row.PT}\nENG_PARAPHRASE: {row['paraphrase-eng']}")
            t_id,attention,tt_ids  =  self.get_encode(row.PT,row['paraphrase-eng'],tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(1)
            
            # Sentences that are NOT considered 'plagiarism' from ENG to PT_ERLA
            pairs.append(f"ENG:{row.ENG}\nNEGATIVE_PT: {row['pt_books_paraphrase__pt_erla']}")
            t_id,attention,tt_ids  =  self.get_encode(row.ENG,row['pt_books_paraphrase__pt_erla'],tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(0) 
            
            # Sentences that are NOT considered 'plagiarism' from PT to ENG_ERLA
            pairs.append(f"PT:{row.PT}\nNEGATIVE_ENG: {row['eng_books__eng_erla']}")
            t_id,attention,tt_ids  =  self.get_encode(row.PT,row['eng_books__eng_erla'],tokenizer)
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(0) 
            

        
        return  token_ids, attention_mask,token_type_ids, labels, pairs

    def get_encode(self,text1,text2,tokenizer):
        
        encode =  tokenizer.encode_plus(text=[text1,text2], max_length=self.max_length,
                                                pad_to_max_length=True,add_special_tokens=True)
        return encode['input_ids'], encode['attention_mask'], encode['token_type_ids']
        
    def __getitem__(self, idx):
                
        return torch.LongTensor(self.token_ids[idx]),\
               torch.LongTensor(self.attention_mask[idx]),\
               torch.LongTensor(self.token_type_ids[idx]),\
               self.labels[idx]

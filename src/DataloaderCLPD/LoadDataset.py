# Comum libraries
import os
import random
from typing import Dict
from typing import List
import pandas as pd
import re
from argparse import Namespace
from tqdm  import trange, tqdm
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer

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
                 data_path=None,
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
        
        elif name == 'capes_scielo':
            self.data_path = None
        
        elif not name in ['capes', 'scielo' ,'capes_scielo', 'books']:
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
        
    def get_organized_pairs(self):
        """
        Function to work with just one tokenizer for both languages in pt or eng.
        """
        
        if self.data_type == 'train':
            if self.name == 'books':
                raise IOError ("Books does not have a train mode")
                
            return  self.train_pairs()
        
        elif self.data_type == 'test':
            return self.test_pairs()
        
    
    def train_pairs(self):
        """
        This function return pairs of sentence labeled as plagiarism or not.

        Return: (sentence_1, sentence_2, label)
        """
         # Load pandas pkl
        if self.name == 'capes_scielo':
            dataset_capes = pd.read_pickle("/work/datasets/capes/TRAINSET.pkl")
            dataset_scielo = pd.read_pickle("/work/datasets/scielo/TRAINSET.pkl")
            # Sample Data
            # Dataset must be equal divided by capes and scielo
            if self.sample_size/2 > len(dataset_capes):
                sample_size_capes = len(dataset_capes)
                sample_size_scielo = sample_size_capes
            else: 
                sample_size_capes = int(self.sample_size/2)
                sample_size_scielo = sample_size_capes
            
            dataset_capes = dataset_capes.sample(sample_size_capes,random_state=self.seed)
            dataset_scielo = dataset_scielo.sample(sample_size_scielo,random_state=self.seed)
            
            dataset = dataset_capes.append(dataset_scielo)
                
        else:
            
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
            trainset_encoded = DataloaderCapesScieloPairs(trainset, self.n_negatives,self.max_length,
                                                          f"{self.name} Train")
            valset_encoded = DataloaderCapesScieloPairs(valset, self.n_negatives,self.max_length,
                                                        f"{self.name} Validation")    
            
            return trainset_encoded, valset_encoded
        
        else:
            trainset_encoded = DataloaderCapesScieloPairs(trainset, self.n_negatives,self.max_length,
                                                          f"{self.name} Train")
            return trainset_encoded
        
    def test_pairs(self):
        
        # Load pandas pkl
        testset = pd.read_pickle(f"{self.data_path}/TESTSET.pkl")
        
        # Assert that index is the row line
        testset = testset.reset_index(drop=True)
        
        if self.name in ['capes','scielo']:
            testset_encoded = DataloaderCapesScieloPairs(testset, 2 ,
                                                    self.max_length, f"{self.name} Test")
        
        elif self.name in ['books']:
            testset_encoded = DataloaderBooksPairs(testset, self.max_length,
                                                   f"{self.name} Test")
        
        return testset_encoded
        
    def get_double_tokenized_data(self,eng_tokenizer,pt_tokenizer):
        """
        Function to work with two diferents tokenizer on each sentence language
        
        Return encoded sentences with the touple (encoded(eng_sentences),encoded(pt_senteces),labels)
        
        """
        if self.data_type == 'train':
            if self.name == 'books':
                raise IOError ("Books does not have a train mode")
            
            # TRAIN DATA
            
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
                
                trainset_encoded = DoubleBertCapesScielo(trainset,eng_tokenizer, pt_tokenizer,
                                                         self.n_negatives,self.max_length,
                                                         f"{self.name} Train")
                valset_encoded = DoubleBertCapesScielo(valset, eng_tokenizer, pt_tokenizer,
                                                       self.n_negatives,self.max_length,
                                                       f"{self.name} Validation")

                return trainset_encoded, valset_encoded

            else:
                trainset_encoded = DoubleBertCapesScielo(trainset,eng_tokenizer, pt_tokenizer,
                                                         self.n_negatives,self.max_length,
                                                         f"{self.name} Train")
                return trainset_encoded
                
        elif self.data_type == 'test':
            # TESTSET
            
             # Load pandas pkl
            testset = pd.read_pickle(f"{self.data_path}/TESTSET.pkl")

            # Assert that index is the row line
            testset = testset.reset_index(drop=True)

            if self.name in ['capes','scielo']:
                testset_encoded = DoubleBertCapesScielo(testset,eng_tokenizer, pt_tokenizer, 2 ,
                                                        self.max_length,f"{self.name} TEST")

            elif self.name in ['books']:
                testset_encoded = DoubleBertBooks(testset, eng_tokenizer, pt_tokenizer,
                                                  self.max_length,f"{self.name} TEST")

            return testset_encoded
      
        
    def get_organized_data(self,tokenizer,tokenizer_type='bert'):
        """
        Function to work with just one tokenizer for both languages in pt or eng.
        """
        
        if tokenizer_type == 'bert' and isinstance(tokenizer,T5Tokenizer):
            raise TypeError("Tokenizer type is 'bert' but It instance of 't5'")
    
        if self.data_type == 'train':
            if self.name == 'books':
                raise IOError ("Books does not have a train mode")
                
            return  self.organize_train(tokenizer,tokenizer_type)
        
        elif self.data_type == 'test':
            return self.organize_test(tokenizer,tokenizer_type)
            
    
    def organize_train(self,tokenizer,tokenizer_type='bert'):
        
        # Load pandas pkl
        if self.name == 'capes_scielo':
            dataset_capes = pd.read_pickle("/work/datasets/capes/TRAINSET.pkl")
            dataset_scielo = pd.read_pickle("/work/datasets/scielo/TRAINSET.pkl")
            # Sample Data
            # Dataset must be equal divided by capes and scielo
            if self.sample_size/2 > len(dataset_capes):
                sample_size_capes = len(dataset_capes)
                sample_size_scielo = sample_size_capes
            else: 
                sample_size_capes = int(self.sample_size/2)
                sample_size_scielo = sample_size_capes
            
            dataset_capes = dataset_capes.sample(sample_size_capes,random_state=self.seed)
            dataset_scielo = dataset_scielo.sample(sample_size_scielo,random_state=self.seed)
            
            dataset = dataset_capes.append(dataset_scielo)
                
              
            
        else:
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
            trainset_encoded = DataloaderCapesScielo(trainset,tokenizer, self.n_negatives,self.max_length,
                                                     f"{self.name} Train",tokenizer_type)
            valset_encoded = DataloaderCapesScielo(valset,tokenizer, self.n_negatives,self.max_length,
                                                   f"{self.name} Validation",tokenizer_type)
            
            return trainset_encoded, valset_encoded
        
        else:
            trainset_encoded = DataloaderCapesScielo(trainset,tokenizer, self.n_negatives,
                                                     self.max_length,f"{self.name} Train",tokenizer_type)
            return trainset_encoded
        
        
    def organize_test(self,tokenizer,tokenizer_type='bert'):
        
        # Load pandas pkl
        testset = pd.read_pickle(f"{self.data_path}/TESTSET.pkl")
        
        # Assert that index is the row line
        testset = testset.reset_index(drop=True)
        
        if self.name in ['capes','scielo']:
            testset_encoded = DataloaderCapesScielo(testset,tokenizer, 2 ,
                                                    self.max_length,f"{self.name} TEST",tokenizer_type)
        
        elif self.name in ['books']:
            testset_encoded = DataloaderBooks(testset, tokenizer,
                                              self.max_length,f"{self.name} TEST",tokenizer_type)
        
        return testset_encoded
    
#########################################
#        DATALOADER CAPES SCIELO       #
#######################################

        
class DataloaderCapesScielo(Dataset):
    
    def __init__(self,dataset,tokenizer,n_negatives,max_length,name,tokenizer_type):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Capes or Scielo Dataset
        """
        
        self.name = name
        self.dataset = dataset
        self.n_negatives = n_negatives
        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        
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
            
        for index in tqdm(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
            # Sentences that are considered 'plagiarism' ENG->PT
            pairs.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.ENG,row.PT,tokenizer,'true')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label)
            
            # Sentences that are considered 'plagiarism' PT->ENG
            pairs.append(f"PT: {row.PT}\nENG:  {row.ENG}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.PT,row.ENG,tokenizer,'true')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label)
            
            # Sentences that not are considered 'plagiarism' from eng to pt
            for neg in range(self.n_negatives):
                text1 = row.ENG
                text2 = eval(f"row.top_{neg+1}_qeng_pt")
                pairs.append(f"ENG: {text1}\nNEGATIVE_{neg+1}_PT: {text2}")
                t_id,attention,tt_ids,label  =  self.get_encode(text1,text2,tokenizer,'false')
                token_ids.append(t_id); attention_mask.append(attention); token_type_ids.append(tt_ids);
                labels.append(label)
                
            # Sentences that not are considered 'plagiarism' from pt to eng
            for neg in range(self.n_negatives):
                text1 = row.PT
                text2 = eval(f"row.top_{neg+1}_qpt_eng")
                pairs.append(f"PT: {text1}\nNEGATIVE_{neg+1}_ENG: {text2}")
                t_id,attention,tt_ids,label  =  self.get_encode(text1,text2,tokenizer,'false')
                token_ids.append(t_id); attention_mask.append(attention); token_type_ids.append(tt_ids);
                labels.append(label)
            

        
        return  token_ids, attention_mask,token_type_ids, labels, pairs

    def get_encode(self,text1,text2,tokenizer,label):
        """
        Get encode of each model, implemented for Bert and T5 encoder
        "bert": Separete text1 and text2 with a token [SEP] and insert as input of model
        
        't5': use the sentence model -> 'plagiarism  sentence1: text1 sentence2: text2'
        
        The label is also converted for both method:
            if method is bert, the label return 0 for not plagiarism
                                         and 1 for plagiarism
            if method is t5, the label return token(false) for not plagiarims
                                        and token(true) for plagiarism
        """
        
        if self.tokenizer_type == 'bert':
            encode =  tokenizer.encode_plus(text=[text1,text2], max_length=self.max_length,
                                                pad_to_max_length=True,add_special_tokens=True)
            if label == 'true':
                gt_label = 1
            elif label == 'false':
                gt_label = 0
            else:
                raise IOError(f"Label can only assume values 'true' or 'false' and it is {label}")
            
            
                
        elif self.tokenizer_type == 't5':
            text = f"plagiarism sentence1: {text1} sentence2: {text2} {tokenizer.eos_token}"
            
            encode =  tokenizer.encode_plus(text=text, max_length=self.max_length,
                                                pad_to_max_length=True,add_special_tokens=True)
            if not label in ['true','false']:
                raise IOError(f"Label can only assume values 'true' or 'false' and it is {label}")
                
            if label == 'true':
                gt_label = tokenizer.encode(f"{1} {tokenizer.eos_token}",max_length=3,
                                            pad_to_max_length=True,add_special_tokens=True)
            elif label == 'false':
                gt_label = tokenizer.encode(f"{0} {tokenizer.eos_token}",max_length=3,
                                            pad_to_max_length=True,add_special_tokens=True)
            
        return encode['input_ids'], encode['attention_mask'], encode['token_type_ids'], gt_label
        
        
    def __getitem__(self, idx):
                
        return torch.LongTensor(self.token_ids[idx]),\
               torch.LongTensor(self.attention_mask[idx]),\
               torch.LongTensor(self.token_type_ids[idx]),\
               self.labels[idx],\
               self.pairs[idx]
    
    
#############################################
#     DATALOADER CAPESCIELO PAIRS          #
###########################################

class DataloaderCapesScieloPairs(Dataset):
    
    def __init__(self,dataset,n_negatives,max_length,name):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Capes or Scielo Dataset
        """
        
        self.name = name
        self.dataset = dataset
        self.n_negatives = n_negatives
        self.max_length = max_length
        
        self.sent1 ,self.sent2, self.pairs, self.labels = self.organize(self.dataset)
        self.labels =  torch.LongTensor(self.labels)
        
        
    def __len__(self):
        return len(self.labels)

    def organize(self, dataframe):
        
        labels = []
        pairs = []
        sent1 = []
        sent2 = []
        # Setup name of tqdm bar
        desc = f"Loading {self.name.upper()}"
            
        for index in tqdm(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
            # Sentences that are considered 'plagiarism' ENG->PT
            pairs.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            sent1.append(row.ENG)
            sent2.append(row.PT)
            labels.append(1)
            
            # Sentences that are considered 'plagiarism' PT->ENG
            pairs.append(f"PT: {row.PT}\nENG:  {row.ENG}")
            sent1.append(row.PT)
            sent2.append(row.ENG)
            labels.append(1)
            
            # Sentences that not are considered 'plagiarism' from eng to pt
            for neg in range(self.n_negatives):
                text1 = row.ENG
                text2 = eval(f"row.top_{neg+1}_qeng_pt")
                pairs.append(f"ENG: {text1}\nNEGATIVE_{neg+1}_PT: {text2}")
                sent1.append(text1)
                sent2.append(text2)
                labels.append(0)
                
            # Sentences that not are considered 'plagiarism' from pt to eng
            for neg in range(self.n_negatives):
                text1 = row.PT
                text2 = eval(f"row.top_{neg+1}_qpt_eng")
                pairs.append(f"PT: {text1}\nNEGATIVE_{neg+1}_ENG: {text2}")
                sent1.append(text1)
                sent2.append(text2)
                labels.append(0)
            
        
        return  sent1, sent2, pairs, labels

    def __getitem__(self, idx):
                
        return self.sent1[idx],\
               self.sent2[idx],\
               self.labels[idx],\
               self.pairs[idx]
        
    
#########################################
#        DATALOADER BOOKS              #
#######################################
class DataloaderBooks(Dataset):
    
    def __init__(self,dataset,tokenizer,max_length,name,tokenizer_type):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Books Dataset
        """
        
        self.name = name
        self.dataset = dataset
        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        
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
            
        for index in tqdm(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
            # Sentences that are considered 'plagiarism' ENG->PT
            pairs.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.ENG,row.PT,
                                                      tokenizer, 'true')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label)
            
            # Sentences that are considered 'plagiarism' ENG->PT_PARAPHRASE
            pairs.append(f"ENG: {row.ENG}\nPT_PARAPHRASE: {row['paraphrase-pt']}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.ENG,row['paraphrase-pt'],
                                                      tokenizer, 'true')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label)
            
              # Sentences that are considered 'plagiarism' PT->ENG
            pairs.append(f"PT: {row.PT}\nENG: {row.ENG}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.PT,row.ENG,
                                                      tokenizer, 'true')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label)
            
            # Sentences that are considered 'plagiarism' PT->ENG_PARAPHRASE
            pairs.append(f"PT: {row.PT}\nENG_PARAPHRASE: {row['paraphrase-eng']}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.PT,row['paraphrase-eng'],
                                                      tokenizer, 'true')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label)
            
            # Sentences that are NOT considered 'plagiarism' from ENG to PT_ERLA
            pairs.append(f"ENG:{row.ENG}\nNEGATIVE_PT: {row['pt_books_paraphrase__pt_erla']}")
            t_id,attention,tt_ids,label  = self.get_encode(row.ENG,row['pt_books_paraphrase__pt_erla'],
                                                           tokenizer,'false')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label) 
            
            # Sentences that are NOT considered 'plagiarism' from PT to ENG_ERLA
            pairs.append(f"PT:{row.PT}\nNEGATIVE_ENG: {row['eng_books__eng_erla']}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.PT,row['eng_books__eng_erla'],
                                                            tokenizer,'false')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label) 
            
            # Sentences that are NOT considered 'plagiarism' (same dataset bookj) from PT to ENG
            pairs.append(f"PT:{row.PT}\nNEGATIVE_ENG: {row['top_1_qpt_eng']}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.PT,row['top_1_qpt_eng'],tokenizer,'false')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label) 
            
            # Sentences that are NOT considered 'plagiarism' (same dataset bookj) from ENG to PT
            pairs.append(f"ENG:{row.ENG}\nNEGATIVE_PT: {row['top_1_qeng_pt']}")
            t_id,attention,tt_ids,label  =  self.get_encode(row.ENG,row['top_1_qeng_pt'],tokenizer,'false')
            token_ids.append(t_id);  attention_mask.append(attention) ; token_type_ids.append(tt_ids)
            labels.append(label) 
            

        
        return  token_ids, attention_mask,token_type_ids, labels, pairs

    def get_encode(self,text1,text2,tokenizer,label):
        """
        Get encode of each model, implemented for Bert and T5 encoder
        "bert": Separete text1 and text2 with a token [SEP] and insert as input of model
        
        't5': use the sentence model -> 'plagiarism  sentence1: text1 sentence2: text2'
        
        The label is also converted for both method:
            if method is bert, the label return 0 for not plagiarism
                                         and 1 for plagiarism
            if method is t5, the label return token(false) for not plagiarims
                                        and token(true) for plagiarism
        """
        
        if self.tokenizer_type == 'bert':
            encode =  tokenizer.encode_plus(text=[text1,text2], max_length=self.max_length,
                                                pad_to_max_length=True,add_special_tokens=True)
            if label == 'true':
                gt_label = 1
            elif label == 'false':
                gt_label = 0
            else:
                raise IOError(f"Label can only assume values 'true' or 'false' and it is {label}")
            
            
                
        elif self.tokenizer_type == 't5':
            text = f"plagiarism sentence1: {text1} sentence2: {text2} {tokenizer.eos_token}"
            
            encode =  tokenizer.encode_plus(text=text, max_length=self.max_length,
                                                pad_to_max_length=True,add_special_tokens=True)
            if not label in ['true','false']:
                raise IOError(f"Label can only assume values 'true' or 'false' and it is {label}")
                
            if label == 'true':
                gt_label = tokenizer.encode(f"{1} {tokenizer.eos_token}",max_length=3,
                                            pad_to_max_length=True,add_special_tokens=True)
            elif label == 'false':
                gt_label = tokenizer.encode(f"{0} {tokenizer.eos_token}",max_length=3,
                                            pad_to_max_length=True,add_special_tokens=True)
            
        return encode['input_ids'], encode['attention_mask'], encode['token_type_ids'], gt_label
        
        
    def __getitem__(self, idx):
                
        return torch.LongTensor(self.token_ids[idx]),\
               torch.LongTensor(self.attention_mask[idx]),\
               torch.LongTensor(self.token_type_ids[idx]),\
               self.labels[idx],\
               self.pairs[idx]


    
#########################################
#        DATALOADER BOOKS   PAIRS      #
#######################################    
class DataloaderBooksPairs(Dataset):
    
    def __init__(self,dataset,max_length,name):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Books Dataset
        """
        
        self.name = name
        self.dataset = dataset
        self.max_length = max_length
        self.sent1, self.sent2, self.pairs, self.labels = self.organize(self.dataset)

        self.labels =  torch.LongTensor(self.labels)
        
        
    def __len__(self):
        return len(self.labels)

    def organize(self, dataframe):
        
        labels = []
        pairs = []
        sent1 = []
        sent2 = []
        
        # Setup name of tqdm bar
        desc = f"Processing {self.name.upper()}"
            
        for index in tqdm(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
            # Sentences that are considered 'plagiarism' ENG->PT
            pairs.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            sent1.append(row.ENG)
            sent2.append(row.PT)
            labels.append(1)
            
            # Sentences that are considered 'plagiarism' ENG->PT_PARAPHRASE
            pairs.append(f"ENG: {row.ENG}\nPT_PARAPHRASE: {row['paraphrase-pt']}")
            sent1.append(row.ENG)
            sent2.append(row['paraphrase-pt'])
            labels.append(1)
            
            # Sentences that are considered 'plagiarism' PT->ENG
            pairs.append(f"PT: {row.PT}\nENG: {row.ENG}")
            sent1.append(row.PT)
            sent2.append(row.ENG)
            labels.append(1)
            
            # Sentences that are considered 'plagiarism' PT->ENG_PARAPHRASE
            pairs.append(f"PT: {row.PT}\nENG_PARAPHRASE: {row['paraphrase-eng']}")
            sent1.append(row.PT)
            sent2.append(row['paraphrase-eng'])
            labels.append(1)
            
            # Sentences that are NOT considered 'plagiarism' from ENG to PT_ERLA
            pairs.append(f"ENG:{row.ENG}\nNEGATIVE_PT: {row['pt_books_paraphrase__pt_erla']}")
            sent1.append(row.ENG)
            sent2.append(row['pt_books_paraphrase__pt_erla'])
            labels.append(0) 
            
            # Sentences that are NOT considered 'plagiarism' from PT to ENG_ERLA
            pairs.append(f"PT:{row.PT}\nNEGATIVE_ENG: {row['eng_books__eng_erla']}")
            sent1.append(row.PT)
            sent2.append(row['eng_books__eng_erla'])
            labels.append(0) 
            
            # Sentences that are NOT considered 'plagiarism' (same dataset bookj) from PT to ENG
            pairs.append(f"PT:{row.PT}\nNEGATIVE_ENG: {row['top_1_qpt_eng']}")
            sent1.append(row.PT)
            sent2.append(row['top_1_qpt_eng'])
            labels.append(0) 
            
            # Sentences that are NOT considered 'plagiarism' (same dataset bookj) from ENG to PT
            pairs.append(f"ENG:{row.ENG}\nNEGATIVE_PT: {row['top_1_qeng_pt']}")
            sent1.append(row.ENG)
            sent2.append(row['top_1_qeng_pt'])
            labels.append(0) 
            

        
        return  sent1, sent2, pairs, labels

    def __getitem__(self, idx):
                
        return self.sent1[idx],\
               self.sent2[idx],\
               self.labels[idx],\
               self.pairs[idx]

    
    
    
    
    
#########################################
#        DATALOADER BOOKS Double Bert  #
#######################################      
class DoubleBertBooks(Dataset):
    
    def __init__(self,dataset,eng_tokenizer, pt_tokenizer,max_length,name):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Capes or Scielo Dataset
        
        Return value of dataset[i]:
            i-th sentence:<tuple> ( <dict> encoded(eng_sentence[i]), <dict> encoded(pt_setence[i]), label)
            The encoded dict has the attributes:
                encode['input_ids'], encode['attention_mask'], encode['token_type_ids']
                
        """
        
        self.name = name
        self.dataset = dataset
        self.max_length = max_length
        
        self.eng_encode, self.pt_encode, self.labels, self.pairs = self.encode(self.dataset, eng_tokenizer,pt_tokenizer)
        self.labels =  torch.LongTensor(self.labels)
        
        
    def __len__(self):
        return len(self.labels)

    def encode(self, dataframe,eng_tokenizer,pt_tokenizer):
        
        eng_encode = []
        pt_encode = []
        
        labels = []
        pairs_debug = []
        # Setup name of tqdm bar
        desc = f"Processing {self.name.upper()}"
            
        for index in tqdm(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
             # Sentences that are considered 'plagiarism' ENG->PT
            label = 1 # true
            text_eng = row.ENG
            text_pt = row.PT
            pairs_debug.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
            pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
            labels.append(label)
            
            # Sentences that are considered 'plagiarism' ENG->PT_PARAPHRASE
            label = 1 # true
            text_eng = row.ENG
            text_pt = row['paraphrase-pt']
            pairs_debug.append(f"ENG: {row.ENG}\nPT_PARAPHRASE: {row['paraphrase-pt']}")
            eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
            pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
            labels.append(label)
            
            # Sentences that are considered 'plagiarism' PT->ENG_PARAPHRASE
            label = 1 # true
            text_eng = row['paraphrase-eng']
            text_pt = row.PT
            pairs_debug.append(f"ENG_PARAPHRASE: {row['paraphrase-eng']}\nPT: {row.PT}")
            eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
            pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
            labels.append(label)
            
            # Sentences that are NOT considered 'plagiarism' from ENG to PT_ERLA
            label = 0 # False
            text_eng = row.ENG
            text_pt = row['pt_books_paraphrase__pt_erla']
            pairs_debug.append(f"ENG:{row.ENG}\nNEGATIVE_PT: {row['pt_books_paraphrase__pt_erla']}")
            eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
            pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
            labels.append(label)
            
            # Sentences that are NOT considered 'plagiarism' from PT to ENG_ERLA
            label = 0 # False
            text_eng = row['eng_books__eng_erla']
            text_pt = row.PT
            pairs_debug.append(f"NEGATIVE_ENG: {row['eng_books__eng_erla']}\nPT:{row.PT}")
            eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
            pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
            labels.append(label)
            
            # Sentences that are NOT considered 'plagiarism' (same dataset book) from PT to ENG
            label = 0 # False
            text_eng = row['top_1_qpt_eng']
            text_pt = row.PT
            pairs_debug.append(f"NEGATIVE_ENG: {row['top_1_qpt_eng']}\nPT:{row.PT}")
            eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
            pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
            labels.append(label)
            
            # Sentences that are NOT considered 'plagiarism' (same dataset book) from ENG to PT
            label = 0 # False
            text_eng = row.ENG
            text_pt = row['top_1_qeng_pt']
            pairs_debug.append(f"ENG:{row.ENG}\nNEGATIVE_PT: {row['top_1_qeng_pt']}")
            eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
            pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
            labels.append(label)
            
        return  eng_encode, pt_encode,labels, pairs_debug

    def get_encode(self,text,tokenizer):
        """
        Get encode of each model, implemented for Bert and T5 encoder
        "bert": Encoded 
        """
        
        return  tokenizer.encode_plus(text=text, max_length=self.max_length, 
                                       return_tensors='pt', pad_to_max_length=True,
                                       add_special_tokens=True)
        
        
    def __getitem__(self, idx):
                
        return self.eng_encode[idx],\
               self.pt_encode[idx],\
               self.labels[idx],\
               self.pairs[idx]


class DoubleBertCapesScielo(Dataset):
    
    def __init__(self,dataset,eng_tokenizer, pt_tokenizer,n_negatives,max_length,name):
        """
        Creates a dataset ready to use on Torch Dataloader basead on the dataframe organized from Capes or Scielo Dataset
        
        Return value of dataset[i]:
            i-th sentence:<tuple> ( <dict> encoded(eng_sentence[i]), <dict> encoded(pt_setence[i]), label)
            The encoded dict has the attributes:
                encode['input_ids'], encode['attention_mask'], encode['token_type_ids']
                
        """
        
        self.name = name
        self.dataset = dataset
        self.n_negatives = n_negatives
        self.max_length = max_length
        
        self.eng_encode, self.pt_encode, self.labels, self.pairs = self.encode(self.dataset, eng_tokenizer,pt_tokenizer)
        self.labels =  torch.LongTensor(self.labels)
        
        
    def __len__(self):
        return len(self.labels)

    def encode(self, dataframe,eng_tokenizer,pt_tokenizer):
        
        eng_encode = []
        pt_encode = []
        
        labels = []
        pairs_debug = []
        # Setup name of tqdm bar
        desc = f"Processing {self.name.upper()}"
            
        for index in tqdm(range(len(dataframe)),desc=desc):
            row = dataframe.iloc[index]
            
            # Sentences that are considered 'plagiarism' ENG->PT
            label = 1 # true
            pairs_debug.append(f"ENG: {row.ENG}\nPT: {row.PT}")
            eng_encode.append( self.get_encode(row.ENG,eng_tokenizer) )
            pt_encode.append( self.get_encode(row.PT,pt_tokenizer))
            labels.append(label)
            
            # Sentences that not are considered 'plagiarism' from eng to pt
            for neg in range(self.n_negatives):
                label = 0 # False
                text_eng = row.ENG
                text_pt = eval(f"row.top_{neg+1}_qeng_pt")
                pairs_debug.append(f"ENG: {text_eng}\nNEGATIVE_{neg+1}_PT: {text_pt}")
                eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
                pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
                labels.append(label)
                
            # Sentences that not are considered 'plagiarism' from pt to eng
            for neg in range(self.n_negatives):
                label = 0 # False
                text_eng = eval(f"row.top_{neg+1}_qpt_eng")
                text_pt = row.PT
                pairs_debug.append(f"NEGATIVE_{neg+1}_ENG: {text_eng}\nPT: {text_pt}")
                eng_encode.append( self.get_encode(text_eng,eng_tokenizer) )
                pt_encode.append( self.get_encode(text_pt,pt_tokenizer))
                labels.append(label)
                
        
        return  eng_encode, pt_encode,labels, pairs_debug

    def get_encode(self,text,tokenizer):
        """
        Get encode of each model, implemented for Bert and T5 encoder
        "bert": Encoded 
        """
        
        return  tokenizer.encode_plus(text=text, max_length=self.max_length, 
                                       return_tensors='pt', pad_to_max_length=True,
                                       add_special_tokens=True)
        
        
    def __getitem__(self, idx):
                
        return self.eng_encode[idx],\
               self.pt_encode[idx],\
               self.labels[idx],\
               self.pairs[idx]
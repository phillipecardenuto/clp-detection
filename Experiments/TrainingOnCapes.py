# Comum libraries
import os
from glob import glob
import random
from typing import Dict
from typing import List
import numpy as np
import pandas as pd
import re
from argparse import Namespace
from tqdm.notebook  import trange, tqdm_notebook

# Dataset
import sys
sys.path.insert(0, "/work/src/DataloaderCLPD/")
from LoadDataset import *

# Torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# HugginFace
from transformers import BertTokenizer,BertTokenizerFast,BertForSequenceClassification
# Sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Tersorflow
import tensorboard


# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Setup seeds
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available(): 
    dev = "cuda:0"
else: 
    dev = "cpu" 

device = torch.device(dev)
print("Device",dev)

if "cuda" in dev:
    print("GPU: ", torch.cuda.get_device_name(0))

class BertFinetuner(pl.LightningModule):

    def __init__(self, hparams=None,train_dataloader=None,val_dataloader=None,test_dataloader=None):
        
        super(BertFinetuner, self).__init__()
        
        #Hiperparameters
        if hparams:
            self.hparams = hparams
             # Learnning Rate and Loss Function
            self.learning_rate = hparams.learning_rate
            self.lossfunc = torch.nn.CrossEntropyLoss()
            # Optimizer
            self.optimizer = eval(self.hparams.optimizer)

            # Retrieve model from Huggingface
            self.model = BertForSequenceClassification.from_pretrained(hparams.model).to(device)


            # freeze bert embeddings
            if hparams.freeze:
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = False
                # freeze bert attention encoders, but release the last five ones
                for layer in self.model.bert.encoder.layer[:-5]:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Dataloaders
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
       


    def forward(self, input_ids, attention_mask, token_type_ids,labels=None):
       
        # If labels are None, It will return a loss and a logit
        # Else it return the predicted logits for each sentence
        return self.model(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     labels=labels)

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label,_ = batch
         
        # fwd
        loss, y_hat = self(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device),label.to(device))
        
        # loss
        # loss = self.lossfunc(y_hat, label) # Using loss from the model
        
        # logs
        tensorboard_logs = {'train_loss': loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label,_ = batch
         
        # fwd
        loss, y_hat = self(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device),label.to(device))
        
        # loss
        #loss = self.lossfunc(y_hat, label) # Using loss from the model
        
        # F1 -score
        _, y_hat = torch.max(y_hat, dim=1)
        val_f1 = f1_score(y_pred=y_hat.cpu(), y_true=label.cpu())
        val_f1 = torch.tensor(val_f1)
        
        return {'val_loss': loss, 'val_f1': val_f1}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_f1': avg_val_f1}
        
        return {'val_loss': avg_loss.item(), 'val_f1': avg_val_f1.item(), 
                'progress_bar': tensorboard_logs, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label, pairs = batch
        
        y_hat = self(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))[0]
        _, y_hat = torch.max(y_hat, dim=1)

        return {'pairs': pairs, 'y_true': label.cpu(), 'y_pred':y_hat.cpu() }

    def test_epoch_end(self, outputs):
        
        
        pairs = [pair for x in outputs for pair in x['pairs']]
        y_true = np.array([ y.item() for x in outputs for y in x['y_true'] ])
        y_pred = np.array([ y.item() for x in outputs for y in x['y_pred'] ])
        
        # Write failure on file
        with open (f"{self.log_path}/FAILURE_TESTSET_{self.testset_name}.txt", 'w') as file:
               for index,pair in enumerate(pairs):
                    if y_true[index] != y_pred[index]:
                        file.write("="*50+f"\n[Y_TRUE={y_true[index]} != Y_PRED={y_pred[index]}]\n"+pair \
                                  +'\n'+"="*50+'\n')
                        
        with open (f"{self.log_path}/METRICS_TESTSET_{self.testset_name}.txt", 'w') as file:
                file.write("="*50+"\n"+
                           "\t\t"+self.testset_name.upper()+"\n"+
                           "="*50+"\n\n\n"+
                           "-"*50+"\n"+
                           "CONFUSION MATRIX:\n"+
                           f'{confusion_matrix(y_true=y_true, y_pred=y_pred)}\n\n'+
                           "-"*50+"\n"+
                           "SKLEARN REPORT:\n"+
                           f'{classification_report(y_true=y_true, y_pred=y_pred)}\n\n'+
                           "-"*50+"\n"+
                           f"F1-SCORE: {f1_score(y_pred=y_pred, y_true=y_true)}\n\n"+
                           "="*50+"\n")
                           
        
        print("CONFUSION MATRIX:")
        print(confusion_matrix(y_true=y_true, y_pred=y_pred))
        
        print("SKLEARN  REPORT")
        print(classification_report(y_true=y_true, y_pred=y_pred))
        
        
        test_f1 =  f1_score(y_pred=y_pred, y_true=y_true)
    
        tensorboard_logs = {'test_f1': test_f1}
        return {'test_f1': test_f1, 'log': tensorboard_logs,
                 'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):

        optimizer =  self.optimizer(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate)
        
        scheduler = StepLR(optimizer, step_size=self.hparams.steplr_epochs, gamma=self.hparams.scheduling_factor)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
    
    
    

def get_all_dataloaders(train_dataset_name,
                        max_length,
                        val_size,
                        sample_size,
                        n_negatives,
                        batch_size,
                        tokenizer,
                        ):
     
    train_clpd = CLPDDataset(name=train_dataset_name,
                            data_type='train',
                            sample_size=sample_size,
                            val_size=val_size,
                            max_length= max_length,
                            n_negatives=n_negatives)
    
    trainset , valset = train_clpd.get_organized_data(tokenizer=tokenizer)
    
    capes_testset = CLPDDataset(name='capes', data_type='test', max_length= max_length).get_organized_data(tokenizer=tokenizer)
    
    scielo_testset = CLPDDataset(name='scielo', data_type='test', max_length= max_length).get_organized_data(tokenizer=tokenizer)
    
    books_testset = CLPDDataset(name='books', data_type='test', max_length= max_length).get_organized_data(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(trainset, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    
    val_dataloader = DataLoader(valset, batch_size=batch_size,
                                  shuffle=False, num_workers=4)
    
    capes_dataloader = DataLoader(capes_testset, batch_size=batch_size,
                                  shuffle=False, num_workers=4)
    
    scielo_dataloader = DataLoader(scielo_testset, batch_size=batch_size,
                                  shuffle=False, num_workers=4)
    
    books_dataloader = DataLoader(books_testset, batch_size=batch_size,
                                  shuffle=False, num_workers=4)
    
    
    return train_dataloader, val_dataloader , capes_dataloader , scielo_dataloader, books_dataloader




# Training will perform a cross-dataset.
# Training on Capes

hyperparameters = {
                    "experiment_name": "CAPES", 
                    "max_epochs": 2,
                    "optimizer": 'torch.optim.Adam',
                    "patience": 1,
                    "steplr_epochs":1,
                    "scheduling_factor": 0.9,
                    "learning_rate": 1e-5,
                    "max_length":200,
                    "batch_size":100,
                    'gpu': 0,
                    'trainset': 'capes',
                    'trainset_len': 200000,
                    'val_size': 0.2,
                    'freeze': True
                   }

# N_negative First arg
# BertModel Second arg
n_negatives = int(sys.argv[1])
bert_model = sys.argv[2]
hyperparameters['model'] = bert_model
hyperparameters['n_negatives'] = n_negatives
# Bert  Tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model)


train_loader, val_loader , capes_loader , scielo_loader, books_loader = get_all_dataloaders(
                                                                    train_dataset_name=hyperparameters['trainset'] ,
                                                                    max_length=hyperparameters['max_length'],
                                                                    val_size=hyperparameters['val_size'],
                                                                    sample_size=hyperparameters['trainset_len'],
                                                                    n_negatives=hyperparameters['n_negatives'],
                                                                    batch_size=hyperparameters['batch_size'],
                                                                    tokenizer=tokenizer)

experiment_name = hyperparameters['experiment_name'].replace("/",'_')
hyperparameters['experiment_name'] = f'{experiment_name}_{bert_model}_N_{n_negatives}'


#------------------------------#
#       Checkpoints / LOG      #
#------------------------------#

log_path = 'logs'
ckpt_path = os.path.join(log_path, hyperparameters["experiment_name"], "-{val_loss:.2f}")  
checkpoint_callback = ModelCheckpoint(prefix="checkpoint",  # prefixo para nome do checkpoint
                                      filepath=ckpt_path,  # path onde será salvo o checkpoint
                                      monitor="val_loss", 
                                      mode="min",
                                      save_top_k=1)   
# Hard coded
logger_path = os.path.join(log_path, hyperparameters["experiment_name"])
logger = TensorBoardLogger(logger_path,name='Tensorboard_logger')

# Lighting Trainer
trainer = pl.Trainer(gpus=[hyperparameters['gpu']],
                     logger=logger,
                     max_epochs=hyperparameters["max_epochs"],
                     check_val_every_n_epoch=1,
                     accumulate_grad_batches=2,
                     checkpoint_callback=checkpoint_callback,
                     amp_level='O2', use_amp=False)
hparams = Namespace(**hyperparameters)
model = BertFinetuner(hparams=hparams,train_dataloader=train_loader,val_dataloader=val_loader, test_dataloader=None)

# Train
trainer.fit(model)


#------------------------------#
#            TEST              #
#------------------------------#

# Get Checkpoints path
checkpoint = glob(f'{trainer.weights_save_path}/checkpoint*')
checkpoint.sort()
checkpoint = checkpoint[0]

model.log_path = trainer.weights_save_path

# Books
model.testset_name = 'books'
tester_books =  pl.Trainer(gpus=[hyperparameters['gpu']],amp_level='O2', use_amp=False)
tester_books.test(model=model,test_dataloaders=books_loader)

# CAPES
model.testset_name = 'capes'
tester_capes =  pl.Trainer(gpus=[hyperparameters['gpu']],amp_level='O2', use_amp=False)
tester_capes.test(model=model,test_dataloaders=capes_loader)

# Scielo
model.testset_name = 'scielo'
tester_scielo =  pl.Trainer(gpus=[hyperparameters['gpu']],amp_level='O2', use_amp=False)
tester_scielo.test(model=model,test_dataloaders=scielo_loader)


import numpy as np
import pandas as pd
from model import BertClassifier
from preprocess import BBCNewsDataset
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
import datetime

np.random.seed(49)

class BertTrainer:
    def __init__(self):
        pass
    def set_params(self,**kwargs):
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['lr']
        self.epochs = kwargs['epochs']
    def customize_dataset(self,df_train,df_val):
        ## training data
        self.train_length = len(df_train)
        self.train_dataset = BBCNewsDataset(df_train)
        self.train_dataloader = DataLoader(self.train_dataset,batch_size=self.batch_size)
        ## testing data
        self.val_length = len(df_val)
        self.val_dataset = BBCNewsDataset(df_val)
        self.val_dataloader = DataLoader(self.val_dataset,batch_size=self.batch_size)    
    def train_model(self, save_file = './BERT_WEIGHTS.pth', load_file = None):
        ## create model
        model = BertClassifier()
        if load_file is not None:
            model.load_state_dict(torch.load(load_file))
            print("load pretrained model successfully")
        ## define criterion & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(),lr=self.lr)
        ## gpu or cpu
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
        ## start training
        for epoch_num in range(self.epochs):
            total_acc_train = 0
            total_loss_train = 0
            ## use tqdm
            for train_input, train_label in tqdm(self.train_dataloader, desc = "Training"):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device) # from torch.Size([2, 1, 512]) to torch.Size([2, 512]), 2 is batch size
                output = model(input_id, mask)
                ## calculate loss, note that dtype(train_label) = long
                batch_loss = criterion(output,train_label)
                total_loss_train += batch_loss.item()
                ## calculate accuracy
                acc = (output.argmax(dim=1) == train_label).sum()
                total_acc_train += acc.item()
                ## update model
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            ## Start to evaluate model
            total_acc_val = 0
            total_loss_val = 0
            with torch.no_grad():
                for val_input, val_label in tqdm(self.val_dataloader,desc="Evaluation"):
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    output = model(input_id,mask)
                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    acc = (output.argmax(dim=1) == val_label).sum()
                    total_acc_val += acc.item()
            
            ## print the result
            print(f'''Epochs: {epoch_num + 1}
                    | Train Loss: {total_loss_train / self.train_length: .3f}
                    | Train Accuracy: {total_acc_train / self.train_length: .3f}
                    | Val Loss: {total_loss_val / self.val_length: .3f}
                    | Val Accuracy: {total_acc_val / self.val_length: .3f}''')
        ## save model
        torch.save(model.state_dict(), save_file)
        print("save model successfully")


if __name__ == '__main__':
    ## load source data and split them into train/test
    df = pd.read_csv("BBC News Train.csv")
    mask = np.random.rand(len(df)) < 0.8
    df_train, df_test = df[mask], df[~mask]
    ## create trainer instance and set params at first
    trainer = BertTrainer()
    trainer.set_params(batch_size=8,
                        lr=1e-6,
                        epochs=15)
    ## create dataset and dataloader after setting batch_size
    ## In this case, we make valid and test same (acutally one should additionally make valid data)
    trainer.customize_dataset(df_train = df_train, 
                                df_val = df_test)
    ## train model and save the result
    trainer.train_model(save_file = './BERT_WEIGHTS.pth')
    





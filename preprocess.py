import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bcc_news_labels = {"business":0,
                    "entertainment":1,
                    "sport":2,
                    "tech":3,
                    "politics":4,}


class BBCNewsDataset(Dataset):
    def __init__(self,df):
        self.labels = [bcc_news_labels[label] for label in df['Category']]
        self.texts = [tokenizer(text,padding = 'max_length',
                                    max_length = 512,
                                    truncation = True,
                                    return_tensors = "pt") for text in df['Text']]
    def get_classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self,idx):
        label = np.array(self.labels[idx])
        label = torch.tensor(label,dtype=torch.long)
        return label
    
    def get_batch_texts(self,idx):
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


if __name__ == '__main__':
    train_file = "BBC News Train.csv"
    df_train = pd.read_csv(train_file)
    print(df_train.head(10))

    newsdataset = BBCNewsDataset(df_train)
    print(newsdataset.get_batch_texts(3))
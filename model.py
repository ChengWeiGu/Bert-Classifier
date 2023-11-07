import numpy as np
import pandas as pd
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5, n_class = 5):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768,n_class)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id,
                                    attention_mask = mask,
                                    return_dict = False)
        droupout_output = self.dropout(pooled_output)
        linear_output = self.linear(droupout_output)
        final_layer = self.relu(linear_output)
        return final_layer




if __name__ == '__main__':
    model = BertClassifier()
    print(model)
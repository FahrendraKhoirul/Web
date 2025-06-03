import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin


class CustomClassifierAspect(nn.Module, PyTorchModelHubMixin):
        def __init__(self, bert, num_labels):
            super(CustomClassifierAspect, self).__init__()
            self.bert = bert
            self.linear38 = nn.Linear(bert.config.hidden_size, 38)
            self.dropout38 = nn.Dropout(0.2)
            self.linear8 = nn.Linear(38, 8)
            self.linear3 = nn.Linear(8, 3)
            self.linearOutput = nn.Linear(3, num_labels)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            logits38 = self.linear38(pooled_output)
            logits38 = self.dropout38(logits38)
            logits8 = self.linear8(logits38)
            logits3 = self.linear3(logits8)
            logits = self.linearOutput(logits3)
            probabilities = self.sigmoid(logits)
            return probabilities
        
class CustomClassifierSentiment(nn.Module, PyTorchModelHubMixin):
        def __init__(self, bert, num_labels):
            super(CustomClassifierSentiment, self).__init__()
            self.bert = bert
            self.linear38 = nn.Linear(bert.config.hidden_size, 38)
            self.dropout38 = nn.Dropout(0.2)
            self.linear8 = nn.Linear(38, 8)
            self.linear3 = nn.Linear(8, 3)
            self.linearOutput = nn.Linear(3, num_labels)
            self.softmax = nn.Softmax()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            logits38 = self.linear38(pooled_output)
            logits38 = self.dropout38(logits38)
            logits8 = self.linear8(logits38)
            logits3 = self.linear3(logits8)
            logits = self.linearOutput(logits3)
            probabilities = self.softmax(logits)
            return probabilities

# from transformers import AutoTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertModel,BertForMaskedLM
import torch.nn.functional as F
import torch.nn as nn
import torch
import json

def infoPraser(intent, conf):
    if conf>0.7:
        op={'status':True,
            'result':
                    {'intent':intent,
                     'confidence':conf}}
    else:
        op={'status':True,
            'result':None}
    return op

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

class IntentClassifier():
    def __init__(self, model_path, tokenizer_path, label_path):
        self.model = BertClassifier()
        self.model.load_state_dict(torch.load("model_chatbot.pth", "cpu"))
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.classes = json.load(open(label_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Bert model loaded successfully")


    def modelPass(self, text):
        encoding = self.tokenizer(text, padding='max_length', max_length = 8, truncation=True,
                                return_tensors="pt")
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        op=self.model.forward(input_ids.to(self.device),attention_mask.to(self.device))
        conf = F.softmax(op).max().item()
        pred = self.classes[str(op.argmax().item())]
        return pred, conf

    def getIntent(self,inp_text):
        
        intent, conf = self.modelPass(inp_text)

        op = infoPraser(intent, conf)
        
        return op



# processor = IntentClassifier("model_chatbot.pth","tokenizer_chatbot","id2label.json")
# processor.getIntent("forgot password")
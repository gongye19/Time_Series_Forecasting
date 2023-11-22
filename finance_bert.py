from transformers import pipeline
import pandas as pd
import torch
import numpy as np


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = './coinmarket.csv'
data = pd.read_csv(path)

print(data.head())
text = data['text']
length = text.map(lambda x: len(x)).max()

text = text.tolist()

specific_model = pipeline("sentiment-analysis",device = device)
# specific_model = pipeline(model="ProsusAI/finbert",device = device)
# text = ["I love you", "I hate you","you are so ugly",'bitcoin will go up']
output = specific_model(text)
print(output)


pos_news = 0
neg_news = 0
for i in range(len(output)):
    if output[i]['label'] == 'NEGATIVE':
        neg_news+=1
    elif output[i]['label'] == 'POSITIVE':
        pos_news+=1
print("POSITIVE account for: "+str((pos_news/len(output))*100)+'%')
print("NEGTIVE account for: "+str((neg_news/len(output))*100)+'%')
























# import torch
# from transformers import BertModel, BertConfig, BertTokenizer
# from torch import nn
# from transformers import BertModel



# class BertClassifier(nn.Module):
#     def __init__(self, path):
#         super(BertClassifier, self).__init__()
#         self.model_config = BertConfig.from_pretrained(path)
#         self.bert = BertModel.from_pretrained(path, config=self.model_config)
#         self.dropout = nn.Dropout(0.5)
#         self.linear = nn.Linear(768, 3)
#         self.relu = nn.ReLU()

#     def forward(self, input_id, mask):
#         _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
#         # dropout_output = self.dropout(pooled_output)
#         # linear_output = self.linear(dropout_output)
#         # final_layer = self.relu(linear_output)
#         return pooled_output
#         # return final_layer
    
# if __name__ == "__main__":

#     path = './finbert'
#     tokenizer = BertTokenizer.from_pretrained(path)  # 用来split句子成token的类方法
#     print('load bert model over')
#     input = 'Ann likes to eat chocolate'
#     # bert_input = {"input_ids": [],"token_type_ids": [], "attention_mask": []}
#     bert_input = tokenizer(input,padding='max_length', 
#                         max_length = 10, 
#                         truncation=True,
#                         return_tensors="pt")
#     input_id = bert_input['input_ids']
#     mask  = bert_input['attention_mask']
#     model = BertClassifier(path)
#     output = model.forward(input_id,mask)
# print(output)
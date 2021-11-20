import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import chain
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import copy
import csv
import json
import logging
import os
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel, RobertaPreTrainedModel, AutoConfig, RobertaModel
import numpy as np
import os 

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class Roberta_WiC(RobertaPreTrainedModel):
    def __init__(self,  model_name, config, dropout_rate):
        super(Roberta_WiC, self).__init__(config)
        self.model = AutoModel.from_pretrained(model_name, config=config)  # Load pretrained XLMRoberta

        self.num_labels = config.num_labels

        #self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.eos_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.entity_fc_layer1 = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.entity_fc_layer2 = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)

        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            dropout_rate,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):
        outputs = self.model(
            input_ids, attention_mask=attention_mask
        )  
        sequence_output = outputs[0] 
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        sentence_representation = self.eos_fc_layer(outputs.pooler_output)
        
        e1_h = self.entity_fc_layer1(e1_h)
        e2_h = self.entity_fc_layer2(e2_h)

        concat_h = torch.cat([sentence_representation, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                #loss_fct = nn.BCEWithLogitsLoss()
                #loss_fct = LabelSmoothingCrossEntropy()
                #loss_fct = Cross_FocalLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    

def test_pred(test_dataset, eval_batch_size, model):
    test_dataset = test_dataset
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,batch_size=eval_batch_size)

    logger = logging.getLogger(__name__)
    init_logger()

    # Eval!
    logger.info("***** Running evaluation on %s dataset *****", "test")
    #logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    nb_eval_steps = 0
    preds = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for batch in tqdm(test_dataloader, desc="Predicting"):
        batch = tuple(batch[t].to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": None,
                "e1_mask": batch[2],
                "e2_mask": batch[3],
            }
            outputs = model(**inputs)
            pred = outputs[0]

        nb_eval_steps += 1

        if preds is None:
            preds = pred.detach().cpu().numpy()
            #out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)
            #out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    # preds = np.around(preds)
    preds_label = np.argmax(preds, axis=1)
    df = pd.DataFrame(preds, columns=['pred_0','pred_1'])
    df['label'] = preds_label
    preds = preds.astype(int)
    return df 


def load_test_data(dataset_dir):
    dataset = pd.read_csv(dataset_dir, delimiter='\t')
    li = []
    for s1, s2 in zip(list(dataset['SENTENCE1']), list(dataset['SENTENCE2'])):
        li.append(s1+' '+s2)
    dataset["ANSWER"] = [0] * len(dataset)
    return dataset

def convert_sentence_to_features(train_dataset, tokenizer, max_len, mode='train'):
    max_seq_len=max_len
    #cls_token=tokenizer.cls_token
    #sep_token=tokenizer.sep_token
    pad_token=tokenizer.pad_token_id
    add_sep_token=False
    mask_padding_with_zero=True
    
    all_input_ids = []
    all_attention_mask = []
    all_e1_mask=[]
    all_e2_mask=[]
    all_label=[]
    m_len=0
    for idx in tqdm(range(len(train_dataset))):
        sentence = '<s>' + train_dataset['SENTENCE1'][idx][:train_dataset['start_s1'][idx]] \
            + ' <e1> ' + train_dataset['SENTENCE1'][idx][train_dataset['start_s1'][idx]:train_dataset['end_s1'][idx]] \
            + ' </e1> ' + train_dataset['SENTENCE1'][idx][train_dataset['end_s1'][idx]:] + '</s>' \
            + ' ' \
            + '<s>' + train_dataset['SENTENCE2'][idx][:train_dataset['start_s2'][idx]] \
            + ' <e2> ' + train_dataset['SENTENCE2'][idx][train_dataset['start_s2'][idx]:train_dataset['end_s2'][idx]] \
            + ' </e2> ' + train_dataset['SENTENCE2'][idx][train_dataset['end_s2'][idx]:] + '</s>'

            
        
        token = tokenizer.tokenize(sentence)
        m_len = max(m_len, len(token))
        e11_p = token.index("<e1>")  # the start position of entity1
        e12_p = token.index("</e1>")  # the end position of entity1
        e21_p = token.index("<e2>")  # the start position of entity2
        e22_p = token.index("</e2>")  # the end position of entity2

        token[e11_p] = "$"
        token[e12_p] = "$"
        token[e21_p] = "#"
        token[e22_p] = "#"

        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        special_tokens_count = 1

        if len(token) < max_seq_len - special_tokens_count:
            input_ids = tokenizer.convert_tokens_to_ids(token)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_e1_mask.append(e1_mask)
            all_e2_mask.append(e2_mask)
            all_label.append(train_dataset['ANSWER'][idx])

    all_features = {
        'input_ids' : torch.tensor(all_input_ids),
        'attention_mask' : torch.tensor(all_attention_mask),
        'e1_mask' : torch.tensor(all_e1_mask),
        'e2_mask' : torch.tensor(all_e2_mask)
    }  
    return RE_Dataset(all_features, all_label)

def softmax(sr):
    
    max_val = np.max(sr)
    exp_a = np.exp(sr-max_val)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

if __name__ == '__main__':
    eval_batch_size = 16
    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large", return_token_type_ids=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    test_dataset = load_test_data("./Data/NIKL_SKT_WiC_Test.tsv")
    test_Dataset = convert_sentence_to_features(test_dataset, tokenizer, max_len= 280, mode='eval')

    n_fold = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for fold in tqdm(range(n_fold)):
        config = AutoConfig.from_pretrained(
                "klue/roberta-large",
                num_labels= 2
            )
        model = Roberta_WiC(
                'klue/roberta-large',
                config= config, 
                dropout_rate = 0.1
            )
        model.load_state_dict(torch.load('./rbt_model'+str(fold)+'/pytorch_model.bin', map_location=device))
        model.eval()
        result = test_pred(test_Dataset, eval_batch_size, model)
        result.to_csv(str(fold)+'_rbt_result.csv', index=False)

    ensemble= pd.DataFrame()
    for fold in range(n_fold):
        df = pd.read_csv(str(fold)+'_rbt_result.csv')
        ensemble['label'+str(fold)]= df['label']


    soft_ensemble= pd.DataFrame()
    soft_ensemble['pred_0'] = ensemble['label0']
    soft_ensemble['pred_1'] = ensemble['label0']
    soft_ensemble['pred_0'] = 0
    soft_ensemble['pred_1'] = 0

    for fold in range(n_fold):
        df = pd.read_csv(str(fold)+'_rbt_result.csv')
        df= df.drop('label',axis=1)
        df = df.apply(softmax,axis=1)
        soft_ensemble['pred_0'] += df['pred_0']
        soft_ensemble['pred_1'] += df['pred_1']

    soft_ensemble['label'] = soft_ensemble['pred_0'] < soft_ensemble['pred_1']

    submission_json = {"wic" : []}
    for i, pred in enumerate(soft_ensemble['label'],0):
        if pred == True:
            submission_json["wic"].append({"idx" : i, "label" : True})
        else:
            submission_json["wic"].append({"idx" : i, "label" : False})

    with open('./submission.json', 'w') as fp:
        json.dump(submission_json, fp)
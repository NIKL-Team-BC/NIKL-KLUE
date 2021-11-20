import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm

class WICDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_data(dataset_dir, mode = 'train'):
    dataset = pd.read_csv(dataset_dir, delimiter='\t')
    li = []
    for s1, s2 in zip(list(dataset['SENTENCE1']), list(dataset['SENTENCE2'])):
        li.append(s1+' '+s2)
    dataset["ANSWER"] = dataset["ANSWER"].astype(int)
    if mode == 'test':
        dataset["ANSWER"] = [0] * len(dataset)
    return dataset

def convert_sentence_to_features(train_dataset, tokenizer, max_len):
    
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
    return WICDataset(all_features, all_label)
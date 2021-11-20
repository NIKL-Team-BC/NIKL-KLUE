import os
import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import logging
import os
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
import torch.nn.functional as F

from model import R_RoBERTa_WiC
from load_data import load_data, convert_sentence_to_features
from transformers import AutoModel, AutoConfig
import argparse

# seed 고정 
def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
    }

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


class Trainer(object):
    def __init__(self, args, model_dir = None,train_dataset=None, dev_dataset=None, test_dataset=None,tokenizer=None):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.model_dir = model_dir 
        self.best_score = 0
        self.hold_epoch = 0

        self.eval_batch_size = args.eval_batch_size
        self.train_batch_size = args.train_batch_size
        self.max_steps = args.max_steps
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate
        self.adam_epsilon= args.adam_epsilon
        self.warmup_steps = args.warmup_steps
        self.num_train_epochs = args.num_train_epochs
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.max_grad_norm = args.max_grad_norm
        self.dropout_rate = args.dropout_rate
        self.classifier_epoch= args.classifier_epoch
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        
        self.config = AutoConfig.from_pretrained(
            "klue/roberta-large",
            num_labels = 2
        )
        self.model = R_RoBERTa_WiC(
           "klue/roberta-large", 
            config=self.config, 
            dropout_rate = self.dropout_rate,
        )

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        
    def train(self):
        init_logger()
        seed_everything(args.seed)
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_batch_size,
        )

        if self.max_steps > 0:
            t_total = self.max_steps
            self.num_train_epochs = (
                self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total,
        )
        
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Total train batch size = %d", self.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.logging_steps)
        logger.info("  Save steps = %d", self.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = tqdm(range(int(self.num_train_epochs)), desc="Epoch")

        for epo_step in train_iterator:
            self.global_epo = epo_step
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(batch[t].to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[4],
                    "e1_mask": batch[2],
                    "e2_mask": batch[3]
                }
                
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                    logger.info("  global steps = %d", global_step)

                if 0 < self.max_steps < global_step:
                    epoch_iterator.close()
                    break
            
            self.evaluate("dev")
            if self.hold_epoch > 4:
                train_iterator.close()
                break
                
            if 0 < self.max_steps < global_step:
                train_iterator.close()
                break
          

        return global_step, tr_loss / global_step
    
   
    def evaluate(self, mode):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        elif mode == "train":
            dataset = self.train_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        # Eval!
        logger.info('---------------------------------------------------')
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(batch[t].to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[4],
                    "e1_mask": batch[2],
                    "e2_mask": batch[3],
                }
                #with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        
        if mode == "dev":
            if result['acc']>self.best_score:
                self.save_model()
                self.best_score = result['acc']
                print('save new best model acc : ',str(self.best_score))
                self.hold_epoch = 0
            else:
                self.hold_epoch += 1
        
        
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))
        logger.info("---------------------------------------------------")
        return results
    
    def test_pred(self):
        test_dataset = self.test_dataset
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler,batch_size=self.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", "test")
        logger.info("  Batch size = %d", self.eval_batch_size)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(test_dataloader, desc="Predicting"):
            batch = tuple(batch[t].to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": None,
                    "e1_mask": batch[2],
                    "e2_mask": batch[3],
                }
                outputs = self.model(**inputs)
                #print(outputs)
                pred = outputs[0]

            nb_eval_steps += 1

            if preds is None:
                preds = pred.detach().cpu().numpy()
            else:
                preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)
                
        preds_label = np.argmax(preds, axis=1)
        df = pd.DataFrame(preds, columns=['pred_0','pred_1'])
        df['label'] = preds_label
        preds = preds.astype(int)
        return df
        

    def save_model(self,new_dir=None):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if new_dir == None:
            pass
        else:
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            self.model_dir = new_dir
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.model_dir)

        # Save training arguments together with the trained model
        logger.info("Saving model checkpoint to %s", self.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")

if __name__ == '__main__':
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--train_batch_size', type=int, default=16)
  parser.add_argument('--eval_batch_size', type=int, default=16)
  parser.add_argument('--max_steps', type=int, default=-1)
  parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout for fully-connected layers")
  
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--warmup_steps', type=int, default=64, help="number of warmup steps for learning rate scheduler")
  parser.add_argument('--seed' , type=int , default = 42, help='random seed (default: 42)')
  
  parser.add_argument('--train_data_dir', default="./WIC/Data/NIKL_SKT_WiC_Train.tsv")
  parser.add_argument('--dev_data_dir', default="./WIC/Data/NIKL_SKT_WiC_Dev.tsv")
  parser.add_argument('--test_data_dir', default="./WIC/Data/NIKL_SKT_WiC_Test.tsv")
   

  args = parser.parse_args()
  train_dataset = load_data(args.train_data_dir)
  dev_dataset = load_data(args.dev_data_dir)
  ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
  tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large", return_token_type_ids=False)
  tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

  dev_dataset['ID'] = dev_dataset['ID']+ train_dataset.shape[0]
  concat_dataset = pd.concat([train_dataset,dev_dataset])
  
  def make_fold(x):
    if x <= concat_dataset.shape[0]*0.2:
        return 0
    elif x > concat_dataset.shape[0]*0.2 and x <= concat_dataset.shape[0]*0.4:
        return 1
    elif x > concat_dataset.shape[0]*0.4 and x <= concat_dataset.shape[0]*0.6 :
        return 2
    elif x > concat_dataset.shape[0]*0.6 and x <= concat_dataset.shape[0]*0.8 :
        return 3
    else:
        return 4
      
  concat_dataset['fold']= concat_dataset['ID'].apply(make_fold)
  concat_dataset = concat_dataset.drop(['ID', 'Target'],axis=1)

  logger = logging.getLogger(__name__)
  for fold in tqdm(range(5)): 
      trn_idx = concat_dataset[concat_dataset['fold'] != fold].index
      val_idx = concat_dataset[concat_dataset['fold'] == fold].index
      
      half_val_len = len(val_idx)//2
      add_trn_idx = val_idx[:half_val_len]
      
      trn_idx.append(add_trn_idx)
      val_idx = val_idx[half_val_len:]

      train_folds = concat_dataset.loc[trn_idx].reset_index(drop=True).drop(['fold'],axis=1)
      valid_folds = concat_dataset.loc[val_idx].reset_index(drop=True).drop(['fold'],axis=1)
      
      train_Dataset = convert_sentence_to_features(train_dataset, tokenizer, max_len = 280)
      valid_Dataset = convert_sentence_to_features(dev_dataset, tokenizer, max_len= 280)


      test_dataset = load_data(args.test_data_dir,mode='test')
      test_Dataset = convert_sentence_to_features(test_dataset, tokenizer, max_len= 280)

      trainer = Trainer(args,
                      train_dataset=train_Dataset,
                      dev_dataset=valid_Dataset,
                      test_dataset=test_Dataset,
                      tokenizer =tokenizer,
                      model_dir = './roberta_model_fold_'+str(fold))

      trainer.train()
      trainer.save_model(new_dir='./roberta_model_final_fold_'+str(fold))
      result = trainer.test_pred()
      submission_json = {"wic" : []}
      for i, pred in enumerate(result['label'],1):
          if pred == 1:
              submission_json["wic"].append({"idx" : i, "label" : 'true'})
          else:
              submission_json["wic"].append({"idx" : i, "label" : 'false'})
          
      with open(str(fold)+'_fold_klue_roberta_submission.json', 'w') as fp:
          json.dump(submission_json, fp)
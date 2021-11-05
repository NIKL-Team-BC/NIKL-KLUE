import pickle as pickle
import json
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
from transformers import MBartModel, MBartConfig
import transformers
from load_data import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import argparse
from importlib import import_module
from pathlib import Path
import glob
import re
from collections import defaultdict
from loss import create_criterion
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import time
from time import sleep
from transformers import PreTrainedTokenizerFast
from model import CustomPreTrainModel

from utils import check_arch

# from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

import wandb

# seed 고정 
def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def increment_output_dir(output_path, exist_ok=False):
  path = Path(output_path)
  if (path.exists() and exist_ok) or (not path.exists()):
    return str(path)
  else:
    dirs = glob.glob(f"{path}*")
    matches = [re.search(rf"%s(\d+)" %path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 2
    return f"{path}{n}"


def train(model_dir, args):

  seed_everything(args.seed)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f"device(GPU) : {torch.cuda.is_available()}")
  num_classes = 2
  
  # load model and tokenizerƒ
  MODEL_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("./copa_data_results/data/SKT_COPA_Train_All.tsv")
  val_dataset = load_data("./copa_data_results/data/SKT_COPA_Dev.tsv")

  # train eval split 20% k-fold (5)
  datasets_ = load_data("./copa_data_results/data/SKT_COPA_Train_All.tsv")
  labels_ = datasets_["label"]
  length = len(labels_)
  kf = args.kfold # 1
  class_indexs = defaultdict(list)
  for i, label_ in enumerate(labels_):
    class_indexs[np.argmax(label_)].append(i) #  class index [0] = [2,3,5,6], class index[1]=[나머지]
  val_indices = set()
  for index in class_indexs: # stratified: key : 0, 1 classindex[0][0/5:1/5]
    val_indices = (val_indices | set(class_indexs[index][int(len(class_indexs[index])*(kf-1)/5) : int(len(class_indexs[index])*kf/5)]))
  train_indices = set(range(length)) - val_indices

  train_dataset = datasets_.loc[np.array(list(train_indices))]
  val_dataset = datasets_.loc[np.array(list(val_indices))]

  train_label = train_dataset['label'].values
  val_label = val_dataset['label'].values

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, check_arch(args.model_type))
  tokenized_val = tokenized_dataset(val_dataset, tokenizer, check_arch(args.model_type))

  # make dataset for pytorch.
  train_dataset = CustomDataset(tokenized_train, train_label)
  val_dataset = CustomDataset(tokenized_val, val_label)
  # -- data_loader
  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=args.valid_batch_size,
      shuffle=False,
      drop_last=False,
  )

  # setting model hyperparameter
  if args.model_type == 'Electra_BoolQ':
    config_module = getattr(import_module("transformers"), "ElectraConfig")
  else:
    config_module = getattr(import_module("transformers"), args.model_type + "Config")
  
  model_config = config_module.from_pretrained(MODEL_NAME)
  model_config.num_labels = 2

  model_module = getattr(import_module("model"), args.model_type)
  if args.custompretrain:
    model = model_module.from_pretrained(args.custompretrain, args=args)
    model = CustomPreTrainModel(config=model_config, model=model)
  else:
    if args.model_type in ["BERT", "Electra"]:
      model = model_module.from_pretrained(MODEL_NAME, config=model_config, args=args)
    else:
      model = model_module(config=model_config, args=args)

  model.parameters
  model.to(device)
  save_dir = increment_output_dir(os.path.join(model_dir, args.name, str(args.kfold)))

  # Freeze Parameter
  for name, param in model.named_parameters():
    if ('cls_fc_layer' not in name) and ('label_classifier' not in name): # classifier layer
      param.requires_grad = False

  # -- loss & metric
  criterion = create_criterion(args.criterion)  # default: cross_entropy
  opt_module = getattr(import_module("transformers"), args.optimizer)
  optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps = 1e-8
    )
  scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=args.warmup_steps, 
    num_training_steps=len(train_loader) * args.epochs, 
    last_epoch=- 1
    )

  # -- logging
  start_time = time.time()
  logger = SummaryWriter(log_dir=save_dir)
  with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=4)

  best_val_acc = 0
  best_val_loss = np.inf
  for epoch in range(args.epochs):
    # train loop
    # unFreeze parameters
    if epoch == args.freeze_epoch:
      for name, param in model.named_parameters():
        param.requires_grad = True
    model.train()
    loss_value = 0
    matches = 0
    for idx, items in enumerate(train_loader):
      item = {key: val.to(device) for key, val in items.items()}

      optimizer.zero_grad()
      outs = model(**item)
      loss = criterion(outs[0], item['labels'])

      preds = torch.argmax(outs[0], dim=-1)

      loss.backward()
      optimizer.step()
      scheduler.step()

      loss_value += loss.item()
      matches += (preds == item['labels']).sum().item()
      if (idx + 1) % args.log_interval == 0:
        train_loss = loss_value / args.log_interval
        train_acc = matches / args.batch_size / args.log_interval
        current_lr = get_lr(optimizer)
        print(
            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
        )

        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
        logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
        logger.add_scalar("LR", current_lr, epoch * len(train_loader) + idx)

        loss_value = 0
        matches = 0

    # val loop
    with torch.no_grad():
      print("Calculating validation results...")
      model.eval()
      val_loss_items = []
      val_acc_items = []
      acc_okay = 0
      count_all = 0
      for idx, items in enumerate(tqdm(val_loader)):
        sleep(0.01)
        item = {key: val.to(device) for key, val in items.items()}

        outs = model(**item)

        preds = torch.argmax(outs[0], dim=-1)
        loss = criterion(outs[0], item['labels']).item()

        acc_item = (item['labels'] == preds).sum().item()

        val_loss_items.append(loss)
        val_acc_items.append(acc_item)
        acc_okay += acc_item
        count_all += len(preds)

      val_loss = np.sum(val_loss_items) / len(val_loss_items)
      val_acc = acc_okay / count_all

      if val_acc > best_val_acc:
        print(f"New best model for val acc : {val_acc:4.2%}! saving the best model..")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(f"{save_dir}/best")
        torch.save(args, os.path.join(f"{save_dir}/best", "training_args.bin"))
        best_val_acc = val_acc

      if val_loss < best_val_loss:
        best_val_loss = val_loss
      print(
        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.4}|| "
        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.4}"
      )

      logger.add_scalar("Val/loss", val_loss, epoch)
      logger.add_scalar("Val/accuracy", val_acc, epoch)
      if args.wandb:
        wandb.log({"train_loss": train_loss, "train_acc":train_acc,
            "lr":current_lr, "valid_loss":val_loss, "valid_acc":val_acc})
      s = f'Time elapsed: {(time.time() - start_time)/60: .2f} min'
      print(s)
      print()
      if epoch > 24:
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(f"{save_dir}/best")
        torch.save(args, os.path.join(f"{save_dir}/best", "training_args.bin"))
        break


if __name__ == '__main__':
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', type=str, default='BertBase')
  parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
  
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--freeze_epoch', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--valid_batch_size', type=int, default=128)
  parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
  parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout for fully-connected layers")

  parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
  parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
  
  parser.add_argument('--lr', type=float, default=1e-6)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--warmup_steps', type=int, default=500)               # number of warmup steps for learning rate scheduler
  parser.add_argument('--seed' , type=int , default = 42, help='random seed (default: 42)')
  parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
  parser.add_argument('--kfold', type=int, default=1, help='k-fold currunt step number')

  parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
  parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './copa_data_results/results'))
  parser.add_argument('--wandb', default=False, help='Use wandb : True, False')
  parser.add_argument('--custompretrain', type=str, default="", help='Use custom pretrain : model dir')

  args = parser.parse_args()


  args.epochs = 30
  args.model_type = "Roberta"
  args.pretrained_model = "klue/roberta-large"
  args.lr = 8e-6
  args.batch_size = 32

  i = 1
  print('='*40)
  print(f"k-fold num : {i}")
  print('='*40)
  args.kfold = i

  args.name = f'TrainAll_{args.model_type}_{args.lr}'

  args.wandb = True
  if args.wandb:
    wandb.login()
    wandb.init(project='NIKL-COPA', name=args.name, config=vars(args))

  train(args.model_dir, args)
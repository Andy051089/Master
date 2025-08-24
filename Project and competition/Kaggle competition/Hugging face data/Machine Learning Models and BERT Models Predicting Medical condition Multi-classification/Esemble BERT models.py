import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding, AutoModel, AutoConfig)
from imblearn.under_sampling import RandomUnderSampler
import random
import re
import os
import cbloss.loss
import balanced_loss
from sklearn.metrics import f1_score, recall_score, precision_score


config = {} 
config['SEED'] = 42

def seed_everything():
    random.seed(config['SEED'])
    os.environ['PYTHONHASHSEED'] = str(config['SEED'])
    np.random.seed(config['SEED'])
    torch.manual_seed(config['SEED'])
    torch.cuda.manual_seed(config['SEED'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything()

data= pd.read_csv('/mnt/d/kaggle_trainset.csv')
test_data= pd.read_csv('/mnt/d/kaggle_testset.csv')

label_map = {"neoplasms": 1,
             "digestive system diseases": 2,
             "nervous system diseases": 3,
             "cardiovascular diseases": 4,
             "general pathological conditions": 5}
data["label"] = data["label"].map(label_map)
data["label"] = data["label"]-1

test_size = 0.2
random_state = 42
train_df, test_df = train_test_split(data,
                                     test_size=test_size,
                                     random_state=random_state,
                                     stratify=data["label"])

test_data = test_data.rename(columns={"condition": "texts"})
common_conditions = data[data['texts'].isin(test_data['texts'])]
train_df = train_df[~train_df['texts'].isin(common_conditions['texts'])]

X_train = train_df["texts"]
y_train = train_df["label"]
X_test = test_df["texts"]
y_test = test_df["label"]

X_train_list = X_train.to_list()
y_train_list = y_train.to_list()
X_test_list = X_test.to_list()
y_test_list = y_test.to_list()

num_labels = len(np.unique(y_train))

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced",
                    classes=np.unique(y_train),
                    y=y_train)

class_weights = torch.tensor(class_weights, dtype=torch.float)


samples_per_class = train_df.groupby("label").count().T.values[0]
samples_per_class = torch.tensor(samples_per_class, dtype=torch.float)


class EnsembleClassifier(torch.nn.Module):
    def __init__(self, model_name_list, num_labels, fusion_method="concat", fusion_dropout_rate=0.1, freeze_base_models=False, loss_fct=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.model_name_list = model_name_list
        self.num_models = len(self.model_name_list)
        self.num_labels = num_labels
        self.fusion_method = fusion_method
        self.model_list = torch.nn.ModuleList()
        self.hidden_size_list = []
        for model_name in model_name_list:
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            self.model_list.append(model)
            self.hidden_size_list.append(config.hidden_size)

        if fusion_method == "concat":
            total_hidden_size = sum(self.hidden_size_list)
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(fusion_dropout_rate),
                torch.nn.Linear(total_hidden_size, num_labels)
                )
        if fusion_method == "mean" or fusion_method == "weighted":
            self.classifiers = torch.nn.ModuleList()
            for hidden_size in self.hidden_size_list:
                self.classifiers.append(torch.nn.Sequential(
                    torch.nn.Dropout(fusion_dropout_rate),
                    torch.nn.Linear(hidden_size, num_labels))
                    )

            if fusion_method == "weighted":
                initial_weights = 1.0/torch.ones(self.num_models)
                self.fusion_weights = torch.nn.Parameter(initial_weights)

        self.freeze_base_models = freeze_base_models
        self.loss_fct = loss_fct

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        if self.freeze_base_models:
            for model in self.base_models:
                for param in model.parameters():
                    param.requires_grad = False

        outputs_list = []
        logits_list = []
        for i, model in enumerate(self.model_list):
            model_kwargs = {"input_ids": input_ids,
                            "attention_mask": attention_mask}
            if token_type_ids is not None and "token_type_ids" in model.forward.__code__.co_varnames:
                model_kwargs["token_type_ids"] = token_type_ids

            outputs = model(**model_kwargs)
            outputs_list.append(outputs.last_hidden_state[:,0,:])

        if self.fusion_method == "concat":
            fused_features = torch.cat(outputs_list, dim=1)
            logits = self.classifier(fused_features)

        if self.fusion_method == "mean":
            for i, output in enumerate(outputs_list):
                logits_list.append(self.classifiers[i](output))
            logits = torch.mean(torch.stack(logits_list), dim=0)

        if self.fusion_method == "weighted":
            for i, output in enumerate(outputs_list):
                logits_list.append(self.classifiers[i](output))

            weights = torch.softmax(self.fusion_weights, dim=0)
            logits = torch.zeros_like(logits_list[0])
            for i, model_logits in enumerate(logits_list):
                logits += weights[i]*model_logits

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output_dict = dict(loss=loss, logits=logits)
        output_dict = {key: value for key, value in output_dict.items() if value is not None}

        return output_dict

class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, padding=False, max_length=512):
        self.encodings = tokenizer(texts,
                                   padding=padding,
                                   truncation=True,
                                   max_length=max_length)
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        if self.labels != None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.texts)

model_name_list = [#'emilyalsentzer/Bio_ClinicalBERT',
                  #  "dmis-lab/biobert-v1.1",
                   "HarshadKunjir/BioBERT_medical_abstract_classification",
                  #  "yerkekz/biobert-v1.1-finetuned-medical-txt",
                  #  "Fatih5555/Bio_ClinicalBERT_for_seizureFreedom_classification",
                   "allenai/scibert_scivocab_uncased",
                  #  "UFNLP/gatortron-base-2k"
                   # "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"
                   # "kamalkraj/bioelectra-base-discriminator-pubmed-pmc-lt",
                   # "menadsa/S-BlueBERT",
                   # "nuvocare/WikiMedical_sent_biobert"
                   ]

class_weights_type = None
# class_weights_type = "weightsCrossEntropyLoss"
# class_weights_type = "ClassBalancedLoss"
# class_weights_type = "FocalLoss"
class_weights_type = "focal_loss_class_balanced"

if class_weights_type == None:
    loss_fct = torch.nn.CrossEntropyLoss()
if class_weights_type == "weightsCrossEntropyLoss":
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
if class_weights_type == "ClassBalancedLoss":
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    loss_fct = cbloss.loss.ClassBalancedLoss(samples_per_cls=samples_per_class,
                                             beta=0.99,
                                             num_classes=len(samples_per_class),
                                             loss_func=loss_func)
if class_weights_type == "FocalLoss":
    loss_fct = cbloss.loss.FocalLoss(num_classes=num_labels,
                                     gamma=2.0,
                                     alpha=0.25)
if class_weights_type == "focal_loss_class_balanced":
    class_balanced = True
    loss_fct = balanced_loss.Loss(loss_type="focal_loss",
                                  samples_per_class=samples_per_class,
                                  class_balanced=class_balanced)

# fusion_method = "concat"
# fusion_method = "mean"
fusion_method = "weighted"

fusion_dropout_rate = 0.1
freeze_base_models = False
model = EnsembleClassifier(model_name_list,
                           num_labels=num_labels,
                           fusion_method=fusion_method,
                           fusion_dropout_rate=fusion_dropout_rate,
                           freeze_base_models=freeze_base_models,
                           loss_fct=loss_fct)

tokenizer = AutoTokenizer.from_pretrained(model_name_list[0])
padding = False
max_length = 512
train_dataset = TextClassificationDataset(texts=X_train_list,
                                          labels=y_train_list,
                                          tokenizer=tokenizer,
                                          padding=padding,
                                          max_length=max_length)

test_dataset = TextClassificationDataset(texts=X_test_list,
                                         labels=y_test_list,
                                         tokenizer=tokenizer,
                                         padding=padding,
                                         max_length=max_length)

X_train_dataset = TextClassificationDataset(texts=X_train_list,
                                            labels=None,
                                            tokenizer=tokenizer,
                                            padding=padding,
                                            max_length=max_length)

X_test_dataset = TextClassificationDataset(texts=X_test_list,
                                           labels=None,
                                           tokenizer=tokenizer,
                                           padding=padding,
                                           max_length=max_length)

def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    predictions = logits.argmax(axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    score_dict = {"f1": f1}
    return score_dict

training_args = TrainingArguments(output_dir= '/mnt/d/model',
                                  report_to= 'none',
                                  per_device_train_batch_size= 2,
                                  per_device_eval_batch_size= 2,
                                  learning_rate= 5e-5,
                                #   optim= optim,
                                #   weight_decay= 1e-5,
                                  num_train_epochs= 1,
                                  metric_for_best_model= 'f1',
                                  greater_is_better= True,
                                  eval_strategy= 'epoch',
                                  save_strategy= 'epoch',
                                  load_best_model_at_end= True,
                                  fp16= True,
                                  gradient_accumulation_steps= 4,
                                  torch_empty_cache_steps= 8,
                                  seed= 42)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
early_stopping_callback= EarlyStoppingCallback(early_stopping_patience=1)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  compute_metrics=compute_metrics,
                  data_collator=data_collator,
                  callbacks=[early_stopping_callback])

trainer.train()
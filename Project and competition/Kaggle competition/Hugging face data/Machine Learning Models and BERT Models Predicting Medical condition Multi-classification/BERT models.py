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
    DataCollatorWithPadding, AutoModel)
from imblearn.under_sampling import RandomUnderSampler
import random
import re

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

samples_per_class = train_df.groupby("label").count().T.values[0]
samples_per_class = torch.tensor(samples_per_class, dtype=torch.float)

model_name = "UFNLP/gatortron-base"
# model_name = "answerdotai/ModernBERT-base"
# model_name = "answerdotai/ModernBERT-large"
# model_name = "Simonlee711/Clinical_ModernBERT"
# model_name = "emilyalsentzer/Bio_ClinicalBERT"
# model_name = "chandar-lab/NeoBERT"
# model_name = "dmis-lab/biobert-v1.1"
# model_name = "dmis-lab/biobert-base-cased-v1.2"
# model_name= 'michiyasunaga/BioLinkBERT-base'
# model_name = "menadsa/S-BlueBERT"
# model_name = "allenai/scibert_scivocab_uncased"
# model_name = "allenai/scibert_scivocab_cased"
# model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
# model_name = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
# model_name = 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'
# model_name = 'ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section'
# model_name = 'Tsubasaz/clinical-pubmed-bert-base-512'
# model_name = "Tsubasaz/clinical-pubmed-bert-base-128"
# model_name = "kamalkraj/bioelectra-base-discriminator-pubmed-pmc-lt"
# model_name = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def tokenize_function(examples, text):
    encode = tokenizer(
        examples[text],
        truncation=True,
        padding=False,
        max_length=512)
    return encode

text = "condition"

train_dataset = datasets.Dataset.from_pandas(train_df)
test_dataset = datasets.Dataset.from_pandas(test_df)
kaggle_dataset = datasets.Dataset.from_pandas(test_data)

train_token_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"text": text}
)

test_token_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    fn_kwargs={"text": text}
)

kaggle_token_dataset = kaggle_dataset.map(
    tokenize_function,
    batched=True,
    fn_kwargs={"text": text}
)

# train_token_dataset1 = train_token_dataset1.remove_columns([text])

train_token_dataset = train_token_dataset.remove_columns([text])
eval_token_dataset = test_token_dataset.remove_columns([text])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# model_name = []
# for name, module in model.named_modules():
#     model_name.append(f"{name}: {module}")
#     print(f"{name}: {module}")

# base_model_parameters = []
# for param in model.base_model.parameters():
#     base_model_parameters.append(f"param.requires_grad={param.requires_grad}")
#     print(f"param.requires_grad={param.requires_grad}")
#     # param.requires_grad 均為 True。

# classifier_parameters = []
# for param in model.classifier.parameters():
#     classifier_parameters.append(f"param.requires_grad={param.requires_grad}")
#     print(f"param.requires_grad={param.requires_grad}")
#     # param.requires_grad 均為 True。

# model_named_parameters = []
# for name, param in model.named_parameters():
#     model_named_parameters.append(f"{name}, param.requires_grad={param.requires_grad}")
#     print(f"{name}:, param.requires_grad={param.requires_grad}")
# # model_named_parameters 多 classifier.weight, classifier.bias

# for param in model.base_model.parameters():
#     param.requires_grad = False


# for name, param in model.named_parameters():
#     if "19" in name or "18" in name or "17" in name or "16" in name or "15" in name:
#         param.requires_grad = True

# for name, param in model.named_parameters():
#     if "27" in name or "20" in name or "21" in name or "22" in name or "23" in name or "24" in name or "25" in name or "26" in name:
#         param.requires_grad = True

# for name, param in model.named_parameters():
#     if "final_norm" in name:
#         param.requires_grad = True

# for name, param in model.named_parameters():
#     if "head" in name:
#         param.requires_grad = True

# for name, param in model.named_parameters():
#     if "pooler.dense" in name:
#         param.requires_grad = True

# for name, param in model.named_parameters():
#     if "classifier" in name:
#         param.requires_grad = True

# model_named_parameters = []
# for name, param in model.named_parameters():
#     model_named_parameters.append(f"{name}, param.requires_grad={param.requires_grad}")
#     if param.requires_grad:
#         print(f"{name} is trainable")


# class CustomTrainer_class_weights(Trainer):
#     def __init__(self, class_weights=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights

#     def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         # logits = outputs.logits
#         logits = outputs.get("logits")

#         loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
#         loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss



# from cbloss.loss import ClassBalancedLoss

# class CustomTrainer_ClassBalancedLoss(Trainer):
#     def __init__(self, samples_per_class=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.samples_per_cls = samples_per_class

#     def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         # logits = outputs.logits
#         logits = outputs.get("logits")

#         loss_func = torch.nn.CrossEntropyLoss(reduction="none")
#         # Please Note "reduction = 'none'" should be set for all base Loss Function, while using ClassBalancedLoss.
#         loss_fct = ClassBalancedLoss(samples_per_cls=self.samples_per_cls,
#                                      beta=0.99,
#                                      num_classes=len(self.samples_per_cls),
#                                      loss_func=loss_func)
#         loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
#         # loss = loss_fct(logits, labels)
#         return (loss, outputs) if return_outputs else loss



# from cbloss.loss import FocalLoss

# class CustomTrainer_FocalLoss(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         # logits = outputs.logits
#         logits = outputs.get("logits")

#         loss_fct = FocalLoss(num_classes=model.config.num_labels,
#                              gamma=2.0,
#                              alpha=0.25)
#         loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
#         # loss = loss_fct(logits, labels)
#         return (loss, outputs) if return_outputs else loss


import balanced_loss

class CustomTrainer_focal_loss_class_balanced(Trainer):
    def __init__(self, samples_per_class=None, class_balanced=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # logits = outputs.logits
        logits = outputs.get("logits")

        loss_fct = balanced_loss.Loss(loss_type="focal_loss",
                                      samples_per_class=self.samples_per_class,
                                      class_balanced=self.class_balanced)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        # loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


from sklearn.metrics import f1_score, recall_score, precision_score

def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    # predictions = np.argmax(logits, axis=-1)
    predictions = logits.argmax(axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    score_dict = {"f1": f1}
    return score_dict

# early_stopping_callback= EarlyStoppingCallback(
#     early_stopping_patience= 2)

# optim = "lion_8bit"
# optim = "lion_32bit"
# optim = "adamw_torch_fused"
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


# CustomTrainer_type = "class_wights"
# CustomTrainer_type = "ClassBalancedLoss"
# CustomTrainer_type = "FocalLoss"
CustomTrainer_type = "focal_loss_class_balanced"

# if CustomTrainer_type == "class_wights":
#     CustomTrainer = CustomTrainer_class_weights
#     trainer = CustomTrainer(model=model,
#                             args=training_args,
#                             train_dataset=train_token_dataset,
#                             eval_dataset=eval_token_dataset,
#                             compute_metrics=compute_metrics,
#                             class_weights=class_weights,
#                             data_collator=data_collator)

# if CustomTrainer_type == "ClassBalancedLoss":
#     CustomTrainer = CustomTrainer_ClassBalancedLoss
#     trainer = CustomTrainer(model=model,
#                             args=training_args,
#                             train_dataset=train_token_dataset,
#                             eval_dataset=eval_token_dataset,
#                             compute_metrics=compute_metrics,
#                             samples_per_class=samples_per_class,
#                             data_collator=data_collator)

# if CustomTrainer_type == "FocalLoss":
#     CustomTrainer = CustomTrainer_FocalLoss
#     trainer = CustomTrainer(model=model,
#                             args=training_args,
#                             train_dataset=train_token_dataset,
#                             eval_dataset=eval_token_dataset,
#                             compute_metrics=compute_metrics)
    
if CustomTrainer_type== 'focal_loss_class_balanced':
    CustomTrainer= CustomTrainer_focal_loss_class_balanced
    trainer= CustomTrainer(model= model,
                            args= training_args,
                            train_dataset= train_token_dataset,
                            eval_dataset= eval_token_dataset,
                            compute_metrics= compute_metrics,
                            samples_per_class= samples_per_class,
                            class_balanced= True,
                            # callbacks= [early_stopping_callback],
                            data_collator=data_collator)
                    
trainer.train() 

y_train_predict = trainer.predict(train_token_dataset)
y_train_pred_logits = y_train_predict.predictions
train_logits_tensor = torch.from_numpy(y_train_pred_logits)
train_probabilities_tensor = torch.nn.functional.softmax(train_logits_tensor, dim=-1)
train_probabilities = train_probabilities_tensor.numpy()
train_probabilities_df = pd.DataFrame(train_probabilities)
y_train_pred = np.argmax(y_train_pred_logits, axis=-1)
y_train_pred = pd.Series(y_train_pred)

y_test_predict = trainer.predict(eval_token_dataset)
y_test_pred_logits = y_test_predict.predictions
test_logits_tensor = torch.from_numpy(y_test_pred_logits)
test_probabilities_tensor = torch.nn.functional.softmax(test_logits_tensor, dim=-1)
test_probabilities = test_probabilities_tensor.numpy()
test_probabilities_df = pd.DataFrame(test_probabilities)
y_test_pred = np.argmax(y_test_pred_logits, axis=-1)
y_test_pred = pd.Series(y_test_pred)

kaggle_predict = trainer.predict(kaggle_token_dataset)
kaggle_pred_logits = kaggle_predict.predictions
kaggle_logits_tensor = torch.from_numpy(kaggle_pred_logits)
kaggle_probabilities_tensor = torch.nn.functional.softmax(kaggle_logits_tensor, dim=-1)
kaggle_probabilities = kaggle_probabilities_tensor.numpy()
kaggle_probabilities_df = pd.DataFrame(kaggle_probabilities)
kaggle_pred = np.argmax(kaggle_pred_logits, axis=-1)
kaggle_pred = pd.Series(kaggle_pred)
kaggle_pred= kaggle_pred+ 1
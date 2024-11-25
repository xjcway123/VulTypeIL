import os
import torch
import datasets
import transformers
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer, MixedTemplate
from openprompt import PromptDataLoader, PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from scipy.spatial import distance
import torch.nn.functional as F
from collections import Counter
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Set parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 32
num_class = 23
max_seq_l = 512
lr = 5e-5
num_epochs = 100
use_cuda = True
model_name = "codet5"
pretrainedmodel_path = "D:/model/codet5-base"  # Path of the pre-trained model
early_stop_threshold = 10
ewc_lambda = 0.4  # EWC regularization term weight

# Define classes
classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

data_paths = [
    'H:\SOTitlePlus\SOTitlePlus\\task1\\train.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task2\\train.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task3\\train.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task4\\train.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task5\\train.xlsx',
]

test_paths = [
    'H:\SOTitlePlus\SOTitlePlus\\task1\\test.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task2\\test.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task3\\test.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task4\\test.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task5\\test.xlsx',

]

valid_paths = [
    'H:\SOTitlePlus\SOTitlePlus\\task1\\valid.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task2\\valid.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task3\\valid.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task4\\valid.xlsx',
    'H:\SOTitlePlus\SOTitlePlus\\task5\\valid.xlsx',
]

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F


class LabelAwareSmoothLoss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        """
        Initializes Label Aware Smooth Loss.

        Args:
        - num_classes (int): Number of classes.
        - smoothing (float): Base smoothing factor.
        """
        super(LabelAwareSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits, target):
        """
        Computes the Label Aware Smooth Loss.

        Args:
        - logits (Tensor): Predicted logits (N, num_classes).
        - target (Tensor): Ground truth labels (N,).

        Returns:
        - Tensor: Computed loss value.
        """
        # Calculate the smoothing factor per class
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        class_counts = target_one_hot.sum(dim=0) + 1e-6  # Avoid division by zero
        class_probs = class_counts / class_counts.sum()
        label_aware_smoothing = self.smoothing * (1 - class_probs)

        # Apply label-aware smoothing to target
        target_smooth = target_one_hot * (1 - label_aware_smoothing[target].unsqueeze(1)) + \
                        label_aware_smoothing[target].unsqueeze(1) / self.num_classes

        # Compute cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(target_smooth * log_probs, dim=-1)

        return loss.mean()



def mahalanobis_distance(features, mean, cov_inv):
    """Compute the Mahalanobis distance for a given feature set to the mean with covariance."""
    return [distance.mahalanobis(f, mean, cov_inv) for f in features]


def compute_mahalanobis(prompt_model, dataloader):
    prompt_model.eval()
    all_features = []
    all_cwe_ids = []

    with torch.no_grad():
        for inputs in dataloader:
            cwe_ids = inputs['tgt_text']  # 假设每个batch包含'cwe_id'字段
            all_cwe_ids.extend(cwe_ids)
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            all_features.append(logits.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    mean_features = np.mean(all_features, axis=0)
    cov_matrix = np.cov(all_features, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)  # 稳定性增加微小噪声
    mahalanobis_distances = mahalanobis_distance(all_features, mean_features, cov_inv)

    return mahalanobis_distances, all_features, all_cwe_ids


def select_uncertain_samples_mahalanobis(prompt_model, dataloader, num_samples=200):
    """Select high-uncertainty samples based on Mahalanobis distance with tail and head data."""
    mahalanobis_distances, all_features, all_cwe_ids = compute_mahalanobis(prompt_model, dataloader)

    # 计算CWE ID的出现次数
    cwe_counts = Counter(all_cwe_ids)
    total_samples = len(all_cwe_ids)
    tail_cwe_ids = {cwe_id for cwe_id, count in cwe_counts.items() if count < 0.05 * total_samples}
    # 分离taildata和headdata
    taildata = []
    headdata = []
    tail_distances = []
    head_distances = []
    for i, (feature, cwe_id, distance) in enumerate(zip(all_features, all_cwe_ids, mahalanobis_distances)):
        if cwe_id in tail_cwe_ids:
            taildata.append(feature)
            tail_distances.append(distance)
        else:
            headdata.append(feature)
            head_distances.append(distance)
    # 转换为numpy数组
    taildata = np.array(taildata)
    headdata = np.array(headdata)
    tail_distances = np.array(tail_distances)
    head_distances = np.array(head_distances)
    # 根据条件选择回放样本
    if len(taildata) >= 100:
        # 如果taildata样本数大于等于100，选择taildata和headdata各100个
        tail_indices = np.argsort(tail_distances)[-100:]
        head_indices = np.argsort(head_distances)[-100:]
    else:
        # 如果taildata样本数小于100，选择所有taildata和剩余headdata
        tail_indices = np.argsort(tail_distances)[-len(taildata):]
        head_indices = np.argsort(head_distances)[-(200 - len(taildata)):]
    selected_indices = np.concatenate((tail_indices, head_indices))

    # 确保taildata和headdata的维度一致
    tail_selected = taildata[tail_indices]
    head_selected = headdata[head_indices]

    # 确保tail_selected和head_selected都是二维数组
    if tail_selected.ndim == 1:
        tail_selected = np.expand_dims(tail_selected, axis=1)
    if head_selected.ndim == 1:
        head_selected = np.expand_dims(head_selected, axis=1)

    # 检查tail_selected和head_selected的列数是否一致
    if tail_selected.shape[1] != head_selected.shape[1]:
        # 将head_selected扩展到与tail_selected相同的列数
        head_selected = np.tile(head_selected, (1, tail_selected.shape[1]))

    selected_features = np.concatenate((tail_selected, head_selected), axis=0)
    return selected_indices, selected_features


# Define function to read examples
def read_prompt_examples(filename):
    examples = []
    data = pd.read_excel(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    type = data['type'].tolist()
    for idx in range(len(data)):
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=int(type[idx]),
            )
        )
    return examples


def read_and_merge_previous_datasets(current_index, data_paths):
    merged_data = pd.DataFrame()
    examples = []
    for i in range(current_index - 1):
        data = pd.read_excel(data_paths[i]).astype(str)
        merged_data = pd.concat([merged_data, data], ignore_index=True)
    desc = merged_data['description'].tolist()
    code = merged_data['abstract_func_before'].tolist()
    type = merged_data['type'].tolist()
    for idx in range(len(merged_data)):
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=int(type[idx]),
            )
        )
    return examples


# Define function to compute Fisher information matrix
def compute_fisher(prompt_model, train_dataloader):
    fisher_dict = {}
    prompt_model.eval()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['tgt_text'].cuda()
        loss = loss_func(logits, labels)
        prompt_model.zero_grad()
        loss.backward()
        for name, param in prompt_model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] = param.grad.data.clone().detach() ** 2
            else:
                fisher_dict[name] = torch.zeros_like(param)
    return fisher_dict


# Define function to compute EWC loss
def ewc_loss(prompt_model, fisher_dict, optpar_dict, ewc_lambda):
    loss = 0
    for name, param in prompt_model.named_parameters():
        if name in fisher_dict:
            fisher = fisher_dict[name]
            optpar = optpar_dict[name]
            loss += (fisher * (param - optpar) ** 2).sum()
    return ewc_lambda * loss


# Define function to test the model
def test(prompt_model, test_dataloader, name):
    num_test_steps = len(test_dataloader)
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = accuracy_score(alllabels, allpreds)
        precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
        precisionma, recallma, f1ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
        mcc = matthews_corrcoef(alllabels, allpreds)
        with open(os.path.join('.\\results', "{}.pred.csv".format(name)), 'w', encoding='utf-8') as f, \
                open(os.path.join('.\\results', "{}.gold.csv".format(name)), 'w', encoding='utf-8') as f1:
            for ref, gold in zip(allpreds, alllabels):
                f.write(str(ref) + '\n')
                f1.write(str(gold) + '\n')
        print("acc: {}   precisionma: {}  recallma: {} recallwei: {} weighted-f1: {}  macro-f1: {} mcc: {}".format(acc,
                                                                                                                   precisionma,
                                                                                                                   recallma,
                                                                                                                   recallwei,
                                                                                                                   f1wei,
                                                                                                                   f1ma,
                                                                                                                   mcc))
    return acc, precisionma, recallma, f1wei, f1ma


# Load model and tokenizer
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)

# Define template
template_text = ('The vulnerability description:  {"placeholder":"text_a"} '
                 'The code snippet: {"placeholder":"text_b"} '
                 'Identify the vulnerability type: {"mask"}.')

mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)
# Define the verbalizer
myverbalizer = ManualVerbalizer(tokenizer, classes=classes, label_words={
    "CWE-125": ["Out-of-bounds Read", "Memory Access Violation", "Read Beyond Boundaries"],
    "CWE-787": ["Out-of-bounds Write", "Buffer Overflow", "Memory Corruption"],
    "CWE-476": ["NULL Pointer Dereference", "Access to Null Pointer", "Dereferencing Null"],
    "CWE-119": ["Improper Memory Operations", "Buffer Overflow", "Memory Violation"],
    "CWE-416": ["Use After Free", "Dangling Pointer", "Memory Use After Deallocation"],
    "CWE-20": ["Improper Input Validation", "Input Sanitization Flaw", "Invalid Input Handling"],
    "CWE-190": ["Integer Overflow", "Integer Wraparound", "Overflow in Numeric Calculations"],
    "CWE-120": ["Classic Buffer Overflow", "Buffer Copy Error", "Unchecked Buffer Size"],
    "CWE-200": ["Exposure of Sensitive Data", "Unauthorized Information Access", "Sensitive Information Leak"],
    "CWE-400": ["Uncontrolled Resource Consumption", "Excessive Resource Allocation", "Denial of Service"],
    "CWE-362": ["Race Condition", "Shared Resource Access", "Improper Synchronization"],
    "CWE-401": ["Memory Leak", "Unreleased Memory", "Memory Management Flaw"],
    "CWE-617": ["Reachable Assertion", "Assertion Failure", "Accessing Unreachable Code"],
    "CWE-835": ["Infinite Loop", "Unreachable Loop", "Loop Without Exit Condition"],
    "CWE-772": ["Resource Management Failure", "Resource Leak", "Missing Resource Cleanup"],
    "CWE-369": ["Divide By Zero", "Division Error", "Mathematical Error in Calculation"],
    "CWE-264": ["Access Control Flaw", "Privilege Escalation", "Permission Violation"],
    "CWE-415": ["Double Free", "Double Memory Deallocation", "Memory Deallocation Error"],
    "CWE-122": ["Heap Overflow", "Buffer Overflow in Heap", "Heap-based Memory Corruption"],
    "CWE-22": ["Path Traversal", "Directory Traversal", "Improper Path Limitation"],
    "CWE-770": ["Unrestricted Resource Allocation", "Resource Overconsumption", "Resource Mismanagement"],
    "CWE-399": ["Resource Management Error", "Improper Resource Handling", "Insufficient Resource Control"],
    "CWE-189": ["Numeric Error", "Numerical Miscalculation", "Mathematical Error"]
})



# Fisher information matrix and old parameters
fisher_dict = {}
optpar_dict = {}


# Define the prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

# 初始化新的损失函数
loss_func = LabelAwareSmoothLoss(num_classes=num_class, smoothing=0.1)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.01}
]
optimizer_grouped_parameters2 = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-5)

# 加载五个任务的测试集
test_dataloader1 = PromptDataLoader(
    dataset=read_prompt_examples(test_paths[0]),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader2 = PromptDataLoader(
    dataset=read_prompt_examples(test_paths[1]),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader3 = PromptDataLoader(
    dataset=read_prompt_examples(test_paths[2]),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader4 = PromptDataLoader(
    dataset=read_prompt_examples(test_paths[3]),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader5 = PromptDataLoader(
    dataset=read_prompt_examples(test_paths[4]),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)

# Training process with EWC and Meta-Learning
global_step = 0
prev_dev_loss = float('inf')
best_dev_loss = float('inf')

# Main loop for each dataset
for i in range(1, 6):
    print("-----------------------第" + str(i) + "次任务---------------------------")
    if i == 1:
        train_dataloader = PromptDataLoader(
            dataset=read_prompt_examples(data_paths[i - 1]),
            template=mytemplate,
            tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
            batch_size=batch_size, shuffle=True,
            teacher_forcing=False, predict_eos_token=False, truncate_method="head",
            decoder_max_length=3)
    else:
        train_dataloader1 = PromptDataLoader(
            dataset=read_and_merge_previous_datasets(i, data_paths),
            template=mytemplate,
            tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
            batch_size=batch_size, shuffle=True,
            teacher_forcing=False, predict_eos_token=False, truncate_method="head",
            decoder_max_length=3)

        indices_to_replay, _ = select_uncertain_samples_mahalanobis(prompt_model, train_dataloader1, num_samples=200)
        examples = read_prompt_examples(data_paths[i - 1])
        for idx in indices_to_replay:
            examples.append(read_and_merge_previous_datasets(i, data_paths)[idx])


        train_dataloader = PromptDataLoader(
            dataset=examples,
            template=mytemplate,
            tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
            batch_size=batch_size, shuffle=True,
            teacher_forcing=False, predict_eos_token=False, truncate_method="head",
            decoder_max_length=3)

    validation_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(valid_paths[i - 1]),
        template=mytemplate,
        tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batch_size, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="head",
        decoder_max_length=3)

    num_training_steps = num_epochs * len(train_dataloader)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    if i >= 2:
        prompt_model.load_state_dict(
            torch.load(os.path.join('H:\SOTitlePlus\SOTitlePlus\\bestmodels1\\label.ckpt'),
                       map_location=torch.device('cuda:0')))
    early_stop_count = 0

    bestmetric = 0
    bestepoch = 0

    # Compute Fisher information matrix and save old parameters before training
    fisher_dict = compute_fisher(prompt_model, train_dataloader)
    for name, param in prompt_model.named_parameters():
        optpar_dict[name] = param.clone().detach()

    for epoch in range(num_epochs):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text'].cuda()

            loss_ce = loss_func(logits, labels)
            loss_ewc = ewc_loss(prompt_model, fisher_dict, optpar_dict, ewc_lambda)
            loss = loss_ce + loss_ewc

            loss.backward()
            tot_loss += loss.detach().float()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)

        eval_results = test(prompt_model, validation_dataloader, 'vuldetect_code_summary_val')
        print("eval_results:", eval_results)
        if eval_results[4] >= bestmetric:
            bestmetric = eval_results[4]
            bestepoch = epoch
            torch.save(prompt_model.state_dict(), 'H:\SOTitlePlus\SOTitlePlus\\bestmodels1\\label.ckpt')
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_threshold:
                break
            # Update Fisher information matrix and old parameters
        fisher_dict = compute_fisher(prompt_model, train_dataloader)
        optpar_dict = {name: param.clone().detach() for name, param in prompt_model.named_parameters()}

    # Load the best model and test it
    print("----------------------Load the best model and test it-----------------------------")
    prompt_model.load_state_dict(
        torch.load(os.path.join("H:\SOTitlePlus\SOTitlePlus\\bestmodels1\\label.ckpt"),
                   map_location=torch.device('cuda:0')))
    print("-------------第" + str(i) + "次任务在第1个数据集上的测试----------------")
    test(prompt_model, test_dataloader1, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第2个数据集上的测试----------------")
    test(prompt_model, test_dataloader2, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第3个数据集上的测试----------------")
    test(prompt_model, test_dataloader3, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第4个数据集上的测试----------------")
    test(prompt_model, test_dataloader4, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第5个数据集上的测试----------------")
    test(prompt_model, test_dataloader5, 'vuldetect_code_summary_test')

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
batch_size = 16
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


class OnlineEWCWithFocalLabelSmoothLoss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=0.4, decay_factor=0.9):
        super(OnlineEWCWithFocalLabelSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ewc_lambda = ewc_lambda
        self.decay_factor = decay_factor
        self.fisher_dict = {}
        self.optpar_dict = {}

    def focal_label_smooth_ce_loss(self, logits, target, w=0.5):
        # Label Smoothing Cross Entropy Loss
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        pred = F.softmax(logits, dim=-1)
        log_probs = torch.log(pred)
        ce_loss = -torch.sum(target_smooth * log_probs, dim=-1).mean()

        # Focal Loss
        true_class_pred = pred.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = ((1 - true_class_pred) ** self.focal_gamma) * self.focal_alpha
        focal_loss = (focal_weight * (-torch.log(true_class_pred))).mean()

        # Combine Focal Loss and Label Smoothing Cross Entropy Loss
        combined_loss = w * focal_loss + (1 - w) * ce_loss
        return combined_loss

    def ewc_loss_online(self, prompt_model):
        # Calculate Online EWC loss using accumulated Fisher information and optimal parameters
        ewc_loss = 0
        for name, param in prompt_model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optpar = self.optpar_dict[name]
                ewc_loss += (fisher * (param - optpar) ** 2).sum()
        return self.ewc_lambda * ewc_loss

    def update_fisher(self, prompt_model, dataloader):
        # Update Fisher information with current task's data and parameters
        current_fisher_dict = {}
        prompt_model.eval()

        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text'].cuda()
            loss = self.focal_label_smooth_ce_loss(logits, labels)
            prompt_model.zero_grad()
            loss.backward()

            for name, param in prompt_model.named_parameters():
                if param.grad is not None:
                    if name not in current_fisher_dict:
                        current_fisher_dict[name] = param.grad.data.clone().detach() ** 2
                    else:
                        current_fisher_dict[name] += param.grad.data.clone().detach() ** 2

        # Average Fisher information for the current task
        for name in current_fisher_dict:
            current_fisher_dict[name] /= len(dataloader)

        # Update global Fisher information with decay
        for name, fisher_value in current_fisher_dict.items():
            if name in self.fisher_dict:
                self.fisher_dict[name] = self.decay_factor * self.fisher_dict[name] + fisher_value
            else:
                self.fisher_dict[name] = fisher_value
            self.optpar_dict[name] = prompt_model.state_dict()[name].clone().detach()

    def forward(self, prompt_model, logits, target):
        # Combine Focal + Label Smoothing CE Loss and Online EWC Loss
        focal_label_smooth_loss = self.focal_label_smooth_ce_loss(logits, target)
        ewc_loss = self.ewc_loss_online(prompt_model)
        total_loss = focal_label_smooth_loss + ewc_loss
        return total_loss


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

# Two-phase training functions
def train_phase_one(prompt_model, train_dataloader, val_dataloader, optimizer1, optimizer2, scheduler1, scheduler2,
                    num_epochs, loss_func_no_ewc, patience=5):
    """Phase 1: Train with Focal Loss + Label Smoothing only (no EWC) to learn task-specific features."""
    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

    for epoch in range(num_epochs):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text'].cuda()
            # Phase 1 loss (without EWC)
            loss = loss_func_no_ewc.focal_label_smooth_ce_loss(logits, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)

        # Validation phase to check performance on validation data
        prompt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_dataloader:
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['tgt_text'].cuda()
                loss = loss_func_no_ewc.focal_label_smooth_ce_loss(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter
            torch.save(prompt_model.state_dict(), 'H:\\SOTitlePlus\\tasks_five\\discussion2\\model\\best005.ckpt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def train_phase_two(prompt_model, train_dataloader, val_dataloader, optimizer1, optimizer2, scheduler1, scheduler2,
                    num_epochs, loss_func_with_ewc, patience=5):
    """Phase 2: Train with Focal Loss + Label Smoothing + EWC to prevent forgetting of previous tasks."""
    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

    for epoch in range(num_epochs):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text'].cuda()
            # Phase 2 loss (with EWC)
            loss = loss_func_with_ewc(prompt_model, logits, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)

        # Validation phase to check performance on validation data
        prompt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_dataloader:
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['tgt_text'].cuda()
                loss = loss_func_with_ewc(prompt_model, logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter
            torch.save(prompt_model.state_dict(), 'H:\\SOTitlePlus\\tasks_five\\discussion2\\model\\best005.ckpt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


# Initialize EWC and non-EWC loss functions
loss_func_no_ewc = OnlineEWCWithFocalLabelSmoothLoss(num_classes=num_class, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=0.0)
loss_func_with_ewc = OnlineEWCWithFocalLabelSmoothLoss(num_classes=num_class, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=ewc_lambda)


# Load model and tokenizer
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)

# Define template
template_text = ('The code snippet: {"placeholder":"text_a"} '
                 'The vulnerability description:  {"placeholder":"text_b"} '
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


# Define the prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

# 初始化损失函数
loss_func = OnlineEWCWithFocalLabelSmoothLoss(num_classes=num_class, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=0.4, decay_factor=0.9)

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
    print(f"----------------------- Task {i} ---------------------------")
    if i == 1:
        train_dataloader = PromptDataLoader(
            dataset=read_prompt_examples(data_paths[i - 1]),
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            batch_size=batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )
    else:
        # Create dataloader with merged previous datasets
        train_dataloader1 = PromptDataLoader(
            dataset=read_and_merge_previous_datasets(i, data_paths),
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            batch_size=batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )

        indices_to_replay, _ = select_uncertain_samples_mahalanobis(prompt_model, train_dataloader1, num_samples=50)
        examples = read_prompt_examples(data_paths[i - 1])
        for idx in indices_to_replay:
            examples.append(read_and_merge_previous_datasets(i, data_paths)[idx])

        train_dataloader = PromptDataLoader(
            dataset=examples,
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            batch_size=batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )

    validation_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(valid_paths[i - 1]),
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_l,
        batch_size=batch_size,
        shuffle=True,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head",
        decoder_max_length=3
    )

    num_training_steps = num_epochs * len(train_dataloader)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)

    if i >= 2:
        prompt_model.load_state_dict(
            torch.load(os.path.join('H:\\SOTitlePlus\\tasks_five\\discussion2\\model\\best005.ckpt'),
                       map_location=torch.device('cuda:0')))

    print(f"Starting Phase 1 for Task {i}: Focal Loss + Label Smoothing")
    train_phase_one(
        prompt_model,
        train_dataloader,
        validation_dataloader,  # Add validation data loader
        optimizer1,
        optimizer2,
        scheduler1,
        scheduler2,
        num_epochs,
        loss_func_no_ewc,
        patience=5  # You can adjust patience as needed
    )

    eval_results_phase1 = test(prompt_model, validation_dataloader, f'task_{i}_val_phase1')
    print(f"Phase 1 evaluation for task {i}: ", eval_results_phase1)

    prompt_model.load_state_dict(
        torch.load(os.path.join('H:\\SOTitlePlus\\tasks_five\\discussion2\\model\\best005.ckpt'),
                   map_location=torch.device('cuda:0')))
    print(f"Starting Phase 2 for Task {i}: Focal Loss + Label Smoothing + EWC")
    train_phase_two(
        prompt_model,
        train_dataloader,
        validation_dataloader,  # Add validation data loader
        optimizer1,
        optimizer2,
        scheduler1,
        scheduler2,
        num_epochs,
        loss_func_with_ewc,
        patience=5  # You can adjust patience as needed
    )


    eval_results_phase2 = test(prompt_model, validation_dataloader, f'task_{i}_val_phase2')
    print(f"Phase 2 evaluation for task {i}: ", eval_results_phase2)
    # Update Fisher Information for EWC after each task
    loss_func_with_ewc.update_fisher(prompt_model, train_dataloader)
    print(f"Testing Task {i} model on previous datasets after Phase 2")
    # Load the best model and test it on all tasks

    print("----------------------Load the best model and test it-----------------------------")
    prompt_model.load_state_dict(
        torch.load(os.path.join("H:\\SOTitlePlus\\tasks_five\\discussion2\\model\\best005.ckpt"),
                   map_location=torch.device('cuda:0')))
    for task_dataloader, task_name in zip(
            [test_dataloader1, test_dataloader2, test_dataloader3, test_dataloader4, test_dataloader5],
            ['task1', 'task2', 'task3', 'task4', 'task5']):
        test(prompt_model, task_dataloader, f'{task_name}_test_task_{i}')

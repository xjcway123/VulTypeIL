import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, MixedTemplate
from openprompt import PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 23
max_seq_l = 512
lr = 5e-5
num_epochs = 5000
use_cuda = True
model_name = "roberta"
pretrainedmodel_path = "D:/model/codebert-base"
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


def read_data_to_dataframe(filename):
    data = pd.read_excel(filename).astype(str)
    return data[['abstract_func_before', 'description', 'type']]


def convert_dataframe_to_dataset(data):
    examples = {
        'text_a': [],
        'text_b': [],
        'label': []
    }
    for idx, row in data.iterrows():
        examples['text_a'].append(' '.join(row['abstract_func_before'].split(' ')[:384]))
        examples['text_b'].append(' '.join(row['description'].split(' ')[:64]))
        examples['label'].append(int(row['type']))
    return Dataset.from_dict(examples)



# Define function to compute Fisher information matrix
def compute_fisher(prompt_model, train_dataloader):
    fisher_dict = {}
    prompt_model.eval()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label'].cuda()
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


def compute_confidence(prompt_model, dataloader):
    prompt_model.eval()
    confidences = []
    with torch.no_grad():
        for inputs in dataloader:
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs, _ = torch.max(softmax_probs, dim=-1)
            confidences.extend(max_probs.cpu().numpy())
    return confidences





test_dataloader1 =[]
test_dataloader2 =[]
test_dataloader3 =[]
test_dataloader4 =[]
test_dataloader5 =[]


train_dataset1 = convert_dataframe_to_dataset(read_data_to_dataframe(data_paths[0]))
valid_dataset1 = convert_dataframe_to_dataset(read_data_to_dataframe(valid_paths[0]))
test_dataset1 = convert_dataframe_to_dataset(read_data_to_dataframe(test_paths[0]))

train_dataset2 = convert_dataframe_to_dataset(read_data_to_dataframe(data_paths[1]))
valid_dataset2 = convert_dataframe_to_dataset(read_data_to_dataframe(valid_paths[1]))
test_dataset2 = convert_dataframe_to_dataset(read_data_to_dataframe(test_paths[1]))

train_dataset3 = convert_dataframe_to_dataset(read_data_to_dataframe(data_paths[2]))
valid_dataset3 = convert_dataframe_to_dataset(read_data_to_dataframe(valid_paths[2]))
test_dataset3 = convert_dataframe_to_dataset(read_data_to_dataframe(test_paths[2]))

train_dataset4 = convert_dataframe_to_dataset(read_data_to_dataframe(data_paths[3]))
valid_dataset4 = convert_dataframe_to_dataset(read_data_to_dataframe(valid_paths[3]))
test_dataset4 = convert_dataframe_to_dataset(read_data_to_dataframe(test_paths[3]))

train_dataset5 = convert_dataframe_to_dataset(read_data_to_dataframe(data_paths[4]))
valid_dataset5 = convert_dataframe_to_dataset(read_data_to_dataframe(valid_paths[4]))
test_dataset5 = convert_dataframe_to_dataset(read_data_to_dataframe(test_paths[4]))

train_val_test1 = {
    'train': train_dataset1,
    'validation': valid_dataset1,
    'test': test_dataset1
}

train_val_test2 = {
    'train': train_dataset2,
    'validation': valid_dataset2,
    'test': test_dataset2
}

train_val_test3 = {
    'train': train_dataset3,
    'validation': valid_dataset3,
    'test': test_dataset3
}

train_val_test4 = {
    'train': train_dataset4,
    'validation': valid_dataset4,
    'test': test_dataset4
}

train_val_test5 = {
    'train': train_dataset5,
    'validation': valid_dataset5,
    'test': test_dataset5
}

dataset1 = {}
for split in ['train', 'validation', 'test']:
    dataset1[split] = []
    for data in train_val_test1[split]:
        input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
        dataset1[split].append(input_example)

dataset2 = {}
for split in ['train', 'validation', 'test']:
    dataset2[split] = []
    for data in train_val_test2[split]:
        input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
        dataset2[split].append(input_example)

dataset3 = {}
for split in ['train', 'validation', 'test']:
    dataset3[split] = []
    for data in train_val_test3[split]:
        input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
        dataset3[split].append(input_example)

dataset4 = {}
for split in ['train', 'validation', 'test']:
    dataset4[split] = []
    for data in train_val_test4[split]:
        input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
        dataset4[split].append(input_example)

dataset5 = {}
for split in ['train', 'validation', 'test']:
    dataset5[split] = []
    for data in train_val_test5[split]:
        input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
        dataset5[split].append(input_example)

# Load PLM
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", pretrainedmodel_path)
# Construct template
template_text = ('The code snippet: {"placeholder":"text_a"} '
                 'The vulnerability description:  {"placeholder":"text_b"} '
                 'Identify the vulnerability type: {"mask"}.')

mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)
# Verbalizer
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

# Prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()


test_dataloader1 = PromptDataLoader(dataset=dataset1['test'], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                       batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="head", decoder_max_length=3)
test_dataloader2 = PromptDataLoader(dataset=dataset2['test'], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                       batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="head", decoder_max_length=3)
test_dataloader3 = PromptDataLoader(dataset=dataset3['test'], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                       batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="head", decoder_max_length=3)
test_dataloader4 = PromptDataLoader(dataset=dataset4['test'], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                       batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="head", decoder_max_length=3)
test_dataloader5 = PromptDataLoader(dataset=dataset5['test'], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                       batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="head", decoder_max_length=3)


for i in range(1, 6):
    # Read and convert data
    train_data = read_data_to_dataframe(data_paths[i - 1])
    valid_data = read_data_to_dataframe(valid_paths[i - 1])
    test_data = read_data_to_dataframe(test_paths[i - 1])

    train_dataset = convert_dataframe_to_dataset(train_data)
    valid_dataset = convert_dataframe_to_dataset(valid_data)
    test_dataset = convert_dataframe_to_dataset(test_data)

    # Create the splits dictionary
    train_val_test = {
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    }

    # Convert to InputExample format
    dataset = {}
    for split in ['train', 'validation', 'test']:
        dataset[split] = []
        for data in train_val_test[split]:
            input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
            dataset[split].append(input_example)


    # DataLoaders
    train_dataloader = PromptDataLoader(dataset=dataset['train'], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                        batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                        predict_eos_token=False, truncate_method="head", decoder_max_length=3)
    validation_dataloader = PromptDataLoader(dataset=dataset['validation'], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                             batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                             predict_eos_token=False, truncate_method="head", decoder_max_length=3)


    # Optimizers and scheduler
    loss_func = torch.nn.CrossEntropyLoss()
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
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)


    # Test function
    def test(prompt_model, test_dataloader, name):
        num_test_steps = len(test_dataloader)
        progress_bar = tqdm(range(num_test_steps))
        allpreds = []
        alllabels = []
        with torch.no_grad():
            for step, inputs in enumerate(test_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']

                progress_bar.update(1)
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            acc = accuracy_score(alllabels, allpreds)
            precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
            precisionma, recallma, f1ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
            mcc = matthews_corrcoef(alllabels, allpreds)
            with open(os.path.join('./results', "{}.pred.csv".format(name)), 'w', encoding='utf-8') as f, \
                    open(os.path.join('./results', "{}.gold.csv".format(name)), 'w', encoding='utf-8') as f1:
                for ref, gold in zip(allpreds, alllabels):
                    f.write(str(ref) + '\n')
                    f1.write(str(gold) + '\n')

            print("acc: {}   precisionma :{}  recallma:{} recallwei{} weighted-f1: {}  macro-f1: {} mcc:{}".format(acc,
                                                                                                                   precisionma,
                                                                                                                   recallma,
                                                                                                                   recallwei,
                                                                                                                   f1wei,
                                                                                                                   f1ma,
                                                                                                                   mcc))
        return acc, precisionma, recallma, f1wei, f1ma


    # Training and evaluation
    output_dir = "vultypeprompt3_log"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    progress_bar = tqdm(range(num_training_steps))
    bestmetric = 0
    bestepoch = 0
    early_stop_count = 0
    for epoch in range(num_epochs):
        tot_loss = 0
        num_examples = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label'].cuda()
            loss_ce = loss_func(logits, labels)
            loss_ewc = ewc_loss(prompt_model, fisher_dict, optpar_dict, ewc_lambda)
            loss = loss_ce + loss_ewc

            loss.backward()
            tot_loss += loss.detach().float()
            num_examples += inputs['label'].size(0)
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)
        tot_loss = tot_loss / num_examples
        print("tot_loss:", tot_loss)

        print('\n\nepoch{}------------validate------------'.format(epoch))
        acc, precision, recall, f1wei, f1ma = test(prompt_model, validation_dataloader, name="dev")
        if f1ma > bestmetric:
            bestmetric = f1ma
            bestepoch = epoch
            torch.save(prompt_model.state_dict(), f"{output_dir}/best.ckpt")
        else:
            early_stop_count += 1
            if early_stop_count == early_stop_threshold:
                print("early stopping!!!")
                break
        fisher_dict = compute_fisher(prompt_model, train_dataloader)
        optpar_dict = {name: param.clone().detach() for name, param in prompt_model.named_parameters()}


    print('\n\nepoch{}------------test------------'.format(epoch))
    prompt_model.load_state_dict(torch.load(os.path.join(output_dir, "best.ckpt"), map_location=torch.device('cuda:0')))
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
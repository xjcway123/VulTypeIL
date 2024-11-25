import os
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer,MixedTemplate
from openprompt import PromptDataLoader, PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import warnings
import time

warnings.filterwarnings("ignore")

# Set parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 23
max_seq_l = 512
lr = 5e-5
num_epochs = 5000
use_cuda = True
model_name = "codet5"
pretrainedmodel_path = "D:/model/codet5-base"  # Path of the pre-trained model
early_stop_threshold = 10

classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

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

# Load model and tokenizer
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)

# Define template
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


# Define function to test the model
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
            labels = inputs['tgt_text']
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
        print("acc: {}   precisionma: {}  recallma: {} recallwei: {} weighted-f1: {}  macro-f1: {} mcc: {}".format(acc,
                                                                                                                   precisionma,
                                                                                                                   recallma,
                                                                                                                   recallwei,
                                                                                                                   f1wei,
                                                                                                                   f1ma,
                                                                                                                   mcc))
    return acc, precisionma, recallma, f1wei, f1ma


# Define the prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()
# Define the optimizer and scheduler
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

# 加载五个任务的测试集
test_dataloader1 = PromptDataLoader(
    dataset=read_prompt_examples(r"H:\SOTitlePlus\SOTitlePlus\task1/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader2 = PromptDataLoader(
    dataset=read_prompt_examples(r"H:\SOTitlePlus\SOTitlePlus\task2/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader3 = PromptDataLoader(
    dataset=read_prompt_examples(r"H:\SOTitlePlus\SOTitlePlus\task3/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader4 = PromptDataLoader(
    dataset=read_prompt_examples(r"H:\SOTitlePlus\SOTitlePlus\task4/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader5 = PromptDataLoader(
    dataset=read_prompt_examples(r"H:\SOTitlePlus\SOTitlePlus\task5/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)

# Main loop for each dataset
for i in range(1, 6):
    start_time =  time.time()
    print("-----------------------第" + str(i) + "次任务---------------------------")
    train_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(r"H:\SOTitlePlus\SOTitlePlus\task" + str(i) + "/train.xlsx"),
        template=mytemplate,
        tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batch_size, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="head",
        decoder_max_length=3)
    validation_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(r"H:\SOTitlePlus\SOTitlePlus\task" + str(i) + "/valid.xlsx"),
        template=mytemplate,
        tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batch_size, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="head",
        decoder_max_length=3)

    num_training_steps = num_epochs * len(train_dataloader)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)

    if i >= 2:
        prompt_model.load_state_dict(
            torch.load(os.path.join("H:\\SOTitlePlus\\SOTitlePlus\\new\\best.ckpt"),
                       map_location=torch.device('cuda:0')))
    # Train and validate the model
    output_dir = "result1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    progress_bar = tqdm(range(num_training_steps))
    bestmetric = 0
    bestepoch = 0
    early_stop_count = 0

    for epoch in range(num_epochs):
        # train
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):

            if use_cuda:
                inputs = inputs.cuda()

            logits = prompt_model(inputs)

            labels = inputs['tgt_text'].cuda()

            loss = loss_func(logits, labels)

            # 在向 param.grad 添加随机噪声之前，检查 param.grad 是否为 None
            for param in prompt_model.parameters():
                if param.requires_grad:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += 0.01 * torch.randn_like(param)

            try:
                loss.backward()
            except:
                print(loss)
                exit()
            tot_loss += loss.item()
            optimizer1.step()
            optimizer1.zero_grad()
            scheduler1.step()
            optimizer2.step()
            optimizer2.zero_grad()
            scheduler2.step()
            progress_bar.update(1)
        print("\nEpoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
        this_epoch_best = False

        # validate
        print('\n\nepoch{}------------validate------------'.format(epoch))
        acc, precision, recall, f1wei, f1mi = test(prompt_model, validation_dataloader, name="dev")
        if f1mi > bestmetric:
            bestmetric = f1mi
            bestepoch = epoch
            this_best_epoch = True
            torch.save(prompt_model.state_dict(), "H:\\SOTitlePlus\\SOTitlePlus\\new\\best.ckpt")
        # if this_epoch_best:
        #     early_stop_count=0
        else:
            early_stop_count += 1
            if early_stop_count == early_stop_threshold:
                print("early stopping!!!")
                break
        # test

        end_time = time.time()

        print("用时：", end_time - start_time)


    # Load the best model and test it
    print("----------------------Load the best model and test it-----------------------------")
    prompt_model.load_state_dict(
        torch.load(os.path.join("H:\\SOTitlePlus\\SOTitlePlus\\new\\best.ckpt"),
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


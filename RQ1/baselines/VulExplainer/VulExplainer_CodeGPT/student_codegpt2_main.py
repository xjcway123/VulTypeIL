from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, GPT2Tokenizer, GPT2Model)
from tqdm import tqdm
from student_codegpt2_model import StudentGPT2
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from textcnn_model import TextCNN
from teacher_model import CNNTeacherModel

global BEST_BETA

cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 group):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.group = group
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, cwe_label_map, group_label_map, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        funcs = df["func_before"].tolist()
        labels = df["CWE ID"].tolist()
        groups = df["cwe_abstract_group"].tolist()
        for i in tqdm(range(len(funcs))):
            label = cwe_label_map[labels[i]][1]
            # count label freq if it's training data
            if file_type == "train":
                cwe_label_map[labels[i]][2] += 1
            self.examples.append(convert_examples_to_features(funcs[i], label, group_label_map[groups[i]], tokenizer, args))
        if file_type == "train":
            self.cwe_label_map = cwe_label_map
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("group: {}".format(example.group))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label).float(), torch.tensor(self.examples[i].group)

def convert_examples_to_features(func, label, group, tokenizer, args):
    # source
    input_ids = tokenizer.encode(func, truncation=True, max_length=args.block_size, padding='max_length')
    source_tokens = tokenizer.tokenize(func)[:args.block_size]
    return InputFeatures(source_tokens, input_ids, label, group)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, teacher_model, tokenizer, eval_dataset, cwe_label_map):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc=0

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            input_ids, labels, groups = [x.to(args.device) for x in batch]
            model.train()
            teacher_logits = teacher_model(input_ids=input_ids, groups=groups, labels=labels, return_logit=True)
            soft_label = nn.functional.log_softmax(teacher_logits, dim=-1)
            probs = torch.softmax(teacher_logits, dim=-1)
            teacher_preds = torch.argmax(probs, dim=1)
            teacher_preds_one_hot = []
            for pred in teacher_preds:
                for _, v in cwe_label_map.items():
                    if v[0] == pred.item():
                        teacher_preds_one_hot.append(v[1])
            teacher_preds_one_hot = torch.tensor(teacher_preds_one_hot).float().to(args.device)
            if args.use_hard_distil:
                loss, loss_dis = model(input_ids=input_ids, 
                                        labels=labels, 
                                        soft_label=None,
                                        hard_label=teacher_preds_one_hot)
            else:
                loss, loss_dis = model(input_ids=input_ids,
                                        labels=labels,
                                        soft_label=soft_label,
                                        hard_label=None)
            if args.n_gpu > 1:
                loss = loss.mean()
                loss_dis = loss_dis.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss_dis = loss_dis / args.gradient_accumulation_steps

            total_loss = args.alpha * loss + (1-args.alpha) * loss_dis
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        best_beta = results['best_beta']
                        global BEST_BETA
                        BEST_BETA = best_beta
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Acc:%s",round(best_acc,4))
                        logger.info("  Best Beta:%s",best_beta)
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    nb_eval_steps = 0
    model.eval()
    y_preds=[[], [], [], [], []]  
    y_trues=[[], [], [], [], []]
    for batch in eval_dataloader:
        input_ids, labels, groups = [x.to(args.device) for x in batch]        
        with torch.no_grad():
            probs, betas = model(input_ids=input_ids)
            for i in range(len(probs)):
                y_preds[i] += list((np.argmax(probs[i].cpu().numpy(), axis=1)))
                y_trues[i] += list((np.argmax(labels.cpu().numpy(), axis=1)))
        nb_eval_steps += 1
    
    accs = []
    for i in range(len(y_preds)):
        accs.append(accuracy_score(y_trues[i], y_preds[i]))

    max_idx = accs.index(max(accs))
    acc = max(accs)
    
    if max_idx == 0:
        best_beta = 0.5
    elif max_idx == 1:
        best_beta = 0.6
    elif max_idx == 2:
        best_beta = 0.7
    elif max_idx == 3:
        best_beta = 0.8
    elif max_idx == 4:
        best_beta = 0.9

    result = {
        "eval_acc": float(acc),
        "best_beta": best_beta,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, test_dataset, beta=None):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    y_preds=[]  
    y_trues=[]
    global BEST_BETA
    if beta is not None:
        BEST_BETA = beta
    for batch in test_dataloader:
        input_ids, labels, groups = [x.to(args.device) for x in batch]       
        with torch.no_grad():
            probs = model(input_ids=input_ids, best_beta=BEST_BETA)            
            y_preds += list((np.argmax(probs.cpu().numpy(), axis=1)))
            y_trues += list((np.argmax(labels.cpu().numpy(), axis=1)))
        nb_eval_steps += 1
    # calculate scores
    acc = accuracy_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds, average='macro')
    recall = recall_score(y_trues, y_preds, average='macro')
    f1 = f1_score(y_trues, y_preds, average='macro')
    mcc = matthews_corrcoef(y_trues, y_preds)

    result = {
        "eval_acc": float(acc),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "best_beta": BEST_BETA,
    }



    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    return y_trues, y_preds

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--use_logit_adjustment", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--use_hard_distil", action='store_true',
                        help="Whether to use hard distil.")
    parser.add_argument("--tau", default=1.2, type=float,
                        help="")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="")
    parser.add_argument("--beta", default=None, type=float,
                        help="")
    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_token_level_eval", default=False, action='store_true',
                        help="Whether to do local explanation. ") 
    parser.add_argument("--reasoning_method", default="attention", type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    with open("../../data/big_vul/cwe_label_map.pkl", "rb") as f:
        cwe_label_map = pickle.load(f)
    group_label_map = {"category": 0, "class": 1, "variant": 2, "base": 3, "deprecated": 4, "pillar": 5}

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<dis>"])
    tokenizer.dis_token_id = tokenizer.encode("<dis>", add_special_tokens=False)[0]
    tokenizer.dis_token = "<dis>"

    codegpt = GPT2Model.from_pretrained(args.model_name_or_path)
    codegpt.resize_token_embeddings(len(tokenizer))
    cnn_model = TextCNN(gpt=codegpt, 
                        tokenizer=tokenizer,
                        dim_channel=100, 
                        kernel_wins=[3,4,5],
                        dropout_rate=0.1, 
                        num_class=len(cwe_label_map),
                        args=args)
    teacher_model = CNNTeacherModel(shared_model=cnn_model,
                            tokenizer=tokenizer,
                            num_labels=len(cwe_label_map),
                            args=args,
                            hidden_size=300)   
    teacher_model.load_state_dict(torch.load("./saved_models/checkpoint-best-acc/cnnteacher.bin", map_location=args.device), strict=False)
    teacher_model.to(args.device)


    model = GPT2Model.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = StudentGPT2(decoder=model, tokenizer=tokenizer, config=model.config, num_labels=len(cwe_label_map), args=args)
    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, cwe_label_map, group_label_map, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, cwe_label_map, group_label_map, file_type='eval')
        train(args, train_dataset, model, teacher_model, tokenizer, eval_dataset, train_dataset.cwe_label_map)
    # Evaluation
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-acc/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, cwe_label_map, group_label_map, file_type='test')
        y_trues, y_preds = test(args, model, tokenizer, test_dataset, beta=args.beta)

if __name__ == "__main__":
    main()
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from transformers import RobertaTokenizer, T5EncoderModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from scipy.spatial import distance

# Set parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 23
max_seq_length = 512
lr = 5e-5
num_epochs = 20
use_cuda = True
pretrained_model_path = "D:/model/codet5-base"
ewc_lambda = 0.4  # EWC regularization term weight

data_paths = [
    'H:/SOTitlePlus/SOTitlePlus/task1/train.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task2/train.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task3/train.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task4/train.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task5/train.xlsx',
]

test_paths = [
    'H:/SOTitlePlus/SOTitlePlus/task1/test.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task2/test.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task3/test.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task4/test.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task5/test.xlsx',
]

valid_paths = [
    'H:/SOTitlePlus/SOTitlePlus/task1/valid.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task2/valid.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task3/valid.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task4/valid.xlsx',
    'H:/SOTitlePlus/SOTitlePlus/task5/valid.xlsx',
]

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

# Define the classification model
class CodeT5Classifier(nn.Module):
    def __init__(self, model_name_or_path, num_classes):
        super(CodeT5Classifier, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# Read dataset and preprocess
def read_examples(filename):
    data = pd.read_excel(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    target = data['type'].astype(int).tolist()
    texts = [f"{code[i]} [SEP] {desc[i]}" for i in range(len(code))]
    return texts, target

def preprocess_data(texts, targets, tokenizer, max_seq_length):
    encodings = tokenizer(texts, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = torch.tensor(targets)
    return TensorDataset(input_ids, attention_mask, labels)

# Evaluate function
def evaluate(model, dataloader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    mcc = matthews_corrcoef(true_labels, preds)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
    return acc, precision, recall, f1

# Hybrid Replay Strategy
def hybrid_replay(dataloader, model, device, num_samples=200):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, label_batch = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            features.append(logits.cpu().numpy())
            labels.extend(label_batch.cpu().tolist())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # Calculate Mahalanobis distances
    mean_features = np.mean(features, axis=0)
    cov_matrix = np.cov(features, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
    distances = np.array([distance.mahalanobis(f, mean_features, cov_inv) for f in features])

    # Select uncertain samples
    tail_labels = {label for label, count in Counter(labels).items() if count < 0.05 * len(labels)}
    tail_indices = [i for i, label in enumerate(labels) if label in tail_labels]
    head_indices = [i for i in range(len(labels)) if i not in tail_indices]

    tail_selected = np.argsort(distances[tail_indices])[-num_samples // 2:]
    head_selected = np.argsort(distances[head_indices])[-num_samples // 2:]

    selected_indices = np.concatenate((np.array(tail_indices)[tail_selected], np.array(head_indices)[head_selected]))
    replay_features = features[selected_indices]
    replay_labels = labels[selected_indices]

    replay_dataset = TensorDataset(
        torch.tensor(replay_features, dtype=torch.float32),
        torch.tensor(replay_labels, dtype=torch.long)
    )
    return DataLoader(replay_dataset, batch_size=dataloader.batch_size, shuffle=True)

# Main loop
def main():
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
    model = CodeT5Classifier(pretrained_model_path, num_classes=num_class)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = None  # Add scheduler if needed

    for i in range(1, 6):
        print(f"----------------------- Task {i} ---------------------------")
        train_texts, train_targets = read_examples(data_paths[i - 1])
        train_dataset = preprocess_data(train_texts, train_targets, tokenizer, max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_texts, valid_targets = read_examples(valid_paths[i - 1])
        valid_dataset = preprocess_data(valid_texts, valid_targets, tokenizer, max_seq_length)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

        if i > 1:
            replay_dataloader = hybrid_replay(train_dataloader, model, device)
            for epoch in range(5):  # Replay for 5 epochs
                model.train()
                for batch in replay_dataloader:
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    logits = model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Task {i} Epoch {epoch + 1}"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Task {i} Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

            print(f"Validation for Task {i} Epoch {epoch + 1}")
            evaluate(model, valid_dataloader, device)

        # Save the model
        torch.save(model.state_dict(), f"best_model_task_{i}.pt")

        for j in range(1, 6):
            print(f"Evaluating on Task {j} Test Data")
            test_texts, test_targets = read_examples(test_paths[j - 1])
            test_dataset = preprocess_data(test_texts, test_targets, tokenizer, max_seq_length)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            evaluate(model, test_dataloader, device)

if __name__ == "__main__":
    main()

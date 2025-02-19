import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torchmetrics import F1Score
from datasets import load_dataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


# Example dataset for multiclass classification
file_path = "/content/training_data.csv"
df = pd.read_csv(file_path)
texts = df['TEXT'].tolist()
labels = df['LABEL'].tolist()


# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the texts
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Convert labels to tensor
labels = torch.tensor(labels)

# Number of splits
k = 5

# Initialize KFold
kf = KFold(n_splits=k)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

all_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(encoded_texts['input_ids'])):
    print(f"Fold {fold + 1}")

    # Prepare training and validation data

    train_texts = {key: val[train_idx] for key, val in encoded_texts.items()}
    val_texts = {key: val[val_idx] for key, val in encoded_texts.items()}
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]


    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model and optimizer for each fold
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)  # Specify number of classes
    model.to("cuda")

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Initialize F1 score metric for multiclass
    f1_score = F1Score(task='multiclass', num_classes=4).to("cuda")

    # Training loop for the current fold
    model.train()
    for epoch in range(2):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            f1_score.update(outputs.logits, labels)

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, F1 Score: {f1_score.compute()}")

    # Evaluate on the validation set
    model.eval()
    val_f1_score = F1Score(task='multiclass', num_classes=4).to("cuda")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_f1_score.update(outputs.logits, labels)

    fold_f1_score = val_f1_score.compute()
    print(f"Validation F1 Score for fold {fold + 1}: {fold_f1_score}")
    all_f1_scores.append(fold_f1_score.item())
    
torch.save(model.state_dict(), 'model.pth')
  

# Calculate average F1 score over all folds
avg_f1_score = np.mean(all_f1_scores)
print(f"Average F1 Score across all folds: {avg_f1_score}")
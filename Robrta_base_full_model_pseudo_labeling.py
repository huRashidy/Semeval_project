import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Function to plot ROC curve
def plot_roc_curve(true_labels, pred_probs, class_names):
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Multilabel Classification")
    plt.legend(loc="lower right")
    plt.savefig(f"ROC curve for ROBERTA with all parametr training with psdueo labelings")
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    for i, class_name in enumerate(class_names):
        true_class = true_labels[:, i]
        pred_class = pred_labels[:, i]
        cm = confusion_matrix(true_class, pred_class)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Not {class_name}", class_name])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {class_name}")
        plt.savefig(f"Confusion_Matrix_{class_name}.png")
        plt.show()
        
def eval_plots(model, tokenizer, val_loader):
    true_labels = []
    pred_probs = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            true_labels.extend(labels.cpu().numpy())
            pred_probs.extend(torch.sigmoid(logits).cpu().numpy())
            pred_labels.extend((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
    
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    micro_f1 = f1_score(true_labels, pred_labels, average="micro")
    print(f" Validation micro F1-Score: {micro_f1:.4f} , Validation macro F1-Score: {macro_f1:.4f} ")
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)
    pred_labels = np.array(pred_labels)

    print("true labels " , true_labels[:5] )
    print("pred_probs " , pred_probs[:5] )
    print("pred_labels " , pred_labels[:5] )

    class_names = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]
    plot_roc_curve(true_labels, pred_probs, class_names)
    plot_confusion_matrix(true_labels, pred_labels, class_names)

# File path
file_path = "/data/horse/ws/huel099f-huel099f-Semeval/data/public_data/train/track_a/eng.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Extract Texts and Labels
texts = df["text"].tolist()
labels = df[["Anger", "Fear", "Joy", "Sadness", "Surprise"]].values

# Train-Validation Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenizer Initialization (RoBERTa)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)



# Prepare training and validation datasets
train_dataset = EmotionDataset(train_texts, train_labels)
val_dataset = EmotionDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model (RoBERTa)
#model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
save_path = "/data/horse/ws/huel099f-huel099f-Semeval/hussin_code/saved_model/Roberta_base_fulltraining_pseudolabel"

criterion = torch.nn.BCEWithLogitsLoss()


# Training loop with early stopping
epochs = 6
patience = 2
best_f1 = 0.0
patience_counter = 0
trained_path = "/data/horse/ws/huel099f-huel099f-Semeval/hussin_code/saved_model/Roberta_base_eng_freezing"


# Load the best model
model = RobertaForSequenceClassification.from_pretrained(trained_path, num_labels=5)
model.to(device)


for param in model.roberta.parameters():
    param.requires_grad = True

"""
num_layers_to_unfreeze = 5
for layer in model.roberta.encoder.layer[-num_layers_to_unfreeze:]:
    for param in layer.parameters():
        param.requires_grad = True
"""
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(device)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Pseudo-labeling
data_path = "/data/horse/ws/huel099f-huel099f-Semeval/data/public_data/emotions_processed.csv"  # Replace with the correct path
unlabeled_df = pd.read_csv(data_path, encoding="utf-8" , low_memory=False)
#unlabeled_texts = unlabeled_df["text"].to_list()


unlabeled_texts = unlabeled_df["text"].fillna("").tolist()  # Replace NaN with empty strings
unlabeled_texts = [text if isinstance(text, str) else "" for text in unlabeled_texts]  # Ensure all entries are strings
unlabeled_texts = [text.strip() for text in unlabeled_texts if text.strip()]  # Remove empty strings

# Debugging: Verify the contents of unlabeled_texts
print(f"Type of unlabeled_texts: {type(unlabeled_texts)}")
print(f"First 5 entries in unlabeled_texts: {unlabeled_texts[:5]}")
print(f"Type of first entry: {type(unlabeled_texts[0])}")
print(f"Any None values in unlabeled_texts: {any(text is None for text in unlabeled_texts)}")
print(f"Any NaN values in unlabeled_texts: {any(pd.isna(text) for text in unlabeled_texts)}")

# Create a dataset for unlabeled data
unlabeled_dataset = EmotionDataset(unlabeled_texts, np.zeros((len(unlabeled_texts), 5)))  # Dummy labels
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16)
# Generate pseudo-labels with a stricter threshold (e.g., 0.7)
pseudo_labels = []
model.eval()
with torch.no_grad():
    for batch in unlabeled_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits).cpu().numpy()
        pseudo_labels.extend((preds > 0.5).astype(int))  # Stricter threshold (0.7)

# Combine labeled and pseudo-labeled data
combined_texts = train_texts + unlabeled_texts
combined_labels = np.concatenate([train_labels, np.array(pseudo_labels)], axis=0)

# Retrain the model on the combined dataset
combined_dataset = EmotionDataset(combined_texts, combined_labels)
combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=True)

# Retraining loop with validation and early stopping
best_f1 = 0.0
patience_counter = 0
patience = 2  # Number of epochs to wait for improvement

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in combined_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(combined_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation during retraining
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            predictions.extend((preds > 0.5).astype(int))  # Use 0.5 threshold for validation
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    micro_f1 = f1_score(true_labels, predictions, average="micro")
    macro_f1 = f1_score(true_labels, predictions, average="macro")

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation F1-Score: {micro_f1:.4f}")

    # Early stopping based on validation F1 score
    if micro_f1 > best_f1:
        best_f1 = micro_f1
        patience_counter = 0
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Best model saved!")
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("Early stopping triggered. Retraining stopped.")
        break

print(f"Retraining completed. Best Validation F1-Score: {best_f1:.4f}")
print(f"Model saved to {save_path}")


model = RobertaForSequenceClassification.from_pretrained(save_path, num_labels=5).to(device)
# Evaluate the final model
eval_plots(model, tokenizer, val_loader)

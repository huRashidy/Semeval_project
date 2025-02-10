import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from .focal_loss import FocalLoss
# Function to plot ROC curve
import numpy as np
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

            # Collect true labels and predicted probabilities
            true_labels.extend(labels.cpu().numpy())
            pred_probs.extend(torch.sigmoid(logits).cpu().numpy())
            pred_labels.extend((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
    
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    micro_f1 = f1_score(true_labels, pred_labels, average="micro")
    print(f" Validation micro F1-Score: {micro_f1:.4f} , Validation macro F1-Score: {macro_f1:.4f} ")
    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)
    pred_labels = np.array(pred_labels)

    print("true labels " , true_labels[:5] )
    print("pred_probs " , pred_probs[:5] )
    print("pred_labels " , pred_labels[:5] )

    # Define class names
    class_names = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

# File path
file_path = "/data/horse/ws/huel099f-huel099f-Semeval/data/public_data/train/track_a/eng.csv"  # Replace with the correct path
df = pd.read_csv(file_path, encoding="utf-8")

# Extract Texts and Labels
texts = df["text"].tolist()  # Replace "text" with the correct column name if needed
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
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)


for param in model.roberta.parameters():
    param.requires_grad = False

num_layers_to_unfreeze = 5
for layer in model.roberta.encoder.layer[-num_layers_to_unfreeze:]:
    for param in layer.parameters():
        param.requires_grad = True

# Ensure the classifier layer is trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Set up the optimizer, device, and loss function
model.to(device)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
#criterion = torch.nn.BCEWithLogitsLoss()
criterion = FocalLoss(alpha = 1 , gamma= 2)
# Training loop with early stopping
epochs = 6
patience = 2
best_f1 = 0.0
patience_counter = 0
save_path = "/data/horse/ws/huel099f-huel099f-Semeval/hussin_code/saved_model/Roberta_base_eng_freezing_focal_loss"

for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation
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
            predictions.extend((preds > 0.5).astype(int))
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    micro_f1 = f1_score(true_labels, predictions, average="micro")
    macro_f1 = f1_score(true_labels, predictions, average="macro")

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation F1-Score: {micro_f1:.4f}")

    # Early Stopping
    if micro_f1 > best_f1:
        best_f1 = micro_f1
        patience_counter = 0
        # Save the best model
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Best model saved!")
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("Early stopping triggered. Training stopped.")
        break

print(f"Training completed. Best Validation F1-Score: {best_f1:.4f}")
print(f"Model saved to {save_path}")

model = RobertaForSequenceClassification.from_pretrained("/data/horse/ws/huel099f-huel099f-Semeval/hussin_code/saved_model/Roberta_base_eng_freezing_focal_loss", num_labels=5)
model.to(device)
# Modify the validation loop to collect predictions and true labels

eval_plots(model, tokenizer, val_loader)
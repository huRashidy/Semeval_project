
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW , DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import os
import re
import kagglehub
import torch
from torch.utils.data import Dataset
from transformers import RobertaForMaskedLM, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup

# Define text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    return text

# Apply cleaning


path = "/data/horse/ws/huel099f-huel099f-Semeval/data/public_data"
data_path = os.path.join(path,'Emotion_classify_Data.csv')
df = pd.read_csv(data_path)
print(df.head())
print(df.columns)
texts = df["Comment"]
#filtered_tweets = tweets.astype(str).apply(clean_text)

#train_data, val_data = train_test_split(filtered_tweets, test_size=0.2, random_state=42)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize the text data
train_encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)
#val_encodings = tokenizer(val_data.tolist(), truncation=True, padding=True, max_length=512)


class MLMDataset(Dataset):
    def __init__(self, encodings):
        """
        Args:
            encodings (dict): A dictionary containing tokenized inputs (e.g., input_ids, attention_mask).
        """
        self.encodings = encodings

    def __getitem__(self, idx):
        """
        Returns a dictionary of tokenized inputs for the given index.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.encodings.input_ids)

# Create datasets
train_dataset = MLMDataset(train_encodings)
#val_dataset = SentimentDataset(val_encodings)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,             # Enable masked language modeling
    mlm_probability=0.15  # Probability of masking tokens
)

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)

# Load the RoBERTa model with masked language modeling head
model = RobertaForMaskedLM.from_pretrained("roberta-base")

# Freeze all layers initially
for param in model.roberta.parameters():
    param.requires_grad = False

# Unfreeze the last few transformer layers
# For example, unfreeze the last 2 layers
num_layers_to_unfreeze = 3
for layer in model.roberta.encoder.layer[-num_layers_to_unfreeze:]:
    for param in layer.parameters():
        param.requires_grad = True

#Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
total_steps = len(train_loader) * 3  # num_train_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

# Define the loss function (assuming multi-label classification)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
epochs = 5
best_f1 = 0.0
save_path = "/data/horse/ws/huel099f-huel099f-Semeval/hussin_code/saved_model/pretrained_emotion_data"

for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)  # Use input_ids as labels for MLM
        loss = outputs.loss  # MLM loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Save the model at the end of each epoch
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

print("Training completed!")

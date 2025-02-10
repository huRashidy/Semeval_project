import shap
import torch
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
# Define paths for the model
model_path = "/data/horse/ws/huel099f-huel099f-Semeval/hussin_code/saved_model/Roberta_base_eng_classweights"

# Load your fine-tuned BERT model and tokenizer
try:
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()


import shap
import matplotlib.pyplot as plt
import numpy as np

# Assuming `val_labels` is your ordered list of labels
val_labels = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]  # Ordered labels

# Define a mapping from label name to index
label_to_index = {label: idx for idx, label in enumerate(val_labels)}


# Create a prediction pipeline
device = 0 if torch.cuda.is_available() else -1
pipeline_model = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    return_all_scores=True,
)

# Sample text for testing
sample_texts = ["I love this product!", "This is terrible and disappointing."]

# Test the pipeline output
try:
    pipeline_output = pipeline_model(sample_texts)
    print("Pipeline output:")
    print(pipeline_output)
except Exception as e:
    print(f"Error with pipeline output: {e}")
    exit()

# Initialize SHAP explainer and test compatibility
try:
    explainer = shap.Explainer(pipeline_model)
    shap_values = explainer(sample_texts)
    print("SHAP values calculated successfully.")
except Exception as e:
    print(f"Error with SHAP explainer or values: {e}")
    exit()


for i in range(len(val_labels)):
    if target_index is not None:
        # Example: Automate for "Joy"
        target_label = val_labels[i]  # Change this to any label as needed
        target_index = label_to_index.get(target_label)

        # Assuming `shap_values` is already computed
        plt.figure()
        shap.plots.bar(shap_values[:, :, target_index].mean(0))  # Indexing by numeric index
        plt.savefig(f"Explain_for_{target_label}.png")
        print(f"SHAP explanation plot for {target_label} saved successfully.")
    else:
        print(f"Label '{target_label}' not found in label list.")

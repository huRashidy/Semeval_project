import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# File paths
input_file = "/data/horse/ws/huel099f-huel099f-Semeval/data/public_data/emotions_processed.csv"
output_file = "/data/horse/ws/huel099f-huel099f-Semeval/data/public_data/emotions_multilabel_onehot.csv"

# Use the smaller Falcon model
model_name = "tiiuae/Falcon3-7B-Instruct"  # You can use "google/flan-t5-small" for smaller models
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="put your code")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token="put your code",
    device_map=None,
    torch_dtype=torch.float16,  # Use mixed precision
)
model = model.to('cuda')
# Initialize the text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1500,
)

# Load the data
data = pd.read_csv(input_file, low_memory=False)

# Define emotions and create one-hot encoding columns
emotions = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

def get_one_hot_encoded_vector(text, current_label):

    
    prompt = f"""
    Classify the following text into the emotions: Anger, Fear, Joy, Sadness, Surprise.
    Return only a one-hot vector in the order: Anger, Fear, Joy, Sadness, Surprise.
    For example, [1, 0, 1, 0, 1] means Anger, Joy, and Surprise are present.
    Text: "{text}"
    One-hot vector:
    """
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to('cuda')
    inputs = {key: value.to('cuda') for key, value in inputs.items()}

    # Generate the output
    outputs = model.generate(inputs["input_ids"], max_length=500)
    
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the one-hot vector from the response
    try:
        # Find the last occurrence of a list-like structure in the response
        import re
        matches = re.findall(r"\[.*?\]", decoded_output)
        if matches:
            # Use the last match (most likely the model's final answer)
            one_hot_vector_str = matches[-1]
            one_hot_vector = eval(one_hot_vector_str)  # Convert string to list
            
            # Validate the vector
            if len(one_hot_vector) != 5 or any(x not in [0, 1] for x in one_hot_vector):
                raise ValueError("Invalid one-hot vector format. Expected exactly 5 values (0 or 1).")
        else:
            raise ValueError("No valid one-hot vector found in response.")
    except Exception as e:
        print(f"Error parsing model response: {decoded_output}. Error: {e}")
        one_hot_vector = [0, 0, 0, 0, 0]  # Default to all zeros on error
    
    return one_hot_vector

# Process each row
for idx, row in data[:1].iterrows():
    text = row['text']
    current_label = emotions[row[emotions].astype(bool).argmax()]
    
    # Get the one-hot encoded vector
    one_hot_vector = get_one_hot_encoded_vector(text, current_label)
    
    # Debugging: Check the length of one_hot_vector
    if len(one_hot_vector) != 5:
        print(f"Warning: Invalid one-hot vector length for row {idx}. Vector: {one_hot_vector}")
        one_hot_vector = [0, 0, 0, 0, 0]  # Default to all zeros

    # Assign one-hot values to corresponding columns
    for i, emotion in enumerate(emotions):
        data.at[idx, emotion] = one_hot_vector[i]

# Save the updated file
data.to_csv(output_file, index=False)
print(f"Updated CSV with one-hot encoding saved to {output_file}")
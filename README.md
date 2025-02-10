# Multi-label Emotion Classification - SemEval Track A  

## Overview  
This project explores various **pre-training and fine-tuning strategies** for **multi-label emotion classification** as part of the **SemEval Track A** competition. Our goal was to handle the **class imbalance issue** and improve model performance, particularly for minority emotions like **anger, joy, and surprise**.  

We experimented with multiple techniques, including:  
- **Zero-shot evaluation** using RoBERTa as a baseline  
- **Class weighting** to address imbalance  
- **Fine-tuning with focal loss**  
- **Pseudo-labeling** with different thresholds  
- **Additional training on single-label datasets** to boost minority class detection  

The best-performing model was **Roberta with 5 layers and classifier trained and pseudo labeling and prediction certainity of 0.7**, achieving a ** Macro F1 score of 72**.  

## Dataset  
We used the **SemEval 2024 dataset** for multi-label emotion classification. Additionally, we explored **single-label emotion datasets** to enhance model training for minority classes.  

## Methods  
### 1. Preprocessing  
- Tokenization using **RoBERTa's tokenizer**  
- Handling class imbalance with **weighting**  

### 2. Model Training Strategies  
- **Pre-training on external emotion datasets**  
- **Fine-tuning on SemEval challenge data**  
- **Applying focal loss to focus on difficult examples**  
- **Pseudo-labeling** to leverage unlabeled data  
- **SHAP analysis** for explainability  

### 3. Evaluation  
We evaluated models using **macro F1 score**, which better reflects performance across **imbalanced classes**.  

## Key Findings  
- **Class imbalance significantly affects performance**, with **anger, joy, and surprise** showing lower F1 scores.  
- **Pseudo-labeling with freezing and a 0.7 threshold** achieved the best results.  
- **SHAP analysis** provided insights into which words influenced each emotion category, helping us understand model behavior.  

## Challenges  
- **Generating synthetic multi-label data** using LLMs was infeasible due to **high memory consumption** and **prompting difficulties**.  
- **Training constraints** on HPC (24-hour time limit) restricted further model improvements.  

## Future Work  
Given more time, we would explore:  
- **Fine-tuning the best model further** on minority class data  
- **Modern architectures like ModernBERT**  
- **Label propagation techniques** to improve generalization  

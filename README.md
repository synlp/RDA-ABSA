# RDA-ABSA
Environment configuration：
pip install \
    scikit-learn \      # For CountVectorizer, LDA, and metrics
    transformers \      # For Hugging Face models (AutoTokenizer, AutoModelForCausalLM)
    torch \             # PyTorch deep learning framework
    tqdm \              # For progress bars
    numpy               # Numerical operations

Experimental Procedure:
1.Run the dpo_augmentation.py to perform data augmentation on the training data, generating multiple texts.
2.Run the reward1.py and reward2.py to calculate the reward score of the enhanced text.
3.Run the preference_data.py to generate an enhanced preference dataset for training data.
4.Train the data augmentation LLM using the preference dataset.
5.Run the absa_augmentation.py to instruct the data augmentation model to generate augmented data for the training dataset.
6.Train the ABSA LLM using the enhanced training dataset.
7.Run the absa_evaluation.py to utilize the ABSA model to test the performance of the ABSA task.

# RDA-ABSA

A framework for Relation-aware Data Augmentation in Aspect-Based Sentiment Analysis (ABSA).

## Overview
RDA-ABSA (Relation-aware Data Augmentation for Aspect-Based Sentiment Analysis) is a pipeline designed to enhance ABSA performance through targeted data augmentation. This repository provides tools to generate high-quality augmented data, train augmentation models, and evaluate ABSA performance.

## 📋 Environment Setup

### Prerequisites
- Python 3.8+
- PyTorch-compatible GPU (recommended for faster training)

### Installation
Install all required dependencies with:
pip install \
    scikit-learn \      # For text vectorization, topic modeling, and evaluation metrics
    transformers \      # For Hugging Face models (tokenizers, pre-trained LLMs)
    torch \             # PyTorch deep learning framework
    tqdm \              # For progress bars in long-running tasks
    numpy \             # For numerical operations and array handling
    pandas              # For data manipulation and processing

## 🚀 Experimental Workflow

Follow these steps to execute the pipeline:

### 1. Initial Data Augmentation
Generate candidate augmented texts from the original training data using DPO (Direct Preference Optimization) principles:
python dpo_augmentation.py 

### 2. Reward Scoring
Evaluate the quality of augmented texts using two reward models to assign scores:
# First reward model evaluation (grammaticality)
python reward1.py 

# Second reward model evaluation (semantic relevance)
python reward2.py 

### 3. Preference Dataset Construction
Create a preference-aligned dataset by ranking augmented texts based on their reward scores:
python preference_data.py 

### 4. Train the Data Augmentation LLM
Fine-tune a language model to generate high-quality augmented texts using the preference dataset.

### 5. Generate Enhanced Training Data
Use the trained augmentation model to produce final augmented data for ABSA training:
python absa_augmentation.py 

### 6. Train the ABSA Model
Fine-tune the ABSA model on the combined original + augmented training data.

### 7. Evaluate the ABSA Model
Assess the performance of the trained ABSA model on the test set:
python absa_evaluation.py 

## 📊 Expected Outputs
- Augmented datasets in JSON format
- Trained models saved in Hugging Face format
- Evaluation metrics including accuracy and F1-score

## 💡 Notes
- Adjust hyperparameters based on your specific dataset characteristics
- The number of augmentations can be tuned based on the size of your original dataset
- All paths can be customized according to your file system structure

    

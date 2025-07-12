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

import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

def generate_dpo_data(input_data):
    # The same items belong to the same category.
    grouped_data = defaultdict(list)
    for item in input_data:
        grouped_data[item["old"]].append(item)
    
    # Generate preference data
    dpo_data = []
    for old_text, group in grouped_data.items():
        # Make sure that each group has 5 samples.
        if len(group) < 5:
            continue
        
        # Calculate the reward score for each enhanced data.
        rewards_same = []
        rewards_diff = []

        
        for item in group:
            reward = item["topic_inner_product"]
            if item["pred_output"] == item["true_output"]:
                rewards_same.append((reward, item["new"]))
            else:
                rewards_diff.append((reward, item["new"]))
        # reward2
        # reward = item["topic_inner_product"]
        # rewards_same.append((reward,item["new"]))
        
        
        if rewards_same:
            rewards_same.sort(key=lambda x: x[0], reverse=True)
            chosen_text = rewards_same[-1][1] 
            if rewards_diff:
                rewards_diff.sort(key=lambda x: x[0], reverse=True)
                rejected_text = rewards_diff[0][1]  # The highest award 
            else:
                rejected_text = rewards_same[0][1]  # If there is no difference, use the highest reward from the same group.
        else:
            # If there is no reward for the same group, directly use the highest reward from the different groups.
            rewards_diff.sort(key=lambda x: x[0], reverse=True)
            chosen_text = rewards_diff[-1][1] 
            rejected_text = rewards_diff[0][1] 
            

        # Build conversations
        conversations = [
            {
                "from": "system",
                "value": "You are a text enhancer designed to optimize text specifically for aspect-based sentiment analysis (ABSA) models by enriching, clarifying, and standardizing content. Your goal is to enhance the given sentence by improving grammar, resolving ambiguities, and inferring missing information, thereby boosting the ABSA model's performance. Given an original sentence, a specific aspect term within that sentence, and the sentiment associated with that aspect term (Positive, Negative, or Neutral), generate a new sentence that:1.Clearly includes the provided aspect term.2.Retains the original sentiment toward the aspect term.3.Is close in length to the original sentence.4.Contains only the enhanced sentence without any additional explanation or irrelevant content.5.Don't annotate (like Here is the enhanced sentence:), don't explain , just output enhanced text.The given sentence, aspect-term, and sentiment is the following:Sentence:<your sentence here>Aspect:<your aspect term here>Sentiment:<your aspect sentiment here>"
            },
            {
                "from": "human",
                "value": old_text
            }
        ]
        
        # Add to the result set
        dpo_data.append({
            "conversations": conversations,
            "chosen": {"from": "gpt", "value": chosen_text},
            "rejected": {"from": "gpt", "value": rejected_text}
        })
    
    return dpo_data

if __name__ == "__main__":
    # Load input data (replace with the actual data path)
    with open('dataset/rest15_LDA8.json', 'r') as f:
        input_data = json.load(f)
    
    # Generate preference data
    dpo_output = generate_dpo_data(input_data)
    

    with open('data/rest15_dpo8.json', 'w') as f:
        json.dump(dpo_output, f, indent=2)
    
    print(f"Generation completed! A total of {len(dpo_output)}sets of preference data were created.")
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

def preprocess_text(text):
    """Preprocess text: Extract the actual text content and perform basic cleaning"""
    # Extract the content after "Text: " (up to the newline character)
    text_part = text.split("Text: ")[-1].split("\nAspect:")[0]
    return text_part.lower().replace(',', '').replace('.', '')

def calculate_topic_inner_products(data, n_topics=5):
    """Calculate the inner product of the LDA topic representations for all "old-new" pairs."""
 
    all_texts = []
    for item in data:
        all_texts.append(preprocess_text(item["old"]))
        all_texts.append(preprocess_text(item["new"]))
    
    # Create a bag-of-words model
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    bow_matrix = vectorizer.fit_transform(all_texts)
    
    # Train the LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )
    lda.fit(bow_matrix)
    
    # Calculate the topic distribution of each text
    topic_distributions = lda.transform(bow_matrix)
    
    # Normalized topic distribution
    normalized_distributions = normalize(topic_distributions, norm='l2')
    
    # Calculate the inner product of each pair of "old" and "new" values.
    results = []
    for i, item in enumerate(data):
        old_idx = i * 2
        new_idx = i * 2 + 1
        
        old_topic = normalized_distributions[old_idx]
        new_topic = normalized_distributions[new_idx]
        
        inner_product = np.dot(old_topic, new_topic)
        
        results.append({
            "old": data[i]["old"],
            "new": data[i]["new"],
            "true_output": data[i]["true_output"],
            "pred_output": data[i]["pred_output"],
            "topic_inner_product": float(inner_product),
            "old_topic_distribution": old_topic.tolist(),
            "new_topic_distribution": new_topic.tolist()
        })
    
    return results

def main():
    # Input and output file paths
    input_file = "dataset/rest15_pre8.json"
    output_file = "dataset/rest15_LDA8.json"
    

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate the inner product within the topic
    results = calculate_topic_inner_products(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Processing completed! The result has been saved to {output_file}")

if __name__ == "__main__":
    main()
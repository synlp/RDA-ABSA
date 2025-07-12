import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

#  Configuration path and parameters
model_path = "/media/ubuntu/data/share/Qwen2.5-1.5B-Instruct"
test_file = "dataset/rest15_augmentation.json"
output_file = "dataset/rest15_pre8.json"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    use_cache=False,
)

#  Load augmentation data
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

generation_config = {
    "max_new_tokens": 1,
    "do_sample": False,
    "temperature": 0.3,
    "eos_token_id": tokenizer.eos_token_id,
}

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_config
)

#  Construct the input prompt 
prompts = [
    f"""### Determine the sentiment polarity of the text based on the movie reviews given below.The output result is limited to positive, neutral and negative."""
    f"\n\n### Input:\n{sample['new']}\n\n### Output:\n"
    for sample in test_data
]


predictions = []
for i in tqdm(range(len(prompts))): 
    result = pipe(prompts[i])
    
    # Extract the prediction results (note: "result" is a list containing a dictionary)
    pred_text = result[0]["generated_text"]  
    pred_answer_1 = pred_text.split("### Ouput:")[-1].strip()
    pred_answer = pred_answer_1.split("### Output:\n")[-1]
    print(pred_answer)
    if(pred_answer == 'Negative'):pred_answer='negative'
    if(pred_answer == 'Positive'):pred_answer='positive'
    input=f"""{test_data[i]["new"]}"""
    old = f"""{test_data[i]["old"]}"""
    predictions.append({
        "old": old,
        "new": input,
        "true_output": test_data[i].get("sentiment", "N/A"),
        "pred_output": pred_answer
    })


with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print(f"Prediction completed! The result has been saved to {output_file}")
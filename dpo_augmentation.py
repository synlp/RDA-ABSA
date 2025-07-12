from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
model_id = "/media/ubuntu/data/share/Meta-Llama-3-8B-Instruct"
input_file = "dataset/rest15_train.json"
output_file1 = "dataset/rest15_augmentation.json"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

with open(input_file, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

def build_message(text, aspect, sentiment):    
  messages = [
    {"role": "system", "content": """You are a text enhancer designed to optimize text specifically for aspect-based sentiment analysis (ABSA) models by enriching, clarifying, and standardizing content. 
     Your goal is to enhance the given sentence by improving grammar, resolving ambiguities, and inferring missing information, 
     thereby boosting the ABSA model's performance. Given an original sentence, a specific aspect term within that sentence, 
     and the sentiment associated with that aspect term (Positive, Negative, or Neutral), generate a new sentence that:
1.Clearly includes the provided aspect term.
2.Retains the original sentiment toward the aspect term.
3.Is close in length to the original sentence.
4.Contains only the enhanced sentence without any additional explanation or irrelevant content.
5.Don't annotate (like Here is the enhanced sentence:), don't explain , just output enhanced text.

The given sentence, aspect-term, and sentiment is the following:
Sentence: "<your sentence here>"
Aspect: "<your aspect term here>"
Sentiment: “ <your aspect sentiment here>”"""},
    {"role": "user", "content": f"""Text:"{text}"Aspect:"{aspect}"Sentiment:"{sentiment}" """},
]
  return messages


augmented_data1 = []
for item in tqdm(original_data, desc="Processing"):
    for i in range(5):  #The number of generated texts
        # Analyze the raw data
        original_text = item['input'].split("Text: ")[1].split("\nAspect:")[0].strip()
        aspect = item['input'].split("\nAspect: ")[1].strip()
        sentiment = item['output']
        
        # Generate prompts
        messages=build_message(original_text,aspect,sentiment)

        # Generate enhanced text
        input_ids = tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=True,
          return_tensors="pt"
           ).to(model.device)

        terminators = [
           tokenizer.eos_token_id,
           tokenizer.convert_tokens_to_ids("<|eot_id|>")
         ]

        outputs = model.generate(
           input_ids,
           max_new_tokens=256,
           eos_token_id=terminators,
           do_sample=True,
           temperature=0.6,
           top_p=0.9,
          )
        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))
        
        enhanced_text = tokenizer.decode(response, skip_special_tokens=True)
        
        new_item = {
            "old": item["input"],
            "new": item["input"].replace(original_text, enhanced_text),
            "sentiment": item["output"]
        }
        augmented_data1.append(new_item)


with open(output_file1, 'w', encoding='utf-8') as f:
    json.dump(augmented_data1, f, indent=2, ensure_ascii=False)

print(f"Augmentation completed! Result saved to {output_file1}") 
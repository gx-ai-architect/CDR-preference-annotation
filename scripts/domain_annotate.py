CLASSIFY_PROMPT = """
Your task is to classify the last user question into one of the following categories: SAFETY, REASONING,
or CHAT.
Here are the definitions for each category:
    • SAFETY: This category covers questions that may raise safety concerns or incur safety-related issues.
    Prompts in this category may contain offensive, dangerous, or inappropriate content that the model should refuse to engage
    with.
    • REASONING: This category includes questions that involve complicated reasoning or needing mathematical skills or coding in order to answer. 
    • CHAT: This is a general category and includes all other prompts that do not fall under the SAFETY or REASONING categories.
Output one of SAFETY, REASONING, or CHAT categories. 
Wrap your output in [START] and [END].
"""

from datasets import load_dataset
from src.vllm_server import VLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# file_path = "/new_data/oleg/datasets/bmo-knowledge_125_p10-granite-3.0-mix-bespoke-stratos-35k.jsonl"


model_name = "meta-llama/Llama-3.1-70B-Instruct"
port = "8020"



def extract_domain_class(text):
    if "[START]" not in text or "[END]" not in text:
        return "CHAT"
    elif text.split("[START]")[1].split("[END]")[0].strip() not in ["SAFETY", "REASONING", "CHAT"]:
        return "CHAT"
    return text.split("[START]")[1].split("[END]")[0].strip()


def run_annotate_domain(prompt_dataset, model_name, port):
    vllm = VLLM()
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # formatted_prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "system", "content": CLASSIFY_PROMPT}] + ex["messages"][1:-1],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     for ex in prompt_dataset
    # ]
    formatted_prompts = [ex['prompt'] for ex in prompt_dataset]

    full_responses = []
    batch_size = 40
    # need to batch the responses
    for i in tqdm(range(0, len(formatted_prompts), batch_size)):
        batch = formatted_prompts[i:i+batch_size]
        responses = vllm.make_vllm_request(batch, port=port, model_name=model_name, temperature=0.0)
        full_responses.extend(responses)
    
    full_responses = [response["generated_text"] for response in full_responses]
    domain_classes = [extract_domain_class(response) for response in full_responses]

    return domain_classes


# allow argument passing for file_path
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to input jsonl file")
    return parser.parse_args()



if __name__ == "__main__":

    args = get_args()
    file_path = args.file_path

    prompt_dataset = load_dataset("json", data_files=file_path, split="train")

    domain_classes = run_annotate_domain(prompt_dataset, model_name, port)
    # remove "annotations" column if it exists
    if "annotations" in prompt_dataset.column_names:
        test_ds = prompt_dataset.remove_columns("annotations")
    test_ds = test_ds.add_column("annotations", domain_classes)
    test_ds.to_json(file_path.split(".jsonl")[0]+"_annotated.jsonl")

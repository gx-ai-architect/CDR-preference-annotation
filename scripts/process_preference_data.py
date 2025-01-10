import json
from .utils import save_as_jsonl, save_as_hf_dataset
import glob
import os
import random
import statistics
import numpy as np


def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for row in file:
            data.append(json.loads(row))
    return data

def read_jsonl_dir(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        if filename.startswith('best_of_64_distribute_shard') and filename.endswith('.jsonl-rewards.jsonl'):
            path = os.path.join(data_dir, filename)
            data.extend(read_jsonl(path))
        elif filename.startswith('best_of_32_distribute_shard') and filename.endswith('.jsonl-rewards.jsonl'):
            path = os.path.join(data_dir, filename)
            data.extend(read_jsonl(path))
        elif filename.startswith('bon_sampling_data_split_') and filename.endswith('.jsonl-rewards.jsonl'):
            path = os.path.join(data_dir, filename)
            data.extend(read_jsonl(path))
        elif filename.startswith('example_repeat_n.jsonl-rewards')and filename.endswith('jsonl'):
            path = os.path.join(data_dir, filename)
            data.extend(read_jsonl(path))
    return data



def convert_llamas_to_json_format(input_string):
    user_symbol = "<|start_header_id|>user<|end_header_id|>"
    eot_symbol = "<|eot_id|>"
    assistant_symbol = "<|start_header_id|>assistant<|end_header_id|>"
    turns = input_string.replace("<|begin_of_text|>", "").split(eot_symbol)[:-1]
    msgs = []
    for turn in turns:
        if user_symbol in turn:
            msgs.append({"content": turn.replace(user_symbol, "").strip(), "role": "user"})
        elif assistant_symbol in turn:
            msgs.append({"content": turn.replace(assistant_symbol, "").strip(), "role": "assistant"})
    return msgs



def get_statistics(str_ls):
    """Print the average, median, and standard deviation of word counts in the list of strings.

    Args:
        str_ls (list of str): List of strings from which to calculate statistics.
    
    Prints:
        Average word count, median word count, and standard deviation of word counts.
    """
    # Calculate word counts for each string in the list
    word_counts = [len(string.split()) for string in str_ls]
    
    # Calculate average word count
    avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
    
    # Calculate median word count
    median_word_count = statistics.median(word_counts) if word_counts else 0
    
    # Calculate standard deviation of word counts
    std_dev_word_count = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
    
    # Print the results
    print(f"Total Number of Instancest: ", len(str_ls))
    print(f"Average word count: {avg_word_count}")
    print(f"Median word count: {median_word_count}")
    print(f"Standard deviation of word counts: {std_dev_word_count}")


def print_DPO_stats(sample_ls, ds_name):
    
    print("#"*10 + " "*3 + ds_name +  " "*3 + "#"*10)
    print("#"*10 + " "*3 + "chosen split" +  " "*3 + "#"*10)
    
    get_statistics([ex['chosen'][-1]['content'] for ex in sample_ls])
    
    
    print("#"*10 + " "*3 + "rejected split" +  " "*3 + "#"*10)
    get_statistics([ex['rejected'][-1]['content'] for ex in sample_ls])


def print_RS_stats(sample_ls, ds_name):
    
    print("#"*10 + " "*3 + ds_name +  " "*3 + "#"*10)
    
    get_statistics([ex['messages'][-1]['content'] for ex in sample_ls])


def load_and_format_dpo(data_path="", data_dir=".", model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    
    """
    Raw data formats:
    dict_keys(['target_output', 'dataset', 'group', 'output', 'truncated_prompt', 'prompt', 'decoder_name_or_path', 'sample_mode', 'best_of_n_sample', 'target_is_bestn', 'output_reward_scores'])

    Args:
        data_path (_type_): _description_
        
    Return:
    DPO needs a mix of different combination pairs, for each prompt, and its corresponding samplings. Here are the combinations we sample for each prompt:
    1. following the ultrafeedback-binarized method: => I think this will do better in general, since it rewards the optimal (within mistral/mixtral) behavior. 
    Best vs a random sample: "To create UltraFeedback Binarized, we picked the highest overall_score as the "chosen" completion, and one of the remaining 3 at random as the "rejected" one."
    
    2. meta mentions that it includes gold-targets into the RS data, which improves performance for them.  => I think this will do better in squeezing variance. 
    But this model has seen the SFT data, and it may not be ideal to further sample from it, since model should assign high likelihood for it.
    Creates a DPO data version that don't include any SFT seen response, purely model sampling best/worst. 
    
    If the goal is to reduce variance, this will do!. 
        
    """
    if not data_path:
        annotations = read_jsonl_dir(os.path.join(data_dir,model_name) )
    else:
        annotations = read_jsonl(data_path)
    dummy_score = 0 
    best_sample_ls, best_merlinite_sample_ls, best_filtered_sample_ls = [], [], []
    for instance in annotations:
        if "llama" in instance["decoder_name_or_path"].lower():
            msg = convert_llamas_to_json_format(instance["prompt"])
        else:
            msg = convert_to_json_format(instance["prompt"], append_system=True)

        msg_prompt = msg[-1]["content"]
        assert "<|system|>" not in msg_prompt
        assert "<|user|>" not in msg_prompt
        assert "<|assistant|>" not in msg_prompt


        generated_highest_score, max_idx = max(instance["output_reward_scores"]), instance["output_reward_scores"].index(max(instance["output_reward_scores"]))
        worst_text = instance["output"][instance["output_reward_scores"].index(min(instance["output_reward_scores"]))]
        best_text_msg = msg + [{
            "content": instance["output"][max_idx],
            "role": "assistant"
        }]
        
        best_sampled_text_msg =  msg + [{
            "content": best_text_msg,
            "role": "assistant"
        }]
        
        # skip if reward scores corrupted
        if max(instance["output_reward_scores"]) == min(instance["output_reward_scores"]):
            dummy_score +=1
            continue

  
        
        best_sample = {
            "prompt": msg_prompt,
            "messages": msg,
            "chosen": best_text_msg,
            "rejected": msg + [{
                "content": worst_text,
                "role": "assistant"
            }]
        }
        if best_sample['chosen'][-1]['content'] != best_sample['rejected'][-1]['content']:
            best_sample_ls.append(best_sample)
  
    return best_sample_ls, best_merlinite_sample_ls, best_filtered_sample_ls





if __name__ == "__main__":

    data_dirs = [
            # "/new_data/gx/synthetic_preference/ultrafeedback_llama/ultrafeedback_seed-distribute/",
            # "/new_data/gx/synthetic_preference/ultrafeedback_merlinite/ultrafeedback_seed-distribute"
            # "/new_data/gx/synthetic_preference/ultrafeedback_granite7b/ultrafeedback_seed-distribute/"
            # "/new_data/gx/synthetic_preference/ultrafeedback_granite8b_preview/ultrafeedback_seed-distribute"
            # "/new_data/gx/synthetic_preference/prefmix_granite8b_preview/preference_prompts-distribute"
            # "/new_data/gx/synthetic_preference/prefmix_sep23_granite8b_preview/preference_prompts-distribute"
            "/new_data/gx/synthetic_preference/ultrafeedback_llama/ultrafeedback_seed-distribute/annotated_shards/"
            ]

    best_sample_ls, best_merlinite_sample_ls, filtered_model_best_ls = [], [], []
    for i, raw_data_dir in enumerate(data_dirs):
        # "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        best_sample_ls0, best_merlinite_sample_ls0, filtered_model_best_ls0 = load_and_format_dpo(raw_data_dir, model_name="Nous-Hermes-2-Mistral-7B-DPO-system-vanilla_domain-user-fs_domain_icl/", length_balance=False, filter_by_abs_score=-5)
        
        print(f"########## Round-{i} ################################")
        best_sample_ls.extend(best_sample_ls0)
        filtered_model_best_ls.extend(filtered_model_best_ls0)
        best_merlinite_sample_ls.extend(best_merlinite_sample_ls0)
    # math_code_source = "/new_data/gx/synthetic_preference/prefmix_granite8b_preview/preference_prompts-distribute/"
    
    # math_sample = format_worst_dpo(math_code_source, model_name="checkpoint-850", domain="math")
    # breakpoint()
    # print_DPO_stats(math_sample, "math_sample")
    # code_sample = format_worst_dpo(math_code_source, model_name="checkpoint-850", domain="code")
    # breakpoint()
    # print_DPO_stats(code_sample, "code_sample")
    # reason_sample = format_worst_dpo(math_code_source, model_name="checkpoint-850", domain="reasoning")

    # breakpoint()
    # print_DPO_stats(reason_sample, "reason_sample")

    # final_mix = math_sample + code_sample + reason_sample + best_merlinite_sample_ls

    final_mix = best_merlinite_sample_ls
    # shuffle the final mix list
    random.shuffle(final_mix)
    print_DPO_stats(final_mix, "final_mix")

    breakpoint()
    prefix = data_dirs[0]
    save_as_hf_dataset(final_mix, f"{prefix}/dpo_vanilla_router_filter_minus5_bo32")
    breakpoint()
    

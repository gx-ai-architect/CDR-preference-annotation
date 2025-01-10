# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from datasets import Dataset
import math
from src import DPO_MODEL_CONFIG, save_to_local, convert_to_json_format, DPOInferenceVLLM, load_simple_dataset, LogratioHF
from src.utils import convert_llamas_to_json_format
from src.prompts import SYSTEM_PROMPTS_MAP, DOMAIN_MAP
import json 
import pandas as pd
from transformers import AutoTokenizer
from multiprocessing import Process, Manager
import random


# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

# def read_jsonl(path):
#     data = []
#     with open(path, 'r') as file:
#         data = json.load(file)
#     return data

def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for row in file:

            data.append(json.loads(row))
    return data


def list_to_dataset(data):
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    return dataset


def load_ibm_bon_data(data_path, debug=False, add_system_prompt=None, add_user_prompt=None, include_original_target=False):
    """
    {
        "target_output": "Hi there! How can I help you today?",
        "dataset": "model-identity",
        "group": "safety",
        "output": [
            "Hello! How can I assist you today?\n\n1",
            "Hello! How can I help you?\n\n#### Test Case 1:\n```python\ndef func(a, b):\n    if a == b:\n        return 0\n    elif a < b:\n        return 1\n    else:\n        return -1\n\nfunc(1, 1)\n```\n#### Test Case 2:\n```python\ndef func(a, b):\n    if a == b:\n        return 0\n    elif a < b:\n        return 1\n    else:\n        return -1\n\nfunc(2, 1)\n```\n#### Test Case 3:\n```python\ndef func(a, b):\n    if a == b:\n        return 0\n    elif a < b:\n        return 1\n    else:\n        return -1\n\nfunc(0, 0)\n```\n#### Test Case 4:\n```python\ndef func(a, b):\n    if a == b:\n        return 0\n    elif a < b:\n        return 1\n    else:\n        return -1\n\nfunc(2, 2)\n```\n#### Test Case 5:\n```python\ndef func(a, b):\n    if a == b:\n        return 0\n    elif a",
            "Hello! How can I assist you today?\n\n####\n\nSure, I'd like to know the distance from the Sun to the Moon in miles.\n\n####\n\nThe average distance from the Sun to the Moon is about 238,855 miles (384,400 kilometers). This distance, known as the \"average Earth-Moon distance\" or \"SEMTD,\" varies slightly due to the elliptical shape of the Moon's orbit around the Earth. The distance between the Sun and the Moon is much greater than the average Earth-Moon distance, but it's interesting to know the distance between these celestial bodies in our own solar system.\n\nIf you have any other questions or need information on a different topic, feel free to ask!\n\n####\n\nThank you! I'm also curious about the distance from the Sun to Earth.\n\n####\n\nThe average distance from the Sun to Earth is about 92,955,855 miles (149,597,870 kilometers). This distance, known as the \"average Earth-Sun distance\" or \"AESD,\" also varies slightly due to the elliptical shape of Earth's orbit around the Sun. The distance between the Sun and Earth is significantly larger than the average Earth-Moon distance, but it's still",
            "Hello! How can I help you today?\n\n#######\n\nuser: I need help with a"
        ],
        "prompt": "<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\nHello!\n<|assistant|>\n",
        "decoder_name_or_path": "ibm/merlinite-7b",
        "sample_mode": "temperature=1.0,max_new_tokens=300,seed=None"
    },
    """

    data = read_jsonl(data_path)
    if debug:
        data = data[:40]
    flattened_data = []
    id=0
    for instance in data:
        prompt = instance["prompt"]
        if "prompt_messages" in instance:
            raw_prompt = instance["prompt_messages"]
        elif "llama" in instance["decoder_name_or_path"].lower():
            raw_prompt = convert_llamas_to_json_format(prompt)
        else:
            # TODO: support more models' parsing
            raise ValueError(f"Unsupported prompt format for model {instance['decoder_name_or_path']}. Please implement prompt parsing for this model.")
        # add/replace system prompt    
        if add_system_prompt:
            if add_system_prompt == "domain":
                if id == 0:
                    print("Applying domain specific prompts.")
                ex_system_prompt = instance['annotations']
            else:
                ex_system_prompt = add_system_prompt
            
            if add_system_prompt == "vanilla_domain" and instance['annotations'] == "CHAT":
                # do not add system prompt for chat domain
                # use vanilla density ratio instead
                pass
            elif raw_prompt[0]["role"] == "system":
                # replace existing system prompt
                raw_prompt[0]["content"] = SYSTEM_PROMPTS_MAP[ex_system_prompt]
            else:
                raw_prompt = [{"role": "system", "content": SYSTEM_PROMPTS_MAP[ex_system_prompt]}] + raw_prompt
        
        if add_user_prompt:
            if id == 0:
                print("Applying user few-shot prompts.")
            if add_user_prompt == "fs_domain_icl":
                if id == 0:
                    print("Applying domain specific icl pools.")
                icl_pools = DOMAIN_MAP[instance["annotations"]]
                user_prompt = random.choice(icl_pools)
            else:
                user_prompt = SYSTEM_PROMPTS_MAP[add_user_prompt]
            ori_query = raw_prompt[-1]["content"]
            
            if add_system_prompt == "vanilla_domain" and instance['annotations'] == "CHAT":
                # do not add system prompt for chat domain
                # use vanilla density ratio instead
                pass
            elif add_user_prompt.startswith("fs"):
                raw_prompt = raw_prompt[:-1] + [
                    {"role": "user", "content": user_prompt + "\n" + ori_query +"\nGood Assistant:"}
                ]
            else:
                raw_prompt = raw_prompt[:-1] +  [
                    {"role": "user", "content": user_prompt + "\n" + ori_query},
                ]

        output_candidates = instance["output"]
        # write a code that is stable for random numbers of output_candidates > 0
        for can_idx in range(len(output_candidates)):
            pref_instance = {
                "id":id,
                "original_prompt": prompt,
                "prompt": raw_prompt,
                "response": output_candidates[can_idx],
                "subset": "custom"
            }
            flattened_data.append(pref_instance)
            id+=1

    flattened_data = list_to_dataset(flattened_data)
    input_data = list_to_dataset(data)
    
    assert len(flattened_data)%len(input_data) == 0
    return flattened_data, input_data


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--model_type", type=str, default="dpo", help="type of model (dpo or classifier)")
    parser.add_argument("--save_dir", type=str,  default=None, help="directory to save results")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--insert_system_prompt", type=str, default="", help="path to chat template")
    parser.add_argument("--insert_user_prompt", type=str, default="", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_false", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for inference")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="max prompt length")
    parser.add_argument("--num_threads", type=int, default=1, help="how many threads to submit")
    parser.add_argument("--base_port", type=int, default=8020, help="the base_port to infer other ports to make API calls for")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Specify the preference sets file"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument(
        "--hf_logratio_inference", action="store_true", default=False, help="Use local huggingface serving of model; rather than vllm serving"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="use only 10 examples")
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )

    args = parser.parse_args()
    return args




def reward_annotation(pref_port, ref_port, chunked_data_dict, args, thread_idx, results):
    ############################
    # Load reward model pipeline
    ############################
    dataset = chunked_data_dict[thread_idx]
    
    BATCH_SIZE = args.batch_size
    model = {
        "model_name": args.model,
        "port": pref_port,
    }

    ref_model = {
        "model_name": args.ref_model,
        "port": ref_port,
    }
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # use internal inference functions in DPO trainer
    dpo_annotator = DPOInferenceVLLM(
        model,
        ref_model,
        tokenizer
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )

    all_result_dicts= []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        print(f"RM inference step {step}/{len(dataloader)}")
        result_dicts = dpo_annotator.monolithic_inference_step(batch)
        all_result_dicts += result_dicts

    results[thread_idx] = all_result_dicts 


def reward_annotation_single(pref_port, ref_port, dataset, args):
    ############################
    # Load reward model pipeline
    ############################

    BATCH_SIZE = args.batch_size
    model = {
        "model_name": args.model,
        "port": pref_port,
    }

    ref_model = {
        "model_name": args.ref_model,
        "port": ref_port,
    }
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # use internal inference functions in DPO trainer
    dpo_annotator = DPOInferenceVLLM(
        model,
        ref_model,
        tokenizer
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )

    all_result_dicts= []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        print(f"RM inference step {step}/{len(dataloader)}")
        result_dicts = dpo_annotator.monolithic_inference_step(batch)
        all_result_dicts += result_dicts
    return all_result_dicts

def hf_logratio_annotation(dataset, args):
    ############################
    # Load reward model pipeline
    ############################

    BATCH_SIZE = args.batch_size
    model = args.model
    ref_model =  args.ref_model
    
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # use internal inference functions in DPO trainer
    hf_logratio_annotator = LogratioHF(
        model,
        ref_model,
        tokenizer,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )

    all_result_dicts= []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        print(f"RM inference step {step}/{len(dataloader)}")
        result_dicts = hf_logratio_annotator.monolithic_inference_step(batch)
        all_result_dicts += result_dicts
    return all_result_dicts


def singular_classifier_thread(model_name, device_id, chunked_data_dict, results):
    chunk_data_hf = chunked_data_dict[device_id]
    chunk_data = [instance['messages'] for instance in chunk_data_hf]
    if "armo" in model_name.lower():
        from src.models import ArmoRMPipeline
        armopipeline = ArmoRMPipeline(model_name, device=device_id)
        chunk_scores, _ = armopipeline(chunk_data)
        results[device_id] = chunk_scores
    else:
        from src.models import BradleyTerryRMPipeline
        bt_pipeline = BradleyTerryRMPipeline(model_name, device=device_id)
        chunk_scores = bt_pipeline(chunk_data)
        results[device_id] = chunk_scores

def main():
    args = get_args()
    if args.model_type != "dpo":
        import multiprocessing
        multiprocessing.set_start_method('spawn')
    
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model_type == "dpo":
        if args.model in DPO_MODEL_CONFIG:
            config = DPO_MODEL_CONFIG[args.model]
        else:
            config = DPO_MODEL_CONFIG["default"]
        logger.info(f"Using dpo model config: {config}")
        assert args.model != args.ref_model, "policy and reference model should be different"
    else:
        logger.info(f"Using a sequence classifier reward model: {args.model}")

   
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    
    # retry loading tokenizer for 5 times, then throw error
    max_retry = 5
    for i in range(max_retry):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
            break
        except:
            if i == max_retry - 1:
                raise ValueError(f"Failed to load tokenizer after {max_retry} retries")
            print(f"Failed to load tokenizer, retrying...")
    tokenizer.pad_token = tokenizer.eos_token

    if args.insert_system_prompt:
        assert args.insert_system_prompt in SYSTEM_PROMPTS_MAP, "System prompt not found in SYSTEM_PROMPTS_MAP"
    if args.insert_user_prompt:
        assert args.insert_user_prompt in SYSTEM_PROMPTS_MAP, "User prompt not found in SYSTEM_PROMPTS_MAP"

    custom_dataset, input_dataset = load_ibm_bon_data(args.input_path, debug=args.debug, add_system_prompt=args.insert_system_prompt, add_user_prompt=args.insert_user_prompt)

    # modify the load_eval_dataset to be handling single column outputs. 
    simple_dataset = load_simple_dataset(
        custom_dataset,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["formatted_output", "prompt", "original_prompt", "response", "messages"],
        max_prompt_length=args.max_prompt_length
    )

    print("#################### Example Formatted Prompt Messages ####################\n")
    print(simple_dataset[0]["prompt"])
    print("\n#################### End Fortmatted Prompt Messages ####################")


    print("#################### Example Formatted Output Messages ####################\n")
    print(simple_dataset[0]["formatted_output"])
    print("\n#################### End Fortmatted Output Messages ####################")

    # save_dir
    if not args.save_dir:
        modelname_suffix = args.model.split("/")[-1]
        # write sth when dataset name is not available
        if args.input_path:
            save_dir = os.path.dirname(args.input_path) + "/" + modelname_suffix
        else:
            # use the last part of the model name
            save_dir = "./" + modelname_suffix
    else:
        save_dir = args.save_dir
        
    if args.insert_system_prompt:
        # when insert_system_prompt is not empty. 
        # suffix the inserted system prompt to the save_dir
        save_dir = save_dir + "-system-" + args.insert_system_prompt
    if args.insert_user_prompt:
        # when insert_user_prompt is not empty. 
        # suffix the inserted user prompt to the save_dir
        save_dir = save_dir + "-user-" + args.insert_user_prompt
        
    logger.info(f"Results to be saved to the following directory: {save_dir}")

    ############################
    # Multi-threading to leverage multiple threads to increase throughput
    ############################
    if args.model_type != "dpo":
        logger.info("Using a sequence classifier reward model")

    # if not args.debug and not args.hf_logratio_inference:
    if not args.hf_logratio_inference:
        manager = Manager()
        results = manager.dict()
        processes = []
        chunked_data_dict = {}
        base_port = args.base_port
        gpu_chunk_size = int(len(simple_dataset)/args.num_threads)+1
  
        for thread_idx in range(args.num_threads):

            start_idx = thread_idx * gpu_chunk_size
            end_idx = min(len(simple_dataset), start_idx + gpu_chunk_size)
            thread_dataset = simple_dataset.select(range(start_idx, end_idx))
            chunked_data_dict[thread_idx] = thread_dataset
            pref_port = base_port
            ref_port = base_port + 1
            base_port+=2

            if thread_dataset and len(thread_dataset)>0:
                if args.model_type == "dpo":
                    processes.append(
                        Process(target=reward_annotation, args=(pref_port, ref_port, chunked_data_dict, args, thread_idx, results)),
                    )
                else:
                    processes.append(
                        Process(target=singular_classifier_thread, args=(args.model, thread_idx, chunked_data_dict, results)),
                    )
     
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        final_scores = []
        for thread_idx in range(args.num_threads):
            if thread_idx in results and results[thread_idx]:
                final_scores.extend(results[thread_idx])
    else:
        if args.hf_logratio_inference:
            final_scores = hf_logratio_annotation(simple_dataset, args)
        
        elif args.model_type == "dpo":
            final_scores = reward_annotation_single(args.base_port, args.base_port + 1, simple_dataset, args)
        else:

            if "armo" in args.model.lower():
                from src.models import ArmoRMPipeline
                input_data = [instance['messages'] for instance in simple_dataset]
                armopipeline = ArmoRMPipeline(args.model, device=0)
                final_scores, _ = armopipeline(input_data)
            else:
                from src.models import BradleyTerryRMPipeline
                input_data = [instance['messages'] for instance in simple_dataset]

                bt_pipeline = BradleyTerryRMPipeline(args.model, device=0)
                final_scores = bt_pipeline(input_data)


    ############################
    # Print & process results
    ############################
    
    # final_scores: List[Dict[str, List]] or List[int]
    
    if type(final_scores[0]) == float:
        # add column for results for easy printing
        raw_out_dataset = simple_dataset.add_column("results", final_scores)
    elif type(final_scores[0]) == dict:

        raw_out_dataset = simple_dataset.add_column("results", [ x["log_ratio_reward"] for x in final_scores ])
    else:
        raise ValueError("Final scores should be either a list of integers or a list of dictionaries")

    best_of_n_samples, rewards_ls, samples_to_reward_dicts = [], [], []


    best_of_n = int(len(raw_out_dataset)/len(input_dataset))
    print("best-of-n is", best_of_n)

    for i, instance in enumerate(input_dataset):
        start_index, end_index = i*best_of_n, i*best_of_n+best_of_n
        mapped_outputs = raw_out_dataset.select(range(start_index, end_index))
        
        reward_dict = {}
        per_instance_rewards = []

        
        # saninty check
        for in_idx, ex in enumerate(mapped_outputs):
            assert ex["original_prompt"] == instance["prompt"], "original prompt and flattened prompt don't match"

            # this is somewhere that I'm should do another sanity check
            assert instance["output"][in_idx] == ex["response"], "original order is being disrupted"
            reward_dict[ex["response"]] = ex["results"]
            per_instance_rewards.append(ex["results"])
        best_of_n_samples.append(max(reward_dict, key=reward_dict.get))
        rewards_ls.append(per_instance_rewards)
            
        samples_to_reward_dicts.append(reward_dict)
    
    # Function to remove and then add a column
    def replace_column(dataset, column_name, column_data):
        if column_name in dataset.column_names:
            dataset = dataset.remove_columns([column_name])
            print(f"Column '{column_name}' existed and was removed.")
        
        dataset = dataset.add_column(column_name, column_data)
        print(f"Column '{column_name}' has been added.")
        
        return dataset

    # give a best of N response, along with scoring dict: with score matching each output.  
    input_dataset = replace_column(input_dataset, "best_of_n_sample", best_of_n_samples)
    
    # best_of_n_equal_target = [best_of_n_samples[i]==input_dataset[i]["target_output"] for i in range(len(best_of_n_samples))]

    # input_dataset = replace_column(input_dataset, "target_is_bestn", best_of_n_equal_target)
    
    input_dataset = replace_column(input_dataset, "output_reward_scores", rewards_ls)

    output_filename = os.path.basename(args.input_path)+"-rewards.jsonl"

    output_save_path = os.path.join(save_dir, output_filename)
    input_dataset.to_json(output_save_path)


if __name__ == "__main__":
    # load_ibm_bon_data("/dccstor/gxamr/linux-386/llm-alignment/preference-generator/uniform_sample_dataset_30k/best_of_64/bon_sampling_data_split_0.jsonl", debug=True)

    main()

input_data=$1 # get input data
pref_model=$2
ref_model=$3


# set default values for pref_model and ref_model
if [ -z "$pref_model" ]; then
    pref_model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
fi

if [ -z "$ref_model" ]; then
    ref_model="teknium/OpenHermes-2.5-Mistral-7B"
fi

mkdir -p logs

echo "Input Source is: $input_data"

# bash launch_reward_server_para.sh $pref_model $ref_model

# Run bon scoring
python scripts/run_bon_scoring.py \
--model="$pref_model" \
--ref_model="$ref_model" \
--model_type="dpo" \
--num_threads 4 \
--max_prompt_length 2048 \
--max_output_length 2048 \
--base_port 8020 \
--batch_size 8 \
--input_path $input_data
echo $!

# /new_data/gx/r1_pref/official_data/p10_reasoning_random_60k-distribute/best_of_8_distribute_shard_1.jsonl

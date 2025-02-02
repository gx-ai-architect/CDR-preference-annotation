
FILE_PATH=(
    "/new_data/gx/r1_pref/official_data/p10_reasoning_random_60k-distribute/best_of_8_distribute_shard_0.jsonl"
    "/new_data/gx/r1_pref/official_data/p10_reasoning_random_60k-distribute/best_of_8_distribute_shard_1.jsonl"
    "/new_data/gx/r1_pref/official_data/p10_reasoning_random_60k-distribute/best_of_8_distribute_shard_2.jsonl"
    "/new_data/gx/r1_pref/official_data/p10_reasoning_random_60k-distribute/best_of_8_distribute_shard_3.jsonl"
)

for file_path in ${FILE_PATH[@]}; do
    python scripts/domain_annotate.py --file_path $file_path
done
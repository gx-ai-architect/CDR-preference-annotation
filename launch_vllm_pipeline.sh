# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi

check_success() {
    local file=$1
    local message="Started server process"
    # Tail the log file and grep for success message, exit when found
    tail -f "$file" | grep -q "$message"
    echo "Server at $file has started successfully."
}

# Define model configurations
# ["mixtral"]="mistralai/Mixtral-8x7B-v0.1|0,1|8020"
declare -A models
models=(
    ["policy"]="meta-llama/Meta-Llama-3-8B-Instruct|2|8021"
    ["strong"]="NousResearch/Nous-Hermes-2-Mistral-7B-DPO|3|8022"
    ["weak"]="teknium/OpenHermes-2.5-Mistral-7B|4|8023"
)

# Initialize empty array for PIDs
declare -a pid_array

# Launch each model
for model_key in "${!models[@]}"; do
    IFS='|' read -r model_name devices port <<< "${models[$model_key]}"
    echo "Launching $model_key model ($model_name) on GPU(s) $devices"
    
    CUDA_VISIBLE_DEVICES=$devices python -u -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --model $model_name \
        --gpu-memory-utilization 0.85 \
        --port $port \
        --tensor-parallel-size $(echo $devices | tr ',' '\n' | wc -l) \
        --load-format auto > "logs/server_${model_key}.log" 2>&1 &
    sleep 10
    
    # Start monitoring each server log
    check_success "logs/server_${model_key}.log" &
    pid_array+=($!)
done

# Wait for all check_success processes
for pid in "${pid_array[@]}"; do
    wait $pid
done
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

ray disable-usage-stats
export OPENBLAS_NUM_THREADS=18
export OMP_NUM_THREADS=18
ray start --head --num-cpus=32 --num-gpus=8

pref_model=$1
ref_model=$2

# set default values for pref_model and ref_model
if [ -z "$pref_model" ]; then
    pref_model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
fi

if [ -z "$ref_model" ]; then
    ref_model="teknium/OpenHermes-2.5-Mistral-7B"
fi

echo "pref_model: $pref_model"
echo "ref_model: $ref_model"


get_gpu_count() {
    echo $(nvidia-smi -L | wc -l)
}
# Get the number of available GPUs
num_gpus=$(get_gpu_count)

echo "Detected $num_gpus GPUs."

####### First server

CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $pref_model \
    --gpu-memory-utilization 0.5 \
    --port 8020 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_0.log 2>&1 &

sleep 10
# Start monitoring each server log
check_success "logs/server_0.log" &
pid_array+=($!)  # Save the PID of the check_success process


CUDA_VISIBLE_DEVICES=1 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $ref_model \
    --gpu-memory-utilization 0.5 \
    --port 8021 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_1.log 2>&1 &

sleep 1
# Start monitoring each server log
check_success "logs/server_1.log" &
pid_array+=($!)  # Save the PID of the check_success process



####### Second server

CUDA_VISIBLE_DEVICES=2 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $pref_model \
    --gpu-memory-utilization 0.5 \
    --port 8022 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_2.log 2>&1 &

sleep 10
# Start monitoring each server log
check_success "logs/server_2.log" &
pid_array+=($!)  # Save the PID of the check_success process


CUDA_VISIBLE_DEVICES=3 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $ref_model \
    --gpu-memory-utilization 0.5 \
    --port 8023 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_3.log 2>&1 &

sleep 1
# Start monitoring each server log
check_success "logs/server_3.log" &
pid_array+=($!)  # Save the PID of the check_success process


####### Third server

CUDA_VISIBLE_DEVICES=4 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $pref_model \
    --gpu-memory-utilization 0.5 \
    --port 8024 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_4.log 2>&1 &

sleep 10
# Start monitoring each server log
check_success "logs/server_4.log" &
pid_array+=($!)  # Save the PID of the check_success process


CUDA_VISIBLE_DEVICES=5 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $ref_model \
    --gpu-memory-utilization 0.5 \
    --port 8025 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_5.log 2>&1 &

sleep 1
# Start monitoring each server log
check_success "logs/server_5.log" &
pid_array+=($!)  # Save the PID of the check_success process



####### Fourth server

CUDA_VISIBLE_DEVICES=6 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $pref_model \
    --gpu-memory-utilization 0.5 \
    --port 8026 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_6.log 2>&1 &

sleep 10
# Start monitoring each server log
check_success "logs/server_6.log" &
pid_array+=($!)  # Save the PID of the check_success process


CUDA_VISIBLE_DEVICES=7 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model $ref_model \
    --gpu-memory-utilization 0.5 \
    --port 8027 \
    --tensor-parallel-size 1 \
    --load-format auto > logs/server_7.log 2>&1 &

sleep 1
# Start monitoring each server log
check_success "logs/server_7.log" &
pid_array+=($!)  # Save the PID of the check_success process










# Wait only for the check_success processes
for pid in "${pid_array[@]}"; do
    wait $pid
done

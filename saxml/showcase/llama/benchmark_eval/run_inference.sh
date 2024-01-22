#!/bin/bash

# Multi-Host vlp (TODO: replace these params for your own config)
NAME="jwyang-tpu-vm"
ACCELERATOR_TYPE="v5litepod-4"
RUNTIME_VERSION="v2-alpha-tpuv5-lite"
PROJECT="tpu-prod-env-small"
ZONE="us-east1-c"

USER=jwyang 
GSBUCKET=${NAME}-${USER}-sax-admin


download_data_file() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --project=${PROJECT} --worker=0 \
    --command="sudo chown -R ${USER}:${USER} /home/${USER}/ && mkdir -p /home/${USER}/llama/data/ && \
               gsutil cp gs://jwyang-data/llama-open-orca-eval/open_orca_gpt4_50k_filtered_tokenized_llama_prompt.pkl /home/${USER}/llama/data/" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

MODEL_ID=/sax/test/llama
run_inference() {
  gcloud compute tpus tpu-vm \
    scp --recurse --zone=${ZONE} \
    $PWD/saxml/showcase/llama/benchmark_eval/openorca_eval.py \
    --worker=0 ${NAME}:~/saxml/saxml/server/ \
    --project=${PROJECT} \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

  # compile SAX client and run inference
  RESULT_DIR=/home/${USER}/llama/${4}/batch${1}_input${2}_output${3}/
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="cd saxml && mkdir -p ${RESULT_DIR} && \
                bazel build saxml/client/python/sax.cc --compile_one_dependency && \
                SAX_ROOT=gs://${GSBUCKET}/sax-root \
                python3.10 saxml/server/openorca_eval.py \
                  --model_id ${MODEL_ID} \
                  --data_pkl /home/${USER}/llama/data/open_orca_gpt4_50k_filtered_tokenized_llama_prompt.pkl \
                  --num_samples 512 \
                  --results_dir ${RESULT_DIR} \
                  --batch_size ${1} \
                  --input_seq_len ${2} \
                  --max_decode_steps ${3} |& tee ${RESULT_DIR}/benchmark.log" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}


eval_accuracy() {
  gcloud compute tpus tpu-vm \
    scp --recurse --zone=${ZONE} \
    $PWD/inference/benchmarks/models/llama/benchmark_eval/eval_rouge.py \
    --worker=0 ${NAME}:~/saxml/saxml/server/ \
    --project=${PROJECT} \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="cd saxml && \
                python3.10 saxml/server/eval_rouge.py" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}
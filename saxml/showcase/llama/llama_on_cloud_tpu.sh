#!/bin/bash

# Multi-Host vlp (TODO: replace these params for your own config)
NAME="jwyang-tpu-vm-mlperf " 
ACCELERATOR_TYPE="v5litepod-4"
RUNTIME_VERSION="v2-alpha-tpuv5-lite"
PROJECT="tpu-prod-env-automated"
ZONE="us-east1-c"

# (TODO: replace these params to your own config)
NUM_WORKERS=1
TPU_NAME="t1v-n-1d662506" 
USER=jwyang 
GSBUCKET=${NAME}-${USER}-sax-admin

SAX_GIT_VERSION="sax-llama-mlperf"

create_tpu() {
  # A temporary solution to clean up the failed and suspended queued resources.
  # Otherwise, there will be a quota error.
  existing_qr=$(gcloud alpha compute tpus queued-resources list \
    --project ${PROJECT} \
    --zone ${ZONE} \
    --quiet)
  while read -r line; do
    name=$(echo $line | awk '{print $1}')
    status=$(echo $line | awk '{print $5}')
    echo ${name}
    echo ${status}
    if [[ ${status} == "SUSPENDED" || ${status} == "FAILED" ]]; then
      gcloud alpha compute tpus queued-resources delete ${name} \
        --project ${PROJECT} \
        --zone ${ZONE} \
        --quiet
    fi
  done <<< ${existing_qr}

  gcloud alpha compute tpus queued-resources create ${NAME} \
    --description noteardown \
    --node-id ${NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --runtime-version=${RUNTIME_VERSION} \
    --reserved;
}

list_tpu() {
  gcloud compute tpus tpu-vm list --project=${PROJECT} --zone=${ZONE};
}

list_queue_resource() {
  gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE};
}

delete_tpu() {
  gcloud alpha compute tpus tpu-vm delete ${NAME} --project=${PROJECT} --zone=${ZONE};
  gcloud alpha compute tpus queued-resources delete ${NAME} --project=${PROJECT} --zone=${ZONE};
}

ssh_to_tpu() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --worker ${1} --project ${PROJECT} -- -o ProxyCommand='corp-ssh-helper %h %p'
}

create_disk() {
  for ((i = 0; i < ${NUM_WORKERS}; i++)); do
    TPU_WORKER_NAME=${TPU_NAME}-w-${i}
    DISK_NAME=${NAME}-w${i}-ssd

    SIZE=35
    if [[ ${i} == 0 ]]
    then
      SIZE=512
    fi

    gcloud compute disks create ${DISK_NAME} \
      --size ${SIZE} \
      --zone ${ZONE} \
      --type pd-ssd \
      --project=${PROJECT}

    # attach disk to tpu
    gcloud alpha compute instances attach-disk ${TPU_WORKER_NAME} \
      --zone=${ZONE} \
      --disk=${DISK_NAME} \
      --mode=rw \
      --project=${PROJECT}

    gcloud compute instances set-disk-auto-delete ${TPU_WORKER_NAME} \
      --zone=${ZONE} \
      --auto-delete \
      --disk=${DISK_NAME} \
      --project=${PROJECT}

    gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --worker ${i} --project=${PROJECT} \
      --command="sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb &&
               sudo mkdir -p /mnt/disks/persist &&
               sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist" \
      -- -o ProxyCommand='corp-ssh-helper %h %p'
  done
}

detach_disks() {
  for ((i = 0; i < ${NUM_WORKERS}; i++)); do
    TPU_WORKER_NAME=${TPU_NAME}-w-${i}
    DISK_NAME=${NAME}-w${i}-ssd

    # attach disk to tpu
    gcloud alpha compute instances detach-disk ${TPU_WORKER_NAME} \
      --zone=${ZONE} \
      --disk=${DISK_NAME} \
      --project=${PROJECT}
  done
}

check_disks() {
  set -o xtrace
  dir_checks=""
  for ((i = 0; i < ${NUM_WORKERS}; i++)); do
    dir_checks="$dir_checks $(
      gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --worker ${i} --project=${PROJECT} \
        --command="if [ -d /mnt/disks/persist ]; then echo "exists"; fi" \
        -- -o ProxyCommand='corp-ssh-helper %h %p'
    )"
  done
  num_dir_exists=$(echo "$dir_checks" | wc -w)
  echo "Number of workers with disks: $num_dir_exists"
  set +o xtrace
}

build_sax() {
  gcloud compute tpus tpu-vm ssh --zone=${ZONE} ${NAME} --project=${PROJECT} --worker=all \
    --command="if [ ! -d saxml ]; then git clone https://github.com/jwyang-google/saxml.git; fi && \
              cd saxml && git fetch && git checkout -- . && git checkout ${SAX_GIT_VERSION} && \
              saxml/tools/init_cloud_vm.sh && sudo apt-get install -y python3-numpy" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'

  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=all --project=${PROJECT} \
    --command="cd saxml && sudo chown -R ${USER}:${USER} /mnt/disks/persist/ && \
              nohup bazel --output_user_root=/mnt/disks/persist/bazel_build build saxml/server/server &" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

build_admin() {
  gcloud storage buckets create gs://${GSBUCKET} --project=${PROJECT}

  gcloud compute tpus tpu-vm ssh --zone=${ZONE} ${NAME} --project=${PROJECT} --worker=0 \
    --command="cd saxml && \
               bazel run saxml/bin:admin_config -- \
                --sax_cell=/sax/test \
                --sax_root=gs://${GSBUCKET}/sax-root \
                --fs_root=gs://${GSBUCKET}/sax-fs-root \
                --alsologtostderr" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

start_admin() {
  # start saxml admin server
  gcloud compute tpus tpu-vm ssh --zone=${ZONE} ${NAME} --project=${PROJECT} --worker=0 \
    --command="cd saxml &&
                nohup bazel run saxml/bin:admin_server -- \
                 --sax_cell=/sax/test \
                 --sax_root=gs://${GSBUCKET}/sax-root \
                 --port=10000 \
                 --alsologtostderr &" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

start_model_servers() {
  ssh-add /usr/local/google/home/jwyang/.ssh/google_compute_engine

  # gcloud compute tpus tpu-vm \
  #   scp --zone=${ZONE} --project=${PROJECT} --worker=all \
  #   $PWD/showcase/llama/praxis/praxis_multi_query_attention.py \
  #   ${NAME}:/mnt/disks/persist/bazel_build/60508e82bf5accbe3bafc1356d9f1998/external/third_party_praxis/site-packages/praxis/layers/multi_query_attention.py \
  #   --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

  # start saxml model server
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=all --project=${PROJECT} \
    --command="cd saxml && SAX_ROOT=gs://${GSBUCKET}/sax-root \
                  nohup bazel --output_user_root=/mnt/disks/persist/bazel_build \
                  run saxml/server/server -- \
                  --sax_cell=/sax/test \
                  --port=10001 \
                  --platform_chip=tpuv4 \
                  --platform_topology=2x2 |& tee debug.log &" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

restart_model_servers() {
  ssh-add /usr/local/google/home/jwyang/.ssh/google_compute_engine
  # kill model server process
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=all --project=${PROJECT} \
    --command="sudo rm /tmp/libtpu_lockfile && sudo lsof -t /dev/vfio/0 > tpu_process_pid.txt && sudo pkill -F tpu_process_pid.txt" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'

  start_model_servers
}

# MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA70BFP16TPUv5e16
# MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA70BFP16TPUv5e32
# MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA70BFP16TPUv5e8
# CHECKPOINT="gs://jwyang-archive/llama/pax_llama2_70b_chat/checkpoint_00000000"

MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA7BFP16TPUv5e4
CHECKPOINT="gs://jwyang-archive/llama/7b_pax_llama2/checkpoint_00000000"

# MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA13BFP16TPUv5e8
# CHECKPOINT="gs://jwyang-archive/llama/llama2_pax_weights/llama-2-13b-pax/checkpoint_00000000"

# MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA13BFP16TPUv5e4
# MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA33BFP16TPUv5e8
# MODEL_CONFIG=saxml.server.pax.lm.params.lm_cloud.LLaMA65BFP16TPUv5e16

# CNS repo: /cns/io-d/home/jwyang/data/llama_70b_pax_weights

publish_model() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="cd saxml &&
                bazel run saxml/bin:saxutil -- \
                --sax_root=gs://${GSBUCKET}/sax-root publish \
                /sax/test/llama ${MODEL_CONFIG} \
                $CHECKPOINT 1" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

unpublish_model() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="cd saxml &&
                bazel run saxml/bin:saxutil -- \
                --sax_root=gs://${GSBUCKET}/sax-root unpublish \
                /sax/test/llama" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

check_model_accuracy() {
  ssh-add /usr/local/google/home/jwyang/.ssh/google_compute_engine
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="git clone https://github.com/facebookresearch/llama.git" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'

  gcloud compute tpus tpu-vm \
    scp --zone=${ZONE} --project=${PROJECT} --worker=0 \
    $PWD/inference/benchmarks/models/llama/praxis/praxis_multi_query_attention.py \
    ${NAME}:~/.local/lib/python3.10/site-packages/praxis/layers/multi_query_attention.py \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"    

  gcloud compute tpus tpu-vm \
    scp --zone=${ZONE} --project=${PROJECT} --worker=0 \
    $PWD/inference/benchmarks/models/llama/check_model_accuracy.py \
    ${NAME}:~/llama/check_model_accuracy.py \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"
}
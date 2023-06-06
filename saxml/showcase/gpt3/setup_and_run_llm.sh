#!/bin/bash

# Multi-Host v4
NAME="sax-tpu-vm" # TODO: replace with some TPU name
PROJECT="tpu-prod-env-one-vm"
ZONE="us-central2-b" # TODO: "us-central2-b"
NUM_WORKERS=16
ACCELERATOR_TYPE="v4-128"
SOFTWARE_VERSION="tpu-vm-v4-base"

create_tpu() {
  gcloud alpha compute tpus tpu-vm create ${NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --version=${SOFTWARE_VERSION} \
    --subnetwork=jwyang-tpusubnet
}

ssh_to_tpu() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone ${ZONE} --worker 0
}

TPU_NAME="t1v-n-dbf8958b" # TODO: replace this
USER=jwyang # TODO: replace this
GSBUCKET=${NAME}-${USER}-sax-admin
CHECKPOINT="gs://mlperf-llm-public2/gpt3_cnndm_3.0.0_seq256_lr1e5/checkpoint_00013200" # replace this
SAX_GIT_VERSION="f134a5863c1f89c4354e7b6c6c2132594478f3d5"

create_disk() {
  set -o xtrace
  for ((i = 0; i < ${NUM_WORKERS}; i++)); do
    TPU_WORKER_NAME=${TPU_NAME}-w-${i}
    DISK_NAME=${NAME}-w${i}-ssd
    gcloud compute disks create ${DISK_NAME} \
      --size 35 \
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
  set +o xtrace
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
    --command="if [ ! -d saxml ]; then git clone https://github.com/google/saxml.git; fi && \
              cd saxml && git checkout ${SAX_GIT_VERSION} && \
              saxml/tools/init_cloud_vm.sh && sudo apt-get install -y python3-numpy" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'

  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=all --project=${PROJECT} \
    --command="cd saxml && git checkout ${SAX_GIT_VERSION} && \
              sudo chown -R ${USER}:${USER} /mnt/disks/persist/ && \
              sudo rm -rf /mnt/disks/persist/bazel_build* && \
              bazel --output_user_root=/mnt/disks/persist/bazel_build build saxml/server/server" \
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
  gcloud compute tpus tpu-vm \
    scp --zone=${ZONE} --project=${PROJECT} --worker=all \
    $PWD/saxml/server/pax/lm/params/lm_cloud.py \
    ${NAME}:~/saxml/saxml/server/pax/lm/params/lm_cloud.py \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

  # start saxml model server
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=all --project=${PROJECT} \
    --command="cd saxml && SAX_ROOT=gs://${GSBUCKET}/sax-root \
                  nohup bazel --output_user_root=/mnt/disks/persist/bazel_build \
                  run saxml/server/server -- \
                  --sax_cell=/sax/test \
                  --port=10001 \
                  --platform_chip=tpuv4 \
                  --platform_topology=2x2 &" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

restart_model_servers() {
  # kill model server process
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=all --project=${PROJECT} \
    --command="sudo rm /tmp/libtpu_lockfile && sudo lsof -t /dev/accel1 > tpu_process_pid.txt && sudo pkill -F tpu_process_pid.txt" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'

  start_model_servers
}

publish_model() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="cd saxml &&
                bazel run saxml/bin:saxutil -- \
                --sax_root=gs://${GSBUCKET}/sax-root publish \
                /sax/test/gpt175b saxml.server.pax.lm.params.lm_cloud.C4SpmdGpt3AdamOrgHP \
                $CHECKPOINT 1" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

unpublish_model() {
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="cd saxml &&
                bazel run saxml/bin:saxutil -- \
                --sax_root=gs://${GSBUCKET}/sax-root unpublish \
                /sax/test/gpt175b" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'
}

run_inference() {
  # copy the inference code file cnndm_eval.py to TPU workers
  gcloud compute tpus tpu-vm \
    scp --recurse --zone=${ZONE} \
    $PWD/saxml/showcase/gpt3/cnndm_eval.py \
    --worker=0 ${NAME}:~/saxml/saxml/server/ \
    --project=${PROJECT} \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"

  # compile SAX client and run inference
  gcloud compute tpus tpu-vm ssh ${NAME} --zone=${ZONE} --worker=0 --project=${PROJECT} \
    --command="cd saxml && \
                bazel build saxml/client/python/sax.cc --compile_one_dependency && \
                SAX_ROOT=gs://${GSBUCKET}/sax-root python3 saxml/server/cnndm_eval.py" \
    -- -o ProxyCommand='corp-ssh-helper %h %p'0
}

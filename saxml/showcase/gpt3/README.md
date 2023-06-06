#  GPT-3 Inferece using SAX

### Checkout SAX repo and code
```
git clone https://github.com/jwyang-google/saxml/tree/main
git checkout llm_gpt3_gpt175b
cd saxml
```

### Install and set up the `gcloud` tool
[Install](https://cloud.google.com/sdk/gcloud#download_and_install_the) the
`gcloud` CLI and set the default account and project:

```
gcloud config set account <your-email-account>
gcloud config set project <your-project>
```

### Create a Cloud TPU VM instance for SAX admin/model servers

Use this [guide](https://cloud.google.com/tpu/docs/users-guide-tpu-vm) to
enable the Cloud TPU API in a Google Cloud project.

```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
create_tpu;
```

### Attach ssd disks to Cloud TPU VM created

SSH to one of the workers in TPU VM created:
```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
ssh_to_tpu;
```

From TPU VM prompt, copy the $TPU_NAME with format ```t1v-n-xxxxxxx``` and replace the $TPU_NAME in the ```setup_and_run_llm.sh``` script.
Replace the $USER with your own gcloud username in the script.
Exit TPU VM. 

Create SSD disk and attach it to the TPU VM:
```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
create_disk;
```


### Build SAX model server code

SSH to the Compute Engine VM instance:

```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
build_sax;
```
This step may take a few minutes to finish. 


### Start the Sax TPU admin server
```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
build_admin;
start_admin;
```

You should see log message "Starting the HTTP server on port 8080"

### Start the SAX TPU model servers
```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
start_model_servers;
```

You should see a log message "Joined [admin server IP:port]" from the model
server to indicate it has successfully joined the admin server.

## Run inference code of cnn_dailymail dataset on GPT-3 model
Publish GPT3 model to SAX model server:
```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
publish_model;
```

Run inference with cnn_dailymail test dataset:
```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
run_inference;
```
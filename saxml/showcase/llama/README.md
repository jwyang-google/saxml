#  MLPerf llama2 Inferece using SAX

### Checkout SAX repo and code
```
git clone https://github.com/jwyang-google/saxml/tree/main
git checkout sax-llama-mlperf
cd saxml/
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
source saxml/showcase/llama/llama_on_cloud_tpu.sh;
create_tpu;
```

### Attach ssd disks to Cloud TPU VM created

SSH to one of the workers in TPU VM created:
```
source saxml/showcase/gpt3/setup_and_run_llm.sh;
ssh_to_tpu;
```

From TPU VM prompt, copy the $TPU_NAME with format ```t1v-n-xxxxxxx``` and replace the $TPU_NAME in the ```llama_on_cloud_tpu.sh``` script.
Replace the $USER with your own gcloud username in the script.
Exit TPU VM. 

Create SSD disk and attach it to the TPU VM:
```
source saxml/showcase/llama/llama_on_cloud_tpu.sh;
create_disk;
```


### Build SAX model server code

SSH to the Compute Engine VM instance:

```
source saxml/showcase/llama/llama_on_cloud_tpu.sh;
build_sax;
```
This step may take a few minutes to finish. 


### Start the Sax TPU admin server
```
source saxml/showcase/llama/llama_on_cloud_tpu.sh;
build_admin;
start_admin;
```

You should see log message "Starting the HTTP server on port 8080"

### Start the SAX TPU model servers
```
source saxml/showcase/llama/llama_on_cloud_tpu.sh;
start_model_servers;
```

You should see a log message "Joined [admin server IP:port]" from the model
server to indicate it has successfully joined the admin server.

## Run inference code dataset on llama model
Publish llama model to SAX model server:
```
source saxml/showcase/llama/llama_on_cloud_tpu.sh;
publish_model;
```
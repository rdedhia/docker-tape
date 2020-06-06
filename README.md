# docker-tape

This repository provides a GPU-enabled Docker deployment for a Flask app around the 
[Tasks Assessing Protein Embeddings](https://github.com/songlab-cal/tape) (TAPE) project. Maintained by researchers
at UC Berkeley, TAPE provides models for a variety of protein prediction tasks.

The Docker/Flask application provides two main capabilities:
* Generating embeddings from protein sequences using one of TAPE's pretrained models by wrapping the `tape-embed`
method of the TAPE cli. These embeddings can be used as features for downstream prediction tasks. Embedding sequences
is very compute intensive, so can be accelerated by using a GPU. For this reason, the `Dockerfile` relies on an
`Nvidia` base image and is GPU compatible. Furthermore, we include instructions for setting up an EC2 instance with
GPUs in Amazon Web Services (AWS) with `AWS Cloudformation` to run the Docker container.
* Visualizing the embeddings and labels for a given dataset with `Principal Component Analysis` (PCA) in 2 or 3 
dimensions, through an interact `plotly` plot.

Along with the server, the application contains a simple user interface for leveraging the two capabilities. 

## Running Locally

### Flask (no Docker)

If you would like to run the Flask application outside of Docker, set up a `Python 3.6` virtual environment, and run
the following from the top level of the project:

```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Then, start the application:

```
python tape/server.py
```

You can access the user interface in your browser at `http://localhost:8443/`

### Local Docker

Build the image:

```
docker build -t tape .
```

Run the container, with a different runtime depending on which device you want to use. **Note:** Trying to run the
container with `--runtime=nvidia` will cause an error if there are no GPUs available on your machine:

CPU:

```
docker run -p 443:8443 tape
```

GPU:

```
docker run --runtime=nvidia -p 443:8443 tape
```

Access the user interface in your browser at `http://localhost:443/`

## Running on EC2

### EC2 Deployment

In order to deploy `docker-tape` on EC2, you need access to an AWS account with the permissions to
* Create and launch cloudformation (cfn) templates
* Permission to create S3 buckets (required to upload a template file to cfn)
* Launch EC2 instances
* Create security groups

Then, follow the steps below:

1. Follow the [AWS guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html) to create an EC2
key pair for connecting to EC2 instances via SSH. Then, download the public key and save it in your `~/.ssh` 
directory as `proteins-ec2.pem`.

2. Navigate to `AWS Cloudformation` in the AWS console, and create a new stack with new resources. 
Under `Prerequisite - Prepare template`, select `Template is ready`, and the `Upload a template file` and upload the
template at `infrastructure/template.yaml` from this repository. In `Step 2: Specify stack details`, specify a name
for the stack, and then modify the parameters. You will need to specify a `KeyName` with the name of the key pair
you created in step 1. You can also overwrite the default AMI (Ubuntu deep learning) or the default instance type
`p2.xlarge`, which has 1 GPU. Click `Next` until you get to step 4, and then click `Create Stack` to create the 
cloudformation stack. After it launches, you should see a new `p2.xlarge` instance in your `EC2` console.
The benefit of the cloudformation template is that it provides a blueprint for deploying
your EC2 instance with networking enabled so that you can access the instance on port `22` for SSH or port `443` for
accessing the Docker/Flask app from your browser.

3. Navigate to the `EC2` service in the AWS console. Under `Instances`, you should have a new `p2.xlarge` instance.
One the `Instance State` is running, you can find the public hostname under `Public DNS (IPv4)`. For example, this
looks like `ec2-54-159-56-66.compute-1.amazonaws.com`. Copy and keep track of this value for your instance, which
we will refer to as `EC2_PUBLIC_HOSTNAME` for the remainder of these steps. SSH into the instance from your local
machine by running the following command:

```
ssh -i proteins-ec2.pem ubuntu@{EC2_PUBLIC_HOSTNAME}
```

4. (Optional) Check the state of the GPUs running on the EC2 instance by running `nvidia-smi` in the EC2 terminal.

5. Clone the `docker-tape` repository:

```
git clone https://github.com/rdedhia/docker-tape.git
cd docker-tape
```

### Docker Deployment on EC2

1. Build the Docker image:

```
docker build -t tape .
```

2. Run the container with the `Nvidia` runtime in the background:

```
docker run --runtime=nvidia -p 443:8443 tape &
```

3. (Optional) You can follow the logs of your container by running `docker ps` to find your CONTAINER_ID, and then run
`docker logs <CONTAINER_ID> --follow` to view the logs

### Access Application from the Browser

1. (Optional) Navigate to `http://<EC2_PUBLIC_HOSTNAME>:443/health/full` to verify that the application is healthy.

2. (Optional) View details about the GPU(s) deployed on your instance by navigating to
`http://<EC2_PUBLIC_HOSTNAME>:443/torch/gpu`. Here's a sample response for a `p3.8xlarge` instance with 4 `Tesla V100
GPUs`:

```
{
    "cuda":true,
    "device_count":4,
    "device_name":"Tesla V100-SXM2-16GB"
}
```

3. Interact with the sequence embedding and PCA visualization capabilities in the user interface at 
`http://<EC2_PUBLIC_HOSTNAME>:443/`. Refer to the `Embedding Sequences` and `PCA Visualization` sections at the
bottom of the README for more details.

### Stopping and Starting EC2

If you would like to stop your EC2 instance while you are not using the application, you can do so from the `EC2`
service in the AWS console. Take note that when you restart your instance, you will have to SSH in again and start
the (already built) container. In addition, your `EC2_PUBLIC_HOSTNAME` will change, so you will have to modify the
URL you use for SSH and to navigate to in the browser.

Whenever you restart the instance, we recommend first running `nvidia-smi`, because we have observed in the past that
there are issues with the GPU after starting the instance again. If you get a `CUDA` or `Nvidia` error, try to run
the `infrastructure/fix_nvidia.sh` script to see if this resolves the issue. The error typically looks like this:

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

## Embedding Sequences

In the landing page for the Docker/Flask app in your browser, refer to the `Embed Fasta file with pretrained models`
section. Upload a `fasta` file to embed. Note that the embedding process can be slow for large numbers of sequences,
especially if are using a CPU and use a smaller batch size. Be aware, however, that using a large batch size can cause
the GPU to run out of memory, so we would recommend starting with a small batch size at first.

Specify the parameters for the model, pretrained model, tokenizer, and batch size. We would recommend starting with 
the defaults of `transformer`, `bert-base`, `iupac`, and `4` (or less), but you can learn more about the different
parameters in the TAPE [README.md](https://github.com/songlab-cal/tape/tree/master/README.md).

Specify the name of an output file, and click `Upload`. After the sequences are finished embedding, a file with the
specified name will be downloaded to your local machine in the form of a Numpy file with a `.npz` suffix.

We have included a test file at `tests/test-files/deeploc-small.fasta` with 6 sequences from the DeepLoc dataset to
quickly test out the embedding functionality. If you would like to embed a larger file, you can use the
[deeploc_data_6000.fasta](https://github.com/rdedhia/tape/blob/master/data/deeploc_data_6000.fasta) file from our
fork of the TAPE repository. Refer to [Notebooks.md](https://github.com/rdedhia/tape/blob/master/Notebooks.md#data)
to understand more about how that dataset was derived from the original DeepLoc dataset.

## PCA Visualization

Once you have sequence embeddings in the form of a `.npz` file, you can visualize them with PCA in 2 or 3 dimensions.
In the landing page for the Docker/Flask app in your browser, refer to the `Visualize Embedded data with PCA` section.
Choose your sequence embeddings under `Input Data (npz file)` and a targets files containing labels for each sequence
under `Targets File (JSON)`, and then click `Upload`. The targets file includes key value pairs from sequence ID to 
label, and looks like the following example:

```
{
	"Q9H400": "0",
	"P83456": "3",
	"Q9GL77": "1",
	...
}
```

If you want to try this with a small (and relatively uninteresting) dataset to verify the functionality, use the `.npz`
file produced by embedding the `tests/test-files/deeploc-small.fasta` file in the section above as the embedding file
and `tests/test-files/labels.json` as a target file.

You can also use a larger dataset from our fork of the TAPE repository. For subcellular location, use
[output_deeploc_6000.npz](https://github.com/rdedhia/tape/blob/master/data/output_deeploc_6000.npz) and
[deeploc_labels_q10.json](https://github.com/rdedhia/tape/blob/master/data/deeploc_labels_q10.json).

For membrane bound vs water soluble classification, use
[output_deeploc_q2_6000.npz](https://github.com/rdedhia/tape/blob/master/data/output_deeploc_q2_6000.npz) and
[deeploc_labels_q2.json](https://github.com/rdedhia/tape/blob/master/data/deeploc_labels_q2.json).Again, refer to 
[Notebooks.md](https://github.com/rdedhia/tape/blob/master/Notebooks.md#data) to understand more about how those 
datasets were derived.

After the upload is complete, you should have a 2D or 3D `plotly` visualize that you an interact with. The legend is
interactive, which allows you to compare and and contrast the embeddings of specific classes.

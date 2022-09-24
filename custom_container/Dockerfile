
# Use pytorch GPU base image
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest

# set working directory
WORKDIR /app

# Install required packages
RUN pip install google-cloud-storage transformers==4.17.0 datasets==2.0.0 nltk rouge-score sentencepiece cloudml-hypertune wandb==0.12.11

# Install git lfs to push huggingface to the hub
RUN \
  apt-get update && \
  apt-get install -y sudo curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs=3.1.2 && git lfs install

# Copies the trainer code to the docker image.
COPY ./trainer/__init__.py /app/trainer/__init__.py
COPY ./trainer/experiment.py /app/trainer/experiment.py
COPY ./trainer/utils.py /app/trainer/utils.py
COPY ./trainer/task.py /app/trainer/task.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]

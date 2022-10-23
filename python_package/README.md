# PyTorch - Python Package Training

## Overview

The directory provides code to fine tune a transformer model ([mT5](https://huggingface.co/google/mt5-small)) from Huggingface Transformers Library for text summarization task.

## Prerequisites
* Setup your project by following the instructions from [documentation](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)
* Change directories to this sample.

## Directory Structure

* `trainer` directory: all Python modules to train the model.
* `scripts` directory: command-line scripts to train the model on Vertex AI.
* `setup.py`: `setup.py` scripts specifies Python dependencies required for the training job. Vertex Training uses pip to install the package on the training instances allocated for the job.

### Trainer Modules
| File Name | Purpose                                                                                                                                                 |
| :-------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [utils.py](trainer/utils.py) | Includes: utility functions such as save model to GCS bucket.                                                                                           |
| [experiment.py](trainer/experiment.py) | Runs the model training and evaluation experiment, and exports the final model. Also include an integration with Weight&Biases to track the experiment. |
| [task.py](trainer/task.py) | Includes: 1) Initialize and parse task arguments (hyper parameters), and 2) Entry point to the trainer.                                                 |

### Versions
This script uses the pre-built PyTorch containers for PyTorch 1.7.
* `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-7:latest`


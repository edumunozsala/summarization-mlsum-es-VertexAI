# This script is inspired in the scripts for Train a Pytorch model on Vertex AI
# https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/community-content/pytorch_text_classification_using_vertex_sdk_and_gcloud

import argparse
import os
import sys

from trainer import experiment


def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """
    parser = argparse.ArgumentParser()


    parser.add_argument('--lr', dest='lr',
                        default=0.00001, type=float,
                        help='Learning rate.')
    parser.add_argument('--warmup_steps',
                        default=500, type=int,
                        help='Warmup steps for AdamW optimizer')
    parser.add_argument('--epochs', dest='epochs',
                        default=2, type=int,
                        help='Number of epochs.')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--model-name',
                        type=str,
                        help='The name of the pretrained model'
                        )
    parser.add_argument('--dataset_name',
                        type=str,
                        help='Dataset name in Huggingface Hub')                    
    parser.add_argument('--train_split',
                        type=str,
                        default='train')
    parser.add_argument('--val_split',
                        type=str,
                        default='validation')
    parser.add_argument('--test_split',
                        type=str,
                        default='test')
    parser.add_argument('--max_input_length',
                            type=int,
                            default=512)
    parser.add_argument('--max_target_length',
                            type=int,
                            default=64)
    parser.add_argument('--wandb-api-key',
                        type=str,
                        help='APi Key to log in W&B'
                        )
    parser.add_argument('--push_to_hub',
                        type=str,
                        help='yes or no to push to hub'
                        )
    parser.add_argument('--hub_model_id',
                        type=str,
                        help='Model id in the Hub'
                        )
    parser.add_argument('--hub_token',
                        type=str,
                        help='Token to log to the Hub'
                        )                        
    parser.add_argument('--trained-model-name',
                        type=str,
                        help='The name of the rtained model'
                        )
    parser.add_argument('--fp16',
                        type=str,
                        help='yes or no to set fp16'
                        )

    parser.add_argument(
        '--seed',
        help='Random seed (default: 42)',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--weight-decay',
        help="""
      The factor by which the learning rate should decay by the end of the
      training.

      decayed_learning_rate =
        learning_rate * decay_rate ^ (global_step / decay_steps)

      If set to 0 (default), then no decay will occur.
      If set to 0.5, then the learning rate should reach 0.5 of its original
          value at the end of the training.
      Note that decay_steps is set to train_steps.
      """,
        default=0.01,
        type=float)

    # Enable hyperparameter
    parser.add_argument(
        '--hp-tune',
        default="n",
        help='Enable hyperparameter tuning. Valida values are: "y" - enable, "n" - disable')
    
    # Saved model arguments
    parser.add_argument(
        '--job-dir',
        default=os.getenv('AIP_MODEL_DIR'),
        help='GCS location to export models')

    return parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    print(args)
    experiment.run(args)


if __name__ == '__main__':
    main()

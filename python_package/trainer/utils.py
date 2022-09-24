# This script is inspired in the scripts for Train a Pytorch model on Vertex AI
# https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/community-content/pytorch_text_classification_using_vertex_sdk_and_gcloud

import os
import datetime

from google.cloud import storage

def save_model(args):
    """Saves the model to Google Cloud Storage or local file system

    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    if args.job_dir.startswith(scheme):
        job_dir = args.job_dir.split("/")
        bucket_name = job_dir[2]
        object_prefix = "/".join(job_dir[3:]).rstrip("/")

        if object_prefix:
            model_path = '{}/{}'.format(object_prefix, args.trained_model_name)
        else:
            model_path = '{}'.format(args.trained_model_name)

        bucket = storage.Client().bucket(bucket_name)    
        local_path = os.path.join("/tmp", args.trained_model_name)
        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        for file in files:
            local_file = os.path.join(local_path, file)
            blob = bucket.blob("/".join([model_path, file]))
            blob.upload_from_filename(local_file)
        print(f"Saved model files in gs://{bucket_name}/{model_path}")
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.trained_model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")


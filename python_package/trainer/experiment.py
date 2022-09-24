# This script is inspired in the scripts for Train a Pytorch model on Vertex AI
# https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/community-content/pytorch_text_classification_using_vertex_sdk_and_gcloud

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EvalPrediction, TrainerCallback
from datasets import load_dataset, load_metric
import numpy as np

import nltk
nltk.download('punkt')

import os
from datetime import datetime

# Import libraries for this project
import hypertune
from trainer import utils

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")

class HPTuneCallback(TrainerCallback):
    """
    A custom callback class that reports a metric to hypertuner
    at the end of each epoch.
    """
    
    def __init__(self, metric_tag, metric_value):
        super(HPTuneCallback, self).__init__()
        self.metric_tag = metric_tag
        self.metric_value = metric_value
        self.hpt = hypertune.HyperTune()
        
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"HP metric {self.metric_tag}={kwargs['metrics'][self.metric_value]}")
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric_tag,
            metric_value=kwargs['metrics'][self.metric_value],
            global_step=state.epoch)

def load_datasets_hub(dataset_name,train_split,val_split,test_split):
    """
        Load the dataset from Huggingface Hub
    """
    raw_dataset_train = load_dataset(dataset_name,"es",split=train_split)
    raw_dataset_val = load_dataset(dataset_name,"es",split=val_split)
    raw_dataset_test = load_dataset(dataset_name,"es",split=test_split)    

    return raw_dataset_train, raw_dataset_val,raw_dataset_test

def run(args):
    """Load the data, train, evaluate, and export the model for serving and
     evaluating.

    Args:
      args: experiment parameters.
    """
    """
    def preprocess_function(examples):

            Prepare the dataset to train the Sequence-2-sequence model
            For mlsum dataset
            
        inputs = [PREFIX + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    """
        
    # Applying padding and -100 labels
    def preprocess_function(examples):
        inputs = [PREFIX + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, padding="max_length", max_length=MAX_INPUT_LENGTH, truncation=True)

        # Setup the tokenizer for targets
        #with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], padding="max_length", max_length=MAX_TARGET_LENGTH, truncation=True)

        #model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in l]
            for l in labels["input_ids"]
        ]
        return model_inputs

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    # Define variables
    MAX_INPUT_LENGTH= args.max_input_length
    MAX_TARGET_LENGTH= args.max_target_length
    WANDB_INTEGRATION=False
    
    # If you are using one of the five T5 checkpoints we have to prefix the inputs with "summarize:" 
    if args.model_name in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        PREFIX = "summarize: "
    else:
        PREFIX = ""
    
    # Open our dataset
    # Load the data from HF hub
    train_data, val_data, test_data = load_datasets_hub(args.dataset_name, args.train_split,args.val_split,args.test_split)
    #Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Tokenize the train data
    tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=["text", "summary", "title", "url", "date", "topic"])
    #Tokenize the validation data
    tokenized_val = val_data.map(preprocess_function, batched=True, remove_columns=["text", "summary", "title", "url", "date", "topic"])
    #Tokenize the test data
    tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=["text", "summary", "title", "url", "date", "topic"])

    # Initialize wandb for logging
    if args.wandb_api_key!="no-logging":
        import wandb
        WANDB_INTEGRATION=True
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        wandb.login()

    # Set the variable to push the model to the hub
    if args.push_to_hub=="y":
        push_model=True
    else:
        push_model=False
        
    # Set logging steps when strategy= steps
    steps_per_epoch = len(tokenized_train) // args.train_batch_size
    num_training_steps = steps_per_epoch * args.epochs
    log_steps = num_training_steps//10
        
    # Fine Tuning the model
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    # Setting the Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join("/tmp", args.trained_model_name),
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy="steps",
        logging_steps= log_steps,
        load_best_model_at_end= False,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        overwrite_output_dir=True, 
        do_train=True,
        do_eval=True,
        push_to_hub=push_model,
        hub_strategy="checkpoint",
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        fp16=True if args.fp16=="y" else False,
    )
    
    # Then, we need a special kind of data collator, which will not only pad the inputs to the maximum length in the batch, but also the labels
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # Load the metric from huggingface
    metric = load_metric("rouge")
    # Then we just need to pass all of this along with our datasets to the `Seq2SeqTrainer`
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if WANDB_INTEGRATION:
        wandb_run = wandb.init(
            project="summarization-"+args.dataset_name+"-es",
            config={
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "weight_decay": training_args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "epochs": args.epochs,
                "max_input_length":args.max_input_length,
                "max_target_length":args.max_target_length,
                "dataset": args.dataset_name + "-es",
            },
            group=args.trained_model_name,
            name = args.trained_model_name+'-'+get_timestamp()
        )
        
    # add hyperparameter tuning callback to report metrics when enabled
    if args.hp_tune == "y":
        trainer.add_callback(HPTuneCallback("rouge2", "eval_rouge2"))

    # We can now finetune our model by just calling the `train` method:
    trainer.train()
    
    # Save the model to the Hub        
    if training_args.push_to_hub:
        trainer.push_to_hub()
    else:
        metrics = trainer.evaluate(eval_dataset=tokenized_test)        
        trainer.save_metrics("all", metrics)
        # Save the trained model
        trainer.save_model(os.path.join("/tmp", args.trained_model_name))

    if WANDB_INTEGRATION:    
        wandb.finish()
        
    # Export the trained model
    #trainer.save_model(os.path.join("/tmp", args.trained_model_name))

    # Save the model to GCS
    if args.job_dir:
        utils.save_model(args)
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.trained_model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from datasets import Dataset, load_dataset
import json
import os
import glob
import re
import sys
import datetime
import GPUtil
import time

torch.cuda.empty_cache()

maximum_temperature=0

gpus = GPUtil.getGPUs()
for gpu in gpus:
    maximum_temperature=max(maximum_temperature, gpu.temperature)


while(maximum_temperature>=70):
    
    maximum_temperature=0
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        maximum_temperature=max(maximum_temperature, gpu.temperature)

    print("Sleeping 60 seconds")
    #sleep for 60 seconds
    time.sleep(60)


config={}

#Loading the setting and paths
with open('setting.json', 'r') as file:
    config = json.load(file)


# --- Configuration ---
MODEL_PATH = config.get("MODEL_PATH")
# BASE FILENAME: The script will now search for the latest file based on this base name.
BASE_FINE_TUNING_DATA_FILE = "./datasets/llama_finetuning_data.json"
OUTPUT_DIR = "./finetuned_lora"

# File to read the path of the LATEST ADAPTER to resume training. (The 'input' path)
INPUT_ADAPTER_PATH_FILE = "./latest_adapter_path.txt"

# File to write the path of the NEWLY CREATED adapter after training. 
# MODIFIED: This file will now store a history of all created adapter paths.
ADAPTER_PATH_HISTORY_FILE = "./adapter_path_history.txt"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 10

# --- Helper Functions ---

def find_latest_data_file(base_filename: str) -> str:
    """
    Finds the most recently created fine-tuning data file based on the base filename
    and the _[index] pattern (e.g., llama_finetuning_data_2.json).
    """
    directory = os.path.dirname(base_filename) or '.'
    base_name, ext = os.path.splitext(os.path.basename(base_filename))
    
    # Pattern to match the base file and all indexed files (e.g., base.json, base_1.json, base_2.json)
    pattern = os.path.join(directory, f"{base_name}*{ext}")
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"!!! ERROR: No fine-tuning data files found matching {pattern} !!!")
        return None

    def get_file_index(file_path):
        # Tries to extract the index (e.g., 1, 2, 3) from the filename
        match = re.search(r'_(\d+)\.', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        
        # If it's the base file without an index
        if os.path.basename(file_path) == os.path.basename(base_filename):
            return 0 
            
        return -1

    # Sort files by the extracted index (descending)
    latest_file = max(all_files, key=get_file_index)
    
    print(f"✅ Found latest data file: {latest_file}")
    return latest_file

def load_finetuning_data(file_path: str):
    if not os.path.exists(file_path):
        print(f"!!! ERROR: Fine-tuning data file not found at {file_path}. !!!")
        return None
    print(f"Loading data from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
        
    print(f"Loaded {len(dataset)} records.")
    
    # Convert 'question' and 'answer' to the 'messages' format required by SFTTrainer
    def format_to_messages(example):
        example['messages'] = [
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": example['answer']}
        ]
        return example
        
    if 'question' in dataset.column_names and 'answer' in dataset.column_names:
        dataset = dataset.map(format_to_messages, remove_columns=['question', 'answer'])
        print("Data successfully formatted into 'messages' column.")
    else:
        print("!!! WARNING: Expected 'question' and 'answer' columns not found. Assuming 'messages' is already present. !!!")

    print("Data successfully loaded into Hugging Face Dataset.")
    return dataset

def find_latest_checkpoint(output_dir):
    """Finds the path to the latest checkpoint directory."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint
    
def read_adapter_path(path_file: str) -> str | None:
    """
    Reads the specified file to get the path to the latest fine-tuned adapter.
    """
    if not os.path.exists(path_file):
        print(f"Info: Adapter path file not found at {path_file}. Assuming fresh training.")
        return None
        
    try:
        # Read the file and get the last line (which should be the latest path)
        with open(path_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if not lines:
            print(f"Info: Adapter path file {path_file} is empty. Assuming fresh training.")
            return None
            
        adapter_path = lines[-1] # Get the very last line/path
            
        if not os.path.isdir(adapter_path):
            print(f"Warning: Path read from {path_file} does not appear to be a valid directory: {adapter_path}")
            return None # Treat as if path not found if directory doesn't exist
            
        return adapter_path
        
    except Exception as e:
        print(f"Error reading adapter path from {path_file}: {e}")
        return None

# --- MODIFIED: write_adapter_path now appends the path to the file. ---
def write_adapter_path(path_file: str, adapter_path: str):
    """
    Writes the path of the newly created adapter to the specified file, 
    appending it to the end without overwriting previous paths.
    """
    try:
        os.makedirs(os.path.dirname(path_file) or '.', exist_ok=True)
        # Use 'a' for append mode
        with open(path_file, 'a', encoding='utf-8') as f:
            # Write the path followed by a newline character
            f.write(adapter_path.strip() + '\n')
        print(f"✅ Successfully appended new adapter path to {path_file}")
    except Exception as e:
        print(f"!!! ERROR: Failed to append new adapter path to {path_file}: {e}")

def find_latest_adapter_index(output_dir):
    """
    Finds the index of the most recently saved adapter (e.g., from 'final_adapters_3') 
    and determines the index for the next save.
    """
    adapter_paths = glob.glob(os.path.join(output_dir, "final_adapters_*"))
    
    if not adapter_paths:
        return 1 # Start index at 1 if none are found

    def get_index(path):
        match = re.search(r'final_adapters_(\d+)$', path)
        # Note: If a non-indexed 'final_adapters' directory exists, this will return None (0 in int)
        return int(match.group(1)) if match else 0

    latest_index = max([get_index(path) for path in adapter_paths])
    
    # Ensure the base 'final_adapters' directory (index 0) is considered if it exists but no numbered one does.
    if latest_index == 0 and os.path.isdir(os.path.join(output_dir, "final_adapters")):
        return 1 # Still start the numbering at 1

    return latest_index + 1


# --- Function for Model Loading (Encapsulates QLoRA setup) ---

def load_model_for_training(model_path, existing_adapter_path=None):
    """
    Loads the base model with 4-bit quantization (QLoRA) and optionally 
    applies an existing LoRA adapter to resume training.
    """
    print("Loading base model with 4-bit quantization...")
    
    # Configuration for 4-bit loading

    # 1. Define the original dictionary (fixing the "not defined" error)
    quantization_config_params = {
        "load_in_4bit": True, 
        "bnb_4bit_quant_type": "nf4", 
        "bnb_4bit_compute_dtype": torch.bfloat16, 
        "bnb_4bit_use_double_quant": True,
        "device_map": "auto",
    }

    # 2. Extract device_map so BitsAndBytesConfig doesn't crash
    device_map = quantization_config_params.pop("device_map", "auto")

    # 3. Safely convert the remaining dictionary items into the proper config
    quantization_config = BitsAndBytesConfig(**quantization_config_params)


    
    # 1. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 # Ensures computational dtype is bfloat16
    )


    
    print("** Starting training: Applying new LoRA config. **")

    #Added it
    model = prepare_model_for_kbit_training(model) 

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    return model, peft_config

# --- Main Fine-Tuning Logic (MODIFIED) ---
def run_fine_tuning():
    print("--- Starting Fine-Tuning Process ---")
    
    # 1. Load Data
    latest_data_file = find_latest_data_file(BASE_FINE_TUNING_DATA_FILE)
    if not latest_data_file:
        print("Aborting fine-tuning due to missing data.")
        return
        
    dataset = load_finetuning_data(latest_data_file)
    if dataset is None:
        print("Aborting fine-tuning due to data loading error.")
        return

    # 2. Get Adapter Paths
    # Read the adapter path to resume from (if available) - reads the last line of the history file
    existing_adapter_path = read_adapter_path(INPUT_ADAPTER_PATH_FILE)
    
    # Determine the index for the NEW adapter to be saved
    next_adapter_index = find_latest_adapter_index(OUTPUT_DIR)
    
    # Set the path for the current save
    current_adapter_save_path = os.path.join(OUTPUT_DIR, f"final_adapters_{next_adapter_index}")
    print(f"New adapter will be saved to: {current_adapter_save_path}")

    # 3. Setup Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting tokenizer.pad_token = tokenizer.eos_token")
    
    # 4. Apply PEFT (LoRA) - Using the new consistent loader
    model, peft_config = load_model_for_training(MODEL_PATH, existing_adapter_path)
    
    print(model.print_trainable_parameters())
    
    # 5. Define Training Arguments
    print("Setting up TrainingArguments...")

    #warm up ratio
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=10, 
        save_steps=50,    
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=False,        
        bf16=True,         
        max_grad_norm=0.3,
        seed=42,
        save_only_model=True,


        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 6. Initialize SFTTrainer
    
    def chat_template_formatting_func(example):
        # This function relies on the 'messages' column created in load_finetuning_data
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config, 
        args=training_arguments,
        formatting_func=chat_template_formatting_func,
        data_collator=None,
    )

    # 7. Start Training
    print("Starting trainer.train()...")
    #latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    latest_checkpoint = None
    trainer.train(resume_from_checkpoint=latest_checkpoint) 
    
    # 8. Save Final LoRA Adapters
    print(f"\nTraining finished. Saving final adapter weights to {current_adapter_save_path}...")
    os.makedirs(current_adapter_save_path, exist_ok=True)
    trainer.model.save_pretrained(current_adapter_save_path)
    tokenizer.save_pretrained(current_adapter_save_path)
    print(f"Adapter and tokenizer for Loop {next_adapter_index} saved.")
    
    # 9. Save the path to the newly created adapter to the history file (APPRENDING)
    write_adapter_path(ADAPTER_PATH_HISTORY_FILE, current_adapter_save_path)


if __name__ == "__main__":
    try:
        run_fine_tuning()
    except Exception as e:
        error_string = f"[{datetime.datetime.now()}] Error: {str(e)}"
        
        # Save to a dedicated error file instead of config
        with open("error_log.txt", "a", encoding='utf-8') as f:
            f.write(error_string + "\n")
            
        print(f"Logged error to error_log.txt")
   
        #For exitting with error so that batch code can notice the error
        sys.exit(1)


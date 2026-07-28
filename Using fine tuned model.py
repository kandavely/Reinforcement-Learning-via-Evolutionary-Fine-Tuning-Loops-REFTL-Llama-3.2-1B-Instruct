import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
import os
import sys
import json
import math

# --- Configuration ---
# IMPORTANT: Use the same paths as in your fine-tuning scripts




config={}

#Loading the setting and paths
with open('setting.json', 'r') as file:
    config = json.load(file)


# --- Configuration ---

MODEL_PATH = config.get("MODEL_PATH")

ADAPTER_PATH = "./finetuned_lora/final_adapters_1"

# --- Generation Parameters for Speed ---
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.5 
TOP_K = 50 
# ---

TRAINING_ROLE = "Chatbot"

weight_num=1.0


# --- Model Loading and Initialization ---
def initialize_model():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.padding_side is None:
            tokenizer.padding_side = "left"


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


            
        # Load Base Model with 4-bit quantization
        print(f"Loading base model from {MODEL_PATH}...")


        # 1. Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 # Ensures computational dtype is bfloat16
        )

        # CRITICAL FOR SPEED: Ensures fast sequential token generation
        base_model.config.use_cache = True 

        # Load Fine-Tuned Adapters

        lora_paths=config["DECIDED_LORA_PATHS"]

        adapter_list=[]
        weight_list=[]

        if(len(lora_paths)>=1 and os.path.exists(lora_paths[0])):
            print(f"Loading adapters from {lora_paths[0]}...")
            model = PeftModel.from_pretrained(base_model, lora_paths[0], adapter_name="adapter1")
            
            density_num = 0.8
            adapter_list.append("adapter1")
            weight_list.append(weight_num)


            for i in range(1, len(lora_paths), 1):


                if os.path.exists(lora_paths[i]):
                    adapter_name1="adapter"+str(i+1)
                    adapter_list.append(adapter_name1)
                    weight_list.append(weight_num)
        
                    print(f"Loading second adapter from {lora_paths[i]}...")
                    model.load_adapter(lora_paths[i], adapter_name=adapter_name1)
        

            print("Merging adapters into a single unified adapter...")

            model.add_weighted_adapter(
                adapters=adapter_list, 
                weights=weight_list, 
                adapter_name="merged_adapter", 
                combination_type="dare_ties",
                density=density_num
            )

            print(lora_paths)
            print(adapter_list)
            print(weight_list)
            
            #Set the merged adapter as the active one for evaluation
            model.set_adapter("merged_adapter")
            print("Successfully merged and set 'merged_adapter' as active.")
        else:
        
            print("Loaded model without adapter")
            model = base_model

        
        model.eval()
        
        return model, tokenizer

    except FileNotFoundError:
        print(f"\nERROR: Model path not found. Please verify MODEL_PATH: {MODEL_PATH}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR during model loading: {e}")
        print("Ensure 'bitsandbytes', 'accelerate', and 'peft' are installed and CUDA is available.")
        sys.exit(1)

# --- Chat Functionality ---
def chat_loop(model, tokenizer):
    print("\n" + "="*80)
    print(f"🚀 Chat Session Started (Persona: {TRAINING_ROLE})")
    print(f"Model Parameters: Max Tokens={MAX_NEW_TOKENS}, Temp={TEMPERATURE}")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*80)
    
    # Initialize the chat history
    chat_history = [
        {"role": "system", "content": f"You are an {TRAINING_ROLE}. Answer all questions in this persona."},
        {"role": "assistant", "content": f"Hello"}
    ]
    
    print(f"\nAssistant: {chat_history[-1]['content']}")
    
    # Store the entire conversation's input_ids for caching across turns
    # Initialize with the system message and first assistant greeting
    full_input_ids = tokenizer.apply_chat_template(
        chat_history,
        tokenize=True,
        add_generation_prompt=False, # We don't want a prompt here yet
        return_tensors="pt"
    ).to(model.device)
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nAssistant: Session ended. Goodbye!")
                break
                
            if not user_input.strip():
                continue

            # 1. Add user message to history
            chat_history.append({"role": "user", "content": user_input})

            # 2. Tokenize the NEWEST turn only, appended to the previous full input
            # We ONLY want the tokens for the new user message plus the Llama prompt structure
            # To preserve the cache, we must concatenate the new input IDs to the old ones.
            # We re-encode the *full* history to ensure the correct chat template markers are present,
            # but then we use the full_input_ids to track which parts are already in the cache.
            
            new_input_ids = tokenizer.apply_chat_template(
                chat_history, 
                tokenize=True, 
                add_generation_prompt=True, # Tells the model to generate the next assistant turn
                return_tensors="pt"
            ).to(model.device)
            
            # The model's internal cache mechanism will automatically ignore the tokens 
            # already processed in the previous step, provided use_cache=True.
            # By passing the *new, full* input_ids, the model re-computes the initial context
            # but uses the cached Key/Value states for the *previous* tokens, which is faster.
            # This is the standard and most robust way to use `use_cache` for chat.

            # 3. Generate response
            with torch.no_grad():
                outputs = model.generate(
                    new_input_ids, # Pass the full, newly tokenized history
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K, 
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True # Ensures the model avoids re-computation of old tokens
                )

            # 4. Decode the new generated response (excluding the input prompt)
            generated_text = tokenizer.decode(outputs[0][new_input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            # 5. Add assistant response to history
            chat_history.append({"role": "assistant", "content": generated_text})
            
            # 6. Print the assistant's response
            print(f"\nAssistant: {generated_text}")
            
            # 7. Update the full_input_ids for the next turn's caching reference (optional, as the model handles it)
            # full_input_ids = outputs[0].unsqueeze(0) # Store the entire sequence for potential future optimization

        except KeyboardInterrupt:
            print("\nAssistant: Session ended by user interrupt. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")
            chat_history.pop() # Clean history on failure
            break

if __name__ == "__main__":
    model, tokenizer = initialize_model()
    chat_loop(model, tokenizer)

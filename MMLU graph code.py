import torch
import json
import os
import sys
import glob 
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
import json
import csv
import time
import math
import random
import numpy as np

config={}


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic=True


IS_LOAD_BASE_MODEL=False

#Loading the setting and paths
with open('setting.json', 'r') as file:
    config = json.load(file)


# --- Configuration ---

MODEL_PATH = config.get("MODEL_PATH")


model_display_name=""

weight_num=2.0


if(IS_LOAD_BASE_MODEL==True):
    model_display_name=os.path.basename(MODEL_PATH)
else:
    model_display_name=os.path.basename(MODEL_PATH)+" loaded with finetuned loras merged with weight "+str(weight_num)



ADAPTER_PATH = ""
#ADAPTER_PATH = ""

DATA_DIR = "./MMLU_data/"

GRAPH_FOLDER = "./MMLU_result_graph"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 10


math_sections=[
    "abstract_algebra_test.csv",
    "college_mathematics_test.csv",
    "high_school_mathematics_test.csv",

    ]
MAX_NEW_TOKENS_MATH = 200



json_output={}

sum_result_accuracy=0.0

sum_result_notanswered=0.0



def initialize_model():
    print(f"Loading tokenizer and model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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



    # 1. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 # Ensures computational dtype is bfloat16
    )

    if(IS_LOAD_BASE_MODEL==True):
        model = base_model
        return tokenizer, model




    lora_paths=config["DECIDED_LORA_PATHS"]

    
    adapter_list=[]
    weight_list=[]

    if(len(lora_paths)>=1 and os.path.exists(lora_paths[0])):
        print(f"Loading adapters from {lora_paths[0]}...")
        model = PeftModel.from_pretrained(base_model, lora_paths[0], adapter_name="adapter1")
        adapter_list.append("adapter1")

        density_num=0.8
        weight_list.append(weight_num)


        for i in range(1, len(lora_paths), 1):


            if os.path.exists(lora_paths[i]):
                adapter_name1="adapter"+str(i+1)
                adapter_list.append(adapter_name1)
                weight_list.append(weight_num)
        
                print(f"Loading second adapter from {lora_paths[i]}...")
                model.load_adapter(lora_paths[i], adapter_name=adapter_name1)
        

        print("Merging adapters into a single unified adapter...")
        if(len(lora_paths))>1:
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
        elif(len(lora_paths)==1):
            print("Loaded adapter from", lora_paths[0])
    else:
        
        print("Loaded model without adapter")
        model = base_model

        
    model.eval()
    return tokenizer, model


def run_evaluation(file_path, tokenizer, model):

    global sum_result_accuracy
    global sum_result_notanswered
    
    file_name = os.path.basename(file_path)
    

    #Making folder for each subsection
    base_name = os.path.basename(file_name)
    subsection_folder = os.path.splitext(base_name)[0]
    subsection_folder = GRAPH_FOLDER+"/"+subsection_folder

    print(subsection_folder)

    
    if not os.path.exists(subsection_folder):
        # Create the directory
        os.makedirs(subsection_folder)
        print(f"Creating folder: {subsection_folder}")
    
    data=[]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader=csv.reader(f)
        
        data = [row for row in reader]

    length=len(data)

    print(length)
    #print(data[0])

    subsection_name = os.path.splitext(base_name)[0]
    json_output[subsection_name]={}

    count=0

    result={"total":0, "correct":0, "wrong":0, "notanswered":0}

    #MMLU csv file format
    #0-Question 1-A 2-B 3-C 4-D 5-Answer Option in ABCD form
    

    while count<length:
        i=0
        print(i+count)

        prompts=[]

        while i<BATCH_SIZE and i+count<length:

            #A B C D is used as options
            prompt="This is a MCQ Question. "
            prompt=prompt+data[i+count][0]
            prompt=prompt+"Choose from the following options "
            prompt=prompt+" A. "+data[i+count][1]
            prompt=prompt+" B. "+data[i+count][2]
            prompt=prompt+" C. "+data[i+count][3]
            prompt=prompt+" D. "+data[i+count][4]
            prompt=prompt+" The correct option is "
    
            #print(prompt)

            prompts.append(prompt)
            i=i+1

        

        #print(prompts)
    
    
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            if(base_name in math_sections):
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_MATH, do_sample=False)
            else:
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)


        i=0
        while i<BATCH_SIZE and i+count<length:
            prediction_text = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            #print(prediction_text)

            predicted_option=""
            
            result["total"]=result["total"]+1



            letter=prediction_text[0]
            
            if(letter.lower()=="a"):
                predicted_option="A"
            elif(letter.lower()=="b"):
                predicted_option="B"
            elif(letter.lower()=="c"):
                predicted_option="C"
            elif(letter.lower()=="d"):
                predicted_option="D"


            #print(data[i+count][5])

            if(predicted_option==""):
                #Checking second letter for in case like (A)
                letter=prediction_text[1]

                if(letter.lower()=="a"):
                    predicted_option="A"
                elif(letter.lower()=="b"):
                    predicted_option="B"
                elif(letter.lower()=="c"):
                    predicted_option="C"
                elif(letter.lower()=="d"):
                    predicted_option="D"

            
            

            # Pattern breakdown:
            # \b      : Word boundary (ensures it's not the 'a' in 'Paris')
            # [a-c]   : Character class (matches a, b, c)
            # [.\)]   : Matches either a literal period or a closing parenthesis
            pattern = r"\b[a-d][.\)]"
            
            #pattern = r"\b[a-d]\b"
            matches = re.findall(pattern, prediction_text, re.IGNORECASE)

            if(len(matches)>=1 and predicted_option==""):

                #print(matches[0])
                
                strings1=matches[0]
                letter=strings1[0]
                
                if(letter.lower()=="a"):
                    predicted_option="A"
                elif(letter.lower()=="b"):
                    predicted_option="B"
                elif(letter.lower()=="c"):
                    predicted_option="C"
                elif(letter.lower()=="d"):
                    predicted_option="D"
                
                

            if(predicted_option==""):

                # Pattern breakdown:
                # \b      : Word boundary
                # \d+     : One or more digits (handles 1, 2, 10, 100, etc.)
                # [.\)]   : Matches either a period or a closing parenthesis
                pattern = r"\b\d+[.\)]"

                matches = re.findall(pattern, prediction_text)
                #print(matches)  # Output: ['1.', '2.', '3)']

                
                if(len(matches)>=1):
                    
                    strings2=matches[0]
                    numstring=strings2[0]
                    
                    if(numstring[0]=="1"):
                        predicted_option="A"
                    elif(numstring[0]=="2"):
                        predicted_option="B"
                    elif(numstring[0]=="3"):
                        predicted_option="C"
                    elif(numstring[0]=="4"):
                        predicted_option="D"

                    
            


            print(predicted_option, data[i+count][5])

            
        
            if(predicted_option==""):
                result["notanswered"]=result["notanswered"]+1

            elif(predicted_option==data[i+count][5]):
                result["correct"]=result["correct"]+1
            else:
                result["wrong"]=result["wrong"]+1

            i=i+1
        

        count=count+i
        
    
        
    accuracy_percentage=result["correct"]/result["total"]*100
    notanswered_percentage=result["notanswered"]/result["total"]*100
    print(result)
    print("accuracy percentage", accuracy_percentage)
    print("notanswered percentage", notanswered_percentage)

    json_output[subsection_name]["accuracy_percentage"]=accuracy_percentage
    json_output[subsection_name]["notanswered_percentage"]=notanswered_percentage

    sum_result_accuracy=sum_result_accuracy+accuracy_percentage
    sum_result_notanswered=sum_result_notanswered+notanswered_percentage


    # --- PLOTTING ---
    labels = ['Accuracy Percentage']
    values = [accuracy_percentage]
    colors = ['#98fb98']
    bars = plt.bar(labels, values, color=colors)


    # Ensure the y-axis has some headroom for the labels
    plt.ylim(0, max(values)+10)

    plt.bar_label(bars, fmt='%.2f%%', padding=5)


    # Logic to include Adapter or Model name in title
    plt.title(f'MMLU Metrics: {file_name}\nModel: {model_display_name}')

    graph_filename = file_name.replace(".csv", "_graph_accuracy.png")
    plt.savefig(os.path.join(subsection_folder, graph_filename), bbox_inches='tight')
    plt.close()

    

    # --- PLOTTING ---
    labels = ['Not Answered Percentage']
    values = [notanswered_percentage]
    colors = ['#98fb98']
    bars = plt.bar(labels, values, color=colors)

    # Ensure the y-axis has some headroom for the labels
    plt.ylim(0, max(values)+10)

    plt.bar_label(bars, fmt='%.2f%%', padding=5)
    

    # Logic to include Adapter or Model name in title
    plt.title(f'MMLU Metrics: {file_name}\nModel: {model_display_name}')

    graph_filename = file_name.replace(".csv", "_graph_not_answered.png")
    plt.savefig(os.path.join(subsection_folder, graph_filename), bbox_inches='tight')
    plt.close()





def main():
    if not os.path.exists(GRAPH_FOLDER):
        os.makedirs(GRAPH_FOLDER)

    tokenizer, model = initialize_model()
    #Getting all csv file paths for MMLU test
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

    count=0
    
    for file_path in csv_files:
        print(file_path)

        if(count>=0):
            run_evaluation(file_path, tokenizer, model)
        count=count+1

    


    average_result_accuracy=sum_result_accuracy/len(csv_files)
    average_result_notanswered=sum_result_notanswered/len(csv_files)

    json_output["average"]={}
    
    json_output["average"]["accuracy_percentage"]=average_result_accuracy
    json_output["average"]["notanswered_percentage"]=average_result_notanswered


    subsection_folder="average"

    subsection_folder = GRAPH_FOLDER+"/"+subsection_folder

    
    if not os.path.exists(subsection_folder):
        # Create the directory
        os.makedirs(subsection_folder)
        print(f"Creating folder: {subsection_folder}")


    

    # --- PLOTTING ---
    labels = ['All Category Average Accuracy Percentage']
    values = [average_result_accuracy]
    colors = ['#98fb98']
    bars = plt.bar(labels, values, color=colors)


    # Ensure the y-axis has some headroom for the labels
    plt.ylim(0, max(values)+10)

    plt.bar_label(bars, fmt='%.2f%%', padding=5)


    # Logic to include Adapter or Model name in title
    plt.title(f'MMLU Metrics:  All Category Average \nModel: {model_display_name}')

    graph_filename = "average"+"_graph_accuracy.png"
    plt.savefig(os.path.join(subsection_folder, graph_filename), bbox_inches='tight')
    plt.close()

    

    # --- PLOTTING ---
    labels = ['All Category Average Not Answered Percentage']
    values = [average_result_notanswered]
    colors = ['#98fb98']
    bars = plt.bar(labels, values, color=colors)

    # Ensure the y-axis has some headroom for the labels
    plt.ylim(0, max(values)+10)

    plt.bar_label(bars, fmt='%.2f%%', padding=5)
    

    # Logic to include Adapter or Model name in title
    plt.title(f'MMLU Metrics: All Category Average \nModel: {model_display_name}')

    graph_filename = "average"+"_graph_not_answered.png"
    plt.savefig(os.path.join(subsection_folder, graph_filename), bbox_inches='tight')
    plt.close()




    json_file_path=GRAPH_FOLDER+"/"
    json_file_path=json_file_path+"MMLU_test_result.json"

    with open(json_file_path, "w") as f:
        json.dump(json_output, f, indent=4)


    

if __name__ == "__main__":
    main()






import json
import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd
from collections import Counter

from llava.model.builder import load_pretrained_model, load_senna_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
from peft import PeftModel

def get_model_response(model, tokenizer, image_processor, image_path, question):
    """
    Queries the VLM with an image and question and returns the text response.
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    conv = conv_templates["vicuna_v1"].copy() # Use the conv template you trained with
    # prompt = f"<image>\n{question}"
    prompt = question
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=16 # We only need a short answer
        )
    
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def clean_prediction(prediction_text, valid_actions):
    """
    Finds the first valid action command in the model's output text.
    """
    for action in valid_actions:
        if action in prediction_text.upper():
            return action
    return "UNKNOWN" # Return if no valid action is found

def evaluate_actions(lora_path, model_base_path, eval_data_path, save_path):
    ACTIONS = ["DRIVE_TO_PILE", "DIG", "DUMP", "WAIT"]
    ACTION_QUESTION_SUBSTRING = "what is the most logical next action"

    print("Loading base model...")
    model_name = get_model_name_from_path(model_base_path)
    tokenizer, model, image_processor, context_len = load_senna_pretrained_model(
        model_base_path, None, model_name='llava', device_map=0)
    

    ##### new, loading lora adapters #######
    if lora_path != None:

        print(f"Applying LoRA adapters from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

        print("Merging LoRA weights with the base model...")
        model = model.merge_and_unload()
    
    else:
        print("No LoRA path provided, using base model only.")

    model.eval()

    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)

    ground_truths = []
    predictions = []
    failed_cases = []

    print(f"Evaluating {len(eval_data)} samples...")
    for sample in tqdm(eval_data):
        question = sample['conversations'][0]['value']
        
        if ACTION_QUESTION_SUBSTRING in question:
            image_path = sample['image']
            gt_answer = sample['conversations'][1]['value'].strip()
            
            pred_text = get_model_response(model, tokenizer, image_processor, image_path, question)
            
            cleaned_pred = clean_prediction(pred_text, ACTIONS)
            
            ground_truths.append(gt_answer)
            predictions.append(cleaned_pred)

            if gt_answer != cleaned_pred:
                failed_cases.append({
                    "image": image_path,
                    "ground_truth": gt_answer,
                    "prediction": cleaned_pred,
                    "raw_output": pred_text
                })

    if not ground_truths:
        print("No action-related questions found in the evaluation data. Exiting.")
        return

    accuracy = accuracy_score(ground_truths, predictions)
    report = classification_report(ground_truths, predictions, labels=ACTIONS, zero_division=0)
    conf_matrix = confusion_matrix(ground_truths, predictions, labels=ACTIONS)
    
    conf_matrix_df = pd.DataFrame(conf_matrix, index=ACTIONS, columns=ACTIONS)

    print("\n--- Evaluation Results ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Prediction):")
    print(conf_matrix_df)
    
    results_summary = {
        "overall_accuracy": accuracy,
        "classification_report": classification_report(ground_truths, predictions, labels=ACTIONS, zero_division=0, output_dict=True),
        "confusion_matrix": conf_matrix_df.to_dict(),
        "failed_cases": failed_cases
    }

    with open(save_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\n✅ Evaluation results saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned VLM on the action prediction task.")
    parser.add_argument("--lora-path", type=str, default="/home/hestia-22/Desktop/Heavy-Machinery-Autonomous-Navigation-with-VLMs/lora_output/", 
                        help="Path to the fine-tuned LoRA checkpoint folder.")
    parser.add_argument("--model-base-path", type=str, default="/home/hestia-22/Senna/data/huggingface", help="Path to the fine-tuned model checkpoint (e.g., your lora_output folder).")
    parser.add_argument("--eval-data-path", type=str, default="/home/hestia-22/Desktop/Heavy-Machinery-Autonomous-Navigation-with-VLMs/output_data/eval_dataset.json", help="Path to the llama_format_dataset.json file.")
    parser.add_argument("--save-path", type=str, default="evaluation_results.json", help="Path to save the detailed evaluation results.")
    args = parser.parse_args()

    evaluate_actions(args.lora_path, args.model_base_path, args.eval_data_path, args.save_path)
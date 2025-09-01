import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from glob import glob
from accelerate import init_empty_weights

from llava_next.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava_next.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava_next.llava.conversation import conv_templates, SeparatorStyle
from llava_next.llava.model.builder import load_pretrained_model
from llava_next.llava.utils import disable_torch_init

# --- LLaVA Model Loading and Querying Functions ---

def load_llava_model():
    """
    Loads the LLaVA model.
    Update the 'model_path' to your specific directory.
    """
    model_path = "/home/hestia-22/QA_data_generation_CARLA/data/llava34b"
    print(f"Loading LLaVA model from: {model_path}")
    model_name = "llava-v1.6-34b"
    
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="auto",
            load_4bit=True,
            attn_implementation=None
        )
        model.eval()
        return tokenizer, model, image_processor
    except Exception as e:
        print(f"Error loading LLaVA model: {e}")
        raise

def query_llava(query_text, image_path, tokenizer, model, image_processor):
    """
    Queries the LLaVA model with an image and a text prompt.
    """
    try:
        disable_torch_init()
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [t.to(dtype=torch.float16, device=model.device) for t in image_tensor]

        image_sizes = [image.size]
        
        conv = conv_templates["chatml_direct"].copy()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + query_text
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        
        input_ids = tokenizer_image_token(
            conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=512
            )

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response

    except Exception as e:
        print(f"Error querying LLaVA for image {os.path.basename(image_path)}: {e}")
        return f"Error processing query: {e}"

# --- QA Generation Functions ---

def generate_localization_qa(image_path, tokenizer, model, image_processor):
    """
    Generates QA pairs for identifying and locating piles of material.
    """
    question = "From the perspective of a wheel loader operator, where is the most prominent pile of material (like sand, gravel, or dirt) that needs to be moved? Describe its position in the image (e.g., center, front-left, far-right)."
    answer = query_llava(question, image_path, tokenizer, model, image_processor)
    return {"question": question, "answer": answer}

def generate_action_qa(image_path, tokenizer, model, image_processor):
    """
    Generates QA pairs for predicting the next action.
    """
    question = "You are operating a wheel loader. Based on the image, what is the most logical next action to perform? Your options are: DRIVE_TO_PILE, DIG, DUMP, or WAIT. The answer should be just the action command."
    
    prompt_for_vlm = (
        "Analyze the image from the point of view of a wheel loader operator. "
        "Determine the next logical action. Is there a clear pile to approach? Then the action is DRIVE_TO_PILE. "
        "Is the loader directly in front of and close to a pile? Then the action is DIG. "
        "Is the loader's bucket full and positioned over a truck or designated area? Then the action is DUMP. "
        "Are there obstacles, people, or no clear task? Then the action is WAIT. "
        "Respond with only one of the following commands: DRIVE_TO_PILE, DIG, DUMP, WAIT."
    )
    answer = query_llava(prompt_for_vlm, image_path, tokenizer, model, image_processor)
    valid_answers = ["DRIVE_TO_PILE", "DIG", "DUMP", "WAIT"]
    cleaned_answer = next((cmd for cmd in valid_answers if cmd in answer.upper()), "WAIT")
    
    return {"question": question, "answer": cleaned_answer}

def generate_scene_description_qa(image_path, tokenizer, model, image_processor):
    """
    Generates a general description of the construction scene.
    """
    question = "Describe the current construction site environment from the wheel loader's point of view. Mention the type of terrain, visible machinery, and overall activity level."
    answer = query_llava(question, image_path, tokenizer, model, image_processor)
    return {"question": question, "answer": answer}

def generate_hazard_qa(image_path, tokenizer, model, image_processor):
    """
    Generates QA pairs for identifying potential hazards.
    """
    question = "Are there any potential safety hazards or obstacles visible in this image that the wheel loader operator should be aware of, such as other vehicles, people, or unstable ground?"
    answer = query_llava(question, image_path, tokenizer, model, image_processor)
    return {"question": question, "answer": answer}

# --- Main Processing Function ---

def process_images_and_generate_qa(image_folder, output_dir, tokenizer, model, image_processor, max_images=None):
    """
    Main processing pipeline. Iterates through images, saves a JSON for each,
    and then saves a consolidated JSON file at the end.
    """
    print(f"Searching for images in: {image_folder}")
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg"))) + \
                  sorted(glob(os.path.join(image_folder, "*.png")))
    
    if not image_paths:
        print("Error: No images found in the specified folder.")
        return

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Found {len(image_paths)} images to process.")
    
    individual_frames_dir = os.path.join(output_dir, "individual_frames")
    os.makedirs(individual_frames_dir, exist_ok=True)
        
    all_qa_data = []

    for image_path in tqdm(image_paths, desc="Generating QA data"):
        try:
            qa_pairs = []
            
            qa_pairs.append(generate_scene_description_qa(image_path, tokenizer, model, image_processor))
            qa_pairs.append(generate_localization_qa(image_path, tokenizer, model, image_processor))
            qa_pairs.append(generate_action_qa(image_path, tokenizer, model, image_processor))
            qa_pairs.append(generate_hazard_qa(image_path, tokenizer, model, image_processor))
            
            entry = {
                "image_file": os.path.basename(image_path),
                "qa_pairs": qa_pairs
            }
            
            image_basename = os.path.basename(image_path)
            json_filename = os.path.splitext(image_basename)[0] + '.json'
            individual_json_path = os.path.join(individual_frames_dir, json_filename)
            with open(individual_json_path, 'w') as f:
                json.dump(entry, f, indent=4)
            
            all_qa_data.append(entry)
            
        except Exception as e:
            print(f"Failed to process {os.path.basename(image_path)}: {e}")
            
    final_json_path = os.path.join(output_dir, "_all_frames.json")
    with open(final_json_path, 'w') as f:
        json.dump(all_qa_data, f, indent=4)
        
    print(f"\nProcessing complete. Generated QA data for {len(all_qa_data)} images.")
    print(f"Individual frame JSONs saved in: {individual_frames_dir}")
    print(f"Consolidated JSON saved to: {final_json_path}")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Generate QA data for VLM fine-tuning using images of construction sites.")
    parser.add_argument("--image_folder", type=str, required=True, 
                        help="Path to the folder containing your JPG or PNG images.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to the directory where output JSON files will be saved.")
    parser.add_argument("--max_images", type=int, default=None, 
                        help="Optional: Maximum number of images to process.")
    args = parser.parse_args()
    
    # Load the VLM
    tokenizer, model, image_processor = load_llava_model()
    
    process_images_and_generate_qa(
        args.image_folder,
        args.output_dir,
        tokenizer,
        model,
        image_processor,
        args.max_images
    )

if __name__ == "__main__":
    main()
import json
import os
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split

def stratified_split_dataset(input_file, output_dir, test_size=0.2, random_state=42):
    """
    Splits the LLaMA-format dataset while preserving the distribution of action commands.
    """
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    action_question_substring = "what is the most logical next action"
    
    images_data = {}
    for conv in data:
        image_file = conv.get("image")
        if not image_file:
            continue
        
        if image_file not in images_data:
            images_data[image_file] = {"conversations": [], "action_label": "UNKNOWN"}
        
        images_data[image_file]["conversations"].append(conv)
        
        question = conv["conversations"][0]["value"]
        if action_question_substring in question:
            answer = conv["conversations"][1]["value"].strip()
            images_data[image_file]["action_label"] = answer

    image_paths = list(images_data.keys())
    action_labels = [images_data[p]["action_label"] for p in image_paths]

    print(f"Performing stratified split with {test_size*100:.0f}% for evaluation...")
    train_paths, eval_paths, _, _ = train_test_split(
        image_paths, 
        action_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=action_labels  # This is the key to preserving distribution
    )

    train_dataset = []
    for path in train_paths:
        train_dataset.extend(images_data[path]["conversations"])

    eval_dataset = []
    for path in eval_paths:
        eval_dataset.extend(images_data[path]["conversations"])

    os.makedirs(output_dir, exist_ok=True)
    train_output_path = os.path.join(output_dir, "train_dataset.json")
    eval_output_path = os.path.join(output_dir, "eval_dataset.json")

    with open(train_output_path, 'w') as f:
        json.dump(train_dataset, f, indent=4)
    with open(eval_output_path, 'w') as f:
        json.dump(eval_dataset, f, indent=4)

    print("\n✅ Split complete!")
    print(f"Training data saved to: {train_output_path}")
    print(f"Evaluation data saved to: {eval_output_path}")

    print("\n--- Action Distribution Verification ---")
    train_actions = [conv['conversations'][1]['value'].strip() for conv in train_dataset if action_question_substring in conv['conversations'][0]['value']]
    eval_actions = [conv['conversations'][1]['value'].strip() for conv in eval_dataset if action_question_substring in conv['conversations'][0]['value']]
    
    print(f"\nOriginal Distribution ({len(action_labels)} samples):")
    for action, count in Counter(action_labels).items():
        print(f"- {action}: {count} ({count/len(action_labels)*100:.1f}%)")

    print(f"\nTraining Set Distribution ({len(train_actions)} samples):")
    for action, count in Counter(train_actions).items():
        print(f"- {action}: {count} ({count/len(train_actions)*100:.1f}%)")

    print(f"\nEvaluation Set Distribution ({len(eval_actions)} samples):")
    for action, count in Counter(eval_actions).items():
        print(f"- {action}: {count} ({count/len(eval_actions)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a LLaMA-format dataset with stratification.")
    parser.add_argument("--input-file", type=str, default="/home/hestia-22/Desktop/Heavy-Machinery-Autonomous-Navigation-with-VLMs/output_data/llama_format_dataset.json",
                        help="Path to the source JSON file.")
    parser.add_argument("--output-dir", type=str, default="/home/hestia-22/Desktop/Heavy-Machinery-Autonomous-Navigation-with-VLMs/output_data",
                        help="Directory to save the split train and eval files.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of the dataset to allocate to the evaluation set (e.g., 0.2 for 20%).")
    args = parser.parse_args()

    stratified_split_dataset(args.input_file, args.output_dir, args.test_size)
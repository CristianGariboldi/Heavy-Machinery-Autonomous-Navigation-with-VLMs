import json
import uuid
import os
import argparse

def convert_to_llama_format(src_path, dst_path, image_base_path):
    """
    Converts the single-camera QA dataset to the LLaMA fine-tuning format.
    """
    try:
        with open(src_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at '{src_path}'")
        return

    output_data = []
    
    print(f"Processing {len(data)} frames...")
    for item in data:
        image_filename = item.get("image_file")
        if not image_filename:
            continue # Skip if there's no image file listed

        full_image_path = os.path.join(image_base_path, image_filename)

        scene_token = os.path.splitext(image_filename)[0]

        qa_pairs = item.get("qa_pairs", [])
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            
            conversation_prompt = f"<image>\n{question}"

            unique_id = uuid.uuid4().hex

            output_data.append({
                "id": unique_id,
                "image": full_image_path, # Path to the single image
                "conversations": [
                    {
                        "from": "human",
                        "value": conversation_prompt
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
                "scene_token": scene_token
            })

    with open(dst_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"✅ Success! Converted {len(data)} frames into {len(output_data)} conversations.")
    print(f"Output saved to '{dst_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert QA dataset to LLaMA fine-tuning format.")
    parser.add_argument("--src-file", type=str, default="./output_data/_all_frames.json",
                        help="Source JSON file path (e.g., _all_frames.json).")
    parser.add_argument("--dst-file", type=str, default="./output_data/llama_format_dataset.json",
                        help="Destination LLaMA-format JSON file path.")
    parser.add_argument("--image-base-path", type=str, default="./video_frames_reduced",
                        help="The base directory where the frame images are stored (e.g., /path/to/video_frames_reduced).")
    
    args = parser.parse_args()

    convert_to_llama_format(args.src_file, args.dst_file, args.image_base_path)
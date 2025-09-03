import json
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def wrap_text(text, font, font_scale, thickness, max_width):
    """
    Wraps text to fit within a specified width.
    """
    lines = []
    words = text.split(' ')
    if not words:
        return []
    current_line = words[0]

    for word in words[1:]:
        size = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0]
        if size[0] < max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def draw_text_with_background(frame, text, position, font, font_scale, font_color, bg_color, thickness):
    """
    Draws text with a semi-transparent background for better readability.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    y1, y2 = y - text_h - baseline, y + baseline
    x1, x2 = x, x + text_w
    
    y1 = max(y1, 0)
    y2 = min(y2, frame.shape[0])
    x1 = max(x1, 0)
    x2 = min(x2, frame.shape[1])

    sub_img = frame[y1:y2, x1:x2]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    
    res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
    frame[y1:y2, x1:x2] = res

    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)


def create_annotated_video(json_file, output_video, fps=5):
    """
    Creates a video by overlaying text from a LLaMA-format JSON file onto image frames.
    """
    print(f"Loading and processing data from {json_file}...")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file}'")
        return

    grouped_annotations = {}
    action_q_sub = "what is the most logical next action"
    hazard_q_sub = "any potential safety hazards"

    for conv_item in data:
        image_path = conv_item.get("image")
        if not image_path:
            continue
        
        if image_path not in grouped_annotations:
            grouped_annotations[image_path] = {
                "action": "ACTION: UNKNOWN",
                "hazard": "HAZARDS: Not found."
            }
            
        question = conv_item["conversations"][0]["value"]
        answer = conv_item["conversations"][1]["value"]
        
        if action_q_sub in question:
            grouped_annotations[image_path]['action'] = f"ACTION: {answer.strip()}"
        elif hazard_q_sub in question:
            grouped_annotations[image_path]['hazard'] = f"HAZARDS: {answer.strip()}"

    sorted_image_paths = sorted(grouped_annotations.keys())
    if not sorted_image_paths:
        print("Error: No valid image paths found in the JSON file.")
        return

    first_frame = cv2.imread(sorted_image_paths[0])
    if first_frame is None:
        print(f"Error: Could not read the first image file at '{sorted_image_paths[0]}'")
        return
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Creating video '{output_video}' with {len(sorted_image_paths)} frames...")
    
    for image_path in tqdm(sorted_image_paths, desc="Processing frames"):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}. Skipping frame.")
            continue
            
        frame_annotations = grouped_annotations.get(image_path)

        if frame_annotations:
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            action_text = frame_annotations['action']
            draw_text_with_background(frame, action_text, (30, 50), font, 1.2, (50, 255, 255), (0,0,0), 2)

            hazard_text_full = frame_annotations['hazard']
            wrapped_lines = wrap_text(hazard_text_full, font, 0.7, 1, width - 60)
            
            y_start = height - 20 - (len(wrapped_lines) * 25)
            for i, line in enumerate(wrapped_lines):
                y_pos = y_start + (i * 25)
                draw_text_with_background(frame, line, (30, y_pos), font, 0.7, (255, 255, 255), (0,0,0), 1)

        video_writer.write(frame)

    video_writer.release()
    print(f"\n✅ Video creation complete! Saved to '{output_video}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create an annotated video from a LLaMA-format JSON dataset.")
    parser.add_argument("--json-file", type=str, default="./output_data/eval_dataset.json", help="Path to the LLaMA-format input JSON file.")
    parser.add_argument("--output-video", type=str, default="./data/demo_video.mp4", help="Path to save the output MP4 video file.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for the output video.")
    
    args = parser.parse_args()
    
    create_annotated_video(args.json_file, args.output_video, args.fps)
import json
from collections import Counter

def count_action_distribution(json_file_path):
    """
    Reads the QA dataset, counts the distribution of actions, and prints a summary.

    Args:
        json_file_path (str): The path to the '_all_frames.json' file.
    """
    action_question_substring = "what is the most logical next action"

    try:
        with open(json_file_path, 'r') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return

    actions_list = []
    for frame in all_frames_data:
        for qa_pair in frame.get("qa_pairs", []):
            if action_question_substring in qa_pair.get("question", ""):
                action = qa_pair.get("answer", "UNKNOWN").strip()
                if action:  # Make sure the answer isn't an empty string
                    actions_list.append(action)
                break
    
    if not actions_list:
        print("No actions found for the specified question.")
        return
        
    action_counts = Counter(actions_list)
    total_actions = len(actions_list)

    print(" Action Command Distribution")
    print("====================================")
    
    for action, count in action_counts.most_common():
        percentage = (count / total_actions) * 100
        print(f"- {action:<15} | Count: {count:<5} ({percentage:.1f}%)")
        
    print("------------------------------------")
    print(f"Total Actions Counted: {total_actions}")


if __name__ == "__main__":
    dataset_file = "./output_data/_all_frames.json"
    count_action_distribution(dataset_file)
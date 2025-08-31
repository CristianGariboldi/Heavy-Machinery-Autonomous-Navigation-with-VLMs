import os
import json
import math
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from glob import glob
from accelerate import init_empty_weights

# LLaVA components (ensure you have these installed)
from llava_next.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava_next.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava_next.llava.conversation import conv_templates, SeparatorStyle
from llava_next.llava.model.builder import load_pretrained_model
from llava_next.llava.utils import disable_torch_init

# Constants
FUT_TS = 18  # Number of future timesteps to consider
CAMERA_NAMES = [
    'front', 'front_left', 'front_right',
    'back', 'back_left', 'back_right'
]
CAMERA_MAPPING = {cam: f"rgb_multi_{cam}" for cam in CAMERA_NAMES}
CATEGORY_MAP = {
    'car': 'car',
    'pedestrian': 'pedestrian',
    'bicycle': 'cyclist',
    'motorcycle': 'motorcycle',
    'pole': 'traffic_cone',
    'obstacle': 'obstacle',
    'trafficsign': 'traffic_sign'
}

# Coordinate Transformation Functions
def get_rotation_matrix(yaw):
    """
    Create a rotation matrix from yaw angle (radians).
    This matrix rotates points around the Z-axis by the given yaw angle.
    """
    return np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

def global_to_ego(p_global, R_ego, t_ego):
    """
    Transform a global position to ego-centric coordinates.
    
    Args:
        p_global: Global position (x, y, z)
        R_ego: Rotation matrix of the ego vehicle
        t_ego: Translation vector (position) of the ego vehicle
    
    Returns:
        Position in ego-centric coordinates
    """
    p_global = np.array(p_global)
    return R_ego.T @ (p_global - t_ego)


#### LLaVA Model Functions
# def load_llava_model(model_size="34b", load_8bit=True):
#     """
#     Load the LLaVA model with appropriate settings.
    
#     Args:
#         model_size: Size of the model to load ('7b' or '34b')
#         load_4bit: Whether to load in 4-bit precision to save memory
    
#     Returns:
#         Tokenizer, model, and image processor
#     """
    
#     model_path = "/home/noah-22/QA_data_generation_CARLA/data/llava34b"
#     print(f"Loading LLaVA model: {model_path}")
#     model_name = "llava-v1.6-34b"
#     try:
#         tokenizer, model, image_processor, _ = load_pretrained_model(
#             model_path,
#             None,
#             model_name,
#             device_map="auto",
#             load_4bit=True, # Quantization is commented-out, I just need this line to run it with our resources in the lab
#             attn_implementation=None
#         )
#         return tokenizer, model, image_processor
#     except Exception as e:
#         print(f"Error loading LLaVA model: {e}")
#         raise



def load_llava_model(model_size="34b"):
    """
    Load the LLaVA model with appropriate settings.
    
    Args:
        model_size: Size of the model to load ('7b' or '34b')
    
    Returns:
        Tokenizer, model, and image processor
    """
    model_path = "/home/10001205699/QA_data_generation_CARLA/data/llava-v1.6-34b"
    print(f"Loading LLaVA model: {model_path}")
    model_name = "llava-v1.6-34b"
    
    try:
        # First, create an empty model using init_empty_weights
        with init_empty_weights():
            tokenizer, empty_model, image_processor, _ = load_pretrained_model(
                model_path,
                None,
                model_name,
                device_map="auto",
                attn_implementation=None
            )
        
        # # Next, load the full pretrained model (with all weights materialized)
        # _, pretrained_model, _, _ = load_pretrained_model(
        #     model_path,
        #     None,
        #     model_name,
        #     device_map="auto",
        #     attn_implementation=None
        # )
        
        # empty_model.load_state_dict(pretrained_model.state_dict(), strict=False)
        empty_model.eval()
        empty_model.tie_weights()
        # empty_model.to("cuda")
        
        return tokenizer, empty_model, image_processor
    except Exception as e:
        print(f"Error loading LLaVA model: {e}")
        raise





def query_llava(args, tokenizer, model, image_processor):
    """
    Query LLaVA model with an image and question.
    
    Args:
        args: Object containing query and img_file attributes
        tokenizer: LLaVA tokenizer
        model: LLaVA model
        image_processor: LLaVA image processor
    
    Returns:
        Generated text response
    """
    try:
        disable_torch_init()  # Ensure model is properly initialized
        image = Image.open(args.img_file).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [t.to(dtype=torch.float16, device=model.device) for t in image_tensor]

        # Get the size of the image
        imagesize = image.size  
        image_sizes = [imagesize]  
        
        conv = conv_templates["chatml_direct"].copy()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + args.query
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
                temperature=0,  # I set it to 0 like Senna does
                max_new_tokens=512
            )

        text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()  
        return text_outputs

    except Exception as e:
        print(f"Error querying LLaVA: {e}")
        return f"Error processing image: {e}"


def discover_scenario_folders(root_path):
    """
    Recursively discover all scenario folders in the dataset.
    A scenario folder is identified by having the required subdirectories:
    - At least one camera folder (e.g., rgb_multi_front)
    - bev_gt_label directory
    - ego_vehicle_data.json file
    
    Args:
        root_path: Root path of the dataset
        
    Returns:
        List of valid scenario folder paths
    """
    scenario_folders = []
    
    # Helper function to check if a folder is a valid scenario folder
    def is_valid_scenario(folder_path):
        has_camera = any(os.path.isdir(os.path.join(folder_path, CAMERA_MAPPING[cam])) 
                         for cam in CAMERA_NAMES)
        
        has_bev_gt = os.path.isdir(os.path.join(folder_path, "bev_gt_label"))
        
        has_ego_data = os.path.isfile(os.path.join(folder_path, "ego_vehicle_data.json"))
        
        return has_camera and has_bev_gt and has_ego_data
    
    # Recursively walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path):
        if is_valid_scenario(dirpath):
            scenario_folders.append(dirpath)
            dirnames[:] = []
    
    return scenario_folders


def load_json(file_path):
    """Load a JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_carla_frame(base_path, frame_id, index):
    """
    Process a single CARLA frame into a standardized format with ego-centric coordinates.
    
    Args:
        base_path: Path to the CARLA dataset
        frame_id: Frame ID to process
    
    Returns:
        Dictionary containing processed frame data
    """
    frame_str = f"{frame_id:08d}"
    
    bev_gt_path = os.path.join(base_path, "bev_gt_label", f"{frame_str}.json")
    # ego_status_path = os.path.join(base_path, "ego_vehicle_status", f"{frame_str}.json")
    imu_ego_path = os.path.join(base_path, "ego_vehicle_data.json")
    
    obj_data = load_json(bev_gt_path)
    # ego_data = load_json(ego_status_path)
    ego_imu_data = load_json(imu_ego_path)
    
    if not obj_data or not ego_imu_data:
        print(f"Skipping frame {frame_id} due to missing data")
        return None
    
    # Ego vehicle transform
    
    ego_location = [
        ego_imu_data["ego_vehicle_status"][index]["planning_status"]["pos_global"][0],
        ego_imu_data["ego_vehicle_status"][index]["planning_status"]["pos_global"][1]
    ]

    yaw = ego_imu_data["imu"][index]["orientation_eular"]["yaw"]
    
    R_current = get_rotation_matrix(yaw)
    t_current = np.array(ego_location)
    
    # Process objects
    gt_boxes = []
    gt_names = []
    gt_velocity = []
    obj_ids = []
    
    # Combine vehicle_pedestrian and obstacle lists
    all_objects = []
    if "object_info" in obj_data:
        if "vehicle_pedestrian" in obj_data["object_info"]:
            all_objects.extend(obj_data["object_info"]["vehicle_pedestrian"])
        if "obstacle" in obj_data["object_info"]:
            all_objects.extend(obj_data["object_info"]["obstacle"])
    
    for obj in all_objects:
        # Get object local position
        obj_location = [
            obj.get("location3D", [0, 0])[0],
            obj.get("location3D", [0, 0])[1] # Don't know why, but ego and other agents have opposite reference for y direction
        ]
        
        # Get object dimensions and rotation
        dims = obj.get("dimention", [1, 1, 1])  # Note: CARLA typo in "dimention"
        obj_yaw = obj.get("rotation", 0)[2]  # Extract yaw from rotation
        
        gt_boxes.append([
            obj_location[0], obj_location[1],  # Position (x, y)
            dims[0], dims[1],     # Dimensions (x, y)
            obj_yaw                         # Yaw
        ])
        
        # Get category name, normalize using mapping
        category = obj.get("category", "unknown").lower()
        gt_names.append(CATEGORY_MAP.get(category, category))
        
        # Get velocity and transform to ego frame
        v_global = obj.get("velocity3D", [0, 0, 0])[:2]  # Take only x,y components IS THIS ABSOLUTE OR RELATIVE?
        gt_velocity.append(v_global)
        obj_ids.append(obj.get("id", f"obj_{len(obj_ids)}"))
    
    # Process ego vehicle features
    ego_features = {
        'speed': ego_imu_data.get('ego_vehicle_status', [])[index].get('planning_status', {}).get('speed', 0),
        'velocities': [
            ego_imu_data.get('ego_vehicle_status', [])[index].get('velocity', {}).get('x', 0),
            ego_imu_data.get('ego_vehicle_status', [])[index].get('velocity', {}).get('y', 0)
        ],
        'steering': ego_imu_data.get('ego_vehicle_status', [])[index].get('control', {}).get('steer', 0),
        'throttle': ego_imu_data.get('ego_vehicle_status', [])[index].get('control', {}).get('throttle', 0),
        'brake': ego_imu_data.get('ego_vehicle_status', [])[index].get('control', {}).get('brake', 0),
        'acceleration': [
            ego_imu_data.get('imu', [])[index].get('linear_acceleration', {}).get('x', 0),
            ego_imu_data.get('imu', [])[index].get('linear_acceleration', {}).get('y', 0)   # these are accelerations in ego local ref frame
        ],
        'yaw': yaw,
        'location': ego_location,
        'command': ego_imu_data.get('ego_vehicle_status', [])[index].get('planning_status', {}).get('command', 0),
        'target_speed': ego_imu_data.get('ego_vehicle_status', [])[index].get('planning_status', {}).get('target_speed', 0)
    }
    
    # Get future ego trajectory
    ego_fut_trajs = []
    ego_fut_masks = []
    
    for i in range(FUT_TS):
        
        loc = [
            ego_imu_data["ego_vehicle_status"][index]["planning_status"]["route"][i][0],
            ego_imu_data["ego_vehicle_status"][index]["planning_status"]["route"][i][1]
        ]
        ego_fut_trajs.append(loc)
        ego_fut_masks.append(1)
    
    # Get future object trajectories
    gt_agent_fut_trajs = []
    gt_agent_fut_masks = []
    
    for obj_idx, obj_id in enumerate(obj_ids):
        traj = []
        masks = []
        # Get current position for calculating offsets
        p_cur = np.array(gt_boxes[obj_idx][:2])  # Current x, y
        
        for i in range(1, FUT_TS + 1):
            future_frame = f"{frame_id + i:08d}"
            future_bev_path = os.path.join(base_path, "bev_gt_label", f"{future_frame}.json")
            
            try:
                if os.path.exists(future_bev_path):
                    future_bev = load_json(future_bev_path)
                    if future_bev:
                        future_objs = {}
                        if "object_info" in future_bev:
                            if "vehicle_pedestrian" in future_bev["object_info"]:
                                for o in future_bev["object_info"]["vehicle_pedestrian"]:
                                    future_objs[o.get("id", "")] = o
                            if "obstacle" in future_bev["object_info"]:
                                for o in future_bev["object_info"]["obstacle"]:
                                    future_objs[o.get("id", "")] = o
                        
                        if obj_id in future_objs:
                            # Object found in future frame
                            fut_obj = future_objs[obj_id]
                            p_fut_global = [
                                fut_obj.get("location3D", [0, 0])[0],
                                fut_obj.get("location3D", [0, 0])[1]
                            ]
                            offset = np.array(p_fut_global[:2]) - p_cur
                            traj.append(offset.tolist())
                            masks.append(1)
                        else:
                            traj.append([0, 0])
                            masks.append(0)
                    else:
                        traj.append([0, 0])
                        masks.append(0)
                else:
                    traj.append([0, 0])
                    masks.append(0)
            except Exception as e:
                print(f"Error processing future object in frame {future_frame}: {e}")
                traj.append([0, 0])
                masks.append(0)
        
        gt_agent_fut_trajs.append(traj)
        gt_agent_fut_masks.append(masks)
    
    # Get image paths
    image_paths = {}
    for cam in CAMERA_NAMES:
        cam_path = os.path.join(base_path, CAMERA_MAPPING[cam], f"{frame_str}.jpg")
        if os.path.exists(cam_path):
            image_paths[cam] = cam_path
    
    # Compile all information
    return {
        'frame_id': frame_id,
        'timestamp': ego_imu_data.get("imu", [])[index].get('timestamp', 0),
        'gt_boxes': np.array(gt_boxes),
        'gt_names': np.array(gt_names),
        'gt_velocity': np.array(gt_velocity),
        'gt_ids': np.array(obj_ids),
        'gt_agent_fut_trajs': np.array(gt_agent_fut_trajs),
        'gt_agent_fut_masks': np.array(gt_agent_fut_masks),
        'gt_ego_fut_trajs': np.array(ego_fut_trajs),
        'gt_ego_fut_masks': np.array(ego_fut_masks),
        'ego_features': ego_features,
        'image_paths': image_paths,
        'ego_rotation': R_current,
        'ego_translation': t_current,
        'fut_valid_flag': int(np.all(ego_fut_masks))
    }



def generate_scene_description(image_paths, cam, tokenizer, model, image_processor):
    """
    Generate a scene description using only the front camera image.
    
    Args:
        image_paths: Dictionary mapping camera names to image paths.
        tokenizer, model, image_processor: LLaVA components.
    
    Returns:
        Scene description from the front camera.
    """
    
    if cam not in image_paths:
        print(f"{cam} camera image path is missing.")
        return f"{cam} camera image path is missing."
    
    img_path = image_paths[cam]
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return f"Image not found: {img_path}"
    
    try:
        args = type('Args', (), {
            "query": f"Suppose you are driving, and I am providing you with the image captured by the car's {cam} camera. Describe this driving scene in detail, including vehicles, pedestrians, road conditions, and traffic elements.",
            "img_file": img_path
        })()
        
        description = query_llava(args, tokenizer, model, image_processor)
        return description
    except Exception as e:
        print(f"Error generating description for {cam}: {e}")
        return f"Error processing image for {cam}"





def get_vru_qa(frame_data):
    """
    Generate Vulnerable Road User (VRU) QA pairs.
    
    Args:
        frame_data: Processed frame data
    
    Returns:
        List of QA pairs about VRUs
    """
    qa_pairs = []
    
    try:
        gt_boxes = frame_data["gt_boxes"]  # [x, y, z, l, w, h, yaw]
        gt_names = frame_data["gt_names"]
        
        vru_present = False
        vru_details = []
        
        for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
            # Check for pedestrians and cyclists (vulnerable road users)
            if name in ["pedestrian", "cyclist"]:
                x, y = box[0], box[1]
                dist = math.sqrt(x**2 + y**2)
                # VRU within reasonable distance (20m ahead, 5m laterally)
                if x > 0 and x < 20 and abs(y) < 5:
                    vru_present = True
                    if y < 0:
                        vru_details.append(f"{name} at {dist:.1f}m distance, {x:.1f}m ahead and {-y:.1f}m to the right")
                    else:
                        vru_details.append(f"{name} at {dist:.1f}m distance, {x:.1f}m ahead and {y:.1f}m to the left")
        # Create QA pair
        question = "Are there any pedestrians or cyclists (vulnerable road users) within 20 meters ahead of you?"
        if vru_present:
            answer = f"Yes. {'; '.join(vru_details)}"
        else:
            answer = f"No, there are no vulnerable road users in the immediate vicinity of our vehicle."
        
        qa_pairs.append({"question": question, "answer": answer})
    except Exception as e:
        print(f"Error generating VRU QA: {e}")
        qa_pairs.append({
            "question": "Are there any pedestrians or cyclists near our vehicle?",
            "answer": "Unable to determine due to processing error."
        })
    
    return qa_pairs


def get_obj_rel_position(loc_x, loc_y):

    x, y = loc_x, loc_y
    angle = math.degrees(math.atan2(y, x))

    if -35 <= angle <= 35:
        return "front"
    elif 20 <= angle <= 90:
        return "front_left"
    elif 75 <= angle <= 145:
        return "back_left"
    elif -90 <= angle <= -20:
        return "front_right"
    elif -145 <= angle <= -75:
        return "back_right"
    elif 125 <= angle <= 180 or -180 <= angle <= -125:
        return "back"  
    else:
        raise Exception("Not in any camera range!")


def get_motion_prediction_qa(frame_data, cam):
    """
    Generate motion prediction QA pairs based on future trajectories.
    
    Args:
        frame_data: Processed frame data
    
    Returns:
        List of QA pairs about predicted motion
    """
    qa_pairs = []
    
    try:
        if not frame_data["fut_valid_flag"]:
            answer = "Future trajectory data is not available for this frame."
            return answer
            
        gt_boxes = frame_data["gt_boxes"]
        gt_names = frame_data["gt_names"]
        gt_agent_fut_trajs = frame_data["gt_agent_fut_trajs"]  # [N, FUT_TS, 2]
        
        # Find significant movements
        motion_details = []
        
        for i, (box, name, fut_traj) in enumerate(zip(gt_boxes, gt_names, gt_agent_fut_trajs)):
            if name in ["car", "pedestrian", "cyclist"]:
                # Only process if we have valid future trajectory data
                if np.any(frame_data["gt_agent_fut_masks"][i]):
                    
                    valid_indices = np.where(frame_data["gt_agent_fut_masks"][i])[0]
                    if len(valid_indices) > 0:
                        last_valid_idx = valid_indices[-1]
                        final_pos = fut_traj[last_valid_idx]  # Last valid future position offset
                        
                        x_cur, y_cur = box[0], box[1]
                        x_fut, y_fut = x_cur + final_pos[0], y_cur + final_pos[1]
                        dist = math.sqrt(x_cur**2 + y_cur**2)
                        
                        dist_change = math.sqrt((final_pos[0])**2 + (final_pos[1])**2)

                        if cam != get_obj_rel_position(x_cur, y_cur): # we select objects only present in the prompted camera
                            continue
                        if dist >= 30: # we don't care about very far objects
                            continue
                        if x_cur == 0.0 and y_cur == 0.0: # we exclude ego vehicle
                            continue
                        
                        # Only include significant movements
                        if dist_change > 1: 
                            # Determine direction of movement
                            angle = math.degrees(box[4])
                            
                            if -45 <= angle <= 0 or -360 <= angle <= -315:
                                direction = "forward"
                            elif -135 <= angle < -45:
                                direction = "right"
                            elif -315 < angle <= -225:
                                direction = "left"
                            else:
                                direction = "backward"
                            
                            
                            motion_details.append(
                                f"{name} at ({x_cur:.1f}m, {y_cur:.1f}m) will move {direction} by {dist_change:.1f}m"
                            )
                        else:
                            motion_details.append(
                                f"{name} at ({x_cur:.1f}m, {y_cur:.1f}m) will not move in the next few seconds."
                            )
        
        # Create QA pair
        question = f"How will the traffic participants in the {cam} view image move in the next few seconds?"
        
        if motion_details:
            answer = "Based on trajectory predictions: " + "; ".join(motion_details)
        else:
            answer = f"No traffic participants are detected in the {cam} view image."
        
        
    except Exception as e:
        print(f"Error generating motion prediction QA: {e}")
        answer = "Unable to predict motion due to processing error."
    
    return answer

def get_planning_qa(frame_data):
    """
    Generate planning QA based on ego trajectory.
    
    Args:
        frame_data: Processed frame data
    
    Returns:
        List of QA pairs about planning decisions (meta action)
    """
    qa_pairs = []
    
    try:
        if not frame_data["fut_valid_flag"]:
            return [{
                "question": "What should the vehicle do next?",
                "answer": "Cannot determine the optimal action because future trajectory data is not available."
            }]
            
        ego_fut_trajs = frame_data["gt_ego_fut_trajs"]  # [FUT_TS, 2]
        ego_fut_masks = frame_data["gt_ego_fut_masks"]
        ego_long_acc = frame_data["ego_features"]["acceleration"][0]
        ego_long_vel = abs(frame_data["ego_features"]["speed"])
        ego_cmd_brake = frame_data["ego_features"]["brake"]
        ego_cmd_throttle = frame_data["ego_features"]["throttle"]
        ego_cmd_steer = frame_data["ego_features"]["steering"] # positive is steering right
        ego_command = frame_data["ego_features"]["command"] # PDM-Lite Logic for META-ACTION_LATERAL computation
        ego_target_speed = frame_data["ego_features"]["target_speed"] # PDM-Lite Logic for target speed
        
        # Find the last valid trajectory point
        valid_indices = np.where(ego_fut_masks)[0]
        if len(valid_indices) > 0:
            last_valid_idx = valid_indices[-1]
            final_offset = ego_fut_trajs[last_valid_idx]  # [x, y]
            
            x_offset, y_offset = final_offset[0], final_offset[1]
            dist = math.sqrt(x_offset**2 + y_offset**2)

            threshold = 1.0  # acc [m/s^2], to be checked
            threshold_vel = 3.0 # difference between target speed and current speed
            
            # Determine command based on trajectory
            if ego_long_vel < 1.0 and abs(ego_long_acc) <= threshold:
                command = "stop or proceed very slowly"
            elif abs(y_offset) >= 2.0:
                if y_offset < 0:
                    command = "turn left"
                else:
                    command = "turn right"
            else:
                if ego_long_acc > threshold:
                    command = "accelerate and go straight"
                elif ego_long_acc < -threshold:
                    command = "decelerate"
                else:
                    command = "proceed straight at current speed"
            

            if ego_target_speed <= 1.0:
                meta_action_long = "STOP"
            elif ego_target_speed > 1.0 and (ego_long_vel - ego_target_speed) < -threshold_vel:
                meta_action_long = "ACCELERATE"
            elif ego_target_speed > 1.0 and (ego_long_vel - ego_target_speed) > threshold_vel:
                meta_action_long = "DECELERATE"
            else:
                meta_action_long = "KEEP"


            # META-ACTION_LAT based on PDM-Lite Logic

            if ego_command == 1:
                meta_action_lat = "LEFT_TURN"
            elif ego_command == 2:
                meta_action_lat = "RIGHT_TURN"
            elif ego_command == 3:
                meta_action_lat = "STRAIGHT"
            elif ego_command == 4:
                meta_action_lat = "LANE_FOLLOW"
            elif ego_command == 5:
                meta_action_lat = "LEFT_CHANGE"
            elif ego_command == 6:
                meta_action_lat = "RIGHT_CHANGE"
            elif ego_command == -1:
                meta_action_lat = "VOID"

            # TO DO: LEFT_CHANGE, RIGHT_CHANGE and check META ACTION generation if is good 

            
            # question = f"Your current speed is {ego_long_vel:.2f} m/s, " \
            #         #    f"the navigation command is to '{command}', " \      this line can be neglected for consistency, to be checked
            #             "based on the understanding of the driving scene " \ 
            #             # and the navigation information " \
            #             "what is your plan for the next three seconds? " \
            #             "Please answer your SPEED plan and your PATH plan. " \
            #             "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, " \
            #             "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN and LANE_FOLLOW. " \
            #             "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."

            question = (
                f"Your current speed is {ego_long_vel:.2f} m/s, "
                # f"the navigation command is to '{command}', "
                "based on the understanding of the driving scene "
                # "and the navigation information "
                "what is your plan for the next three seconds? "
                "Please answer your SPEED plan and your PATH plan. "
                "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, "
                "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN and LANE_FOLLOW. "
                "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
            )

            
            answer = meta_action_long + ', ' + meta_action_lat + '\n'

            qa_pairs.append({"question": question, "answer": answer})
        else:
            qa_pairs.append({
                "question": "What should the vehicle do next?",
                "answer": "Cannot determine the optimal action because future trajectory data is invalid."
            })
            
    except Exception as e:
        print(f"Error generating planning QA: {e}")
        qa_pairs.append({
            "question": "What should the vehicle do next?",
            "answer": "Unable to determine due to processing error."
        })
    
    return qa_pairs

def get_planning_explanation(frame_data, image_paths, tokenizer, model, image_processor):
    """
    Generate motion planning of ego vehicle explanations (explainability)
    """
    
    img_path = image_paths["front"]
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return f"Image not found: {img_path}"

    ego_fut_trajs = frame_data["gt_ego_fut_trajs"]  # [FUT_TS, 2]
    ego_fut_masks = frame_data["gt_ego_fut_masks"]
    ego_long_acc = frame_data["ego_features"]["acceleration"][0]
    ego_long_vel = abs(frame_data["ego_features"]["speed"])
    ego_cmd_brake = frame_data["ego_features"]["brake"]
    ego_cmd_throttle = frame_data["ego_features"]["throttle"]
    ego_cmd_steer = frame_data["ego_features"]["steering"] # positive is steering right
    ego_command = frame_data["ego_features"]["command"] # PDM-Lite Logic for META-ACTION_LATERAL computation
    ego_target_speed = frame_data["ego_features"]["target_speed"] # PDM-Lite Logic for target speed

    valid_indices = np.where(ego_fut_masks)[0]
    if len(valid_indices) > 0:
        last_valid_idx = valid_indices[-1]
        final_offset = ego_fut_trajs[last_valid_idx]  # [x, y]
        
        x_offset, y_offset = final_offset[0], final_offset[1]
        dist = math.sqrt(x_offset**2 + y_offset**2)
        
        threshold = 1.0  # acc [m/s^2], to be checked
        threshold_vel = 3.0 # difference between target speed and current speed
        
        # Determine command based on trajectory
        if ego_long_vel < 1.0 and abs(ego_long_acc) <= threshold:
            command = "stop or proceed very slowly"
        elif abs(y_offset) >= 2.0:
            if y_offset < 0:
                command = "turn left"
            else:
                command = "turn right"
        else:
            if ego_long_acc > threshold:
                command = "accelerate and go straight"
            elif ego_long_acc < -threshold:
                command = "decelerate"
            else:
                command = "proceed straight at current speed"
        

        if ego_target_speed <= 1.0:
            meta_action_long = "STOP"
        elif ego_target_speed > 1.0 and (ego_long_vel - ego_target_speed) < -threshold_vel:
            meta_action_long = "ACCELERATE"
        elif ego_target_speed > 1.0 and (ego_long_vel - ego_target_speed) > threshold_vel:
            meta_action_long = "DECELERATE"
        else:
            meta_action_long = "KEEP"


        # META-ACTION_LAT based on PDM-Lite Logic

        if ego_command == 1:
            meta_action_lat = "LEFT_TURN"
        elif ego_command == 2:
            meta_action_lat = "RIGHT_TURN"
        elif ego_command == 3:
            meta_action_lat = "STRAIGHT"
        elif ego_command == 4:
            meta_action_lat = "LANE_FOLLOW"
        elif ego_command == 5:
            meta_action_lat = "LEFT_CHANGE"
        elif ego_command == 6:
            meta_action_lat = "RIGHT_CHANGE"
        elif ego_command == -1:
            meta_action_lat = "VOID"


    pedal_decision = {
        'KEEP': 'maintain the current speed',
        'ACCELERATE': 'accelerate',
        'DECELERATE': 'decelerate',
        'STOP': 'stop the car'
    }

    path_decision = {
        'RIGHT_TURN': 'turn right',
        'RIGHT_CHANGE': 'change to the right lane',
        'LEFT_TURN': 'turn left',
        'LEFT_CHANGE': 'change to the left lane',
        'STRAIGHT': 'go straight',
        'LANE_FOLLOW': 'follow the lane',
        'VOID': 'not do any action'
    }

    if meta_action_long == 'STOP':
        decision = pedal_decision[meta_action_long]
    elif meta_action_lat == 'VOID':
        decision = pedal_decision[meta_action_lat]
    else:
        decision = pedal_decision[meta_action_long] + ' and ' + path_decision[meta_action_lat]


    # question = "You are driving, " \
    #            f"your current speed is {ego_long_vel:.2f} m/s, " \
    #          #  f"and the navigation command is '{command}', " \    this line can be neglected for consinstency, to be checked
    #            "your driving decision for the next three seconds is to " \
    #            f"{decision}. " \
    #            "Based on the provided image of the driving environment, " \
    #            "explain the most likely reason for this decision in one or two concise sentence." \

    question = (
    "You are driving, "
    f"your current speed is {ego_long_vel:.2f} m/s, "
    # f"and the navigation command is '{command}', "
    "your driving decision for the next three seconds is to "
    f"{decision}. "
    "Based on the provided image of the driving environment, "
    "explain the most likely reason for this decision in one or two concise sentences."
    )

    try:
        args = type('Args', (), {
            "query": question,
            "img_file": img_path
        })()
        
        description = query_llava(args, tokenizer, model, image_processor)
        return description, question
    except Exception as e:
        print(f"Error generating description for front camera: {e}")
        return "Error processing image for front camera"



def get_traffic_rule_qa(image_paths, tokenizer, model, image_processor):
    """
    Generate QA about traffic lights detection.
    
    Args:
        image_paths: Dictionary mapping camera names to image paths.
        tokenizer, model, image_processor: LLaVA components.
    
    Returns:
        Pseudo label inferred by the LLM (possible integration of gt labels from gt data)
    """
    img_path = image_paths["front"]
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return f"Image not found: {img_path}"

    question = "Given the provided forward-facing image from a car's perspective, " \
               "identify if there is a traffic light that affects the car's behavior. " \
               "Respond with ONLY one word such as 'Red', 'Green', 'Yellow', or 'None'." \
               "For example, a correct answer format is like 'Green'."

    try:
        args = type('Args', (), {
            "query": question,
            "img_file": img_path
        })()
        
        description = query_llava(args, tokenizer, model, image_processor)
        return description
    except Exception as e:
        print(f"Error generating description for front camera: {e}")
        return "Error processing image for front camera"


def generate_drive_qa(frame_data, tokenizer, model, image_processor):
    """
    Combine all QA generation functions.
    
    Args:
        frame_data: Processed frame data
        tokenizer, model, image_processor: LLaVA components
    
    Returns:
        List of QA pairs
    """
    qa_data = []
    
    try:
        for cam in CAMERA_NAMES:
            # Generate scene description first
            scene_desc = generate_scene_description(
                frame_data['image_paths'], 
                cam,
                tokenizer, 
                model, 
                image_processor
            )
        
            # Basic scene description QA
            qa_data.append({
                "question": f"Suppose you are driving, and I am providing you with the image captured by the car's {cam} camera. Describe this driving scene in detail, including vehicles, pedestrians, road conditions, and traffic elements.",
                "answer": scene_desc
            })

        for cam in CAMERA_NAMES:
            answer = get_motion_prediction_qa(frame_data, cam)
            qa_data.append({
                "question": f"How will the traffic participants in the {cam} view image move in the next few seconds?",
                "answer": answer
            })
        
        # Add specialized QA pairs
        qa_data.extend(get_vru_qa(frame_data))
        tl_qa = get_traffic_rule_qa(frame_data['image_paths'], tokenizer, model, image_processor)
        qa_data.append({
            "question": "Given the provided forward-facing image from a car's perspective, " \
                        "identify if there is a traffic light that affects the car's behavior. " \
                        "Respond with 'Red', 'Green', 'Yellow', or 'None'.",
            "answer": tl_qa
        }
        )
        qa_data.extend(get_planning_qa(frame_data))
        plan_qa = get_planning_explanation(frame_data, frame_data['image_paths'], tokenizer, model, image_processor)
        qa_data.append({
            "question": plan_qa[1],
            "answer": plan_qa[0]
        })
        
        # Add ego vehicle status QA (OPTIONAL)
        ego = frame_data['ego_features']
        qa_data.append({
            "question": "What is the current status of our vehicle?",
            "answer": f"Current speed: {ego['speed']:.2f} m/s, Steering angle: {ego['steering']:.2f} radians, Throttle: {ego['throttle']:.2f}, Brake: {ego['brake']:.2f}"
        })
        
        # Add object detection summary QA (OPTIONAL)
        obj_counts = {}
        for name in frame_data['gt_names']:
            obj_counts[name] = obj_counts.get(name, 0) + 1
        
        obj_text = ", ".join([f"{count} {name}{'s' if count>1 else ''}" 
                            for name, count in obj_counts.items()])
                            
        
    except Exception as e:
        print(f"Error in generate_drive_qa: {e}")
        qa_data.append({
            "question": "What is in this driving scene?",
            "answer": "Error generating QA data for this frame."
        })
    
    return qa_data


def angular_difference(angle_1, angle_2):
    """
    calculate smallest angular difference between 2 angles in radians, addressing the 2pi problem when reaching one full loop
    """
    diff = abs(angle_1 - angle_2)
    if diff > math.pi:
        diff = 2*math.pi - diff
    return diff



# Welford accumulator for online mean/std calculation
class WelfordAccumulator:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences
        
    def update(self, x):
        """Update statistics with new data point"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
    def get_std(self):
        """Get standard deviation (returns 0 for <2 samples)"""
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.n - 1))
    
    def get_stats(self):
        """Return all statistics in a dictionary"""
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.get_std()
        }
    
    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


# Main Processing Function
def process_dataset(data_paths, output_file, tokenizer, model, image_processor, max_frames=None):
    """
    Main processing pipeline for the entire dataset.
    
    Args:
        data_paths: List of paths to scenario folders
        output_file: Path to save the output JSON
        tokenizer, model, image_processor: LLaVA components
        max_frames: Optional limit on number of frames to process per scenario
    """
    print(f"Processing {len(data_paths)} scenario folders")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_datasets = []
    
    # Process each scenario folder
    for scenario_path in data_paths:
        scenario_name = os.path.basename(scenario_path)
        print(f"\nProcessing scenario: {scenario_name}")
        
        # Get frame list for this scenario
        frame_files = sorted(glob(os.path.join(scenario_path, "bev_gt_label", "*.json")))
        
        if max_frames:
            frame_files = frame_files[:max_frames]
            print(f"Processing {len(frame_files)} frames (limited by max_frames)")
        else:
            print(f"Processing {len(frame_files)} frames")
        
        scenario_dataset = []
        
        # Create directory for threshold logs
        thresholds_dir = os.path.join(output_dir, "threshold_logs")
        os.makedirs(thresholds_dir, exist_ok=True)
        thresholds_log = []

        prev_ego_long_vel = 0.0
        prev_ego_steering = 0.0
        prev_ego_command = -1
        lane_follow = 4
        straight = 3
        # initialize previous actors state variables
        prev_actor_states = {}

        # Online filter parameters
        acc_ego_vel = WelfordAccumulator()
        acc_ego_steer = WelfordAccumulator()
        acc_actor_yaw = WelfordAccumulator()
        acc_actor_vel = WelfordAccumulator()
        
        # Sensitivity factors (k) - can be tuned (lower k, more sensitive updates)
        K_EGO_VEL = 0.5
        K_EGO_STEER = 0.005 # test
        K_ACTOR_YAW = 0.008 # test
        K_ACTOR_VEL = 1.0
        
        # Threshold bounds (min, max)
        THRESH_BOUNDS = {
            'ego_vel': (0.5, 1.5),
            'ego_steer': (0.001, 0.1), # 0.01
            'actor_yaw': (0.005, 0.15), # 0.02
            'actor_vel': (2.0, 3.0)
        }
        
        # Initial thresholds
        INIT_THRESH = {
            'ego_vel': 1.0,
            'ego_steer': 0.05,
            'actor_yaw': 0.08,
            'actor_vel': 2.5
        }

        for index, frame_file in enumerate(tqdm(frame_files, desc=f"Processing {scenario_name}")):
            frame_thresholds = {
                "frame_id": None,
                "ego_vel_threshold": None,
                "ego_steer_threshold": None,
                "actor_yaw_threshold": None,
                "actor_vel_threshold": None,
                "ego_vel_diff": None,
                "ego_steer_diff": None,
                "skipped": False,
                "vru_present": False,
                "actor_changed": False
            }
            
            skip = False
            try:
                frame_id = int(os.path.basename(frame_file).split('.')[0])
                frame_data = process_carla_frame(scenario_path, frame_id, index)
                
                if frame_data:
                    # Store frame ID in thresholds log
                    frame_thresholds["frame_id"] = frame_id
                    
                    # Filter Frames based on ego kinematic data and command
                    ego_command = frame_data["ego_features"]["command"]
                    ego_long_vel = abs(frame_data["ego_features"]["speed"])
                    ego_cmd_steer = frame_data["ego_features"]["steering"]
                    gt_boxes = frame_data["gt_boxes"]
                    obj_ids = frame_data["gt_ids"]
                    gt_names = frame_data["gt_names"]

                    any_actor_changed = False
                    vru = False

                    ego_vel_diff = abs(ego_long_vel - prev_ego_long_vel)
                    ego_steer_diff = abs(ego_cmd_steer - prev_ego_steering)
                    
                    # Update accumulators
                    acc_ego_vel.update(ego_vel_diff)
                    acc_ego_steer.update(ego_steer_diff)

                    # Store differences in log
                    frame_thresholds["ego_vel_diff"] = ego_vel_diff
                    frame_thresholds["ego_steer_diff"] = ego_steer_diff

                    # Calculate adaptive thresholds for current frame
                    def get_threshold(acc, thresh_key):
                        """Get threshold with bounds handling"""
                        if acc.n < 2:
                            return INIT_THRESH[thresh_key]
                        
                        # thresh = acc.mean + {
                        #     'ego_vel': K_EGO_VEL,
                        #     'ego_steer': K_EGO_STEER,
                        #     'actor_yaw': K_ACTOR_YAW,
                        #     'actor_vel': K_ACTOR_VEL
                        # }[thresh_key] * acc.get_std()

                        thresh = {
                            'ego_vel': K_EGO_VEL,
                            'ego_steer': K_EGO_STEER,
                            'actor_yaw': K_ACTOR_YAW,
                            'actor_vel': K_ACTOR_VEL
                        }[thresh_key] / (acc.mean + acc.get_std() + 1E-6) # avoid division by zero
                        
                        return max(THRESH_BOUNDS[thresh_key][0], 
                                min(thresh, THRESH_BOUNDS[thresh_key][1]))

                    # Calculate all thresholds upfront for logging
                    ego_vel_thresh = get_threshold(acc_ego_vel, 'ego_vel')
                    ego_steer_thresh = get_threshold(acc_ego_steer, 'ego_steer')
                    actor_yaw_thresh = get_threshold(acc_actor_yaw, 'actor_yaw')
                    actor_vel_thresh = get_threshold(acc_actor_vel, 'actor_vel')
                    
                    # Store thresholds in log
                    frame_thresholds["ego_vel_threshold"] = ego_vel_thresh
                    frame_thresholds["ego_steer_threshold"] = ego_steer_thresh
                    frame_thresholds["actor_yaw_threshold"] = actor_yaw_thresh
                    frame_thresholds["actor_vel_threshold"] = actor_vel_thresh

                    # Ego vehicle processing (SKIPPED FOR THE MOMENT -- WE APPLY THE FILTER IN ANY CASE, NOT JUST LANE_FOLLOW OR STRAIGHT)
                    # if (ego_command == lane_follow and prev_ego_command == lane_follow) or \
                    #    (ego_command == straight and prev_ego_command == straight):
                        
                        # ego_vel_diff = abs(ego_long_vel - prev_ego_long_vel)
                        # ego_steer_diff = abs(ego_cmd_steer - prev_ego_steering)
                        
                        # # Store differences in log
                        # frame_thresholds["ego_vel_diff"] = ego_vel_diff
                        # frame_thresholds["ego_steer_diff"] = ego_steer_diff
                        
                    if ego_vel_diff < ego_vel_thresh and ego_steer_diff < ego_steer_thresh:
                        prev_ego_command = ego_command
                        prev_ego_long_vel = ego_long_vel
                        prev_ego_steering = ego_cmd_steer
                        skip = True

                        # # Update accumulators
                        # acc_ego_vel.update(ego_vel_diff)
                        # acc_ego_steer.update(ego_steer_diff)

                    # Actors processing
                    for i, (box, actor_id, name) in enumerate(zip(gt_boxes, obj_ids, gt_names)):
                        actor_x, actor_y = box[0], box[1]
                        if name in ["pedestrian", "cyclist"]:
                            if 0 < actor_x < 15 and abs(actor_y) < 3:
                                vru = True
                                frame_thresholds["vru_present"] = True
                                print(f"VRU {name} detected at ({actor_x:.2f}, {actor_y:.2f}) in frame {frame_id}")

                        if 0 < abs(actor_x) < 15 and 0 < abs(actor_y) < 3:
                            current_yaw = box[4]
                            current_velocity = box[5]

                            if actor_id in prev_actor_states:
                                prev_yaw = prev_actor_states[actor_id]["yaw"]
                                prev_velocity = prev_actor_states[actor_id]["velocity"]

                                yaw_diff = angular_difference(current_yaw, prev_yaw)
                                velocity_diff = abs(current_velocity - prev_velocity)
                                
                                # Update accumulators
                                acc_actor_yaw.update(yaw_diff)
                                acc_actor_vel.update(velocity_diff)

                                if skip and (yaw_diff >= actor_yaw_thresh or velocity_diff >= actor_vel_thresh):
                                    any_actor_changed = True
                                    frame_thresholds["actor_changed"] = True
                                    print(f"Actor {actor_id} in frame {frame_id} has significant kinematic changes")

                            else:
                                any_actor_changed = True
                                frame_thresholds["actor_changed"] = True
                                print(f"New actor {actor_id} detected in frame {frame_id}")

                            prev_actor_states[actor_id] = {
                                "yaw": current_yaw,
                                "velocity": current_velocity,
                                "last_seen_frame": frame_id
                            }
                    
                    if any_actor_changed:
                        skip = False
                        print(f"Not skipping frame {frame_id} due to significant actor changes")
                    
                    if vru:
                        skip = False
                        print(f"Not skipping frame {frame_id} due to VRU presence")
                        
                    # Cleanup actor states
                    frames_to_keep = 20
                    prev_actor_states = {
                        actor_id: state for actor_id, state in prev_actor_states.items()
                        if frame_id - state["last_seen_frame"] <= frames_to_keep
                    }

                    # Log skip decision
                    frame_thresholds["skipped"] = skip
                    
                    if skip:
                        print(f"Skipped frame {frame_id}")
                        # Add to thresholds log and continue without processing
                        thresholds_log.append(frame_thresholds)
                        # update ego states even when skipping
                        prev_ego_command = ego_command
                        prev_ego_long_vel = ego_long_vel
                        prev_ego_steering = ego_cmd_steer
                        continue

                    # Update state if frame is processed
                    prev_ego_command = ego_command
                    prev_ego_long_vel = ego_long_vel
                    prev_ego_steering = ego_cmd_steer

                    # Generate QA pairs
                    qa_pairs = generate_drive_qa(
                        frame_data,
                        tokenizer,
                        model,
                        image_processor
                    )
                    
                    # Create final entry
                    entry = {
                        "scenario": scenario_name,
                        "frame_id": frame_id,
                        "timestamp": frame_data["timestamp"],
                        "images": frame_data['image_paths'],
                        "qa_pairs": qa_pairs,
                        "ego": {
                            "features": frame_data["ego_features"],
                            "future_trajectory": frame_data["gt_ego_fut_trajs"].tolist() if frame_data["fut_valid_flag"] else []
                        }
                    }
                    
                    scenario_dataset.append(entry)
                    all_datasets.append(entry)
            
            except Exception as e:
                print(f"Error processing frame {frame_id} in {scenario_name}: {e}")
            
            # Always add to thresholds log
            thresholds_log.append(frame_thresholds)
        
        # Save scenario-specific dataset
        scenario_output = f"{os.path.splitext(output_file)[0]}_{scenario_name}.json"
        with open(scenario_output, 'w') as f:
            json.dump(scenario_dataset, f, indent=2)
        
        # Save thresholds log for this scenario
        thresholds_file = os.path.join(thresholds_dir, f"thresholds_{scenario_name}.json")
        with open(thresholds_file, 'w') as f:
            json.dump(thresholds_log, f, indent=2)
        
        print(f"Scenario dataset saved to {scenario_output} with {len(scenario_dataset)} frames processed")
        print(f"Thresholds log saved to {thresholds_file} with {len(thresholds_log)} entries")
    
    # Save complete dataset
    with open(output_file, 'w') as f:
        json.dump(all_datasets, f, indent=2)
    
    print(f"Complete dataset saved to {output_file} with {len(all_datasets)} total frames processed")
    return all_datasets

def main():
    """Main function to run the process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process CARLA data and generate QA pairs")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to CARLA dataset root directory")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Output JSON file")
    parser.add_argument("--max_frames", type=int, default=None, 
                        help="Maximum number of frames to process per scenario")
    parser.add_argument("--process_single", type=str, default=None,
                        help="Process only a specific scenario subfolder")
    parser.add_argument("--process_type", type=str, default=None,
                        help="Process only scenarios of a specific type (e.g., 'Accident')")
    args = parser.parse_args()
    
    # Load LLaVA model
    tokenizer, model, image_processor = load_llava_model()
    
    
    if args.process_single:
        scenario_paths = [args.process_single]
    else:
        all_scenario_paths = discover_scenario_folders(args.data_path)
        
        if args.process_type:
            scenario_paths = [path for path in all_scenario_paths 
                             if args.process_type in path]
        else:
            scenario_paths = all_scenario_paths
    
    if not scenario_paths:
        print(f"No valid scenario folders found in {args.data_path}")
        return
    
    print(f"Found {len(scenario_paths)} valid scenario folders to process")
    for path in scenario_paths:
        print(f"  - {path}")
    
    # Process dataset
    process_dataset(
        scenario_paths,
        args.output_file,
        tokenizer,
        model,
        image_processor,
        args.max_frames
    )

if __name__ == "__main__":
    main()
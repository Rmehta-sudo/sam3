import torch
import cv2
import argparse
import os
import numpy as np

def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']

def extract_mask_from_pt(output, frame_idx=0):
    current_mask = None
    try:
        if isinstance(output, list) and frame_idx < len(output):
            frame_data = output[frame_idx]
            if isinstance(frame_data, dict):
                if "masks" in frame_data:
                    current_mask = frame_data["masks"]
                elif "out_binary_masks" in frame_data:
                    current_mask = frame_data["out_binary_masks"]
            elif isinstance(frame_data, torch.Tensor):
                current_mask = frame_data
                
        elif isinstance(output, dict):
            if frame_idx in output:
                current_mask = output[frame_idx]
            elif str(frame_idx) in output:
                current_mask = output[str(frame_idx)]
            elif "out_binary_masks" in output:
                # If only one mask for the whole sequence or just frame 0
                if frame_idx == 0:
                    current_mask = output["out_binary_masks"]
            elif "masks" in output:
                if frame_idx == 0:
                    current_mask = output["masks"]
    except Exception as e:
        pass

    if current_mask is not None:
        if hasattr(current_mask, "cpu"):
            current_mask = current_mask.cpu().numpy()
            
        if current_mask.size == 0:
            return None
            
        current_mask = np.squeeze(current_mask)
        
        # If still multi-dimensional (e.g., multiple objects), take the first one
        if current_mask.ndim > 2:
            current_mask = current_mask[0]
            
    return current_mask

def apply_mask_overlay(frame, mask_data):
    if mask_data is None:
        return frame
        
    height, width = frame.shape[:2]
    current_mask = mask_data
    
    if current_mask.shape != (height, width):
        current_mask = cv2.resize(current_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 2] = 255 # Red channel
    
    binary_mask = current_mask > 0
    alpha = 0.5
    
    overlayed_frame = frame.copy()
    overlayed_frame[binary_mask] = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)[binary_mask]
    return overlayed_frame

def visualize_pt_masks(input_path, pt_path, output_path=None):
    is_image = is_image_file(input_path)
    
    if output_path is None:
        out_dir = "overlay/image" if is_image else "overlay/video"
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.basename(input_path)
        output_path = os.path.join(out_dir, base_name)
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
    print(f"Loading masks from {pt_path}...")
    output = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    if is_image:
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error loading image {input_path}")
            return
            
        mask_data = extract_mask_from_pt(output, frame_idx=0)
        frame = apply_mask_overlay(frame, mask_data)
        
        cv2.imwrite(output_path, frame)
        print(f"Finished! Saved visualized image to {output_path}")
        
    else:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps == 0 or fps != fps:
            fps = 30.0 # fallback
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            mask_data = extract_mask_from_pt(output, frame_idx=frame_idx)
            frame = apply_mask_overlay(frame, mask_data)
            
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"Finished! Saved visualized video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Original video or image path (legacy argument name)")
    parser.add_argument("--image", help="Original image path")
    parser.add_argument("--input", help="Original video or image path")
    parser.add_argument("--pt", required=True, help="Path to the .pt file")
    parser.add_argument("--out", help="Path to save the output (optional)")
    args = parser.parse_args()
    
    input_path = args.input or args.video or args.image
    if not input_path:
        print("Error: Must provide --input, --video, or --image")
        exit(1)
        
    visualize_pt_masks(input_path, args.pt, args.out)

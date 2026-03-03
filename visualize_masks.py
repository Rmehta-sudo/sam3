import torch
import cv2
import argparse
import os
import numpy as np

def visualize_pt_masks(video_path, pt_path, output_path):
    print(f"Loading masks from {pt_path}...")
    output = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    # 1. Open the original video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps == 0 or fps != fps:
        fps = 30.0 # fallback
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # SAM-3 video outputs are usually lists of dictionaries per frame, 
    # or a dictionary mapping frame_idx -> masks.
    # Let's inspect it to safely extract the masks.
    print(f"Mask object type: {type(output)}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_mask = None
        
        # Extract the mask for the current frame depending on the SAM-3 structure
        try:
            if isinstance(output, list) and frame_idx < len(output):
                # If it's a list per frame
                frame_data = output[frame_idx]
                if isinstance(frame_data, dict) and "masks" in frame_data:
                    current_mask = frame_data["masks"]
                elif isinstance(frame_data, torch.Tensor):
                    current_mask = frame_data
                    
            elif isinstance(output, dict):
                # If it maps frame index/object id
                if frame_idx in output:
                    current_mask = output[frame_idx]
                elif str(frame_idx) in output:
                    current_mask = output[str(frame_idx)]
        except Exception as e:
            pass

        # If a mask was found for this frame, overlay it
        if current_mask is not None:
            if hasattr(current_mask, "cpu"):
                current_mask = current_mask.cpu().numpy()
                
            # Flatten/squeeze out extra dimensions (like batch size or object id)
            current_mask = np.squeeze(current_mask)
            
            # If multiple objects, just take the first one or combine them
            if current_mask.ndim > 2:
                current_mask = current_mask[0] # take first object's mask
                
            # Ensure it matches frame size
            if current_mask.shape != (height, width):
                current_mask = cv2.resize(current_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                
            # Create a red overlay for the mask
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 2] = 255 # Red channel
            
            # Blend the frame and the mask where the mask is active (> 0)
            binary_mask = current_mask > 0
            alpha = 0.5
            frame[binary_mask] = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)[binary_mask]

        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Finished! Saved visualized video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Original video path")
    parser.add_argument("--pt", required=True, help="Path to the .pt file")
    parser.add_argument("--out", required=True, help="Path to save the output mp4")
    args = parser.parse_args()
    
    visualize_pt_masks(args.video, args.pt, args.out)

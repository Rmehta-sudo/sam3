import torch
import argparse
import os
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

#################################### For Video ####################################
from sam3.model_builder import build_sam3_video_predictor

def infer_image(image_path, text_prompt, use_bfloat16=False, device="cuda"):
    print(f"Loading image model...")
    model = build_sam3_image_model()
    if use_bfloat16:
        model.to(dtype=torch.bfloat16)
    model.to(device=device)
    processor = Sam3Processor(model)
    
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    
    print(f"Running inference with text prompt: '{text_prompt}'...")
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(f"Inference complete! Found {len(masks)} masks. Max score: {max(scores) if len(scores) > 0 else 'N/A'}")

    if len(masks) > 0:
        mask = masks[0]
        # Handle if it is a torch Tensor
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
            
        import numpy as np
        # Convert boolean or 0-1 mask to 0-255 uint8 image
        mask_img = Image.fromarray((mask.squeeze() * 255).astype(np.uint8))
        
        basename = os.path.basename(image_path)
        name, _ = os.path.splitext(basename)
        safe_prompt = "".join(c if c.isalnum() else "_" for c in text_prompt)[:10].strip("_")
        out_dir = os.path.join("outputs", "images")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{name}_{safe_prompt}_mask.png")
        
        mask_img.save(save_path)
        print(f"Successfully saved mask to {save_path}!")

    return masks, boxes, scores

def infer_video(video_path, text_prompt, use_bfloat16=False, offload_video=False, device="cuda"):
    print(f"Loading video model...")
    video_predictor = build_sam3_video_predictor()
    if use_bfloat16:
        video_predictor.model.to(dtype=torch.bfloat16)
    video_predictor.model.to(device=device)
    
    print(f"Starting session for video from {video_path}...")
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
            offload_video_to_cpu=offload_video,
        )
    )
    
    print(f"Running inference with text prompt: '{text_prompt}' on frame 0...")
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=response["session_id"],
            frame_index=0, 
            text=text_prompt,
        )
    )
    output = response["outputs"]
    print("Inference complete!")
    
    basename = os.path.basename(video_path)
    name, _ = os.path.splitext(basename)
    safe_prompt = "".join(c if c.isalnum() else "_" for c in text_prompt)[:10].strip("_")
    out_dir = os.path.join("outputs", "videos")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{name}_{safe_prompt}_mask.pt")
    
    torch.save(output, save_path)
    print(f"Successfully saved video output to {save_path}!")
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-3 Inference test script")
    parser.add_argument("--type", choices=["image", "video"], required=True, help="Inference type (image or video)")
    parser.add_argument("--path", required=True, help="Path to image or video")
    parser.add_argument("--prompt", required=True, help="Text prompt for segmentation")
    parser.add_argument("--bfloat16", action="store_true", help="Cast the model to bfloat16 to save VRAM")
    parser.add_argument("--offload_video", action="store_true", help="Keep video frames on CPU until needed to prevent OOM")
    parser.add_argument("--device", default="cuda", help="Execution device (default: cuda)")
    
    args = parser.parse_args()
    
    if args.type == "image":
        infer_image(args.path, args.prompt, use_bfloat16=args.bfloat16, device=args.device)
    elif args.type == "video":
        infer_video(args.path, args.prompt, use_bfloat16=args.bfloat16, offload_video=args.offload_video, device=args.device)

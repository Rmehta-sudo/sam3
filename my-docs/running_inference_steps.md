# Running SAM-3 Inference

This document details how to run the `run_sam3_inference.py` script provided in the root directory. Since you have already logged in with your token on this system, you can proceed directly to running the model.

## Prerequisites

1. Ensure your conda environment where you installed the SAM-3 dependencies is active. Based on our previous setups, this environment should be named `sam3`.
   ```bash
   conda activate sam3
   ```

2. Alternatively, if your environment is already active, simply ensure you are in the project root directory (`/home/rachitmehta/workspace/research-work/SAM-3`).

## Running Image Inference

To run inference on an image using a text prompt, use the `--type image` flag. You can test it with the provided `sample_image.jpg` in the repository.

```bash
python run_sam3_inference.py \
    --type image \
    --path sample_image.jpg \
    --prompt "a description of an object in the image"
```

## Running Video Inference

To run inference on a video, use the `--type video` flag, passing in the path to your video and the initial text prompt for the first frame (frame 0).

```bash
python run_sam3_inference.py \
    --type video \
    --path /path/to/your/video.mp4 \
    --prompt "a description of the object to track"
```

### Script Arguments Explained
- `--type`: Specifies whether the input is an `image` or `video` (required).
- `--path`: The absolute or relative path to the image or video file you want to process (required).
- `--prompt`: A descriptive text prompt specifying the object you want SAM-3 to segment/track (required).

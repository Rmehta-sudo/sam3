import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
image = Image.open("sample_image.jpg")
inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="car")
masks = output["masks"]
print(type(masks))
if isinstance(masks, torch.Tensor):
    print(masks.shape)
elif isinstance(masks, list):
    print(f"List of length {len(masks)}")
    print(type(masks[0]))
    if hasattr(masks[0], "shape"):
        print(masks[0].shape)

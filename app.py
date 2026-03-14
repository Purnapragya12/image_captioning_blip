import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load pretrained BLIP model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image caption function
def caption_image(input_image: np.ndarray):

    # Convert numpy array to PIL Image
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process image
    inputs = processor(raw_image, return_tensors="pt")

    # Generate caption
    output = model.generate(**inputs)

    # Decode tokens to text
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


# Gradio Interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using BLIP model."
)

# Launch app
iface.launch(server_name="0.0.0.0", server_port=7860)

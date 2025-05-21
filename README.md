## Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

### PROBLEM STATEMENT:To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

### DESIGN STEPS:

#### STEP 1:
Install necessary libraries: Begin by installing the required Python packages: transformers, gradio, torchvision, and torch. This is typically done using pip install.
#### STEP 2:
Import modules: Import the BlipProcessor and BlipForConditionalGeneration classes from the transformers library. Also, import Image from PIL (Pillow) and torch.
#### STEP 3:
Load pre-trained models: Initialize the processor using BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") and the model using BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"). These lines load the pre-trained BLIP model and its associated processor.
#### STEP 4:
Define the caption generation function: Create a Python function named generate_caption that takes an image as input.
#### STEP 5:
Process the image: Inside generate_caption, use the processor to prepare the input image for the model. This involves converting the image into a format suitable for the BLIP model, specifying return_tensors="pt" to get PyTorch tensors.
#### STEP 6:
Generate output from the model: Pass the processed inputs to the model.generate() method. This step uses the BLIP model to generate a sequence of tokens representing the image caption.
#### STEP 7:
Decode the caption: Use the processor.decode() method to convert the generated output tokens back into a human-readable string. The skip_special_tokens=True argument ensures that any special tokens used by the model are removed from the final caption.
#### STEP 8:
Return the caption: The generate_caption function returns the decoded caption string.
#### STEP 9:
Create and launch Gradio interface: Import gradio as gr. Then, create a Gradio Interface instance, specifying fn=generate_caption, inputs=gr.Image(type="pil"), outputs="text", and providing a title and description for the web application. Finally, launch the interface using interface.launch().

### PROGRAM:
```
!pip install transformers
!pip install gradio
!pip install torchvision
!pip install torch

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

import gradio as gr
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="BLIP Image Captioning",
    description="Upload an image to generate a caption using the BLIP model."
)
interface.launch()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/3dd0f915-2b13-463b-b6be-9dd9a371abef)

### RESULT:
Therefore the code is excuted successfully.

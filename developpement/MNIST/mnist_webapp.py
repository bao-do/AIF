#%%
import gradio as gr
from PIL import Image
import requests
import io
import numpy as np
import matplotlib.pyplot as plt


def recognize_digit(image):
    if image['composite'] is not None:
        image = Image.fromarray(image['composite'].astype('uint8'))
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        # Send request to the API
        response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
        return response.json()["prediction"]

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs=gr.Sketchpad(type="numpy", image_mode='L'), 
                outputs=gr.Label(value={"Waiting for input...": 1.0}),
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True)
# %%
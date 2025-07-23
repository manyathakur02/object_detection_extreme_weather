from ultralytics import YOLO
import gradio as gr

model = YOLO("best.pt")  # this file should be in same folder

def detect(image):
    results = model(image)
    return results[0].plot()

gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Extreme Weather Object Detection",
    description="Upload an image to detect objects under extreme weather conditions."
).launch()

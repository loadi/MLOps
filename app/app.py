import os.path

import gradio as gr
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "1736192989.onnx"
)
onnx_model = ort.InferenceSession(model_path)

input_name = onnx_model.get_inputs()[0].name
output_name = onnx_model.get_outputs()[0].name

classes = ['N/A', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def img_to_tensor(img):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img.numpy()


def predict(image):
    input_data = img_to_tensor(image)
    pred = onnx_model.run([output_name], {input_name: input_data})[0]
    predicted_class = np.argmax(pred, axis=1)[0]

    return classes[predicted_class]


def classify_image(image):
    return f"На изображении буква: {predict(image)}"


interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=8080)
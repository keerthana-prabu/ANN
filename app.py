import numpy as np
import gradio as gr
import tensorflow as tf
import pickle

from tensorflow.keras.models import load_model


model = load_model("iris_ann_model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


def predict_iris(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):

    data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    data_scaled = scaler.transform(data)

    probabilities = model.predict(data_scaled, verbose=0)[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = probabilities[predicted_index] * 100

    return (
        predicted_class,
        f"{confidence:.2f}%"
    )


demo = gr.Interface(
    fn=predict_iris,

    inputs=[
        gr.Number(label="Sepal Length (cm)", value=5.1),
        gr.Number(label="Sepal Width (cm)", value=3.5),
        gr.Number(label="Petal Length (cm)", value=1.4),
        gr.Number(label="Petal Width (cm)", value=0.2)
    ],

    outputs=[
        gr.Textbox(label="Predicted Species"),
        gr.Textbox(label="Confidence")
    ],

    title="Iris Flower Classifier",
    description="Enter the four flower measurements to predict the Iris species."
)


demo.launch()
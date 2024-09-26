import streamlit as st
import jcopdl

from PIL import Image
import torch

from MobilenetV2.mobilenetv2 import CustomMobileNetV2
from transforms.transform import test_transform as tf

st.markdown("<h2 style='text-align: center; color: black;'>Prediksi</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Gunting Batu Kertas</h2>", unsafe_allow_html=True)

def display(cent_co, image):
    image = Image.open(image)
    result = predict_class(image)

    with cent_co:
        st.markdown(f"<h3 style='text-align: center;'>Prediksi: {result}</h3>", unsafe_allow_html=True) 
        st.image(image, width=200)

def main():
    file_uploaded = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])
    
    left_co, cent_co, last_co = st.columns(3)
    if file_uploaded is not None:
        display(cent_co, file_uploaded)

def predict_class(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = torch.load('model/configs.pth', map_location='cpu')
    weights = torch.load('model/weights_best.pth', map_location='cpu')

    model = CustomMobileNetV2(config.output_size).to(device)
    model.load_state_dict(weights)

    img = tf(image)
    img = img.unsqueeze(0)
    img = img.to(device)

    label2cat = ['paper', 'rock', 'scissors']
    with torch.no_grad():
        model.eval()
        output = model(img)
        preds = output.argmax(1)
        preds = label2cat[preds.item()]
    return preds

if __name__ == "__main__":
    main()

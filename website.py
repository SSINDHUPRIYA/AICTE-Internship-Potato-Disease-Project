import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model_path = "models/trained_plant_disease_model.keras"  # Correct path
    model = tf.keras.models.load_model(model_path)  # Load from models/
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch format
    
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return class index

# Streamlit UI
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])


from PIL import Image
st.title("Potato Disease Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
else:
    st.warning("Please upload an image.")
if(app_mode=='HOME'):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)

elif(app_mode=='Disease Recognition'):
    st.header('Plant Disease Detection System For Sustainable Agriculture')


test_image= st.file_uploader('Choose an image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_column_width=True)

if (st.button('Predict')):
    st.snow()
    st.write('our prediction')
    result_index = model_prediction(test_image)
    class_name=['Potato___Early_blight','Potato___Late_blight','Potato___healthy']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))
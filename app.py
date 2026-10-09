import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title='Signature Analysis', page_icon='✍️', layout='wide')
st.title('AI-Powered Signature Analysis System')
st.caption('Compare the visual structure of two signatures.')

def preprocess(upload):
    image = ImageOps.exif_transpose(Image.open(upload)).convert('RGBA')
    background = Image.new('RGBA', image.size, 'white')
    image = Image.alpha_composite(background, image).convert('RGB')
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is None:
        raise ValueError('No signature detected. Upload a clear signature.')
    x, y, w, h = cv2.boundingRect(points)
    if w < 10 or h < 10:
        raise ValueError('Signature is too small. Upload a clearer image.')
    cropped = ink[y:y+h, x:x+w]
    scale = min(560 / w, 210 / h)
    resized = cv2.resize(cropped, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((250, 600), dtype=np.uint8)
    rh, rw = resized.shape
    left, top = (600-rw)//2, (250-rh)//2
    canvas[top:top+rh, left:left+rw] = resized
    return rgb, gray, canvas

def compare_signatures(first, second):
    kernel = np.ones((5, 5), dtype=np.uint8)
    expanded_first = cv2.dilate(first, kernel)
    expanded_second = cv2.dilate(second, kernel)
    first_coverage = np.mean(expanded_second[first > 0] > 0)
    second_coverage = np.mean(expanded_first[second > 0] > 0)
    return float(100 * (first_coverage + second_coverage) / 2)

left, right = st.columns(2)
with left:
    reference = st.file_uploader('Reference Signature', type=['png', 'jpg', 'jpeg'], key='reference')
    if reference:
        st.image(reference, caption='Reference Signature', width=450)
with right:
    uploaded = st.file_uploader('Uploaded Signature', type=['png', 'jpg', 'jpeg'], key='uploaded')
    if uploaded:
        st.image(uploaded, caption='Uploaded Signature', width=450)
threshold = st.slider('Similarity threshold', 0, 100, 75)
st.caption('The default threshold is for demonstration and has not been validated.')
if st.button('Analyze Signatures', type='primary'):
    if reference is None or uploaded is None:
        st.warning('Please upload both signature images.')
    else:
        try:
            reference.seek(0)
            uploaded.seek(0)
            _, _, reference_binary = preprocess(reference)
            _, uploaded_gray, uploaded_binary = preprocess(uploaded)
            score = compare_signatures(reference_binary, uploaded_binary)
            st.metric('Visual similarity score', f'{score:.1f}%')
            if score >= threshold:
                st.success('Result: Visually Similar')
            else:
                st.warning('Result: Visually Different')
            st.progress(min(1.0, max(0.0, score / 100)))
            edges = cv2.Canny(uploaded_binary, 50, 150)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(uploaded_gray, caption='Grayscale')
            with col2:
                st.image(255-uploaded_binary, caption='Binary Image')
            with col3:
                st.image(255-edges, caption='Stroke Edges')
        except Exception as error:
            st.error(f'Could not analyze the images: {error}')
st.info('This prototype measures image similarity. It cannot establish identity or determine whether a signature is genuine. It does not use a trained deep learning model.')

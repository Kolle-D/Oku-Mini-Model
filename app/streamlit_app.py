import streamlit as st
import sys

sys.path.append("../src")

from inference import (
    translate_oku_to_english,
    translate_english_to_oku
)

st.title("Oku ↔ English Translator")

direction = st.selectbox(
    "Direction",
    [
        "Oku → English",
        "English → Oku"
    ]
)

text = st.text_area(
    "Enter text"
)

if st.button("Translate"):

    if direction == "Oku → English":

        result = translate_oku_to_english(text)

    else:

        result = translate_english_to_oku(text)

    st.success(result)
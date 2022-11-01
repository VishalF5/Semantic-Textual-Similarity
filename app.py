import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

encoder_url = 'https://tfhub.dev/google/universal-sentence-encoder/4' 

encoder = hub.load(encoder_url)



# Calculating Cosine Similarity between two sentences
def get_similarity(sentence_a, sentence_b):
    embed_a = encoder([sentence_a])
    embed_b = encoder([sentence_b])
    similarity = cosine_similarity(embed_a, embed_b)[0][0]
    return f'The similarity score between sentence A and sentence B is: "{similarity:.2f}"'


st.title("Text Similarity")

input_text1 = st.text_area("Enter First sentencet")
input_text2 = st.text_area("Enter  Second Sentence")

if st.button('Calculate'):
    res = get_similarity(input_text1 ,input_text2)

    st.write(res)
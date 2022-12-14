import streamlit as st
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


# Encoder
encoder_url = 'https://tfhub.dev/google/universal-sentence-encoder/4' 

encoder = hub.load(encoder_url)



# Calculating Cosine Similarity between two sentences
def get_similarity(sentence_a, sentence_b):
    embed_a = encoder([sentence_a])
    embed_b = encoder([sentence_b])
    similarity = cosine_similarity(embed_a, embed_b)[0][0]
    return f'The similarity score is : "{similarity:.2f}"'

# Interface
st.title("Text Similarity")

input_text1 = st.text_input("Enter First Sentence : " ,label_visibility = 'visible')
input_text2 = st.text_input("Enter Second Sentence : " ,label_visibility = 'visible')

res = get_similarity(input_text1 ,input_text2)


if st.button('Predict'):

    st.success(res)

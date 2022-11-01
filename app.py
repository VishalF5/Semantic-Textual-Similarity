import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


cv = CountVectorizer(max_features=5000,stop_words='english')


def similarity(sen1 ,sen2):
    score = cosine_similarity(sen1 ,sen2)
    return score


input_text1 = st.text_area("Enter")
input_text2 = st.text_area("Enter your text here")




if st.button('Predict'):
    input_text1 = cv.fit_transform(input_text1).toarray()
    input_text2 = cv.fit_transform(input_text2).toarray()
    res = similarity(input_text1 ,input_text2)
    st.write(res)


import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()

# write the function for converting lower case
def transform_text(text):
    text = text.lower()  # lower case convert
    text = nltk.word_tokenize(text)  # convert the sentence into tokenize word

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]  # here we can't assigning list like text =y so we have to do cloning cos its mutable data type
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf =pickle.load(open('vectorizer.pkl','rb'))
model =pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms =st.text_input("Enter the message here")
if st.button('Predict'):
     # 1. preprocessing
      transform_sms =transform_text(input_sms)
     # 2. vectorize
      vector_input =tfidf.transform([transform_sms])


      result =model.predict(vector_input)[0]
     # 4. Display
      if result ==1:
        st.header("spam")
      else:
         st.header("Not spam")

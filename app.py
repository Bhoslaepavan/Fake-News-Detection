import streamlit as st
import pickle

# Load trained model and TF-IDF vectorizer
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Streamlit UI
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is Fake or Real.")

news_text = st.text_area(
    "Enter News:",
    height=200,
    placeholder="Type or paste news article here..."
)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Convert text into TF-IDF features
        news_vector = vectorizer.transform([news_text])

        # Prediction
        prediction = model.predict(news_vector)[0]

        # Display result
        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

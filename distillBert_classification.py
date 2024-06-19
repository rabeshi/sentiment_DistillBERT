import streamlit as st
from transformers import pipeline

# Initialize the pipeline
pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.write("Enter some text below to see the sentiment analysis result.")

    # User input
    user_input = st.text_area("", height=200)

    # Analyze button
    if st.button("Analyze"):
        if user_input:
            # Get sentiment analysis result
            result = pipe(user_input)
            sentiment = result[0]['label']
            score = result[0]['score']
            
            # Display result
            if sentiment == "POSITIVE":
                st.success(f"Sentiment: {sentiment} with a score of {score:.2f}")
            else:
                st.error(f"Sentiment: {sentiment} with a score of {score:.2f}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()

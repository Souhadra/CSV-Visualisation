import os
import subprocess
import sys
import streamlit as st
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Interactive CSV Chat and Visualization", layout="wide")


def install_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("Installing SpaCy and en_core_web_sm model...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


install_spacy_model()

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


def csv_read(csv_file):
    return pd.read_csv(csv_file)


def get_text_from_dataframe(df):
    return df.to_string(index=False)


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def process_simple_stats(question, df):
    question = question.lower()

    if 'average' in question or 'mean' in question:
        for col in df.columns:
            if col.lower() in question:
                if df[col].dtype in ['int64', 'float64']:
                    return f"The average of {col} is {df[col].mean():.2f}"
                else:
                    return f"Cannot calculate average for non-numeric column {col}"

    if 'sum' in question:
        for col in df.columns:
            if col.lower() in question:
                if df[col].dtype in ['int64', 'float64']:
                    return f"The sum of {col} is {df[col].sum():.2f}"
                else:
                    return f"Cannot calculate sum for non-numeric column {col}"

    return None


def get_conversational_chain(retrieval_chain, ques, df):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        api_key = st.text_input("Gemini API key not found. Please enter your API key:", type="password")
        if not api_key:
            st.error("Gemini API key is required to proceed.")
            return

    try:
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        # Include basic data info in the prompt
        data_info = f"The CSV file has {df.shape[0]} rows and {df.shape[1]} columns. "
        data_info += f"The columns are: {', '.join(df.columns)}. "

        # Include some sample data
        data_info += f"Here's a sample of the data:\n{df.head().to_string()}\n"

        prompt = f"{data_info}\nBased on this CSV data, please answer the following question: {ques}"

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        # Try to handle simple statistical queries directly
        simple_stats = process_simple_stats(ques, df)
        if simple_stats:
            return simple_stats

        return response.text

    except Exception as e:
        return f"Error configuring Generative AI: {e}"


def plot_data(df, x_col, y_col, plot_type):
    plt.figure(figsize=(10, 6))

    if plot_type == "Scatter":
        plt.scatter(df[x_col], df[y_col])
        plt.title(f"Scatter plot: {x_col} vs {y_col}")
    elif plot_type == "Line":
        plt.plot(df[x_col], df[y_col])
        plt.title(f"Line plot: {x_col} vs {y_col}")
    elif plot_type == "Bar":
        df.groupby(x_col)[y_col].mean().plot(kind='bar')
        plt.title(f"Bar plot: Average {y_col} by {x_col}")
    elif plot_type == "Histogram":
        plt.hist(df[x_col], bins=20)
        plt.title(f"Histogram of {x_col}")
    elif plot_type == "Box":
        sns.boxplot(x=x_col, y=y_col, data=df)
        plt.title(f"Box plot: {y_col} by {x_col}")
    elif plot_type == "Violin":
        sns.violinplot(x=x_col, y=y_col, data=df)
        plt.title(f"Violin plot: {y_col} by {x_col}")
    else:
        st.error("Unsupported plot type")
        return

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    st.pyplot(plt)


def user_input(user_question, df):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "csv_extractor",
                                            "This tool is to give answer to queries from the csv")
    response = get_conversational_chain(retrieval_chain, user_question, df)
    st.write("Reply: ", response)


def automatic_visualizations(df):
    st.subheader("Automatic Data Visualizations")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 0:
        # Line chart
        st.write("Line Chart of Numeric Columns")
        st.line_chart(df[numeric_cols].head(50))

        # Bar chart
        st.write("Bar Chart of Numeric Columns")
        st.bar_chart(df[numeric_cols].head(50))

        # Area chart
        st.write("Area Chart of Numeric Columns")
        st.area_chart(df[numeric_cols].head(50))

        # Correlation heatmap
        st.write("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    else:
        st.write("No numeric columns found for automatic visualization.")

    # Distribution plots for numeric columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        st.write(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)


def main():
    st.title("Interactive CSV Chat and Visualization")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = csv_read(uploaded_file)
        st.session_state['df'] = df

        st.write("Data Preview:")
        st.dataframe(df.head())

        st.write("Data Description:")
        st.write(df.describe())

        # Process the data for RAG
        raw_text = get_text_from_dataframe(df)
        text_chunks = get_chunks(raw_text)
        vector_store(text_chunks)

        # Automatic visualizations
        automatic_visualizations(df)

        # Chat interface
        st.header("Chat with your data")
        user_question = st.text_input("Ask a question about your data:")
        if user_question:
            user_input(user_question, df)

        # Interactive plotting
        st.header("Interactive Data Visualization")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Select X-axis", options=df.columns)
        with col2:
            y_col = st.selectbox("Select Y-axis", options=df.columns)
        with col3:
            plot_type = st.selectbox("Select Plot Type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin"])

        if st.button("Generate Plot"):
            plot_data(df, x_col, y_col, plot_type)


if __name__ == "__main__":
    main()
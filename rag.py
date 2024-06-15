import os
import chromadb
import pandas as pd
from PyPDF2 import PdfFileReader

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection
collection = client.create_collection("documents")

# Function to ingest PDF data
def ingest_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extract_text()
        collection.add({"content": text, "source": file_path})

# Function to ingest CSV data
def ingest_csv(file_path):
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        content = " ".join(row.astype(str))
        collection.add({"content": content, "source": file_path})

# Function to read all CSVs in a directory and ingest them
def ingest_all_csvs_in_directory(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            ingest_csv(file_path)

# Example usage: Ingest all CSVs in the 'nba_data' directory
ingest_all_csvs_in_directory("makeup_data")

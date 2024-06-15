import ollama
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOllama
from unstructured.partition.csv import partition_csv
from langchain.schema import Document

# Function to load and partition all CSVs in a directory using unstructured
def load_and_partition_all_csvs(directory_path):
    documents = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            try:
                elements = partition_csv(file_path)
                for element in elements:
                    documents.append(Document(page_content=element.text, metadata={"source": file_name}))
                print(f"Successfully partitioned data from {file_path}")
            except Exception as e:
                print(f"Failed to partition data from {file_path}: {e}")
    return documents

# Function to split documents into smaller chunks
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = []
    for doc in documents:
        for chunk in text_splitter.split_text(doc.page_content):
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return split_docs

# Function to get user input and query Ollama
def query_ollama():
    # Load and partition CSV data
    nba_data_directory = "nba_data"
    all_documents = load_and_partition_all_csvs(nba_data_directory)
    print("CSV data loaded and partitioned.")
    print(f"Total documents loaded: {len(all_documents)}")

    if len(all_documents) > 0:
        print(f"First document sample: {all_documents[0]}")

    # Split documents into smaller chunks
    print("Splitting documents into smaller chunks...")
    split_documents_list = split_documents(all_documents)
    print(f"Total split documents: {len(split_documents_list)}")

    # Initialize embeddings
    try:
        print("Initializing Ollama embeddings...")
        oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
        print("Ollama embeddings initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Ollama embeddings: {e}")
        return

    # Initialize vectorstore
    try:
        print("Initializing vectorstore...")
        vectorstore = Chroma(collection_name="nba_data", embedding_function=oembed)
        print("Vectorstore initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize vectorstore: {e}")
        return

    # Add documents to vectorstore in smaller batches
    try:
        print("Adding documents to vectorstore...")
        batch_size = 10  # Adjust batch size as needed
        for i in range(0, len(split_documents_list), batch_size):
            batch = split_documents_list[i:i + batch_size]
            vectorstore.add_documents(batch)
            print(f"Added batch {i // batch_size + 1} to vectorstore.")
        print("All documents added to vectorstore successfully.")
    except Exception as e:
        print(f"Failed to add documents to vectorstore: {e}")
        return

    # Get user input
    user_message = input("Enter your question about player stats: ")
    print(f"User message: {user_message}")

    # Perform similarity search
    try:
        print("Performing similarity search...")
        query_docs = vectorstore.similarity_search(user_message, k=3)  # Increase k to get more relevant documents
        print(f"Query documents: {query_docs}")
    except Exception as e:
        print(f"Failed to perform similarity search: {e}")
        return

    # Initialize the LLM and summarize chain
    try:
        print("Initializing LLM and summarization chain...")
        llm = ChatOllama(model="llama3")
        chain = load_summarize_chain(llm, chain_type="map_reduce")  # Change to map_reduce for better summarization
        print("Running summarization chain...")
        result = chain.run(input_documents=query_docs)
        print(f"Summarization result: {result}")
    except Exception as e:
        print(f"Failed to initialize LLM or run summarization chain: {e}")
        return

    # Send the user message to Ollama
    try:
        print("Sending user message to Ollama for final response...")
        stream_2 = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': user_message}, {'role': 'system', 'content': result}],
            stream=True,
        )

        # Print the response from Ollama
        print("Receiving response from Ollama...")
        for chunk in stream_2:
            print(chunk['message']['content'], end='', flush=True)
    except Exception as e:
        print(f"Failed to send user message to Ollama or receive response: {e}")

# Example usage
if __name__ == "__main__":
    query_ollama()

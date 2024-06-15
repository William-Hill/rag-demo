import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain
from langchain_community.chat_models import ChatOllama

# Load the CSV file
csv_file_path = 'nba_data/nba_season_totals_2010_2024.csv'
df = pd.read_csv(csv_file_path)

# Preprocess and clean the data
# Example: Fill missing values, convert data types, etc.
df.fillna('', inplace=True)

# Convert rows to Document objects
def convert_row_to_document(row):
    content = ', '.join([f"{key}: {row[key]}" for key in row.index])
    return Document(page_content=content, metadata={"Player": row['name'], "Season": row['Year']})

documents = [convert_row_to_document(row) for _, row in df.iterrows()]

# Split documents into smaller chunks if needed
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents_list = []
for doc in documents:
    split_documents_list.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in text_splitter.split_text(doc.page_content)])

# Initialize embeddings
try:
    print("Initializing Ollama embeddings...")
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    print("Ollama embeddings initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Ollama embeddings: {e}")
    exit(1)

# Initialize vectorstore
try:
    print("Initializing vectorstore...")
    vectorstore = Chroma(collection_name="nba_data", embedding_function=oembed)
    print("Vectorstore initialized successfully.")
except Exception as e:
    print(f"Failed to initialize vectorstore: {e}")
    exit(1)

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
    exit(1)

# Set up LangChain components
llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_template("Tell me the point totals for {player}")

# Create a StuffDocumentsChain
combine_docs_chain = StuffDocumentsChain(
    prompt_template=prompt,
    llm=llm,
    output_parser=StrOutputParser(),
)

# Create a Conversational Retrieval Chain
chain = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=combine_docs_chain,
    return_intermediate_steps=True,
)

# User input
user_message = "LeBron James"

# Using the chain to get a response
response = chain.invoke({"question": user_message})

# Print the response
print(response)

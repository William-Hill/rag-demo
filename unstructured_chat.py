from unstructured.partition.html import partition_html
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

cnn_lite_url = "https://lite.cnn.com/"
elements = partition_html(url=cnn_lite_url)
links = []

for element in elements:
    if element.metadata.link_urls:
        relative_link = element.metadata.link_urls[0][1:]
        if relative_link.startswith("2024"):
            links.append(f"{cnn_lite_url}{relative_link}")


loaders = UnstructuredURLLoader(urls=links, show_progress_bar=True)
docs = loaders.load()


# embeddings = OpenAIEmbeddings()
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = Chroma.from_documents(docs, oembed)
query_docs = vectorstore.similarity_search("What happened in Atlanta recently?", k=1)

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm = ChatOllama(model="llama3")
chain = load_summarize_chain(llm, chain_type="stuff")
# chain.run(query_docs)
print(chain.invoke(query_docs))
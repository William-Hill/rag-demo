# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# /docs/concepts/#langchain-expression-language-lcel

#Chaining can mean making multiple LLM calls in a sequence. Language models are often non deterministic and can make errors, 
# so making multiple calls to check previous outputs or to break down larger tasks into bite-sized steps can improve results.
#OutputParser that parses LLMResult into the top likely string.
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
user_message = input("Enter a topic to make a joke about: ")
print(chain.invoke({"topic": user_message}))
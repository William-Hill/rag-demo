# Query the llama 3 model.  The response is all printed out at once. Not a great user experience
import ollama

user_message = input("Enter a prompt: ")
response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': user_message,
  },
])
print(response['message']['content'])
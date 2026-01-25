import ollama

def chat_with_llama():
    model_name = 'llama3.2:1b'
    while True:
        user_input = input("\nType your prompt here: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        response = ollama.generate(model = model_name, prompt = user_input)
        print(f"Response:\n{response['response']}")

if __name__ == "__main__":
    chat_with_llama()
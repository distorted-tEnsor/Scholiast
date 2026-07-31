# local_llm.py
# A reusable "phone" for talking to our local model.
# Any part of the project can use this instead of rewriting the connection code.

import requests


class LocalLLM:
    """A simple handle to the local model served by Ollama.

    Create one, then call .ask() as many times as you like:
        llm = LocalLLM()
        print(llm.ask("What is a transformer?"))
    """

    def __init__(self, model="qwen2.5:3b"):
        # Remember which model to use and where Ollama is listening.
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def ask(self, question):
        """Send a question to the model and return its answer as text."""
        payload = {
            "model": self.model,
            "prompt": question,
            "stream": False,
        }
        response = requests.post(self.url, json=payload)
        return response.json()["response"]


# This block only runs if you run THIS file directly (a built-in self-test).
# It won't run when other parts of the project import LocalLLM.
if __name__ == "__main__":
    llm = LocalLLM()
    print("Testing the reusable LLM tool...\n")
    print(llm.ask("In one sentence, what is a research paper abstract?"))
    print()
    print(llm.ask("In one sentence, what is a citation?"))
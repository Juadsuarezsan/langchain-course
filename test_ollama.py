from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def test_ollama():
    # Crear el modelo Ollama (usando el que ya tienes)
    llm = OllamaLLM(model="gemma3:270m")
    
    # Crear un prompt simple
    prompt = PromptTemplate(
        template="Responde en español: {question}",
        input_variables=["question"]
    )
    
    # Crear la cadena
    chain = prompt | llm
    
    # Probar
    response = chain.invoke({"question": "¿Cuál es la capital de Francia?"})
    print("Respuesta:", response)

if __name__ == "__main__":
    test_ollama()

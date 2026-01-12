from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = Flask(__name__)
load_dotenv()

# ------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ------------------------------------------------------------------
# LLM (Hugging Face)
# ------------------------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)


# Vector DB
# ------------------------------------------------------------------
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


# Prompt + RAG

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


# Routes

@app.route("/")
def index():
    return render_template("bot.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")

    if not user_input:
        return "Please ask a medical question."

    response = rag_chain.invoke({"input": user_input})

    answer = response["answer"]

    # 🔹 Extract source info
    sources = []
    for doc in response.get("context", []):
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", None)

        if source:
            filename = source.split("\\")[-1]
            if page is not None:
                sources.append(f"{filename} (Page {page})")
            else:
                sources.append(filename)

    # Remove duplicates
    sources = list(set(sources))

    # 🔹 Format final response
    if sources:
        answer += "\n\n📄 Sources:\n"
        for src in sources:
            answer += f"- {src}\n"

    return answer.strip()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

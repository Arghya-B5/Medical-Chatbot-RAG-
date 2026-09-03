# from flask import Flask, render_template, request
# from dotenv import load_dotenv
# import os

# from src.helper import download_hugging_face_embeddings
# from src.prompt import system_prompt

# from langchain_pinecone import PineconeVectorStore
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# # ------------------------------------------------------------------
# # App setup
# # ------------------------------------------------------------------
# app = Flask(__name__)
# load_dotenv()

# # ------------------------------------------------------------------
# # ENV
# # ------------------------------------------------------------------
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# if not PINECONE_API_KEY:
#     raise ValueError("PINECONE_API_KEY not set")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# # ------------------------------------------------------------------
# # LLM (Hugging Face)
# # ------------------------------------------------------------------
# # llm = HuggingFaceEndpoint(
# #     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
# #     task="text-generation",
# #     temperature=0.3,
# #     max_new_tokens=512,
# # )
# from langchain_groq import ChatGroq

# llm = ChatGroq(
#     model="openai/gpt-oss-20b",
#     temperature=0.2
# )

# # model = ChatHuggingFace(llm=llm)


# # Vector DB
# # ------------------------------------------------------------------
# embeddings = download_hugging_face_embeddings()

# docsearch = PineconeVectorStore.from_existing_index(
#     index_name="medical-chatbot",
#     embedding=embeddings,
# )

# retriever = docsearch.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )


# # Prompt + RAG

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# # qa_chain = create_stuff_documents_chain(model, prompt)
# # rag_chain = create_retrieval_chain(retriever, qa_chain)
# question_answer_chain = create_stuff_documents_chain(
#     llm,
#     prompt
# )

# rag_chain = create_retrieval_chain(
#     retriever,
#     question_answer_chain
# )

# # Routes

# @app.route("/")
# def index():
#     return render_template("bot.html")

# @app.route("/get", methods=["POST"])
# def chat():
#     user_input = request.form.get("msg")

#     if not user_input:
#         return "Please ask a medical question."

#     response = rag_chain.invoke({"input": user_input})

#     answer = response["answer"]

#     # 🔹 Extract source info
#     sources = []
#     for doc in response.get("context", []):
#         source = doc.metadata.get("source", "")
#         page = doc.metadata.get("page", None)

#         if source:
#             filename = source.split("\\")[-1]
#             if page is not None:
#                 sources.append(f"{filename} (Page {page})")
#             else:
#                 sources.append(filename)

#     # Remove duplicates
#     sources = list(set(sources))

#     # 🔹 Format final response
#     if sources:
#         answer += "\n\n📄 Sources:\n"
#         for src in sources:
#             answer += f"- {src}\n"

#     return answer.strip()


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)

from flask import Flask, render_template, request
from dotenv import load_dotenv

import os
import time
import mlflow

# LangChain
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



# Load Environment Variables


load_dotenv()



# Flask App


app = Flask(__name__)



# MLFLOW + DAGSHUB


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TRACKING_URI:
    raise ValueError("MLFLOW_TRACKING_URI not set")

if not MLFLOW_TRACKING_USERNAME:
    raise ValueError("MLFLOW_TRACKING_USERNAME not set")

if not MLFLOW_TRACKING_PASSWORD:
    raise ValueError("MLFLOW_TRACKING_PASSWORD not set")

# DagsHub MLflow URL
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create / use experiment
mlflow.set_experiment("Medical-Chatbot-RAG")



# PINECONE API KEY


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



# GROQ LLM


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)



# HUGGING FACE EMBEDDINGS


embeddings = download_hugging_face_embeddings()



# PINECONE VECTOR DATABASE


docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)



# RETRIEVER


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)



# PROMPT


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)



# RAG CHAIN


question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)



# HOME PAGE


@app.route("/")
def index():
    return render_template("bot.html")



# CHAT ROUTE


@app.route("/get", methods=["POST"])
def chat():

    print("\n==============================")
    print("CHAT REQUEST RECEIVED")
    print("==============================")

    user_input = request.form.get("msg")

    print("User input:", user_input)

    if not user_input:
        return "Please ask a medical question."

    try:

        start_time = time.time()

        print("Starting MLflow run...")

        with mlflow.start_run():

            print("MLflow run started")

            
            # Log parameters
            

            mlflow.log_params({
                "llm_model": "openai/gpt-oss-20b",
                "temperature": 0.2,
                "vector_database": "Pinecone",
                "retrieval_method": "similarity",
                "top_k": 3
            })

            print("Parameters logged")

            
            # Run RAG
            

            print("Calling RAG chain...")

            response = rag_chain.invoke({
                "input": user_input
            })

            print("RAG response received")

            
            # Answer
           

            answer = response["answer"]

            print("Answer generated")

            
            # Retrieved documents
            

            retrieved_docs = response.get("context", [])

            print(
                "Retrieved documents:",
                len(retrieved_docs)
            )

           
            # Latency
            

            latency = time.time() - start_time

            
            # Metrics
           

            mlflow.log_metric(
                "response_latency_seconds",
                latency
            )

            mlflow.log_metric(
                "retrieved_documents",
                len(retrieved_docs)
            )

            mlflow.log_metric(
                "answer_length",
                len(answer)
            )

           
            # Sources
           

            sources = []

            for doc in retrieved_docs:

                source = doc.metadata.get(
                    "source",
                    ""
                )

                page = doc.metadata.get(
                    "page",
                    None
                )

                if source:

                    filename = source.split("\\")[-1]

                    if page is not None:

                        sources.append(
                            f"{filename} (Page {page})"
                        )

                    else:

                        sources.append(
                            filename
                        )

            # Remove duplicates

            sources = list(set(sources))

            mlflow.log_metric(
                "unique_sources",
                len(sources)
            )

           
            # Tags
           
            mlflow.set_tags({
                "Author": "Arghya",
                "Project": "Medical Chatbot",
                "Framework": "LangChain",
                "LLM": "Groq",
                "Vector_Database": "Pinecone",
                "Application": "RAG"
            })

           
            # Add sources
            

            if sources:

                answer += "\n\n📄 Sources:\n"

                for src in sources:

                    answer += f"- {src}\n"

        print("MLflow run completed")
        print("Returning answer")

        return answer.strip()

    except Exception as e:

        print("\n==============================")
        print("ERROR")
        print("==============================")

        print(type(e).__name__)
        print(str(e))

        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )    
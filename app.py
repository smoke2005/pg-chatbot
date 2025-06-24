import pandas as pd
import uuid
import json
import os
from flask import Flask, request, jsonify, render_template
from google.cloud import aiplatform
import google.generativeai as genai
from vertexai.language_models import TextEmbeddingModel
from dotenv import load_dotenv
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_DIM = 768
CSV_PATH = "vector_store/faq_data.csv"

# Setup
app=Flask(__name__)

# Configuration
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
ENDPOINT_RESOURCE_NAME = os.getenv("ENDPOINT_RESOURCE_NAME")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

aiplatform.init(project=PROJECT_ID, location=REGION)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
chat_model = genai.GenerativeModel("gemini-1.5-pro")

# Process and Upload Embeddings 
def process_and_upload_embeddings():
    df = pd.read_csv(CSV_PATH)
    vectors = []

    for _, row in df.iterrows():
        q = str(row['User Question'])
        a = " ".join(filter(None, [
            str(row.get('First Touch Answer', '')),
            str(row.get('Second Touch Answer', '')),
            str(row.get('Conclusion', ''))
        ]))

        if q.strip() and a.strip():
            embedding = embedding_model.get_embeddings([q])[0].values
            vectors.append({
                "id": str(uuid.uuid4()),
                "embedding": embedding,
                "metadata": {"question": q, "answer": a}
            })

    with open("embedding_data.jsonl", "w") as f:
        for item in vectors:
            f.write(json.dumps(item) + "\n")

    os.system(f"gsutil cp embedding_data.jsonl gs://{BUCKET_NAME}/embedding_data.jsonl")
    return f"gs://{BUCKET_NAME}/embedding_data.jsonl"

# Get Existing Endpoint 
def get_existing_endpoint():
    return aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_RESOURCE_NAME)

#To get
def load_local_metadata(jsonl_path):
    metadata_lookup = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            metadata_lookup[entry["id"]] = entry["metadata"]
    return metadata_lookup

#Fallback find_neighbors if endpoint.find_neighbors does not work
def find_neighbors(query_embedding, embeddings, metadata, top_k=3):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(similarities[idx]),
            "question": metadata[idx]["question"],
            "answer": metadata[idx]["answer"]
        })

    return results

# Retrieve Context
def retrieve_context(query):
    query_vector = embedding_model.get_embeddings([query])[0].values
    endpoint1=aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_RESOURCE_NAME)
    response = endpoint1.find_neighbors(
        deployed_index_id="pg_cb_1750666469107",
        queries=[query_vector],
        num_neighbors=3,
    )
    
    context_lines = []

    if not response:
        return "Sorry, coouldn't find context"

    for idx, neighbor in enumerate(response[0]):
        neighbor_id=neighbor.id
        metadata = metadata_lookup.get(neighbor_id)
        if metadata:
            q = metadata.get("question", "[Missing Question]")
            a = metadata.get("answer", "[Missing Answer]")
            context_lines.append(f"Q: {q}\nA: {a}")
        else:
            context_lines.append(f"[No metadata found for ID: {neighbor_id}]")

    return "\n\n".join(context_lines) if context_lines else "No context found."
    

# Gemini Chat
def chat_with_context(query, context):
    prompt = f"""
        You are ParentGeenee, a kind, supportive virtual assistant built to help parents navigate digital safety and parenting questions. You speak with empathy and clarity, and your role is to gently guide parents using the most relevant, accurate information provided.

        Instructions:
        - Carefully read the **Context** below before answering.
        - Use a warm, encouraging tone, like a friendly expert speaking to a concerned parent.
        - Keep language simple and jargon-free.
        - If steps or advice are involved, use friendly bullet points or numbers.
        - Be concise but informative — prioritize reassurance and clarity.
        - If the query is unrelated to ParentGeenee, gently say:
        > “I'm here to help with questions specifically about ParentGeenee. Please ask me something related to the app or its features.”
        - If you’re unsure or the context is too limited, reply:
        > “That’s a great question. Could you please rephrase it or provide more details so I can assist you better?”

        Context:
        {context}

        User Question:
        {query}

        Your Response:

        """
    response = chat_model.generate_content(prompt)
    return response.text.strip()

# Manual Test 
def ask():
    user_input = "how can i add a parent"
    context=retrieve_context(user_input)
    print(f"\nContext:\n{context}")
    answer = chat_with_context(user_input, context)
    print(f"\nAnswer:\n{answer}")

# Main 
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query", "")
    context=retrieve_context(user_query)
    print(f"\nContext:\n{context}")
    answer = chat_with_context(user_query, context)
    print(f"\nAnswer:\n{answer}")
    return jsonify({"response": answer})

if __name__ == "__main__":
    metadata_lookup = load_local_metadata("embedding_data.jsonl")
    app.run(host="0.0.0.0", port=5000, debug=True)


    

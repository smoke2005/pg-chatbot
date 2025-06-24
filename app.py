import pandas as pd
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import uuid
import json
import os

# Configuration
PROJECT_ID = "abstract-pod-382707"
REGION = "us-central1"
BUCKET_NAME = "pg-cb-bucket"
INDEX_DISPLAY_NAME = "pg-cb-index"
ENDPOINT_DISPLAY_NAME = "parentgeenee-faq-endpoint-1"
DEPLOYED_INDEX_ID = "pg-cb"
EMBEDDING_DIM = 768

# Init vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# Process CSV and upload embeddings to GCS
def process_and_upload_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    vectors = []

    for _, row in df.iterrows():
        question = str(row['User Question'])
        answer = " ".join(filter(None, [
            str(row.get('First Touch Answer', '')),
            str(row.get('Second Touch Answer', '')),
            str(row.get('Conclusion', ''))
        ]))

        if question.strip() and answer.strip():
            embedding = embedding_model.get_embeddings([question])[0].values
            unique_id = str(uuid.uuid4())
            vectors.append({
                "id": unique_id,
                "embedding": embedding,
                "metadata": {
                    "question": question,
                    "answer": answer
                }
            })

    json_path = "embedding_data.json"
    with open(json_path, "w") as f:
        json.dump(vectors, f, indent=2)

    # Upload to GCS
    os.system(f"gsutil cp {json_path} gs://{BUCKET_NAME}/embedding_data.json")
    return f"gs://{BUCKET_NAME}/embedding_data.jsonl", len(vectors)

# Create Matching Engine Index
def create_index(gcs_uri):
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=gcs_uri,
        dimensions=EMBEDDING_DIM,
        approximate_neighbors_count=150,
        distance_measure_type="DOT_PRODUCT_DISTANCE"
    )
    return index

# Deploy Index or reuse existing endpoint
def deploy_index(index):
    endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f"display_name='{ENDPOINT_DISPLAY_NAME}'"
    )
    print(endpoints)
    if endpoints:
        endpoint = endpoints[0]
        print(f"Reusing existing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=ENDPOINT_DISPLAY_NAME,
            public_endpoint_enabled=True
        )
        print(f"Created new endpoint: {endpoint.resource_name}")

    if not endpoint.deployed_indexes:
        endpoint.deploy_index(
            index=index,
            deployed_index_id=DEPLOYED_INDEX_ID
        )
        print("Index deployed to endpoint.")
    else:
        print("Index already deployed.")

    return endpoint


if __name__ == "__main__":
    csv_path = "vector_store/faq_data.csv"
    gcs_uri, num_vectors = process_and_upload_embeddings(csv_path)
    index = create_index(gcs_uri)
    endpoint = deploy_index(index)



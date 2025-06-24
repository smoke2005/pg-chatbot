# ParentGeenee Chatbot

## How It Works

1. Converts FAQ data into text embeddings using `text-embedding-005`.
2. Saves the embeddings in `.jsonl` format and uploads them to Cloud Storage.
3. A Matching Engine index is created from this data to enable semantic search.
4. When the user submits a query:

   * The system finds the most relevant FAQs from the index.
   * These are passed as context to Gemini Pro.
   * A friendly and supportive response is generated for the parent.

---

## Setup Option 1: Run via Python (Locally)

### 1. Clone the repository

Clone this repo to your machine using GitHub.

### 2. Install Python dependencies

Install the required libraries using `pip install -r requirements.txt`. This includes Flask, Pandas, Google Cloud SDKs, dotenv, and more.

### 3. Create a `.env` file

Add your project configuration like GEMINI API key, project ID, bucket name, and endpoint details.

### 4. Run the 'setup.py' script

This script processes the CSV, generates embeddings, saves them in `.jsonl`, uploads to Cloud Storage, and creates + deploys the Matching Engine index.

### 5. Launch the Flask chatbot

Run the main app file and open your browser to interact with the chatbot.

---

## Setup Option 2: Using Google Cloud Console 

You can also set everything up through the GCP web interface without running Python scripts.

### Step 1: Enable Required APIs

Enable the following APIs from the Google Cloud Console:

* Vertex AI API
* Cloud Storage API
* Matching Engine API

### Step 2: Create a Cloud Storage Bucket

Create a bucket named something like `pg-cb-bucket` with region `us-central1`. This will store your vector file.

### Step 3: Upload the Embedding File

Upload your `embedding_data.json` file to this bucket. Copy the full Cloud Storage path.

### Step 4: Create the Matching Engine Index

* Go to Vertex AI → Indexes
* Click “Create Index”
* Use Matching Engine
* Choose your `.json` file in Cloud Storage
* Set embedding dimension to 768 and distance type to DOT_PRODUCT_DISTANCE

Wait for the index to be created.

### Step 5: Deploy the Index to an Endpoint

* Go to the newly created index
* Click “Deploy to Endpoint”
* Either create a new endpoint or reuse an existing one
* Once deployed, note down the Endpoint Resource Name and Deployed Index ID

Update these values in your `.env` file or code.

### 6. Launch the Flask chatbot

Run the main app file and open your browser to interact with the chatbot.


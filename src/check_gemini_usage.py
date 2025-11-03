import requests
import json
import os

# 1️⃣  Replace this with your Gemini API key from https://aistudio.google.com/app/apikey
API_KEY = os.getenv("GEMINI_API_KEY") or"AIzaSyB9DvxYb-b-rO4Mm-mV12s9PRq1FtqqmWw"

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# 2️⃣  List all available models for your key
def list_models():
    url = f"{BASE_URL}/models"
    headers = {"x-goog-api-key": API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print("\n✅ Available models:")
        for m in models:
            print(f" - {m['name']} ({m.get('displayName', 'No display name')})")
    else:
        print("❌ Error fetching models:", response.text)

# 3️⃣  Test call to embeddings model to ensure usage appears in dashboard
def test_embedding():
    url = f"{BASE_URL}/models/gemini-embedding-001:embedContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }
    data = {
        "model": "models/gemini-embedding-001",
        "content": { "parts": [ { "text": "hello world" } ] }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("\n🔍 Embedding API call result:")
    print("Status:", response.status_code)
    print("Response headers (may contain rate limit info):")
    for k, v in response.headers.items():
        if "quota" in k.lower() or "limit" in k.lower() or "remaining" in k.lower():
            print(f"  {k}: {v}")
    print("\nResponse body:", response.text)

if __name__ == "__main__":
    print("Checking Gemini models and embeddings availability...\n")
    # list_models()
    test_embedding()
    print("\n✅ Done! Now go to https://aistudio.google.com/app/apikey → View usage to see updated quota.")

from fastapi.testclient import TestClient
import app as appmod

# Simple stub document type
class StubDoc:
    def __init__(self, source, page, content):
        self.metadata = {"source": source, "page": page}
        self.page_content = content

class StubRetriever:
    def get_relevant_documents(self, query):
        return [StubDoc("test.pdf", 1, "This is a test snippet about the subject.")]


def stub_qa(question, history):
    return "This is a stubbed answer."


def test_query_with_stubs():
    # Inject stubs into the running app module
    appmod._retriever = StubRetriever()
    appmod._qa = stub_qa

    client = TestClient(appmod.app)
    resp = client.post("/query", json={"query": "What is this?", "top_k": 3, "style": "summary"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "This is a stubbed answer."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "test.pdf"

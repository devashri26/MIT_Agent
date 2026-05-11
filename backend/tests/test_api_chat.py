from fastapi.testclient import TestClient

from backend.api.main import app


# Pin every test to provider="mock" so they don't depend on which API keys are exported
# in the developer's environment. The real-provider path is exercised manually.
def _post(client: TestClient, path: str, body: dict):
    body = {**body, "provider": "mock"}
    return client.post(path, json=body)


def test_answer_endpoint_returns_grounded_answer() -> None:
    client = TestClient(app)
    response = _post(
        client, "/answer", {"query": "What is MCA eligibility?", "top_k": 3, "run_judge": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "confidence" in data
    assert "hallucination" in data


def test_chat_endpoint_creates_session_and_returns_state() -> None:
    client = TestClient(app)
    response = _post(
        client, "/chat", {"query": "What is MCA eligibility?", "top_k": 3, "run_judge": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"]
    assert "answer" in data
    assert "conversation_state" in data
    assert len(data["conversation_state"]["turns"]) >= 2


def test_chat_endpoint_followup_carries_session() -> None:
    client = TestClient(app)
    first = _post(
        client, "/chat", {"query": "What is MCA eligibility?", "top_k": 3, "run_judge": False}
    ).json()
    session_id = first["session_id"]
    second = _post(
        client,
        "/chat",
        {
            "query": "What about hostel fees?",
            "session_id": session_id,
            "top_k": 3,
            "run_judge": False,
        },
    ).json()
    assert second["session_id"] == session_id
    assert len(second["conversation_state"]["turns"]) >= 4


def test_chat_stream_returns_sse_stream() -> None:
    client = TestClient(app)
    with client.stream(
        "POST",
        "/chat/stream",
        json={"query": "What is MCA eligibility?", "top_k": 3, "provider": "mock"},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = b"".join(chunk for chunk in response.iter_bytes())
    assert b"event: meta" in body
    assert b"data:" in body


def test_conversation_reset() -> None:
    client = TestClient(app)
    created = _post(
        client, "/chat", {"query": "MCA eligibility", "top_k": 3, "run_judge": False}
    ).json()
    session_id = created["session_id"]
    state = client.get(f"/conversation/{session_id}").json()
    assert state.get("session_id") == session_id
    client.delete(f"/conversation/{session_id}")
    after = client.get(f"/conversation/{session_id}").json()
    assert after["exists"] is False

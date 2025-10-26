from langchain_core.documents import Document
import json
from typing import Any

# Copiar la función helper del event_translator
def _serialize_tool_response(response: Any) -> str:
    """Serialize tool response to JSON, handling LangChain Documents and other objects."""
    def convert_to_dict(obj):
        """Recursively convert objects to dict."""
        if isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif hasattr(obj, 'model_dump'):
            # LangChain Document and Pydantic models
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            # Older Pydantic models
            return obj.dict()
        else:
            return obj

    return json.dumps(convert_to_dict(response))

# Test 1: Single Document
print("=" * 60)
print("TEST 1: Single Document")
print("=" * 60)
d = Document(page_content="test content", metadata={"source": "test.txt", "page": 1})

try:
    result = _serialize_tool_response(d)
    print("✅ SUCCESS")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: List of Documents (como retorna el RAG tool)
print("\n" + "=" * 60)
print("TEST 2: List of Documents (RAG tool response)")
print("=" * 60)
docs = [
    Document(page_content="doc 1", metadata={"source": "file1.txt"}),
    Document(page_content="doc 2", metadata={"source": "file2.txt"}),
    Document(page_content="doc 3", metadata={"source": "file3.txt"}),
]

try:
    result = _serialize_tool_response(docs)
    print("✅ SUCCESS")
    print(f"Result length: {len(result)} chars")
    # Parse to verify it's valid JSON
    parsed = json.loads(result)
    print(f"Parsed {len(parsed)} documents")
    print(f"First doc: {parsed[0]}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Regular objects (should still work)
print("\n" + "=" * 60)
print("TEST 3: Regular Python objects")
print("=" * 60)
regular_response = {"status": "ok", "count": 42, "items": ["a", "b", "c"]}

try:
    result = _serialize_tool_response(regular_response)
    print("✅ SUCCESS")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)

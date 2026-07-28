import inspect
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
print('QDRANT_URL', os.getenv('QDRANT_URL'))
print('QDRANT_API_KEY', bool(os.getenv('QDRANT_API_KEY')))
print('qdrant-client module loaded')

client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
print('collection exists', client.collection_exists('document_chunks'))
info = client.get_collection('document_chunks')
print('points_count', info.points_count)

query = 'what is an organization'
response = client.query_points(
    collection_name='document_chunks',
    query=query,
    using='dense',
    limit=3,
    with_payload=True,
    with_vectors=False,
)

print('response type', type(response))
print('has result', hasattr(response, 'result'))
print('result type', type(response.result) if hasattr(response, 'result') else None)
print('result repr', repr(response.result)[:200])
for hit in response.result or []:
    print('hit type', type(hit))
    print('has payload', hasattr(hit, 'payload'))
    print('payload', getattr(hit, 'payload', None))
    print('score', getattr(hit, 'score', None))
    print('id', getattr(hit, 'id', None))
    print('vector', getattr(hit, 'vector', None))
    print('---')

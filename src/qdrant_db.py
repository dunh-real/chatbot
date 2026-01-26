from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

class Qdrant_database():
    def __init__(self, collection_name, record_size):
        self.collection_name = collection_name
        self.record_size = record_size
        self.client = QdrantClient(url = "http://localhost:6333")
    
    def create_collection(self):
        self.client.create_collection(
            collection_name = self.collection_name,
            vectors_config = VectorParams(seize = self.record_size, distance = Distance.DOT),
        )
    
    def add_vectors(self, tenant_id, chunk_id, vector, payload):
        self.client.upsert(
            collection_name = self.collection_name,
            wait = True,
            points = PointStruct(tenant_id = tenant_id, chunk_id = chunk_id, vector = vector, payload = payload)
        )
    
    def query(self, query, limit, tenant_id):
        search_results = self.client.query_points(
            collection_name = self.collection_name,
            query = query,
            query_filter = Filter(
                must = [FieldCondition(key = "tenant_id", match = MatchValue(value = tenant_id))]
            ),
            with_payload = True,
            limit = limit
        ).points
        
        return search_results
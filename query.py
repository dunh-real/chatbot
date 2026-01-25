from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
import json


"""
Output format:
type: list
number of json string: 3
json info: id (tenant ID), score: ranking for top k search, payload: dictionary contains text, vector, shared_key, order_value
"""

# initialize the client
client = QdrantClient(url = 'http://localhost:6333')

query_text = str(input("Enter you prompt: "))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(query_text)
print(type(embeddings))

# run a query
search_result = client.query_points(
    collection_name = "test1_collection",
    query = embeddings,
    with_payload = True,
    limit = 3
).points
print(type(search_result[0]))


# convert output to json
def convert_output_to_json(output_text):
    output_text = search_result
    final_str = []
    for i in output_text:
        data_dict = i.__dict__
        json_str = json.dumps(data_dict)
        final_str.append(json_str)
    
    return final_str

json_str = convert_output_to_json(search_result)
print(json_str)


# run a query with a filter
# from qdrant_client.models import Filter, FieldCondition, MatchValue

# search_result = client.query_points(
#     collection_name = "test1_collection",
#     query = [0.2, 0.1, 0.9, 0.7],
#     query_filter = Filter(
#         must = [FieldCondition(key = "city", match = MatchValue(value = "Berlin"))]
#     ),
#     with_payload = True,
#     limit = 3,
# ).points


from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import csv
import uuid
import config
from config import get_prompt_template, PromptTemplate
import time, os

pc = Pinecone(api_key=config.PINECONE_API_KEY)
dims = 3072
spec = ServerlessSpec(
    cloud="aws", region="us-east-1"  # us-east-1
)

# check if index already exists (it shouldn't if this is first time)
existing_indexes = pc.list_indexes()

if config.PINECONE_INDEX_NAME not in [item["name"] for item in existing_indexes]:
    # if does not exist, create index
    print("creating index on pinecone...")
    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=dims,  # dimensionality of embed 3
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(config.PINECONE_INDEX_NAME).status['ready']:
        time.sleep(1)
else:
    print(f"Index with name '{config.PINECONE_INDEX_NAME}' already exists.")
    # user_input = input("Would you like to delete and recreate the index? (y/n): ").lower()
    # if user_input == 'y':
    #     print(f"Deleting index '{config.PINECONE_INDEX_NAME}'...")
    #     pc.delete_index(config.PINECONE_INDEX_NAME)
    #     print("Creating new index...")
    #     pc.create_index(
    #         name=config.PINECONE_INDEX_NAME,
    #         dimension=dims,
    #         metric='cosine',
    #         spec=spec
    #     )
    #     while not pc.describe_index(config.PINECONE_INDEX_NAME).status['ready']:
    #         time.sleep(1)
    #     print("Index recreated successfully!")
    # else:
    #     print("Using existing index.")

# connect to index
index = pc.Index(config.PINECONE_INDEX_NAME)
print("index status:")
print(index.describe_index_stats())

embed_model = OpenAIEmbeddings(
    model=config.ModelType.embedding,
    openai_api_key=config.OPENAI_API_KEY
)

# embed and index all our our data!
def import_csv_to_vector(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        id = 0
        
        for row in reader:
            id = id + 1
            embedding_id = str(uuid.uuid4())
            vector = [{
                'id': embedding_id,
                'values':embed_model.embed_documents([get_prompt_template(PromptTemplate.SAVED_REPLY).format(title=row[3], details=row[4])])[0],
                'metadata': {
                    'id': id
                },
            }]
            index.upsert(vectors=vector)
            print(f"{id}: done")
        
    print("CSV data imported successfully into pinecone vector database.")

def format_rag_contexts(matches: list):
    contexts = []
    ids = []
    for x in matches:
        ids.append(x['metadata']['id'] + 1)
    with open(os.path.join("fixture", "info.csv"), 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for index, row in enumerate(reader, start = 2):
            if index in ids:
                text = get_prompt_template(PromptTemplate.SAVED_REPLY).format(title=row[3], details=row[4])
                contexts.append(text)
    context_str = "\n---\n".join(contexts)
    # print(context_str)
    return context_str

def query_pinecone(query: str, top_k = 5):
    #query pinecone and return list of records
    xq = embed_model.embed_documents([query])

    # initialize the vector store object
    xc = index.query(
        vector=xq[0], top_k=top_k, include_values=True, include_metadata=True
    )

    context_str = format_rag_contexts(xc["matches"])
    # print(context_str)
    return context_str

import typesense
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables, even if they are already loaded
load_dotenv()

API_KEY = os.getenv("TYPESENSE_API_KEY")
COLLECTION_NAME = os.getenv("TYPESENSE_COLLECTION_NAME")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST")
CHUNK_SIZE = 40

client = typesense.Client({
  'api_key': API_KEY,
  'nodes': [{
    'host': TYPESENSE_HOST,
    'port': '8108',
    'protocol': 'http'
  }],
  'connection_timeout_seconds': 600
})

    

# I need here a df or dictionary with doi and list of entities

# Read pickle file with sentences
df = pd.read_parquet("../datasets/ready_to_index.parquet")

# Delete rows where predictions is an empty list
def list_is_empty(predictions: list):
    
    if len(predictions) == 0:
        return True
    else:
        return False
    
df = df[~df["predictions"].apply(list_is_empty)]

# Convert from array to list, because numpy arrays cannot be converted to json
df["predictions"] = df["predictions"].apply(lambda x: list(x))

# Convert dataframe to list of dictionaries
data = df.to_dict('records')

# Split data into chunks using a generator
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

print("Total chunks: ", len(list(chunks(data, CHUNK_SIZE))))

# Split data into chunks
chunk_counter = 0
for chunk in chunks(data, CHUNK_SIZE):
    # Send chunk to Typesense
    response = client.collections[COLLECTION_NAME].documents.import_(chunk, {'action': 'create'})

    # Check if there was an error in any of the responses
    for res in response:
        if res['success'] == False:
            print("ERROR")
            print(res['error'])
            break
    print("Sending chunk to Typesense")

    chunk_counter += 1
    print("Chunk: ", chunk_counter)

print("Finished sending data to Typesense")


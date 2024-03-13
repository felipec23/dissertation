"""
# For starting the Typesense server:
1. Start running docker in Macbook (Docker Desktop)
2. Run this from the terminal: 

export TYPESENSE_API_KEY=XXXXX
    
mkdir $(pwd)/tesis-data

docker run -p 8108:8108 -v$(pwd)/tesis-data:/data typesense/typesense:0.24.1 \
  --data-dir /data --api-key=$TYPESENSE_API_KEY --enable-cors

  
To end:
docker compose down  
"""

# !pip install --upgrade typesense

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()

COLLECTION_NAME = os.getenv("TYPESENSE_COLLECTION_NAME")
HOST = os.getenv("TYPESENSE_HOST")

import typesense
client = typesense.Client({
  'api_key': os.getenv("TYPESENSE_API_KEY"),
  'nodes': [{
    'host': HOST,
    'port': '8108',
    'protocol': 'http'
  }],
  'connection_timeout_seconds': 600
})

# Try to delete it, if exists
# Drop pre-existing collection if any
try:
    client.collections[COLLECTION_NAME].delete()
except Exception as e:
    pass

create_response = client.collections.create({
  "name": COLLECTION_NAME,
  "fields": [
    {"name": "title", "type": "string", "index": True, "optional": False },
    {"name": "abstract", "type": "string", "index": True, "optional": False },
    {"name": "doi", "type": "string", "index": False, "optional": True },
    {"name": "authors", "type": "string", "index": False, "optional": True },
    {"name": "predictions", "type": "string[]", "index": True, "optional": False },
    {"name": "link", "type": "string", "index": False, "optional": True },
    {"name": "date", "type": "string", "index": False, "optional": True }
  ]
})

# List existing collections
print(client.collections.retrieve())
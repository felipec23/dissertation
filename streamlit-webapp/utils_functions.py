import pandas as pd
import streamlit as st
import typesense
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load Typesense client
TYPSENSE_HOST = os.getenv("TYPESENSE_HOST")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
COLLECTION_NAME = os.getenv("TYPESENSE_COLLECTION_NAME")

@st.cache_resource
def load_typesense_client():

    global client
    print("Loading model and typesense client...")

    client = typesense.Client({
    'api_key': TYPESENSE_API_KEY,
    'nodes': [{
        'host': TYPSENSE_HOST,
        'port': '8108',
        'protocol': 'http'
    }],
    'connection_timeout_seconds': 600
    })

    return client

client = load_typesense_client()

def process_entities(entities: list) -> str:

    cleaned_entities = []
    for entity in entities:
        try:
            entity, value, unit = entity.split("|")
            cleaned_entities.append("{}: {} {}".format(entity, value, unit))
        except:
            print("Error with entity: ", entity)

    # Now, convert to a string
    cleaned_entities = "; ".join(cleaned_entities)
    return cleaned_entities


@st.cache_data
def read_data():
    df = pd.read_pickle("results-tensile.pkl")

    # Convert to list of dictionaries
    response = []
    for n_row, row in df.reset_index().iterrows():
        dictionary = {
            "title": row['title'],
            "authors": 'Felipe Caceres',
            "abstract": row['abstract'],
            "url": 'https://www.google.com',
            "tags": process_entities(row['entities']),
            "publishedDate": 1620000000,
            "publishedDate_string": "May, 2021"
        }
        response.append(dictionary)

    return response




def process_hits(response):

    data = []
    for hit in response['hits']:
        
        match_snippets = hit['highlights'][0]['snippets']
        match_snippets = [snippet.replace("<mark>", "").replace("</mark>", "") for snippet in match_snippets]

        for match in match_snippets:

            try:
                name, value, unit = match.split(" | ")
                # print("match: ", match)
                data.append({
                    "title": hit["document"]["title"],
                    "variable": name,
                    "value": value,
                    "unit": unit
                    })
                
            except Exception as e:
                print("Error processing document: ", e)

    return data


@st.cache_data
def perform_query(search_config):

    # Perform the query for the first page
    response = client.collections[COLLECTION_NAME].documents.search(search_config)

    print("Page requested:", response["page"])

    all_pages_data = []

    # Process data for first page
    data = process_hits(response)

    # Append to the list
    all_pages_data.extend(data)

    total_hits = response['found']

    # st.write("Total hits: ", total_hits)

    # Number of pages
    N_pages = total_hits // 200

    # Create a list with the pages, starting from 2
    pages = list(range(2, N_pages+1))

    i = 0
    for page in pages:
        search_config['page'] = page
        response = client.collections[COLLECTION_NAME].documents.search(search_config)
        print("Page requested:", response["page"])

        # Process data
        data = process_hits(response)

        # Append to the list
        all_pages_data.extend(data)

        i += 1

        if i == 5:
            print("Max pages reached")
            break



    # Convert to dataframe
    df = pd.DataFrame(all_pages_data)
    return df


def set_page_to_1():
    """
    This one is for the main search bar
    """
    print("New query, 1")
    st.session_state.current_page = 1
    st.session_state.new_query = True




def run_next_page():

    # Update the current page
    st.session_state['current_page'] += 1
    st.session_state['new_query'] = False
    print("Current page updated, next: ", st.session_state['current_page'])

    response = client.collections[COLLECTION_NAME].documents.search({
        'q': st.session_state['search_input'],
        'query_by': 'title,abstract',
        'page': st.session_state['current_page']
    })

    st.session_state['response'] = response
    print("Response updated")

def run_previous_page():

    # Update the current page
    st.session_state['current_page'] -= 1 
    if st.session_state['current_page'] <= 0:
        st.session_state['current_page'] = 1
    st.session_state['new_query'] = False
    print("Current page updated, previous: ", st.session_state['current_page'])

    response = client.collections[COLLECTION_NAME].documents.search({
        'q': st.session_state['search_input'],
        'query_by': 'title,abstract',
        'page': st.session_state['current_page']
    })
    st.session_state['response'] = response
    print("Response updated")



def scroll_if_needed():
    if st.session_state['new_query'] == False:

        # If its not a new query, then we are just changing the page, thus, we must scroll to the top

        print("Scrolling to top")

        string1 = f"""<p>{st.session_state['current_page']}</p>"""
        string2 = """
        <script>
                    window.parent.document.querySelector('section.main').scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
        </script>
                """
        string2_no_scroll = """
        <script>
                    window.parent.document.querySelector('section.main').scrollTo({
                    top: 0,
                    behavior: 'auto'
                });
        </script>
                """
        string3 = string1 + string2
        st.components.v1.html(
            string3,
            height=0
        )

@st.cache_data
def make_request_to_typesense(search_config_papers):

    print("Making a request to Typesense")

    response = client.collections[COLLECTION_NAME].documents.search(search_config_papers)

    return response



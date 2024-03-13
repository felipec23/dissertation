import pandas as pd
from llama_cpp import Llama
import time

GGML_PATH = '/Users/estebanfelipecaceresgelvez/Documents/tesis/ggml-model-openllama3-4bit-q4_1.bin'
# df = pd.read_excel('sentences-tensile.xlsx')
df = pd.read_pickle('sentences-tensile.pkl')


llm = Llama(model_path=GGML_PATH)

PROMPT = \
"""### Instruction:
Extract the property names, quantities and units.

### Input:
{}

### Response:
property_name|property_value|property_unit"""

def run_inference(prompt: str) -> str:

    time_start = time.time()
    output = llm(
        prompt, 
        max_tokens=1020, 
        echo=True, 
        temperature=0.8, 
        top_p=0.90, 
        top_k=60, 
        repeat_penalty=1.0,
        # penalize_nl=False
        )
    print(f"Time elapsed: {time.time() - time_start}")
    full_prompt = output["choices"][0]["text"]
    response = full_prompt[full_prompt.find("### Response:")+len("### Response:"):].strip()
    no_header = response[response.find("\n")+1:].strip()

    return no_header

# run_inference(prompt)
# Applying predictions to the abstracts
def extract_entities(sentences: list) -> list:
    """
    Extracts entities from a list of sentences.
    """
    entities = []
    for sentence in sentences:
        # add prompt
        prompt = PROMPT.format(sentence)

        # run inference
        response = run_inference(prompt)

        # split response into lines
        lines = response.split("\n")
        
        entities.extend(lines)
    return entities


# Parse string to list
# df['sentences'] = df['sentences'].apply(lambda x: x.split('\n'))

# pri= df['sentences'].iloc[0]

# extract_entities(df['sentences'].iloc[0])

# Extract entities from the abstracts
df['entities'] = df['sentences'].apply(lambda x: extract_entities(x))

df.to_pickle("results-tensile.pkl")


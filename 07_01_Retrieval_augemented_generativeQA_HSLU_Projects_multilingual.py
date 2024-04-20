#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Retrieval-Augmented-Generative-Question-Answering-(GQA)" data-toc-modified-id="Retrieval-Augmented-Generative-Question-Answering-(GQA)-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Retrieval-Augmented Generative Question Answering (GQA)</a></span><ul class="toc-item"><li><span><a href="#Libaries" data-toc-modified-id="Libaries-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Libaries</a></span></li><li><span><a href="#Set-API" data-toc-modified-id="Set-API-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Set API</a></span><ul class="toc-item"><li><span><a href="#Exkurs:-Ask-about-HSLU-without-knowledgebase" data-toc-modified-id="Exkurs:-Ask-about-HSLU-without-knowledgebase-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Exkurs: Ask about HSLU without knowledgebase</a></span></li></ul></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Building-a-Knowledge-Base:-Embeed-data" data-toc-modified-id="Building-a-Knowledge-Base:-Embeed-data-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Building a Knowledge Base: Embeed data</a></span></li><li><span><a href="#Create-Chunks" data-toc-modified-id="Create-Chunks-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Create Chunks</a></span></li><li><span><a href="#Create-Embeedings-(ada-002)" data-toc-modified-id="Create-Embeedings-(ada-002)-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Create Embeedings (ada-002)</a></span></li><li><span><a href="#Vector-database" data-toc-modified-id="Vector-database-1.7"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Vector database</a></span></li><li><span><a href="#Indexing" data-toc-modified-id="Indexing-1.8"><span class="toc-item-num">1.8&nbsp;&nbsp;</span>Indexing</a></span></li><li><span><a href="#Query-embeeding-and-get-context" data-toc-modified-id="Query-embeeding-and-get-context-1.9"><span class="toc-item-num">1.9&nbsp;&nbsp;</span>Query embeeding and get context</a></span></li><li><span><a href="#Set-prompt-and-results" data-toc-modified-id="Set-prompt-and-results-1.10"><span class="toc-item-num">1.10&nbsp;&nbsp;</span>Set prompt and results</a></span><ul class="toc-item"><li><span><a href="#Example-1:-German" data-toc-modified-id="Example-1:-German-1.10.1"><span class="toc-item-num">1.10.1&nbsp;&nbsp;</span>Example 1: German</a></span></li><li><span><a href="#Example-2:-French" data-toc-modified-id="Example-2:-French-1.10.2"><span class="toc-item-num">1.10.2&nbsp;&nbsp;</span>Example 2: French</a></span></li><li><span><a href="#Example-3:-German" data-toc-modified-id="Example-3:-German-1.10.3"><span class="toc-item-num">1.10.3&nbsp;&nbsp;</span>Example 3: German</a></span></li><li><span><a href="#Example-4:-Chinese" data-toc-modified-id="Example-4:-Chinese-1.10.4"><span class="toc-item-num">1.10.4&nbsp;&nbsp;</span>Example 4: Chinese</a></span></li><li><span><a href="#Example-5:-German-again" data-toc-modified-id="Example-5:-German-again-1.10.5"><span class="toc-item-num">1.10.5&nbsp;&nbsp;</span>Example 5: German again</a></span></li><li><span><a href="#Example-6:-Spanisch" data-toc-modified-id="Example-6:-Spanisch-1.10.6"><span class="toc-item-num">1.10.6&nbsp;&nbsp;</span>Example 6: Spanisch</a></span></li><li><span><a href="#Example7:-Japanese" data-toc-modified-id="Example7:-Japanese-1.10.7"><span class="toc-item-num">1.10.7&nbsp;&nbsp;</span>Example7: Japanese</a></span></li><li><span><a href="#Example-8:-Swedish" data-toc-modified-id="Example-8:-Swedish-1.10.8"><span class="toc-item-num">1.10.8&nbsp;&nbsp;</span>Example 8: Swedish</a></span></li><li><span><a href="#Outcome-withouth-primer-for-correct-answers-(Example-1)" data-toc-modified-id="Outcome-withouth-primer-for-correct-answers-(Example-1)-1.10.9"><span class="toc-item-num">1.10.9&nbsp;&nbsp;</span>Outcome withouth primer for correct answers (Example 1)</a></span></li></ul></li></ul></li><li><span><a href="#Gradio:-HSLU-multilingual-project-Chatbot-(retrieval-augmentet-GQA)" data-toc-modified-id="Gradio:-HSLU-multilingual-project-Chatbot-(retrieval-augmentet-GQA)-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Gradio: HSLU multilingual project Chatbot (retrieval-augmentet GQA)</a></span></li></ul></div>

# # Retrieval-Augmented Generative Question Answering (GQA)

# ## Libaries

# In[2]:


import pandas as pd
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
from IPython.display import Markdown
import gradio as gr
import warnings
warnings.filterwarnings("ignore")


# ## Set API

# In[3]:


# Set API key
openai.api_key = os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key here and hit enter:")


# ### Exkurs: Ask about HSLU without knowledgebase

# In[42]:


query = "Welche Projekte gibt es auf der HSLU in Luzern zum Thema Senioren?"
res = openai.Completion.create(
    engine='text-davinci-003',
    prompt=query,
    temperature=0,
    max_tokens=400,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)

res['choices'][0]['text'].strip()


# ## Load data

# In[4]:


# Load data
df1 = pd.read_excel("4_preprocess.xlsx")
# Create dataframe
df1 = pd.DataFrame(df1, columns=['Projekt ID', "Projekttitel", "Teaser",  "Abstract", "url_new", "Projektstatus", "year_start", "year_end", "SAP Interne Organisationen", "corpus"])


# In[5]:


# Rename text column
df1.rename(columns={"corpus": "text"}, inplace=True)


# In[6]:


# Convert dataframe to a list of dict for Pinecone data upsert
data = df1.to_dict('records')


# In[7]:


# Inspect
data[6]


# In[8]:


new_data = data


# In[9]:


new_data[2]


# ## Building a Knowledge Base: Embeed data

# Example

# In[10]:


embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)


# In[11]:


# vector embeddings are stored within the 'data' key
res.keys()


# In[12]:


# we have created two vectors (one for each sentence input)
len(res['data'])


# In[13]:


# we have created two 1536-dimensional vectors
len(res['data'][0]['embedding']), len(res['data'][1]['embedding'])


# ## Create Chunks

# In[14]:


tokenizer = tiktoken.get_encoding('cl100k_base') 

# Create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


# In[15]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]) 


# In[16]:


chunks = text_splitter.split_text(data[6]['text'])[:3]
chunks


# In[18]:


tiktoken_len(chunks[0]), tiktoken_len(chunks[1])


# ## Create Embeedings (ada-002)

# In[19]:


import os
os.environ["OPENAI_API_KEY"] = "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"
#os.environ["COHERE_API_KEY"] = ""
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
#os.environ["SERPAPI_API_KEY"] = ""


# In[20]:


from getpass import getpass
#OPENAI_API_KEY = getpass("OpenAI API Key: ")
OPENAI_API_KEY = "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"


# In[21]:


from langchain.embeddings.openai import OpenAIEmbeddings
model_name = 'text-embedding-ada-002' 
embed = OpenAIEmbeddings()


# In[22]:


# embed
texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]
res = embed.embed_documents(texts)
len(res), len(res[0])
   


# ## Vector database

# In[24]:


# Initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(api_key="Pass your key", environment="us-west1-gcp")

index_name = 'lekgrag'

# If the index does not exist, we create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension= len(res[0]), #dimension=shape[1], 
        metric='cosine')  ##dotproduct


# In[25]:


# Connect to index
index = pinecone.Index(index_name)
index.describe_index_stats()


# ## Indexing

# In[26]:


# Indexing
batch_limit = 100

texts = []
metadatas = []

for i, record in enumerate(tqdm(data)):
    # first get metadata fields for this record
    metadata = {
        'Projekt_ID': str(record['Projekt ID']),
        'source': record['url_new'],
        'title': record['Projekttitel'],
        'abstract': record['Abstract'],
    }
    # now we create chunks from the record text
    record_texts = text_splitter.split_text(record['text'])
    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    # append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []


# In[67]:


# Inspect index
index.describe_index_stats()


# ## Query embeeding and get context
# 

# In[103]:


query = "welche projekte gibt es aktuell zu senioren?"


# In[104]:


# Embeed Query
embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=3, include_metadata=True)


# In[105]:


# Inspect
res


# In[106]:


# Retrieval function

# Set limit
limit = 3750

# Function
def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # Retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # Get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # Build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # Append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt


# In[107]:


# Firstly retrieve relevant items from Pinecone
query_with_contexts = retrieve(query)
query_with_contexts


# In[110]:


# Function to complete Prompt
def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()


# In[111]:


# Complete the context-infused query
complete(query_with_contexts)


# In[112]:


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[86]:


# Get list of retrieved text
#contexts = [item['metadata']['text'] for item in res['matches']]
#augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query


# In[83]:


print(augmented_query)    


# ## Set prompt and results

# ### Example 1: German

# In[115]:


# system message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}])


# In[116]:


from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# In[102]:


from IPython.display import display, Markdown
def print_result(result):
  output_text = f"""### Question: 
  {query}
  ### Answer: 
  {res['choices'][0]['message']['content']}
  ### Sources: 
  {res['sources']}
  ### All relevant sources:
  #{' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """
  display(Markdown(output_text))


# ### Example 2: French

# In[125]:


query = "Quels sont les projets sur le thème de l'énergie?"

# Embeed Query
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(input=[query],engine=embed_model)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=4, include_metadata=True)

#top_k=10 -->InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 6187 tokens. Please reduce the length of the messages.

query_with_contexts = retrieve(query)


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[126]:


# system message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# ### Example 3: German

# In[127]:


query = "Welche Projekte gibt es zum Thema Fassadenbegrünung?"

# Embeed Query
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(input=[query],engine=embed_model)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=4, include_metadata=True)

#top_k=10 -->InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 6187 tokens. Please reduce the length of the messages.

query_with_contexts = retrieve(query)


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[130]:


# system message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users {query} and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# ### Example 4: Chinese

# In[132]:


query = "有哪些关于虚拟现实的项目"

# Embeed Query
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(input=[query],engine=embed_model)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=4, include_metadata=True)

#top_k=10 -->InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 6187 tokens. Please reduce the length of the messages.

query_with_contexts = retrieve(query)


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[133]:


# system message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}])

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# 
# ***Translation - manually with Deepl Translate for readers who do not understand Chinese***
# 
# Welche Projekte gibt es zum virtual reality?  
# 
# An der HSLU gibt es folgende Projekte zur virtuellen Realität:
# 
# "Virtuelles Wasser": Ziel des Projektes ist es, die Attraktivität der Brunnenkarte der Stadt Luzern im Bereich des Tourismus mittels Augmented Reality zu verbessern, indem bestehende Informationen interaktiv und unterhaltsam aufbereitet werden. Das Projekt untersucht auch die Bedeutung von mobilen Geräten im Tourismus und das Potenzial von Virtual und Augmented Reality Technologien.
# 
# "Virtual and Augmented Reality Zentrum für KMU": Ziel des Projektes ist es, ein Zentrum für Virtual- und Augmented-Reality-Technologien einzurichten, das KMU die Möglichkeit bietet, diese Technologien kennenzulernen und zu erleben, Prototypen und Projekte in ihrer eigenen Umgebung durchzuführen sowie Schulungen für ihre Mitarbeiter (und Kunden) anzubieten. Das Zentrum kann auch von Schulen und Colleges für die Ausbildung genutzt werden.
# 
# "VR Vaccine Production Demo": Ziel dieses Projekts ist es, die Technologie der virtuellen Realität zu nutzen, um neue Laborprozesse vor Ort zu erleben und zu erlernen, ohne lange Wege zurücklegen und den Laborbetrieb unterbrechen zu müssen. Ziel des Projekts ist es, das Oculus Quest Headset vor Ort für die Mitglieder zur Verfügung zu stellen.
# 
# "XUND Futurelab": Das Projekt sieht vor, in der neu errichteten Schule des XUND ein Futurelab einzurichten, um Zukunftstechnologien im Bereich der Pflege zu demonstrieren, wozu auch die Demonstration von Virtual- und Augmented-Reality-Anwendungen gehört. Das Immersive Realities-Team der HSLU wird Unterstützung und Beratung bei der Einrichtung und Konfiguration des Futurelab sowie beim Aufbau von Kapazitäten für Virtual- und Augmented-Reality-Technologien leisten. .
#     

# In[135]:


# system message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn {query} and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# ### Example 5: German again

# In[136]:


query = "Wissensmanagement, gibt es dazu Projekte?"

# Embeed Query
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(input=[query],engine=embed_model)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=4, include_metadata=True)

#top_k=10 -->InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 6187 tokens. Please reduce the length of the messages.

query_with_contexts = retrieve(query)


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[137]:


# system message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn {query} and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# ### Example 6: Spanisch

# In[140]:


query = "Gestión del conocimiento, ¿hay algún proyecto al respecto?"

# Embeed Query
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(input=[query],engine=embed_model)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=4, include_metadata=True)

#top_k=10 -->InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 6187 tokens. Please reduce the length of the messages.

query_with_contexts = retrieve(query)


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[141]:


# System message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn {query} and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# ***Translation - manually with Deepl Translatefor readers who do not understand Spanisch***
#     
# Knowledge management, are there projects for this?    
#     
# At HSLU there are several projects related to knowledge management. One of them is the "Know and Share" project, which develops a vision for a global knowledge sharing platform. This platform aims to facilitate the exchange of know-how for the operation and maintenance of equipment and technical devices. The exchange of information between manufacturers of equipment and/or technical services and local users/operators is done through rich content, such as text or video tutorials. The project seeks to support the entire life cycle of a piece of equipment and allow the efficient performance of installation, commissioning, operation and maintenance activities by local personnel. (SOURCE: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/detail/?pid=373)
# 
# Another related project is the "Competence Management Platform for PHZG (Pilot Project)". In collaboration with PHZG, an online platform for managing competencies in the new personalized and individualized study format "pi" is being developed. HSLU experts are involved in the conception, software engineering, user experience and project management of this project. (SOURCE: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/detail/?pid=5928)
# 
# In addition, the "Sustainable NGO Governance" project investigates the relationship between NGO governance and the impact they achieve through their activities. A governance-impact model has been developed and tested in the field of development cooperation. The project has also developed a change management framework that promotes the sustainable implementation of governance approaches and increases individual and organizational learning capacity as well as NGO development impact orientation. (SOURCE: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/detail/?pid=558)
# 
# These are just a few examples of projects related to knowledge management at HSLU. For more information, you can visit the HSLU website at the following link: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/    

# ### Example7: Japanese

# In[142]:


query = "機会均等のためのプロジェクトにはどのようなものがありますか?"

# Embeed Query
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(input=[query],engine=embed_model)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=4, include_metadata=True)

#top_k=10 -->InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 6187 tokens. Please reduce the length of the messages.

query_with_contexts = retrieve(query)


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[143]:


# System message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn {query} and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# ***Translation - manually with Deepl Translatefor readers who do not understand Japanese***
# 
# 
# HSLU has the following equal opportunity projects:
# 
# ITC-Vorprojekt: SNF-Projekteingabe: Chancengleichheit in der Sozialhilfe in verschiedenen Regionen der Schweiz: This project aims to develop SNF projects on equal opportunities in social welfare in different regions of Switzerland. The project will involve HSLU-SA and W interdisciplinary teams, practice partners and (international) research partners. This project aims to trace the mechanisms of (re)production of social inequalities by gender, race and class in connection with regional and socio-spatial inequalities. It also aims to show whether innovative mechanisms already exist to promote equal opportunity. The project is based on methodological triangulation at various policy implementation levels including national, state, institutional, and counselor. We plan to make comparisons between different regions (urban, rural, language regions) to highlight differences in service delivery. Based on this, we formulate recommendations for improving equality of opportunity in social welfare.
# 
# Beziehungen zwischen Investoren*innen und von Frauen geführten Start-ups erfolgreich gestalten (JUFIN): This project develops and implements measures to improve equal opportunities for women entrepreneurs in Switzerland. International research shows that women are at a distinct disadvantage in obtaining investment capital compared to men. The project aims to raise awareness of equal opportunities, develop the capacity of investors, women entrepreneurs and education officers, and network and facilitate funding partnerships between women entrepreneurs and investors. The project is funded by the Swiss Federal Office for Gender Equality. The implementation of the developed measures is carried out by the company Sandborn.
# 
# gleichstellen.ch: The aim of this project is to promote debate on gender equality in the Swiss labor market and to propose ways to improve equality. This project is based on Lucia M. Lanfranconi's paper "Geschlechtergleichstellung durch Wirtschaftsnutzendiskurs?" (2014). In this project, the film "Gleichstellen – eine Momentaufnahme" was developed, showing possible approaches and current problems of equality in the Swiss labor market. The film will be screened at various events and used in events such as panel discussions. In addition, E-Learning boxes based on the film have also been developed, covering aspects of pay equity, careers, parenting, female and male occupations, part-time work, and equality policies. This box provides a starting point for designing and conducting short workshops (90 or 120 minutes) in companies and educational institutions.

# ### Example 8: Swedish

# In[144]:


query = "Vilka projekt finns det för rekommendatorsystem?"

# Embeed Query
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(input=[query],engine=embed_model)

# Retrieve from Pinecone
xq = res['data'][0]['embedding']

# Get relevant contexts (including the questions)
res = index.query(xq, top_k=4, include_metadata=True)

#top_k=10 -->InvalidRequestError: This model's maximum context length is 4097 tokens. However, your messages resulted in 6187 tokens. Please reduce the length of the messages.

query_with_contexts = retrieve(query)


# Get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]
contexts_2 = [item['metadata']['source'] for item in res['matches']]
contexts_3 = [item['metadata']['title'] for item in res['matches']]
contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

augmented_query = "\n\n".join(contexts)+"\n  \n".join(contexts_2)+"\n\n-----\n\n"+"\n  \n".join(contexts_3)+"\n\n-----\n\n"+"\n  \n".join(contexts_4)+"\n\n-----\n\n"+query


# In[145]:


# System message to 'prime' the model
primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn {query} and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information
provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
"""

res = openai.ChatCompletion.create( 
    model = "gpt-3.5-turbo", temperature=0,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

from IPython.display import Markdown
display(Markdown(res['choices'][0]['message']['content']))


# ### Outcome withouth primer for correct answers (Example 1)

# In[93]:


res = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are Q&A bot. A highly intelligent system that answers user questions"},
        {"role": "user", "content": query}
    ]
)
display(Markdown(res['choices'][0]['message']['content']))
     


# # Gradio: HSLU multilingual project Chatbot (retrieval-augmentet GQA)

# In[ ]:


import openai
import gradio as gr

openai.api_key = "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"

def respond(query):
    # Embeed Query
    embed_model = "text-embedding-ada-002"
    res = openai.Embedding.create(input=[query], engine=embed_model)

    # Retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # Get relevant contexts (including the questions)
    res = index.query(xq, top_k=4, include_metadata=True)

    # Get list of retrieved text
    contexts = [item['metadata']['text'] for item in res['matches']]
    contexts_2 = [item['metadata']['source'] for item in res['matches']]
    contexts_3 = [item['metadata']['title'] for item in res['matches']]
    contexts_4 = [item['metadata']['abstract'] for item in res['matches']]

    augmented_query = "\n\n".join(contexts) + "\n  \n".join(contexts_2) + "\n\n-----\n\n" + "\n  \n".join(contexts_3) + "\n\n-----\n\n" + "\n  \n".join(contexts_4) + "\n\n-----\n\n" + query  # System message to 'prime' the model
    primer = f"""You are a friendly Q&A bot. A highly intelligent system that answers user questions based only on the information provided by the user above each question. You can detect the language of the users questionn {query} and respond in the language you detected. You start your answer always in the following format according to the language detected: "At HSLU there are following projects related to {query}". Provide the identical title and summarizse the abstract from information provided by the user. Take note of the query language and provide anser in the same language as the querry. Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources. If the information can not be found in the information provided by the user you truthfully say "I don't know. I use just the top 10 semantic search results, please have a look on the website: https://www.hslu.ch/de-ch/hochschule-luzern/forschung/projekte/".
    """

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", temperature=0,
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ]
    )

    return res['choices'][0]['message']['content']

# Create the Gradio Interface

# Creating a Gradio interface for the chatbot
with gr.Blocks() as demo:
    title = gr.HTML("<h1>HSLU Project Chatbot</h1>")
    input = gr.Textbox(label="What would you like to know?")  # Textbox for user input
    output = gr.Textbox(label="Inspect answer...")  # Textbox for chatbot response
    btn = gr.Button("Get answer")  # Button to trigger the agent call
    btn.click(fn=respond, inputs=input, outputs=output)

# Launching the Gradio interface
demo.launch(share=True, debug=True)


import json
import boto3
import botocore
import faiss
from faiss import write_index, read_index
import time
import LLMREDIAL_dataset
from Memory import  Memory
import numpy as np
import pandas as pd
import re
#memory = Memory()

annotation_file= 'Annotations/annotated_movies_P5.json'#'Annotations/sample_annotations.json'#

def load_data(file):
    with open(file) as f:
        df=json.load(f)
    return df

def embed_str(prompt):

    try:

        kwargs = {
            "modelId": "amazon.titan-embed-text-v2:0",#"amazon.titan-embed-text-v2",  #try "titan-text-v2"
            "contentType": "application/json",
            "body": json.dumps(
                {
                    "inputText": prompt
                }
            )
        }
        session = boto3.Session(profile_name='bedrock-profile')
        config = botocore.config.Config(
            read_timeout=900, connect_timeout=900, retries={"max_attempts": 3}
        )
        client = session.client(
            service_name="bedrock-runtime", region_name="us-east-1", config=config
        )
        response = client.invoke_model(**kwargs)
        response_body = json.loads(response.get('body').read())
        return response_body['embedding']
    except Exception as ex:
        print('Error in embedding: ', ex)
        return 'Error'


def embed_str_ann(ann_str):
    embs=[]
    re_patterns = r'\[(.*?)\]<(.*?)>'
    matches = re.findall(re_patterns, ann_str)
    for ann_k,ann_v  in matches:
        embs.append(embed_str('[%s]<%s>'%(ann_k,ann_v)))  # embedd each annotation seperately
    return np.mean(embs, axis=0)


def vector_search(query,item_list,embeddings_file,k=1,annotation_f='',threshold=None):


    if len(item_list) == 0 or query =='':
        print("Empty key list")
        return None,None,None

    #annotated_movies = load_data(annotation_file)
    annotation_vectors = read_index(embeddings_file)

    # measure retrieval latency
    t_start = time.time()

    query_embed = query
    #query_embed = embed_str(query)
    query = np.array(query_embed).reshape(1, -1)
    dist, pos = annotation_vectors.search(query, k=k) #config.cs_top_k_retrieval
    retrieval_latency = time.time() - t_start

    if threshold != None:
        filtered_indices = []
        filtered_distances = []
        for i in range(len(dist)):
            for j in range(len(dist[i])):
                if dist[i][j] > threshold:
                    filtered_indices.append(pos[i][j])
                    filtered_distances.append(dist[i][j])
        matched = [list(item_list)[i] for i in filtered_indices]
        #print(len(matched),' memory retrieved')
        return matched,retrieval_latency,filtered_distances

    matched = [list(item_list)[i] for i in pos[0]]
    #print('Successfully Retrieved: ',len(matched))
    return matched,retrieval_latency,dist

def count_all_att(annotations_file):
    with open(annotations_file, 'r') as f:
        df = json.load(f)
    att={}
    for k, v in df.items():
        for i,vv in v.items():
            if i in att:
                att[i]+=1
            else:
                att[i]=1

    labels=list(att.keys())
    cnts=list(att.values())

    return att
#movie attribute string with no movie name, all attributes
def embedding_1_list(list_movies,fname):
    annotated_movies = load_data(annotation_file)
    movie_embeddings=[]
    for movie_id in list_movies:
        movie_embeddings.append(embed_str(str(annotated_movies[movie_id])))
    movie_embeddings=np.array(movie_embeddings)
    d = movie_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(movie_embeddings)
    print("Saving")
    write_index(index,fname)

    return
def embedding_1_list(list_items,fname):
    annotated_movies = load_data(annotation_file)
    movie_embeddings=[]
    print("Embedding")
    for movie_id,annotations in annotated_movies.items():
        if annotations is not None:
            movie_embeddings.append(embed_str(str(annotations)))

    movie_embeddings=np.array(movie_embeddings)
    d = movie_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(movie_embeddings)
    print("Saving")
    write_index(index, fname)

    return


#movie attribute string with movie name, all attributes
def embedding_2(annotation_file):

    annotated_movies = load_data(annotation_file)
    movie_embeddings = []
    print("Embedding")
    for movie_id, annotations in annotated_movies.items():
        print("Embedding:", memory.get_item_name_by_id(movie_id))
        movie_embeddings.append(embed_str("["+memory.get_item_name_by_id(movie_id)+"]"+str(annotations)))

    movie_embeddings = np.array(movie_embeddings)
    d = movie_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(movie_embeddings)
    print("Saving")
    write_index(index, "embedding_2.index")

    return

def embedding_2_list(list_movies, memory=None,fname=''):

    annotated_movies = load_data(annotation_file)
    movie_embeddings = []
    print("Embedding")
    for movie_id in list_movies:
        movie_embeddings.append(embed_str("["+memory.get_item_name_by_id(movie_id)+"]"+str(annotated_movies[movie_id])))

    movie_embeddings = np.array(movie_embeddings)
    d = movie_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(movie_embeddings)
    print("Saving")
    write_index(index, fname)

    return

#embedd each annotation seperately
def embedding_3(annotation_file):
    annotated_movies = load_data(annotation_file)
    movie_embeddings = []
    print("Embedding")
    for movie_id, annotations in annotated_movies.items():
        ann_emb=[embed_str(memory.get_item_name_by_id(movie_id))]
        for ann in annotations:
            ann_emb.append(embed_str(str(ann))) #embedd each annotation seperately

        movie_embeddings.append(np.average(ann_emb))
    d = movie_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(movie_embeddings)
    print("Saving")
    write_index(index, "embedding_3.index")

#ordering attributes based on relevance
def embedding_4(annotation_file):
    annotated_movies = load_data(annotation_file)
    movie_strs={}
    movie_embeddings = []
    print("Embedding")
    for movie_id, annotations in annotated_movies.items():
        movie_str=memory.get_item_name_by_id(id=movie_id)
        for att in count_all_att(annotation_file).keys():
            if att in annotations:
                movie_str+=f'[{att}]<{annotations[att]}'

        movie_emb = embed_str(movie_str)
        movie_embeddings.append(movie_emb)

    d = movie_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(movie_embeddings)
    write_index(index, "embedding_4.index")
if __name__== "main":
    print("Embedding 1")
    #embedding_1('Annotations/annotated_movies_P5.json')#()
    #embedding_2_list(["630352138X","630434063X"],'temp.index')
    #print(vector_search("[year]<2006>[genre]<Crime, Drama, Thriller>[director]<Martin Scorsese>",'temp.index'))
    #embedding_4('Annotations/sample_annotations.json')
    print(embed_str("[ Constantine [Blu-ray] [Blu-ray] (2008)]{\[genre]:<action, fantasy, horror, thriller>,[rating]: <r>, [actors]:<keanu reeves, rachel weisz, shia labeouf>, [director]: <francis lawrence>, [based on]: <hellblazer comic book series, [release year]: <2005>")) #runtime': '121 minutes', 'language': 'english', 'country of origin': 'usa', 'production company': 'warner bros. pictures', 'special effects': 'cgi', 'plot keywords': 'exorcism, supernatural, heaven vs. hell, redemption'}


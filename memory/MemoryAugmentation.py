import logging

from ..memory.LTMBackbone import LTMBackbone
import augmentation_prompt as PRMPT
import json
import re
import boto3
import botocore
import numpy as np
import faiss
from faiss import write_index
import os
from retry import retry

logger = logging.getLogger(__name__)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler('mem_aug.log', mode='w')
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - [%(lineno)d] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


MODEL2ID = {
    "v3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "v3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    #  "v3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    #  "gpt-4": "gpt-4",
    #  "gpt-4-turbo": "gpt-4-turbo-preview"
}

ANNOTATED_MOVIES="/local/home/ranasal/Mem_Auig_v2/Annotations/annotated_movies_P6.json" #path to annotated items
DATASET_FINAL="/local/home/ranasal/Mem_Auig_v2/datasets/LLM-REDIAL/Movie/final_data.jsonl"
MOVIE_ITEMS="/local/home/ranasal/Mem_Auig_v2/datasets/LLM-REDIAL/Movie/item_map.json"

class MemoryAugmentation:
    def __init__(self):
        self.items_map=[]
        self.conversations_map=[]
        self.users_map=[]
        self.attributes_bank={"items_attributes":[],"user_attributes":[]}
        self.llm_engine= LTMBackbone()

    def annotate_input_dialogue(self, dialogue):
        response=self.llm_engine.generate_text_claude(PRMPT.AUGMENT_INPUT_DIALOGUE4+" "+' '.join(dialogue))
        return response


    '''def annotate_items(self,items,file): #annotate movie, book, electronic or sport
        items_attributes={"":""}
        #atts=[]
        items_count={}
        i=0
        for item in items:
            response=self.llm_engine.generate_text_claude(PRMPT.AUGMENT_MOVIE3+" "+item)
            attributes=re.findall(r'\[(.*?)\]',response)
            items_attributes[item]=attributes
            #atts.extend(attributes)
            #print("item",item)
            #print(attributes)
            i+=1
            if i==10: return
        #items_count=Counter(atts)
        #print(items_count)
        df=pd.DataFrame(items_attributes)
        df.to_csv(file,sep='\t')
        return
    '''

    def annotate_movie_2(self, items):
        out_file = open(ANNOTATED_MOVIES + "_P6.json", 'a')
        items_annotations = {}
        print("annotating")

        for key, item in items.items():
            response = self.llm_engine.generate_text_claude(PRMPT.AUGMENT_MOVIE_5 + ": " +item)
            response = response.lower()
            re_patterns = r'\[(.*?)\]<(.*?)>'
            matches = re.findall(re_patterns, response)
            att = {}
            if matches:
                for a, v in matches:
                    att[a] = v
            items_annotations[key] = att  # {att:val for att,val in matches}  else {}#TODO check mulitple values for same attribute
        json.dump(items_annotations, out_file, indent=4)
        return

    def annotate_item(self, items,type="movie"):
        out_file = open(ANNOTATED_MOVIES + f"_{type}_annotated_3.json", 'a')
        items_annotations = {}
        print("annotating")
        #cont=False
        for key, item in items.items():
            #if str(key) == "B004LAGFLS":
            #    cont=True
            #if cont is False:
            #    continue
            response = self.llm_engine.generate_text_claude(PRMPT.annotate_prompt(type) + ": " +item)
            response = response.replace('\n','').lower()
            re_patterns = r'\[(.*?)\]<(.*?)>'
            matches = re.findall(re_patterns, response)
            att = {}
            if matches:
                for a, v in matches:
                    att[a] = v
            items_annotations[key] = att  # {att:val for att,val in matches}  else {}#TODO check mulitple values for same attribute
        json.dump(items_annotations, out_file, indent=4)
        return

    def annotate_movie_batch(self,items,batch_size=10):
        out_file= open(ANNOTATED_MOVIES+"_P5.json",'a')
        items_annotations={}
        movies_batch=""
        movies_batch_id={}
        i=0
        print("annotating")
        for key,item in items.items():
            if i==batch_size:
                response=self.llm_engine.generate_text_claude(PRMPT.AUGMENT_MOVIE_5+": "+movies_batch)
                response=response.lower()
                pattern = r'<movie>(.*?)</movie>'
                movies=re.findall(pattern,response)
                if movies:
                    for movie in movies:
                        name= re.findall(r'\{(.*?)\}', movie)[0]
                        re_patterns = r'\[(.*?)\]<(.*?)>'
                        matches = re.findall(re_patterns,movie)
                        att={}
                        if matches:
                            for a,v in matches:
                                att[a]=v
                        items_annotations[name]=att#{att:val for att,val in matches}  else {}#TODO check mulitple values for same attribute
                i=0
                movies_batch=""
            else:
                i+=1
                movies_batch=movies_batch+"\n"+item
        json.dump(items_annotations, out_file, indent=4)
        return
    def annotate_movie(self,items):
        out_file= open(ANNOTATED_MOVIES+"_P4.json",'w')
        items_annotations={}
        for key,item in items.items():
            response=self.llm_engine.generate_text_claude(PRMPT.AUGMENT_MOVIE_4+": "+item)
            response=response.lower()
            re_patterns = r'\[(\w+)\]<(\w+)>'
            matches = re.findall(re_patterns,response)
            items_annotations[key]={att:val for att,val in matches} if matches else {}#TODO check mulitple values for same attribute
        json.dump(items_annotations,out_file,indent=4)
        return
    def annotate_movie_json(self,items):
        out_file= open(ANNOTATED_MOVIES+"_PR.json",'w')
        items_annotations={}
        responses={}
        for key,item in items.items():
            response=self.llm_engine.generate_text_claude(PRMPT.AUGMENT_MOVIE_REASONING+": "+item)
            responses[key]=response
        json.dump(responses,out_file,indent=4)
        return

    def retrieve(self,history_items,annotations):
        with open(ANNOTATED_MOVIES,'r') as f:
            annotations_file=json.loads(f)
        history_relevant=[]
        for key,att_dict in annotations_file:
            if key in history_items:
                for att,value in att_dict:
                    if '['+att+']<'+value+'>'.lower() in annotations:
                        history_relevant.append(self.get_items_by_id(key))

    def annotate_conversation(self):
        print("")

    @retry()
    def embed_str(self,prompt):#todo remove to llm backbone
        if prompt=='':
            return ''
        try:

            kwargs = {
                "modelId": "amazon.titan-embed-text-v2:0",  # "amazon.titan-embed-text-v2",  #try "titan-text-v2"
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
            print('Error', ex)


    def embedd_annotations(self,fname):  # todo move to memory later
        # embed all annotation file and store to DB
        with open(ANNOTATED_MOVIES,'r') as f:
            annotations_file=json.load(f)

        movie_keys=[]
        movie_att_embeddings=[]
        i=0

        for key,att_dict in annotations_file.items():
            #create embedding string
            to_embedd=''
            for k,v in att_dict.items():
                to_embedd+='['+k+']<'+v+'>'
            movie_keys.append(key)
            embedded_att = self.embed_str(to_embedd)
            movie_att_embeddings.append(embedded_att)
            if i%1000 == 0: print(f"Completed {i} movies")
            i+=1

        movie_embeddings = np.array(movie_att_embeddings)
        print('movies to store',movie_embeddings.shape)
        d = movie_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(movie_embeddings)
        print("Saving")
        write_index(index, fname)


    def embedd_user_memories(self,dataset,annotated_movies):  # todo move to memory later

        dataset_final = dataset+"/final_data.jsonl"
        movie_items = dataset+"/item_map.json"
        # embed all annotation file and store to DB
        with open(dataset_final,'r') as fr:
            final_data = fr.readlines()

        with open(annotated_movies,'r') as f:
            annotations_file=json.load(f)

        with open(movie_items,'r') as f:
            movies=json.load(f)

        Total_len = len(final_data)
        print(Total_len)
        for i in range(Total_len):

            per_data = json.loads(final_data[i])
            user_id, user_information = next(iter(per_data.items()))

            if os.path.exists('DB2/'+str(user_id)+'.index'): #DB3_movie_name_P6
                print('file exists')
                continue

            # read user's history_interaction
            movie_keys = []
            movie_att_embeddings = []
            history_interaction = user_information['history_interaction']
            for h in history_interaction:
                h_annotations = annotations_file[h]
                to_embedd = ''

                if len(h_annotations.items()) == 0:
                    movie_name=movies[h]
                    to_embedd+='[title]'+'<'+str(movie_name)+'>'
                else:
                    for k, v in h_annotations.items():#excluidng every movies item in embedding
                        #movie_name = movies[h]
                        #to_embedd += '[title]' + '<' + str(movie_name) + '>'
                        to_embedd += '[' + k + ']<' + v + '>'
                #movie_keys.append(h) #movie id
                embedded_att = self.embed_str(to_embedd)
                print('to embedd',len(to_embedd))
                movie_att_embeddings.append(embedded_att)
                to_embedd = ''

            #all history embedded
            #create DB for user
            #print(movie_att_embeddings)
            #print(len(movie_att_embeddings))
            movie_embeddings = np.array(movie_att_embeddings)
            print('movies to store', movie_embeddings.shape)
            d = movie_embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(movie_embeddings)
            print("Saving")
            write_index(index,'DB2/'+str(user_id)+'.index') #DB3_movie_name_P6


if __name__ == "__main__":
    with open('/local/home/ranasal/Mem_Auig_v2/datasets/LLM-REDIAL/Movie/item_map.json','r') as file:
        data=json.load(file)
    mem=MemoryAugmentation()
    #mem.annotate_item(data,type='sport')
    mem.embedd_user_memories(data)
    #mem.annotate_movie_2(data)

#i=0
#items={}
#for k,v in data.items():
#    items[k]=v
#    if i>10:
#        break
#    else:
#        i+=1
#mem.annotate_movie_json(data)

#with open(ANNOTATED_MOVIES+'_P4.json','r') as file:
#    data=json.load(file)



# tools
# memory instance from memory aug
# add new history by annotation
# retrieve old history by attributes
# recommendations output

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import time
from http.client import responses
from typing import *
import boto3
import pandas as pd
import re
import datetime

from Memory import Memory
import botocore
from ..data.LLMREDIAL_dataset import load_dialogue_ids, Dialogue, DATASET, getUser, User, Conversation
import json
import movie_embedding_retrieval
from retry import retry
from Evaluation import RecoMetric
# from anaconda3.bin.mturk import list_hits


MODEL2ID = {
    "v3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "v3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    #  "v3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    #  "gpt-4": "gpt-4",
    #  "gpt-4-turbo": "gpt-4-turbo-preview"
}


class RecommenderAgent:

    def __init__(self, temperature: float = 0.0, max_tokens: int = 150000):
        self.engine = "v3-sonnet"
        self.memory = Memory()
        self.prompt = ""
        self.dialogues_list = []
        self.dialogues_id_list = []
        self.masked_dialogues_list = []
        self.setup_llm()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def setup_llm(self):
        session = boto3.Session(profile_name='bedrock-profile')
        config = botocore.config.Config(
            read_timeout=900, connect_timeout=900, retries={"max_attempts": 3}
        )
        self.client = session.client(
            service_name="bedrock-runtime", region_name="us-east-1", config=config
        )
        self.model_id = MODEL2ID[self.engine]

    def init_recommendations(self, dialogues_count=50):
        random.seed(32)
        list_of_ids = random.sample(range(0, 10088), dialogues_count)  # conversations id range from 0-10088
        list_of_ids.sort()
        self.dialogues_id_list = list_of_ids
        self.dialogues_list = load_dialogue_ids(list_of_ids)  # list of Dialogues class
        return list_of_ids

    def recommend(self, aug_usage=False, k=1, vs_k=5, cut=False, filter=False, reason=True, type='movie',embedding=False):

        df = pd.DataFrame(columns=['label', 'response'])

        timestamp = datetime.datetime.now().strftime("%d-%H%M%S")

        results_file = f"results_noannotation_k=_{k}_" + f"d_{len(self.dialogues_list)}_" + f"{timestamp}.csv"

        if aug_usage:
            self.memory.init_memory_augmentation()
            results_file = f"results_annotated_k=_{k}_" + f"d_{len(self.dialogues_list)}_" + f"{timestamp}.csv"

        # Use the current dialogue conversation to retrieve information about the user and relevant history if needed
        for dialogue in self.dialogues_list:
            hist_len = 0
            user_id = getUser(dialogue.dialogue_id)
            if user_id is None:
                print('User is None')
                continue

            labels, dial_label, hist_len, hist,prompt = self.generate_prompt(user_id, dialogue, k=k, vs_k=vs_k,
                                                                                annotations=aug_usage, filter=filter,
                                                                                type=type,embedding=embedding)
            #print('history',hist)
            if prompt is None:
                continue
            recommendation = self.llm_recommend(prompt)

            diag_comp = []
            pattern = ""
            if type == "electronic":
                pattern = r'<electronic>(.*?)</electronic>'  # movies
            elif type == "sport":
                pattern = r'<sport>(.*?)</sport>'
            elif type == "book":
                pattern = r'<book>(.*?)</book>'
            elif type == "movie":
                pattern = r'<movie>(.*?)</movie>'
            recomm_movies = re.findall(pattern, recommendation)

            pattern = r'<dialogue>(.*?)</dialogue>'
            match = re.search(pattern, recommendation, re.DOTALL)
            if match:
                diag_comp = match.group(1)

            reasoning = ""
            pattern = r'<Reasoning>(.*?)</Reasoning>'
            match = re.search(pattern, recommendation, re.DOTALL)
            if match:
                reasoning = match.group(1)

            #print(len(recomm_movies),' recommended')
            if reason:
                new_result = {'label': labels, 'diag_label': dial_label, 'diag_comp': diag_comp,
                              'response': recomm_movies, 'reasoning': reasoning, 'history len': hist_len}
            else:
                new_result = {'label': labels, 'diag_label': dial_label, 'diag_comp': diag_comp,
                              'response': recomm_movies, 'history len': hist_len}

            #df = df.append(new_result, ignore_index=True)
            df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)

        print('Writing to file')
        df.to_csv(results_file, sep='\t')
        return results_file

    def generate_prompt(self, user_id, dialogue, k=1,vs_k=5, annotations=False, filter=False, type='movie', reason=True,embedding=False):

        dialogue_cut = ""

        #get cut dialogues
        with open(DATASET + '/conversations_cut.json', 'r') as file:
            df_cuts = json.load(file)

        diag_text = dialogue.text_str if dialogue.text_str != '' else ' '.join(dialogue.text)
        user_data = self.memory.get_user_data_by_id((user_id))

        if annotations:
            #get annotations for input dialogue to filter
            annotations = self.memory.get_annotations(diag_text)
            cut_annotations = self.memory.get_annotations(df_cuts[str(dialogue.dialogue_id)]['Cut'])
            #print('dialogues, annotations', diag_text,'\n>>>>>>>>>>>',annotations)
            if not filter:
                history = self.memory.retrieve_all_attributes(user_data.history_interaction,
                                                              annotations)  # todo return histories with annotations
            elif not embedding:
                history = self.memory.retrieve_by_attributes(user_data.history_interaction,
                                                             annotations)  # todo return histories with annotations
            else:

                diag_annotation = movie_embedding_retrieval.embed_str(annotations.replace('\n','').strip().lower())
                '''
                movie_embedding_retrieval.embedding_1_list(user_data.history_interaction,fname='temp.index')#,memory=self.memory  memory=self.memory,fname='temp.index')
                print('retrieveing')
                '''
                #print('retrieveing for user',user_id)
                #get user DB and find most relevant history
                history,latency,dist = movie_embedding_retrieval.vector_search(diag_annotation,self.memory.get_items_by_id(user_data.history_interaction), 'DB/'+user_id+'.index',k=vs_k,threshold= 0.2) #rertieveing from user D
                df_retrieved=pd.DataFrame(columns=['matched','retrieval_latency','dist'])

                if history is None or dist is None or len(dist)==0:
                        return None,None,None,None,None
                new_row= {'matched':history,
                          'retrieval_latency':latency,
                          'dist':sum(dist)/len(dist)}
                df_retrieved = pd.concat([df_retrieved, pd.DataFrame([new_row])], ignore_index=True)
                df_retrieved.to_csv(f"movie_retrieval_k=_{k}_vs_k={vs_k}.csv", mode='a', header=False, index=False)
                #print('retrieved history',history
        else:
            history = self.memory.get_items_by_id(user_data.history_interaction)

        # user_might_like = self.memory.get_items_by_id(user_data.user_might_likes)
        labels, old_conversations = self.memory.history_conversations(user_data.conversations, dialogue.dialogue_id)
        labels = re.sub(r'[^a-zA-Z0-9\s]', '', str(labels)).lower().strip()
        cut = df_cuts[str(dialogue.dialogue_id)]['Cut']
        # v4
        prompt_ann = ""
        if annotations:
            prompt_ann = " and their attributes and values provided as [attribute]<value> format for every movie"
        # v7
        prompt = (
            f"Pretend you are a {type} recommender system tasked with generating a list of the top {k} most relevant {type} recommendations for the dialogue below. Use the annotated history of interacted {type}s to guide your recommendation. Also, conisder the provided annotations of the input dialogue "
            f"Your goal is to provide a comma-separated list of {type} titles, sorted in descending order based on their relevance to the user's interest, request and historical interactions."
            f"Annotated History for interacted {type}s: {history}."
            f"Dialogue to complete: {cut}."
            # f"Annotations of Dialogue: {cut_annotations}"
            "Please follow this format to respond:"
            f"<Reasoning>Provide your reasoning for recommendation</Reasoning>"
            f"<{type}>[Provide a comma-separated list of the top {k} {type} titles .Make sure to include {k} {type}s and don't include any commas in the {type} name and don't include any [attribute]<value> in the {type} list ]</{type}>"
            f"<dialogue>Complete the Dialogue as user and agent turns until a valid {type} is recommeneded</dialogue>"
            "Don't include anything else in the repsonse.")
        if history is None:
            return None,None,None,None,None
        return labels, df_cuts[str(dialogue.dialogue_id)]['label'], len(
            history),history, prompt  # TODO uppdate to include label in the right place

    def llm_recommend(self, prompt):
        body = json.dumps({"messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # "top_p": 0.95,
            # "top_k": 250
        })
        accept = "application/json"
        contentType = "application/json"
        try:
            response = self.client.invoke_model(
                body=body, modelId=self.model_id, accept=accept, contentType=contentType
            )
            response = json.loads(response.get('body').read()).get("content")[0]['text']
            return response
        except:
            return ""


if __name__ == "__main__":
    rc = RecommenderAgent()
    toeval=['']
    for k in [10]:
        for vs_k in [5,10]:
            diag_prmpt=4
            p='sonnet'
            n=200
            test = f" att-val no name - diagprmpt = {diag_prmpt} vector search k={vs_k} recomm k={k} annotatrions model=sonnet type=movie n={n}, {p}, reasoning. Note: has conv history and no mightlike as before"
            print("Initiating recommendation for", test)
            rc.init_recommendations(dialogues_count=n)
            rf = rc.recommend(k=k,vs_k=vs_k, aug_usage=True, cut=True, filter=True, reason=True, type='movie',embedding=True)
            print(rf)
            print(test)
            with open('to_test.txt', 'a') as f:
                f.write("\nFile explanation: " + test + " \n File name: " + str(rf))
                # rc.recommend(k=10,aug_usage=True
            toeval.append(str(rf))

    rm = RecoMetric(k_list=[1, 5, 10])
    for fname in toeval:
        rm.persuasive(fname)
        df = pd.read_csv(fname, sep='\t')  # "results_annotated_27-1555.csv"
        labels = df['label']
        recomm = df['response']
        rm.evaluate([str(r).split(',') for r in recomm], labels,fname)
        report = rm.report(fname)
        print(report)
        with open("evaluation_results.txt", "a") as file:
            file.write("\n" + fname + ":\n" + str(report))
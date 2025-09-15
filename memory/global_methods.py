import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import openai
import numpy as np
import json
import time
import sys
import os

import boto3
import botocore
from retry import retry


def get_openai_embedding(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   return np.array([openai.Embedding.create(input = texts, model=model)['data'][i]['embedding'] for i in range(len(texts))])

def set_anthropic_key():
    pass

def set_gemini_key():

    # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def set_openai_key():
    openai.api_key = os.environ['OPENAI_API_KEY']


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_claude_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
                                                    temperature=temperature).strip()
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
            output = output.replace('json', '') # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def get_client():
    session = boto3.Session(profile_name='bedrock-profile')
    config = botocore.config.Config(
        read_timeout=900, connect_timeout=900
    )
    client = session.client(
        service_name="bedrock-runtime", region_name="us-east-1", config=config
    )
    return client


def run_llama(query, max_new_tokens=4000):

    model_id='meta.llama3-70b-instruct-v1:0'

    client=get_client()
    body = json.dumps({
        "prompt": query,
        "max_gen_len": max_new_tokens,
        "temperature": 0.1,
    })
    try:
        response=''
        while response == '' or response is None:

            response = client.invoke_model(
                body=body, modelId=model_id
            )
            if response == '':
                time.sleep(5000)

        response_body = json.loads(response.get("body").read())

        return response_body["generation"]

    except Exception as err:
        print("Error!!"+str(err))
        return ''

def run_mistral(query, max_new_tokens=4000):

    model_id='mistral.mistral-7b-instruct-v0:2'
    client = get_client()
    body = json.dumps({
        "prompt": query,
        "max_tokens": max_new_tokens,
        "temperature":0
    })

    try:
        response=''
        while response == '':

            response = client.invoke_model(
                body=body, modelId=model_id
            )

            if response == '':
                time.sleep(5000)

        response_body = json.loads(response.get("body").read())
        return response_body["outputs"][0]["text"]

    except Exception as err:
        print("Error!!"+str(err))
        return ''


def run_claude (query, max_new_tokens=150000, model_name='claude-sonnet'):
    #try:
    if model_name == 'claude-sonnet':
        model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
    elif model_name == 'claude-haiku':
        model_name = "anthropic.claude-3-haiku-20240307-v1:0"

    session = boto3.Session(profile_name='bedrock-profile')
    config = botocore.config.Config(
        read_timeout=1200, connect_timeout=1200, retries={"max_attempts": 10}
    )
    client = session.client(
        service_name="bedrock-runtime", region_name="us-east-1", config=config
    )
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_new_tokens,
       })
    accept = "application/json"
    contentType = "application/json"
    response=''
    while response=='':

        response = client.invoke_model(
            body=body, modelId=model_name,accept=accept,contentType=contentType
        )
        if response == '': time.sleep(7000)

    response_body = json.loads(response.get("body").read())
    try:
        return response_body["content"][0]["text"]
    except json.decoder.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        text_block = response_body["content"][0]["text"]
        return text_block


def run_claude_with_examples(query,  examples, input, num_gen=1, max_new_tokens=100000, wait_time=1,
                             temperature=0.0,model_name = 'claude-sonnet'):
    if model_name == 'claude-sonnet':
        model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
    elif model_name == 'claude-haiku':
        model_name = "anthropic.claude-3-haiku-20240307-v1:0"

    client = get_client()

    response = ''

    messages=[{"role": "user", "content": query}]

    for inp, out in examples:
        messages.append({"role": "assistant", "content": inp})
        messages.append({"role": "user", "content": out})

    messages.append({"role": "assistant", "content": input.strip()})


    body = json.dumps({
        "messages": messages,
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_new_tokens,
        })

    accept = "application/json"
    contentType = "application/json"

    while response == '':
        try:
            wait_time=wait_time*2
            response = client.invoke_model(
                    body=body, modelId=model_name, accept=accept,contentType=contentType
                )

            response_body = json.loads(response.get("body").read())
            print("response",response_body)
            if response_body['content'] is None:
                response=None
                time.sleep(wait_time)
                print("continuing")
                continue
            print('returning')
            return response_body["content"][0]["text"]

        except Exception as e:
            print('Exception :( :',e)
            response=None
            time.sleep(wait_time)
            pass

        except json.decoder.JSONDecodeError as e:
            print(f"JSONDecodeError: {e.msg}")
            text_block = response_body["content"][0]["text"]
            return text_block


def run_claude_for_annotations(dialog_turn,temperature=0.0,model_name = 'claude-haiku',max_new_tokens=100000,type='turn'):

    if model_name == 'claude-sonnet':
        model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
    elif model_name == 'claude-haiku':
        model_name = "anthropic.claude-3-haiku-20240307-v1:0"


    client=get_client()

    level='dialogue turn'
    if type=='session': level='dialogue'
    if type == 'question': level = 'question'

    prompt= (f"You are an expert annotator who generates the most relevant attributes in a {level}. Given the {level} below, identify the key attributes and their values."
             "The values will be used for future dialogue retrieval."
             "Attributes should be specfic with most relevant values."
             "Don't include speaker name."
             "Include value information that you find relevant and their names if mentioned."#If their is an event attribute mention the type of event."#like names of place, sport, hobbies, interests, etc.."
             "Don't include any unspecified or none values. If an attribute has a none or unspecified value skip this attribute."# or general values. Include relevant information that might to be used for future retrieval"
             "Respond in the format [attribute]<value>. Make sure the attribuite name is between [ ]  and the value between < >."
             "Don't include anything else."
             f"{level}:{dialog_turn}")

    prompt2=(
            "You are an expert annotator who annotates a conversation by identifying the key attributes the speakers are mentioing and their corresponding value."
            "The generated attributes and values will be used later to revtrieve relevant conversations."
            "Annotate the conversation below generating specific and relevant key attributes and their corresponding values."
            #"The attributes should be concise and describe the caetgory of its corresponding value. Consider specific and consice attribute and values only."
            #"Values should be specific as much as possible. Include value name if mentioned. don't include parts of the conversation."#If their is an event attribute mention the type of event."#like names of place, sport, hobbies, interests, etc.."
            #"The values will be used for future dialogue retrieval."
            "Don't include any unspecified or none values. If an attribute has a none or unspecified value skip this attribute."# or general values. Include relevant information that might to be used for future retrieval"
            "Each turn in the conversation contains a dialogue id within square brackets." 
            "Make sure to include the dialogue id from which the values are taken."
            "Exclude speakers name from the attributes and values"
            "Important: Respond only in the format [Dialog id]:[attribute]<value>. Make sure the attribuite name is between [ ]  and the value between < >."
            f"Conversation:{dialog_turn}"
            )
    #best till now
    prompt3= (f"You are an expert annotator who generates the most relevant attributes in a conversation. Given the conversation below, identify the key attributes and their values on a turn by turn level."
             "Attributes should be specfic with most relevant values only. attributes should be generic and values should be detailed, descriptive in few words"
             "Don't include speaker name."
             "Include value information that you find relevant and their names if mentioned. use few words only."#If their is an event attribute mention the type of event."#like names of place, sport, hobbies, interests, etc.."
             "Don't include any unspecified or none values. If an attribute has a none or unspecified value skip this attribute."# or general values. Include relevant information that might to be used for future retrieval"
             "Each dialogue turn contains a dialogue id between [ ]. Make sure to include the dialogue the attributes and values are extracted form."
             "Important: Respond only in the format [{speaker name:[Dialog id]:[attribute]<value>}]. Make sure the attribuite name is between [ ]  and the value between < > with muliple values seperated with a comma."
             "if a turn has no attributes skip it."
             "Don't include anything else."
             f"{level}:{dialog_turn}")

    if type =='question':
        #print('in question')
        '''prompt= (f"You are an expert annotator who generates the most relevant attributes in a question. Given the question below, identify the key attributes and their values."
             "Your annotations should inlcude 2 main attributes which are the name of the person as [person]<name> and what the inquiry is about as [inquiry attribute]<value>." 
             "Include other attributes that are general and relevant with very specific values."
             #"Make sure to include the name of person the question is asking about"    
             "Include value information that you find relevant and their names if mentioned."#If their is an event attribute mention the type of event."#like names of place, sport, hobbies, interests, etc.."
             "Respond in the format: [person]<name>[inquiry]<?>[attribute]<value>]. Include only one person and name. Choose the most relevant name if there are more tahn one name."
             "Also make sure that the attribuite name is between [ ]  and the value between < >."
             "Important: you must include [person]<name>[inquiry attribute]<value> in your respond with other attributes"
             "Don't include anything else."
             f"question:{dialog_turn}")
            '''
        prompt = (f"You are an expert annotator who rewrites a question in terms of a set of attributes and theor values. Given the question below, identify the key attributes and their values."
             "Attributes should be specfic with most relevant values only. attributes should be generic and values should be detailed, descriptive in few words"
             "Include value information that you find relevant. The value for the inquired attribute should be <?>" 
             f"Respond in the format:[attribute]<value>. person is one of the attributes and the person name is its value. "
             f"Important: You must include a person attribute and only one person name. Choose the most relevant person name if there are more than one person name."
             "Sort with most relevant attributes first."
             "Also make sure that the attribuite name is between [ ]  and the value between < >."
             "Don't include anything else. use only lower case letters"
             f"question:{dialog_turn}"
                  )
        '''
        f"Rewrite the following question as a set of relevant attributes and their values that represent the person the question is related to and the attributes and values being asked for."
        f"Respond in the format:[attribute]<value>. person is one of the attributes and the person name is its value. Include only one person and name. Choose the most relevant name if there are more than one name."
         "Also make sure that the attribuite name is between [ ]  and the value between < >."
         "Don't include anything else."
        f"question:{dialog_turn}")
        '''
    messages=[{"role": "user", "content": prompt}]

    body = json.dumps({
        "messages": messages,
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_new_tokens,
        # "temperature": 0.7,
        # "top_p": 0.95,
        # "top_k": 250
    })

    accept = "application/json"
    contentType = "application/json"
    try:

        response = client.invoke_model(
                    body=body, modelId=model_name, accept=accept,contentType=contentType
                )

        response_body = json.loads(response.get("body").read())
        return response_body["content"][0]["text"]

    except Exception as e:
        print('Exception :( :',e)
        return ''

    except json.decoder.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        text_block = response_body["content"][0]["text"]
        return text_block

def run_gemini(model, content: str, max_tokens: int = 0):

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return None


def run_chatgpt(query, num_gen=1, num_tokens_request=1000, 
                model='chatgpt', use_16k=False, temperature=1.0, wait_time=1):

    completion = None
    while completion is None:
        wait_time = wait_time * 2
        try:
            # if model == 'davinci':
            #     completion = openai.Completion.create(
            #                     # model = "gpt-3.5-turbo",
            #                     model = "text-davinci-003",
            #                     temperature = temperature,
            #                     max_tokens = num_tokens_request,
            #                     n=num_gen,
            #                     prompt=query
            #                 )
            if model == 'chatgpt':
                messages = [
                        {"role": "system", "content": query}
                    ]
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature = temperature,
                    max_tokens = num_tokens_request,
                    n=num_gen,
                    messages = messages
                )
            elif 'gpt-4' in model:
                completion = openai.ChatCompletion.create(
                    model=model,
                    temperature = temperature,
                    max_tokens = num_tokens_request,
                    n=num_gen,
                    messages = [
                        {"role": "user", "content": query}
                    ]
                )
            else:
                print("Did not find model %s" % model)
                raise ValueError
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        # except Exception as e:
        #     if e:
        #         print(e)
        #         print(f"Timeout error, retrying after waiting for {wait_time} seconds")
        #         time.sleep(wait_time)
    

    if model == 'davinci':
        outputs = [choice.get('text').strip() for choice in completion.get('choices')]
        if num_gen > 1:
            return outputs
        else:
            # print(outputs[0])
            return outputs[0]
    else:
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):

    completion = None
    
    messages = [
        {"role": "system", "content": query}
    ]
    for inp, out in examples:
        messages.append(
            {"role": "user", "content": inp}
        )
        messages.append(
            {"role": "system", "content": out}
        )
    messages.append(
        {"role": "user", "content": input}
    )   
    
    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature = temperature,
                max_tokens = num_tokens_request,
                n=num_gen,
                messages = messages
            )
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
    
    return completion.choices[0].message.content

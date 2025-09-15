from abc import abstractmethod
import json
import logging
import boto3
import configparser
import boto3, botocore
import os
#from retry import retry

MODEL2ID = {
    "v3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "v3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    #  "v3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    #  "gpt-4": "gpt-4",
    #  "gpt-4-turbo": "gpt-4-turbo-preview"
}

class LTM_LLM:
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response to the given prompt
        """
        raise NotImplementedError("generate method must be implemented")

    def get_name(self) -> str:
        return self.__class__.__name__


class LTMBackbone (LTM_LLM):
    def __init__(self,
                 model="v3-sonnet"):

        session = boto3.Session(profile_name='bedrock-profile')
        config = botocore.config.Config(
            read_timeout=900, connect_timeout=900, retries={"max_attempts": 3}
        )
        self.client = session.client(
            service_name="bedrock-runtime", region_name="us-east-1", config=config
        )
        self.model_id = MODEL2ID[model]

        self.max_tokens = 1024

        #self.system_prompt = "Respond helpfully."

        #self.api_key = os.environ.get('OPENAI_API_KEY')


    def generate_text_claude(self,
                             prompt: str,
                             model_id: str = "anthropic.claude-v2"
                             ) -> str:
        body = json.dumps({
            "messages": [
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
            "max_tokens": 8000,
            #"temperature": 0.7,
            #"top_p": 0.95,
            #"top_k": 250
        })
        try:
            modelId = model_id
            accept = "application/json"
            contentType = "application/json"
            response = self.client.invoke_model(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
            response = json.loads(response.get('body').read()).get("content")[0]['text']
            return response
        except:
            return ""

#client=LtmClient()
#print(client.generate_text_claude("Recommend a movie to watch that is emotional romantic and new"))
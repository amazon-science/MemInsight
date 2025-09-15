DIALOGUE_TURN_ANNOTATION_BASIC=(f"You are an expert annotator who generates the most relevant attributes in a conversation. Given the conversation below, identify the key attributes and their values on a turn by turn level."
             "Attributes should be specific with most relevant values only. attributes should be generic and values should be detailed, descriptive in few words"
             "Don't include speaker name."
             "Include value information that you find relevant and their names if mentioned. use few words only."#If their is an event attribute mention the type of event."#like names of place, sport, hobbies, interests, etc.."
             "Don't include any unspecified or none values. If an attribute has a none or unspecified value skip this attribute."# or general values. Include relevant information that might to be used for future retrieval"
             "Each dialogue turn contains a dialogue id between [ ]. Make sure to include the dialogue the attributes and values are extracted form."
             "Important: Respond only in the format [{speaker name:[Dialog id]:[attribute]<value>}]. Make sure the attribute name is between [ ]  and the value between < > with multiple values separated with a comma."
             "if a turn has no attributes skip it."
             "Don't include anything else."
             "Dialogue Turn:{}")

DIALOGUE_TURN_ANNOTATION_BASIC_ORDERED=  (
            "You are an expert dialogue annotator, given the following dialogue turn generate a list of attributes and values for relevant information in the text."
            "Generate the annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text."
            "and values for relevant information in this dialogue turn with respect to each person. Be concise and direct."
            "Include person name as an attribute and value pair."
            "Please make sure you read and understand these instructions carefully."
            "1- Identify the key attributes in the dialogue turn and their corresponding values."
            "2- Arrange attributes descendingly with respect to relevance from left to right."
            "3- Generate the sorted annotations list in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text."
            "4- Skip all attributes with none vales"
            "Important: YOU MUST  attribute name is between [ ] and value between < >. Only return a list of [attribute]<value> nothing else. "
            "Don't include anything else or extra text in your response."
            "Dialogue Turn: {}"
        )

DIALOGUE_TURN_ANNOTATION_MISTRAL=  ("[INST]"
            "You are an expert dialogue annotator, Your task is to generate a list of attributes and values for relevant information in the given dialogue turn."
            "You must generate and respond with the annotations in the format: [attribute_name]<value> where attribute_name is the attribute name and value is its corresponding value from the dialogue turn."
            "Only return a list of [attribute_name]<value>. "
            "Don't include anything else or extra text in your response."
            #"Just generate the JSON object without explanations:"
            "Input Dialogue Turn: {}"
            "[/INST]"


        )

'''Please make sure you read and understand these instructions carefully."
            "1- Identify the key attributes in the dialogue turn and their corresponding values."
            "2- Generate annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text."
            "3- Skip all attributes with none vales"
            '''
DIALOGUE_TURN_ANNOTATION_BASIC_V3=  (
            "You are an expert dialogue annotator, given the following dialogue turn generate a list of attributes and values for relevant information "
            "like events or attributes for learning, memory, sensation, perception, attention, emotion, language, and motivation for the speaker."
            "Generate the annotations in the format: [attribute_name]<value> where attribute is the attribute name and value is its corresponding value from the text."
            "and values for relevant information in this dialogue turn with respect to each person. Be concise and direct."
            "Include person name as an attribute and value pair."
            "Please make sure you read and understand these instructions carefully."
            "1- Identify the key attributes in the dialogue turn and their corresponding values."
            "2- Generate annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text."
            " Important: Ensure attribute name is between [ ] and value between < >. Only return a list of [attribute]<value> nothing else. "
            " Don't include anything else or extra text in your response."
            "Dialogue Turn: {}"
        )


DIALOGUE_TURN_ANNOTATION_LLAMA_SHORT=(
    "You are an AI assistant designed to annotate dialogue turns"
    "Identify the key attributes in the dialogue turn and represent as [attribute]<value>."
    "Be concise and specific "
    "Format: generate the annotation for the dialogue turn as  [attribute]<value> list."
    "Input Dialogue turn:  \n\n {} \n\n "
)
DIALOGUE_TURN_ANNOTATION_LLAMA_FORMATTED=(
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>"
    "You are an AI assistant designed to annotate dialogue turns"
    "Identify the key attributes in the dialogue turn and represent as [attribute]<value>."
    "Be concise and specific "
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
    "Generate the annotation for the dialogue turn as  [attribute]<value> list for the given dialogue turn:  \n\n {} \n\n ?"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)
DIALOGUE_TURN_ANNOTATION_LLAMA=(
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>"
    "You are an AI assistant designed to annotate dialogue turns"
    "Please make sure you read and understand these instructions carefully."
    "1- Identify the key attributes in the dialogue turn. "
    "2- Identify the corresponding values for each attribute. Be concise and specific "
    "3- Represent each annotation as: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text."
    "Important: Ensure attribute name is between [ ] and value between < >. "
    "Don't include anything else or extra text in your response."
    "Don't include the dialogue or parts of it."
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
    "What is the [attribute]<value> list for the given dialogue turn:  \n\n {} \n\n ?"
    "<|eot_id|>"
)

DIALOGUE_ANNOTATION=(
   "You are an expert annotator who generates the most relevant attributes in a conversation. Given the conversation below with dialogue ids for each turn."
   "identify relevant key attributes and their values that describe most important information in the conversation like events, emotions, intent, etc."
   "Attributes should be specific with most relevant and specific values only."
   "For every attribute mention the reference dialogue id and person name."
   "Group related attributes together with order of relevance from left to right"
   "Important: Return a json file including the format [%speaker name%:{Dialog_id:$[attribute]<value>$}]. Make sure the attribute name is between [ ]  and the value between < > with multiple values separated with a comma."
   "if a dialogue turn has no attributes skip it."
   "Don't include anything else. Don't include special characters"
   "Dialogue:"
)

REFLECT_ANNOTATION=(
   "You are an expert evaluator for conversation annotations."
   "Given a list of attributes and their values with turn id generated for the dialogue below in the format [%speaker name%:{Dialog_id:$[attribute]<value>$}]"
   "Evaluate the efficacy of these annotations to represent most important information like events, emotions and intent, etc., in the dialogue, correctly reference related dialogue turns and their usage to generate back the dialogue."
   "Update these attributes and their values to be more descriptive of the dialogue, specific and contain relevant information and their values."
   "Important: Return back the annotations in a json file in the format [speaker name:{[attribute]<value>}]. Make sure the attribute name is between [ ]  and the value between < > with multiple values separated with a comma."
   "if a dialogue turn has no attributes skip it."
   "Don't include anything else. Don't include special characters."
   "Dialogue:{}"
   "Annotations:{}"
)
#AR:Annotate Reflect
DIALOGUE_ANNOTATION_AR=(
   "You are a dialogue annotator who generates the most relevant attributes in a conversation. Given the conversation below"
   "identify relevant key attributes and their values that describe most important information in the conversation like events, emotions, intent, etc."
   "Attributes should be specific with most relevant and specific values only."
   "For every attribute mention the speaker name."
   "mention attributes with order of relevance from left to right"
   "Important: the response format should be in json format like [speaker name:[attribute]<value>]. Make sure the attribute name is between [ ]  and the value between < >."
   "Don't include anything else. Don't include special characters or new lines"
   "Dialogue:{}"
)


REFLECT_ANNOTATION_AR=(
   "You are a dialogue generator and evaluator for conversation annotations."
   "Given a list of attributes and their values generated for the dialogue below."
   "Evaluate the efficacy of these annotations to regenerate back the dialogue. Update the annotations if needed to represent the most important information like events, emotions and intent, etc.."
   "Update these attributes and their values to be more descriptive of the dialogue, specific and contain relevant information and their values."
   "Important: the response format should be in json format like [speaker name:[attribute]<value>]. Make sure the attribute name is between [ ]  and the value between < >."
   "Don't include anything else. Don't include special characters or new lines."
   "Annotations:{}"
   "Dialogue Turn:{}"
)

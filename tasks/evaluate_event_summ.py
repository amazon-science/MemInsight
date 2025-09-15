import sys
import time
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import os, json
from ..memory import global_methods as glbl  # import run_claude, run_claude_for_annotations
import pandas as pd
import re

from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
# from augmentation_prompts import DIALOGUE_ANNOTATION, REFLECT_ANNOTATION
from evaluation import rougel_score

#####################################Files###############################################
#*****Added
with open('config.json', 'r') as f:
    config = json.load(f)
ANNOTATIONS_FILE= config["annotations_dir"]
TYPE= config["type"]
#*****

#####################################Prompts#############################################
DIALOGUE_ANNOTATION = (
    "You are a dialogue annotator who generates the most relevant attributes in a conversation. Given the conversation below"
    "identify relevant key attributes and their values that describe most important information in the conversation like events, emotions, intent, etc."
    "Attributes should be specific with most relevant and specific values only."
    "For every attribute mention the speaker name."
    "mention attributes with order of relevance from left to right"
    "Important: the response format should be in json format like {speaker name:{[attribute]<value>}. Make sure the attribuite name is between [ ]  and the value between < >."
    "Don't include anything else. Don't include special characters or new lines"
    "Dialogue:"
)
REFLECT_ANNOTATION = (
    "You are aa dialogue generator and evaluator for conversation annotations."
    "Given a list of attributes and their values generated for the dialogue below."
    "Evaluate the efficacy of these annotations to regenerate back the dialogue. Update the annotations if needed to represent the most important information like events, emotions and intent, etc.."
    "Update these attributes and their values to be more descriptive of the dialogue, specific and contain relevant information and their values."
    "Important: the response format should be in json format like {speaker name:{[attribute]<value>}. Make sure the attribuite name is between [ ]  and the value between < >."
    "Don't include anything else. Don't include special characters or new lines."
)

EVENTS_SUMMARY_LABEL = (
    "Generate an event summary paragraph for the events listed below for the given speaker name and dates. Input:")

EVENTS_SUMMARY_AUG = (
    "Given a list of attributes and values listed below in the format [attribute name]<value>. Your job is to generate an event summary paragraph for the included events. Include speaker name and dates"
    "Include only text and no special characters or brackets. Input:")

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--prompt-dir', type=str, default="")
    args = parser.parse_args()
    return args


#TODO remove and use the one in global_methods file
# Standard method for all model to run a given query on the given model
def run_model(model, query):
    while True:
        try:
            if 'claude' in model:
                if 'haiku' in model:
                    return glbl.run_claude(query, model_name='claude-haiku')
                else:
                    return glbl.run_claude(query)

            if 'llama' in model:
                return glbl.run_llama(query)

            if 'mistral' in model:
                return glbl.run_mistral(query)
        except Exception as e:
            print(e)
            continue

# Helper function to get a conversation in a textual form from a given sample
def get_conversation(data):
    conversations = []
    session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                    'session' in k and 'date_time' not in k]
    for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                  desc='Generating Annotations for %s' % data['sample_id']):

        date_time = data['conversation'][('session_%s_date_time') % i]

        conversation = ""
        conversation += date_time + '\n'
        for ii, dialog in enumerate(data['conversation']['session_%s' % i]):
            d_id = dialog["dia_id"]
            d_speaker = dialog['speaker'].lower()
            try:
                d_text = dialog['clean_text'].lower()
                conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + d_text + '\"'
            except KeyError:
                d_text = dialog['text'].lower()
                conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + d_text + '\"'

            if 'blip_caption' in dialog:
                d_blip = dialog['blip_caption']
                conversation += ' and shared ' + dialog['blip_caption']

            conversation += '\n'

        conversation = conversation.lower().strip()
        conversations.append(conversation)

    return conversations


"""""""""""""""""""""""
|*Generating Summaries*|
"""""""""""""""""""""""


# Generate event summaries in a format similar to the LoCoMo dataset
# Input:  ann_file: An augmented file with LLM generated annotations, model: the model used for summary generation
# Output: a dictionary for events in each session (session level)
def get_event_summaries(ann_file, model):
    prompt = (
        "Given the following attributes and values that annotate a dialogue for every speaker in the format  {speaker name:{[attribute]<value>},"
        "generate a summary for the event attributes only to describe the main and important events represented in these annotations. Refrain from mentionaing any minimal event. Include any event related details and speaker. "
        "Format: a bullet paraghraph for major life events for every speaker with no special characters."
        "Don't include anything else in your response or extra text or lines."
        "Don't include bullets."
        "Input annotations:"
    )
    with open(ann_file, 'r') as f:
        ann = json.load(f)
    events = {}
    for sample_id, sessions in ann.items():
        events[sample_id] = {}
        for session, atts in sessions.items():
            summary = run_model(model, prompt + str(atts))
            events[sample_id].update({session: summary})
    return events


def get_all(model,datafile,annfile=''):
    prompt_annfile = (
        "Given the following conversation between two speakers and its annotations on the turn level in form of Annotations: [attribute]<value> for each dialogue turn."
        "Generate a concise summary for the most important events based on the events attributes and the given conversation."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Input:"
    )
    prompt_ann_tl = (
        "You are an expert dialogue annotator, given the following dialogue turns between 2 speakers, generate a list of relevant attributes for major events and relevant information."
        "Please make sure you read and understand these instructions carefully."
        "1- Identify the key attributes for each dialog turn and their corresponding values. Be concise and specific"
        "2- Generate a list of annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text. "
        "Its important to ensure that attribute name is between [ ] and value between < >."
        "Don't include anything else in your response."
        "Dialogue turn: {}")

    prompt_tl = (
        "Given the following conversation between two speakers and its annotations on the turn level in form of Annotations: [attribute]<value> for each dialogue turn."
        "Generate a concise summary for the most important events based on the events attributes and conversation."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Input:"
    )

    prompt_ann_sl = (
        "You are an expert dialogue annotator, given the following conversation between 2 speakers generate a list of relevant attributes"
        "and values for major events and relevant information in this conversation with respect to each person. "
        "Please make sure you read and understand these instructions carefully."
        "1- Identify the key attributes in the conversation and their corresponding values. Be concise and specific"
        "2- Generate a list of annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text. Its important to ensure that attribute name is between [ ] and value between < >. Don't include anything else in your response."
        " Conversation: {}")

    prompt_sl = (
        "Given the following conversation between two speakers and its annotations on the conversation level in form of [attribute]<value>."
        "Generate a concise summary for the most important events based on the given annotations and dialogue."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Conversation:{}"
        "Annotations:{}"
    )
    prompt_ann_atl = (
        "You are an expert dialogue annotator, given the following dialogue turn generate a list of relevant attributes"
        " for major events and relevant information. Generate the annotations in the format: [attribute]<value> where attribute is teh attribute name and value is its corresponding value from the text."
        "Important: make sure to include attributes names between [ ] and value between < >. "
        " Don't inlcude anything else in your response."
        " Dialogue turn: {}")

    prompt_atl= (
        "Given the following annotations for a conversation between two speakers. The annotations are on the turn level in form of a list of [attribute]<value> for each dialogue turn."
        "Generate a concise summary for the most important events based on the events attributes."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Annotations:"
        # "Annotations:{}"
    )

    prompt_bl = (
        "Given the following conversation between two speakers. Each turn is seperated by a newline"
        "Generate a summary of the most important events."
        "Don't include anything else in your response or extra text or lines."
        "Conversation:"
    )

    prompt_ann_asl = (
        "You are an expert dialogue annotator, given the following dialogue turns generate a list of relevant attributes"
        " for major events and relevant information. Generate the annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text."
        "Important: make sure to include attributes names between [ ] and value between < >. "
        " Don't include anything else in your response."
        " Dialogue: {}")

    prompt_asl = (
        "Given the following annotations for a conversation between two speakers. The annotations are in the form of a list of [attribute]<value> for a whole conversation."
        "Generate a concise summary for the most important events based on the events attributes."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Annotations:"
        # "Annotations:{}"
    )
    with open(datafile, 'r') as dfile:
        samples = json.load(dfile)

    if annfile != '':
        with open(annfile, 'r') as afile:
            samples_ann = json.load(afile)

    sessions_summaries_annfile = {}
    sessions_summaries_tl= {}
    sessions_summaries_sl = {}
    sessions_summaries_atl = {}
    sessions_summaries_asl = {}
    sessions_summaries_bl = {}

    for data in samples:

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        sessions_summaries_annfile[data['sample_id']] = {}
        sessions_summaries_tl[data['sample_id']] = {}
        sessions_summaries_sl[data['sample_id']] = {}
        sessions_summaries_atl[data['sample_id']] = {}
        sessions_summaries_asl[data['sample_id']] = {}
        sessions_summaries_bl[data['sample_id']] = {}

        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Generating Event Summaries for %s' % data['sample_id']):

            date_time = data['conversation'][('session_%s_date_time') % i]
            conversation = ''
            conversation += date_time + '\n'

            annotations_atl=''
            annotations_atl += date_time + '\n'

            annotations_asl = ''
            annotations_asl += date_time + '\n'

            for _, dialog in enumerate(data['conversation']['session_%s' % i]):
                d_id = dialog["dia_id"]
                d_speaker = dialog['speaker'].lower()
                try:
                    d_text = dialog['clean_text'].lower()
                    conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
                except KeyError:
                    d_text = dialog['text'].lower().strip()
                    conversation += dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

                if 'blip_caption' in dialog:
                    d_blip = dialog['blip_caption']
                    conversation += ' and shared ' + dialog['blip_caption']

                conversation_ann=conversation
                conversation_ann += ' Annotations:' + str(
                    # samples_ann[data['sample_id']]['session_%s' % i][d_speaker][d_id.lower()]) #locomo_annotations
                    samples_ann[data['sample_id']]['session_%s' % i])
                conversation_ann +='\n'

                conversation_tl=conversation
                conversation_tl += ' Annotations:' + run_model(model, prompt_ann_tl.format(
                    dialog['speaker'] + ' said, \"' + d_text + '\"'))
                conversation_tl+='\n'

                conversation_sl=conversation
                annotations_sl = run_model(model, prompt_ann_sl.format(conversation))
                summary = run_model(model, prompt_sl.format(conversation, annotations_sl))
                sessions_summaries_sl[data['sample_id']].update({'session_%s' % i: summary})
                conversation_sl+='\n'

                annotations_atl += run_model(model, prompt_ann_atl.format(conversation)) + '\n'

                conversation += '\n'  # keep annotation to files

            summary_annfile = run_model(model, prompt_annfile + str(conversation_ann))
            sessions_summaries_annfile[data['sample_id']].update({'session_%s' % i: summary_annfile})

            summary_tl = run_model(model, prompt_tl + str(conversation))
            sessions_summaries_tl[data['sample_id']].update({'session_%s' % i: summary_tl})

            summary_atl = run_model(model, prompt_atl + str(annotations_atl))
            sessions_summaries_atl[data['sample_id']].update({'session_%s' % i: summary_atl})

            annotations_asl += run_model(model, prompt_ann_asl.format(conversation)) + '\n'
            summary_asl = run_model(model, prompt_asl + str(annotations_asl))
            sessions_summaries_asl[data['sample_id']].update({'session_%s' % i: summary})

            baseline_summary = run_model(model, prompt_bl + conversation)
            sessions_summaries_bl[data['sample_id']].update({'session_%s' % i: baseline_summary})


    # Store results
    with open(f'Evaluation_summary/event_diag_using_{annfile}_summaries_{model}.json', 'w') as f:
        json.dump(sessions_summaries_annfile, f)

    with open(f'Evaluation_summary/event_diag_turn_level_summaries_{model}_{datetime.datetime.now()}.json', 'w') as f:
        json.dump(sessions_summaries_tl, f)

    with open(f'Evaluation_summary/event_diag_session_level_summaries_{model}_{datetime.datetime.now()}.json','w') as f:
        json.dump(sessions_summaries_sl, f)

    with open(f'Evaluation_summary/event_ann_only_summaries_tl_{model}_{datetime.datetime.now()}.json', 'w') as f:
        json.dump(sessions_summaries_atl, f)

    with open(f'Evaluation_summary/event_ann_only_summaries_sl_{model}_{datetime.datetime.now()}.json', 'w') as f:
        json.dump(sessions_summaries_asl, f)

    with open(f'Evaluation_summary/baseline_summaries_{model}_{time.time()}.json', 'w') as f:
        json.dump(sessions_summaries_bl, f)

    return sessions_summaries_annfile,sessions_summaries_tl,sessions_summaries_sl,sessions_summaries_atl,sessions_summaries_asl,sessions_summaries_bl


# Generating event summaries based on dialogues (turn level or session depending on annotation file) and pre-generatred annotations, both files provided
def get_event_and_dialog_summaries(datafile, annfile, model):
    prompt = (
        "Given the following conversation between two speakers and its annotations on the turn level in form of Annotations: [attribute]<value> for each dialogue turn."
        "Generate a concise summary for the most important events based on the events attributes and the given conversation."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Input:"
    )

    with open(datafile, 'r') as dfile:
        samples = json.load(dfile)

    with open(annfile, 'r') as afile:
        samples_ann = json.load(afile)

    sessions_summaries = {}

    for data in samples:

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        sessions_summaries[data['sample_id']] = {}

        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Generating Event Summaries for %s' % data['sample_id']):

            date_time = data['conversation'][('session_%s_date_time') % i]
            conversation = ""
            conversation += date_time + '\n'
            for _, dialog in enumerate(data['conversation']['session_%s' % i]):
                d_id = dialog["dia_id"]
                d_speaker = dialog['speaker'].lower()
                try:
                    d_text = dialog['clean_text'].lower()
                    conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
                except KeyError:
                    d_text = dialog['text'].lower().strip()
                    conversation += dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

                if 'blip_caption' in dialog:
                    d_blip = dialog['blip_caption']
                    conversation += ' and shared ' + dialog['blip_caption']

                conversation += ' Annotations:' + str(
                    # samples_ann[data['sample_id']]['session_%s' % i][d_speaker][d_id.lower()]) #locomo_annotations
                    samples_ann[data['sample_id']]['session_%s' % i])
                conversation += '\n'  # keep annotation to files

            summary = run_model(model, prompt + str(conversation))
            sessions_summaries[data['sample_id']].update({'session_%s' % i: summary})

    # Store results
    with open(f'Evaluation_summary/event_diag_using_{annfile}_summaries_{model}.json', 'w') as f:
        json.dump(sessions_summaries, f)

    return sessions_summaries


# Generating event summaries based on dialogues (turn level) with on the fly annotations generation
def get_event_and_dialog_turn_level_summaries(datafile, model):
    prompt_ann = (
        "You are an expert dialogue annotator, given the following dialogue turns between 2 speakers, generate a list of relevant attributes for major events and relevant information."
        "Please make sure you read and understand these instructions carefully."
        "1- Identify the key attributes for each dialog turn and their corresponding values. Be concise and specific"
        "2- Generate a list of annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text. "
        "Its important to ensure that attribute name is between [ ] and value between < >."
        "Don't include anything else in your response."
        "Dialogue turn: {}")

    prompt = (
        "Given the following conversation between two speakers and its annotations on the turn level in form of Annotations: [attribute]<value> for each dialogue turn."
        "Generate a concise summary for the most important events based on the events attributes and conversation."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Input:"
    )

    with open(datafile, 'r') as dfile:
        samples = json.load(dfile)

    sessions_summaries = {}

    for data in samples:

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        sessions_summaries[data['sample_id']] = {}

        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Generating Event Summaries (Turn Level Annotations) for %s' % data['sample_id']):

            date_time = data['conversation'][('session_%s_date_time') % i]
            conversation = ""
            conversation += date_time + '\n'
            for _, dialog in enumerate(data['conversation']['session_%s' % i]):
                d_id = dialog["dia_id"]
                d_speaker = dialog['speaker'].lower()
                try:
                    d_text = dialog['clean_text'].lower()
                    conversation += dialog['speaker'] + ' said, \"' + d_text + '\"'
                except KeyError:
                    d_text = dialog['text'].lower().strip()
                    conversation += dialog['speaker'] + ' said, \"' + d_text + '\"'

                if 'blip_caption' in dialog:
                    d_blip = dialog['blip_caption']
                    conversation += ' and shared ' + dialog['blip_caption']

                conversation += ' Annotations:' + run_model(model, prompt_ann.format(
                    dialog['speaker'] + ' said, \"' + d_text + '\"'))
                conversation += '\n'

            summary = run_model(model, prompt + str(conversation))
            sessions_summaries[data['sample_id']].update({'session_%s' % i: summary})

    with open(f'Evaluation_summary/event_diag_turn_level_summaries_{model}_{datetime.datetime.now()}.json', 'w') as f:
        json.dump(sessions_summaries, f)

    return sessions_summaries


# Generating event summaries based on dialogues (Session level) with on the fly annotations generation
def get_event_and_session_level_summaries(datafile, model):
    prompt_ann = (
        "You are an expert dialogue annotator, given the following conversation between 2 speakers generate a list of relevant attributes"
        "and values for major events and relevant information in this conversation with respect to each person. "
        "Please make sure you read and understand these instructions carefully."
        "1- Identify the key attributes in the conversation and their corresponding values. Be concise and specific"
        "2- Generate a list of annotations in the format: [attribute]<value> where attribute is the attribute name and value is its corresponding value from the text. Its important to ensure that attribute name is between [ ] and value between < >. Don't include anything else in your response."
        " Conversation: {}")

    prompt = (
        "Given the following conversation between two speakers and its annotations on the conversation level in form of [attribute]<value>."
        "Generate a concise summary for the most important events based on the given annotations and dialogue."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Conversation:{}"
        "Annotations:{}"
    )

    with open(datafile, 'r') as dfile:
        samples = json.load(dfile)

    sessions_summaries = {}

    for data in samples:

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        sessions_summaries[data['sample_id']] = {}

        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Generating Event Summaries (Session Level Annotations) for %s' % data['sample_id']):
            print('**In Session %s**' % i)
            date_time = data['conversation'][('session_%s_date_time') % i]
            conversation = ""
            conversation += date_time + '\n'
            for _, dialog in enumerate(data['conversation']['session_%s' % i]):
                try:
                    d_text = dialog['clean_text'].lower()
                    conversation += dialog['speaker'] + ' said, \"' + d_text + '\"'
                except KeyError:
                    d_text = dialog['text'].lower().strip()
                    conversation += dialog['speaker'] + ' said, \"' + d_text + '\"'

                if 'blip_caption' in dialog:
                    d_blip = dialog['blip_caption']
                    conversation += ' and shared ' + dialog['blip_caption']

            conversation += '\n'  # keep annotation to files

            annotations = run_model(model, prompt_ann.format(conversation))
            summary = run_model(model, prompt.format(conversation, annotations))
            sessions_summaries[data['sample_id']].update({'session_%s' % i: summary})

    print('Number of generated summaries', len(sessions_summaries))
    with open(f'Evaluation_summary/event_diag_session_level_summaries_{model}_{datetime.datetime.now()}.json',
              'w') as f:
        json.dump(sessions_summaries, f)
    return sessions_summaries


def get_event_from_annotations_turn_level_summaries(datafile, model):
    prompt_ann = (
        "You are an expert dialogue annotator, given the following dialogue turn generate a list of relevant attributes"
        " for major events and relevant information. Generate the annotations in the format: [attribute]<value> where attribute is teh attribute name and value is its corresponding value from the text."
        "Important: make sure to include attributes names between [ ] and value between < >. "
        " Don't inlcude anything else in your response."
        " Dialogue turn: {}")

    prompt = (
        "Given the following annotations for a conversation between two speakers. The annotations are on the turn level in form of a list of [attribute]<value> for each dialogue turn."
        "Generate a concise summary for the most important events based on the events attributes."
        "Don't include anything else in your response, Don't include extra text, lines, numbers, bullets or special characters."
        "Annotations:"
        # "Annotations:{}"
    )
    with open(datafile, 'r') as dfile:
        samples = json.load(dfile)

    sessions_summaries = {}

    for data in samples:

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        sessions_summaries[data['sample_id']] = {}

        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Generating Event Summaries for %s' % data['sample_id']):

            date_time = data['conversation'][('session_%s_date_time') % i]
            annotations = ''
            annotations += date_time + '\n'
            conversation = ''
            for _, dialog in enumerate(data['conversation']['session_%s' % i]):
                try:
                    d_text = dialog['clean_text'].lower()
                    conversation += dialog['speaker'] + ' said, \"' + d_text + '\"'
                except KeyError:
                    d_text = dialog['text'].lower().strip()
                    conversation += dialog['speaker'] + ' said, \"' + d_text + '\"'

                if 'blip_caption' in dialog:
                    d_blip = dialog['blip_caption']
                    conversation += ' and shared ' + dialog['blip_caption']

                annotations += run_model(model, prompt_ann.format(conversation)) + '\n'

            summary = run_model(model, prompt + str(annotations))
            sessions_summaries[data['sample_id']].update({'session_%s' % i: summary})

    with open(f'Evaluation_summary/event_ann_only_summaries_{model}_{datetime.datetime.now()}.json', 'w') as f:
        json.dump(sessions_summaries, f)

    return sessions_summaries


def get_baseline_summaries(datafile, model):
    prompt = (
        "Given the following conversation between two speakers. Each turn is seperated by a newline"
        "Generate a summary of the most important events."
        "Don't include anything else in your response or extra text or lines."
        "Conversation:"
    )

    with open(datafile, 'r') as dfile:
        samples = json.load(dfile)

    sessions_summaries = {}

    for data in samples:

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        sessions_summaries[data['sample_id']] = {}

        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Generating Event Summaries for %s' % data['sample_id']):

            date_time = data['conversation'][('session_%s_date_time') % i]
            conversation = ""
            conversation += date_time + '\n'
            for _, dialog in enumerate(data['conversation']['session_%s' % i]):
                d_id = dialog["dia_id"]
                d_speaker = dialog['speaker'].lower()
                try:
                    d_text = dialog['clean_text'].lower()
                    conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
                except KeyError:
                    d_text = dialog['text'].lower().strip()
                    conversation += dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

                if 'blip_caption' in dialog:
                    d_blip = dialog['blip_caption']
                    conversation += ' and shared ' + dialog['blip_caption']

                conversation += '\n'  # keep annotation to files

            baseline_summary = run_model(model, prompt + conversation)
            sessions_summaries[data['sample_id']].update({'session_%s' % i: baseline_summary})

    print('Number of generated summaries', len(sessions_summaries))
    with open(f'Evaluation_summary/baseline_summaries_{model}_{time.time()}.json', 'w') as f:
        json.dump(sessions_summaries, f)

    return sessions_summaries


"""""""""""""""""
 |**Evaluation**|
"""""""""""""""""


# evaluates a generated list of events vs event summaries in data samples
def evaluate_events(event_ann, data, model):
    with open(event_ann, 'r') as f:
        events_ann = json.load(f)
    with open(data, 'r') as f2:
        sample = json.load(f2)
    labels = []
    aug = []
    for data in sample:
        print('**********Evaluating**************')
        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]
        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Evaluating events for %s' % data['sample_id']):
            label_session = data['event_summary']['events_session_%s' % i]
            aug_events = events_ann[data['sample_id']]['session_%s' % i]
            labels.append(run_model(model, EVENTS_SUMMARY_LABEL + str(label_session)))
            aug.append(run_model(model, EVENTS_SUMMARY_AUG + str(aug_events)))
    print(rougel_score(aug, labels))


# Evaluate event summaries
# Input:  events dictionary, data file, model used for generation, usage of summary of events from labels, evaluation against raw dialogue
# Metric: Rouge, BERTScore, G-Eval(Relevance, Coherence, Consistency)
def evaluate_summaries(events, datafile, model, descrp='', events_summary=True, dialogue=False):
    label_events_to_summary_prompt = (
        "Given the following list of events for every speaker,"
        "generate a summary of the represented events."
        "Format: a bullet paraghraph for major life events for every speaker with no special characters."
        "Don't include anything else in your response or extra text or lines."
        "Don't inlcude bullets."
        "Input:"
    )

    label_summary = []
    pred_summary = []

    with open(datafile, 'r') as f:
        samples = json.load(f)

    for data in samples:
        if dialogue:  # the summaries will be evaluated with raw dialogues
            label_summary.extend(get_conversation(data))

        for session, label_events in data['event_summary'].items():

            pred_summary.append(events[data['sample_id']]['session_' + session.split('_')[-1]])

            if dialogue: continue  # conversations already packed in label_summary

            if events_summary:  # the summaries will be evaluated with summary of event labels
                label_summary.append(run_model(model, label_events_to_summary_prompt + str(label_events)))


            else:
                label_summary.append(str(label_events))
    scores_all = {}
    scores = []
    scores_1 = []
    scores_2 = []
    scores_l = []
    missed = 0

    # Evaluation
    for aug, label in zip(pred_summary, label_summary):
        score1, score2, scorel, score = rougel_score(aug, label)
        if score == {}:
            missed += 1
            continue
        scores_1.append(score1)
        scores_2.append(score2)
        scores_l.append(scorel)
        #scores.append(score)

    # TODO-Update remove and get scores from returned score once, remove extra variables
    print('Rouge-1:', sum(scores_1) / len(scores_1))
    print('Rouge-2:', sum(scores_2) / len(scores_2))
    print('Rouge-l:', sum(scores_l) / len(scores_l))

    scores_all['Exp. Description'] = [descrp]
    scores_all['Time'] = [datetime.datetime.now()]
    scores_all['Rouge ALl'] = [scores]
    scores_all['Rouge 1'] = [sum(scores_1) / len(scores_1)]
    scores_all['Rouge 2'] = [sum(scores_2) / len(scores_2)]
    scores_all['Rouge l'] = [sum(scores_l) / len(scores_l)]

    scores_bert = bert_score(pred_summary, label_summary)
    print(sum(scores_bert) / len(scores_bert))

    scores_all['Bert Score'] = [str(scores_bert)]
    scores_all['Avg Bert Score'] = [sum(scores_bert) / len(scores_bert)]

    g_eval_scores = g_eval(pred_summary, label_summary, model)

    scores_all['GEval'] = [g_eval_scores]

    scores_all['Missed'] = [missed]

    # Store results to csv file
    df = pd.DataFrame(scores_all)
    df.to_csv('Evaluation_summary/' + model + '_all_final.csv',mode='a', sep='\t', index=False, header=False)
    print('total number of evaluated items', len(pred_summary))
    return scores_all


#########Added Evaluation Functions##########

def g_eval(pred, label, model):
    # Reference: https://github.com/nlpyang/geval/blob/main/prompts/summeval/rel_detailed.txt
    scores = ['relevance', 'coherence', 'consistency']
    all_scores = {}
    folder = "g_eval_prompts/"

    for score in scores:
        tmp_scores = []
        for pred_summ, label_summ in zip(pred, label):
            prompt = open(folder + score + '.txt', 'r').readlines()
            try:
                response = run_model(model, '\n'.join(prompt).format(label_summ, pred_summ))
                tmp_scores.append(int(response))
            except ValueError:
                response = re.sub(r'\D', '', response)
                try:
                    int_response = int(response)
                    if int_response >= 1 and int_response <= 5:
                        tmp_scores.append(int_response)
                    else:
                        tmp_scores.append(0)
                except:
                    print('Error! Response received%s' % response)

        avg_score = sum(tmp_scores) / len(tmp_scores)
        print(f'Average Score for {score} is {avg_score}')
        all_scores[score] = avg_score

    return all_scores


def bert_score(pred, label):
    scores = []
    for pred_summ, label_summ in zip(pred, label):
        reference = [label_summ.split()]
        candidate = pred_summ.split()
        scores.append(sentence_bleu(reference, candidate))
    return scores


if __name__ == "__main__":

    # tmp_file,final_file = annotate_and_reflect('/local/home/ranasal/Mem_Auig_v2/locomo/data/locomo10.json')

    final_annotation_file = 'final_reflect_locomo_annotations_all.json'
    locomo_file = '../data/locomo10.json'
    locomo_annotation_file = 'locomo_annotations.txt'

    models = ['mistral']  # 'mistral','claude-haiku','llama''claude-sonnet'

    for model in models:
        '''
        print("***********Evaluating Annotations from files %s *************" % model)
        print(locomo_annotation_file)
        event_summary=get_event_and_dialog_summaries(locomo_file,locomo_annotation_file,model)
        print('No Dialogue and Event Summary Evaluation')
        evaluate_summaries(event_summary, locomo_file, model,descrp='annotations from file %s Summary Evaluation'%locomo_annotation_file)
        print('Label Events Evaluation')
        evaluate_summaries(event_summary, locomo_file, model,
                           descrp='annotations from file %s with Label Events Evaluation' % locomo_annotation_file)

        #########################################################################################################
        '''
        (sessions_summaries_annfile, sessions_summaries_tl, sessions_summaries_sl, sessions_summaries_atl,
        sessions_summaries_asl, sessions_summaries_bl) = get_all(model,locomo_file,annfile=final_annotation_file)


        print("***********Evaluating Annotations from files %s *************" % model)

        print('No Dialogue and Event Summary Evaluation')
        evaluate_summaries(sessions_summaries_annfile, locomo_file, model,
                           descrp='annotations from file %s Summary Evaluation' % final_annotation_file)
        print('Label Events Evaluation')
        evaluate_summaries(sessions_summaries_annfile, locomo_file, model,
                           descrp='annotations from file %s with Label Events Evaluation' % final_annotation_file)


        #########################################################################################################
        print("***********Evaluating Turn Level Annotations for %s *************" % model)

        print('No Dialogue and Event Summary Evaluation')
        evaluate_summaries(sessions_summaries_tl, locomo_file, model,
                           descrp='annotations turn level on the fly [Summary Evaluation]')

        print('Label Events Evaluation')
        evaluate_summaries(sessions_summaries_tl, locomo_file, model,
                           descrp='annotations turn level on the fly [Events Label Evaluation]')

        #########################################################################################################
        print("***********Evaluating Session Level Annotations for %s *************" % model)

        print('No Dialogue and Event Summary Evaluation')
        evaluate_summaries(sessions_summaries_sl, locomo_file, model,
                           descrp='annotations session level on the fly [Summary Evaluation]')

        print('Label Events Evaluation')
        evaluate_summaries(sessions_summaries_sl, locomo_file, model,
                           descrp='annotations session level on the fly [Events Label Evaluation]')


        #########################################################################################################
        print("***********Evaluating Annotation only summaries TL for %s *************" % model)


        print('Event Summary Evaluation')
        evaluate_summaries(sessions_summaries_atl, locomo_file, model,
                           descrp='Annotation only summaries[Summary Evaluation]')

        print('Label Events Evaluation')
        evaluate_summaries(sessions_summaries_atl, locomo_file, model,
                           descrp='Annotation only summaries [Events Label Evaluation]')

        #########################################################################################################
        print("***********Evaluating Annotation only summaries SL for %s *************" % model)

        print('Event Summary Evaluation')
        evaluate_summaries(sessions_summaries_asl, locomo_file, model,
                           descrp='Annotation only summaries SL[Summary Evaluation]')

        print('Label Events Evaluation')
        evaluate_summaries(sessions_summaries_asl, locomo_file, model,
                           descrp='Annotation only summaries SL [Events Label Evaluation]')

        #########################################################################################################
        print("***********Evaluating Baseline for %s *************" % model)

        print('No Dialogue and Event Summary Evaluation')
        evaluate_summaries(sessions_summaries_bl, locomo_file, model,
                           descrp='Baseline [Summary Evaluation]')

        print('Label Events Evaluation')
        evaluate_summaries(sessions_summaries_bl, locomo_file, model,
                           descrp='Baseline [Events Label Evaluation]')



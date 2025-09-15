import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import re
import embedding_utils
import random
import os, json
from tqdm import tqdm
from locomo.global_methods import run_claude, run_claude_for_annotations, run_llama,run_mistral
from task_eval.rag_utils import get_embeddings
from retry import retry
from Github.prompts import augmentation_prompts
import numpy as np
import faiss
from faiss import write_index

#TODO Eliminate redundant code segments to improve performance and optimize resource utilization.
#*****Added
with open('config.json', 'r') as f:
    config = json.load(f)
ANNOTATIONS_DIR= config["annotations_dir"]
TYPE= config["type"]
#*****


MAX_LENGTH={'claude-sonnet': 2000000, 'claude-haiku': 2000000}
PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.
"""

QA_PROMPT_BATCH_ANN = """
Based on the above conversations attributes, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.
"""

# If no information is available to answer the question, write 'No information available'.

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

CONV_START_PROMPT_AUG = "Below is a some parts of a conversation between two people: {} and {}. The conversatiaon takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

CONV_START_PROMPT_ANN = "Below are the main attributes and values for a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation attributes is added to the annotations.\n\n"


#*****Added
#Standard method for all model to run a given query on the given model
@retry()
def run_model(model, query):
    if 'claude' in model:
        if 'haiku' in model:
            return run_claude(query, model_name='claude-haiku')
        else:
            return run_claude(query)

    if 'llama' in model:
        return run_llama(query)

    if 'mistral' in model:
        return run_mistral(query)
#*****Added


def process_ouput(text):
    text = text.strip()
    if text[0] != '{':
        start = text.index('{')
        text = text[start:].strip()
    return json.loads(text)


def get_cat_5_answer(model_prediction, answer_key):

    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if 'a' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    elif len(model_prediction) == 3:
        if '(a)' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    else:
        return model_prediction


def get_input_context(data, num_question_tokens, model, args):

    query_conv = ''
    stop = False
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in data:
            query_conv += "\n\n"
            for dialog in data['session_%s' % i][::-1]:
                turn = ''
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'
                query_conv = turn + query_conv

            query_conv = '\nDATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        if stop:
            break

    return query_conv

#*****Added
def aug_filtering_v1(in_data,qa):

    prompt = (
        'Given the follwoing question, determine what are the main inquiry attribute to look for and the person the question is for.'
        'Important to reply in the format: Person:[names]Attributes:[]. Don\'t include anything else in your response')

    question_attribute = run_claude(prompt + qa['question'])

    with open(in_data['sample_id'] + '_sample.json', 'r') as f:
        session_ann = json.load(f)

    context = []
    for session_id, person_att in session_ann.items():
        for person, p_att in person_att.items():
            if person not in question_attribute:
                continue
            for diag_id, diag_att in p_att.items():
                for k, v in diag_att.items():
                    if k in question_attribute or k == 'date':
                        context.append(v)
    return context


def aug_filtering_v2(data,qa,args):

    #get question attributes for filtering
    prompt = (
        'Given the follwoing question, determine what are the main inquiry attribute to look for and the person the question is for.'
        'Important to reply in the format: Person:[names]Attributes:[]. Don\'t include anything else in your response')

    question_attribute = run_claude(prompt + qa['question'])
    context = []

    #filter in all matching context
    session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                    'session' in k and 'date_time' not in k]

    # go over all sessions to find related attributes and filter
    for i in range(min(session_nums), max(session_nums) + 1):

        date_time = data['conversation']['session_%s_date_time' % i]  # session data and time
        # dialog level (in session)
        for dialog in data['conversation']['session_%s' % i]:
            if dialog['speaker'] not in question_attribute:
                continue
            dia_text = dialog['text']

            if 'blip_caption' in dialog.keys():
                dia_text +=' the speaker shared: ' + dialog['blip_caption']
            att_val = get_annotation_attributes(dia_text, args)

            for k, v in att_val.items():
                if k in question_attribute or k == 'date':
                    context.append(str(v))
    return context
#*****

#*****Updated
def get_claude_answers(in_data, out_data, prediction_key, args):

    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))
    speakers_names = list(set([d['speaker'] for d in in_data['conversation']['session_1']]))

    # start instruction prompt
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_prompt_ann=CONV_START_PROMPT_ANN.format(speakers_names[0], speakers_names[1])

    start_tokens =100

    if args.rag_mode and 'aug_att' not in args.rag_mode:
        assert args.batch_size == 1, "Batch size need to be 1 for RAG and embedding-based evaluation."
        context_database, query_vectors = prepare_for_rag(args, in_data)

    else:
        context_database, query_vectors = None, None

    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='Generating answers'):
        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            if i>=len(in_data['qa']):
                break

            qa = in_data['qa'][i] #getting question by question


            if prediction_key not in out_data['qa'][i] or args.overwrite:
                #print('in pred cond adding',i)
                include_idxs.append(i)
            else:
                continue

            #CHOOSE ANSWER FORMAT WITH RESPECT TO QUESTION CA
            if qa['category'] == 2:
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')

            elif qa['category'] == 5:
                question = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
                if random.random() < 0.5:
                    question = question.format('Not mentioned in the conversation', qa['adversarial_answer'])
                    answer = {'a': 'Not mentioned in the conversation', 'b': qa['adversarial_answer']}
                else:
                    question = question.format(qa['adversarial_answer'], 'Not mentioned in the conversation')
                    answer = {'b': 'Not mentioned in the conversation', 'a': qa['adversarial_answer']}

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)

            else:
                questions.append(qa['question'])


        if questions == []:
            continue

        context_ids = None
        distances = []

        if args.use_rag and args.rag_mode == 'dialog':
            query_conv, context_ids = get_rag_context_dialog(context_database, query_vectors[include_idxs][0],
                                                      args)  # rag mode is set to batch size 1

        elif args.use_rag and 'aug' in args.rag_mode  and 'aug_att' not in args.rag_mode:

            query_conv, context_ids,avg_dist = get_rag_context(context_database, query_vectors[include_idxs[0]], args)  # rag mode is set to batch size 1
            distances.append(avg_dist)

        elif args.use_rag and 'aug_att' in args.rag_mode and 'aug_models' not in args.model:

            context = aug_filtering_v2(in_data,qa,args)
            question_prompt = QA_PROMPT_BATCH_ANN + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            query_conv = start_prompt_ann + str(context)


        else:

            question_prompt =  QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            num_question_tokens=100
            query_conv = get_input_context(in_data['conversation'], num_question_tokens + start_tokens, None, args)
            query_conv = start_prompt + query_conv
        

        if args.batch_size == 1:

            query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])
            answer = run_claude(query)
            
            if len(cat_5_idxs) > 0:
                answer = get_cat_5_answer(answer, cat_5_answers[0])

            out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()
            if args.use_rag:
                out_data['qa'][include_idxs[0]][prediction_key + '_context'] = context_ids
                out_data['qa'][include_idxs[0]]['Avg Distance'] = str(distances)
        else:

            query = query_conv + '\n' + question_prompt

            trials = 0
            answer=""
            while trials < 5:
                try:

                    trials += 1
                    print("Trial %s" % trials)
                    print("Trying with answer token budget = %s per question" % PER_QA_TOKEN_BUDGET)
                    answer = run_claude(query, PER_QA_TOKEN_BUDGET * args.batch_size, args.model)

                    try:
                        answers = json.loads(answer).get("content")[0]['text']

                    except:
                        answers = process_ouput(answer.strip())

                    break

                except json.decoder.JSONDecodeError:
                    print('in except')
                    pass
            
            for k, idx in enumerate(include_idxs):
                try:
                    answers = process_ouput(answer.strip())

                    if k in cat_5_idxs:
                        predicted_answer = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_idxs.index(k)])
                        out_data['qa'][idx][prediction_key] = predicted_answer
                    else:
                        try:
                            out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
                        except:
                            out_data['qa'][idx][prediction_key] = ', '.join([str(n) for n in list(answers[str(k)].values())])
                except:
                    try:
                        answers = json.loads(answer.strip())
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answers[k], cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = answers[k].replace('(a)', '').replace('(b)', '').strip()
                    except:
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answer.strip(), cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = json.loads(answer.strip().replace('(a)', '').replace('(b)', '').split('\n')[k])[0]

    return out_data


def get_question_annotations(question):
    ann = run_claude_for_annotations(question,type='question')

    return ann#get_attributes(ann)

#*******added
def get_rag_context_dialog(context_database, query_vector, args):
    output = np.dot(query_vector, context_database['embeddings'].T)
    sorted_outputs = np.argsort(output)[::-1]
    sorted_context = [context_database['context'][idx] for idx in sorted_outputs[:args.top_k]]

    sorted_context_ids = []
    for idx in sorted_outputs[:args.top_k]:
        context_id = context_database['dia_id'][idx]
        if type(context_id) == str:
            if ',' in context_id:
                context_id = [s.strip() for s in context_id.split(',')]
        if type(context_id) == list:
            sorted_context_ids.extend(context_id)
        else:
            sorted_context_ids.append(context_id)

    sorted_date_times = [context_database['date_time'][idx] for idx in sorted_outputs[:args.top_k]]
    if args.rag_mode in ['dialog', 'observation']:
        query_context = '\n'.join(
            [date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])
    else:
        query_context = '\n\n'.join(
            [date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])

    return query_context, sorted_context_ids


def get_rag_context_summary(context_database, query_vector, args):

    output = np.dot(query_vector, context_database['embeddings'].T)
    sorted_outputs = np.argsort(output)[::-1]
    sorted_context = [context_database['context'][idx] for idx in sorted_outputs[:args.top_k]]

    sorted_context_ids = []
    for idx in sorted_outputs[:args.top_k]:
        context_id = context_database['dia_id'][idx]
        if type(context_id) == str:
            if ',' in context_id:
                context_id = [s.strip() for s in context_id.split(',')]
        if type(context_id) == list:
            sorted_context_ids.extend(context_id)
        else:
            sorted_context_ids.append(context_id)

    sorted_date_times = [context_database['date_time'][idx] for idx in sorted_outputs[:args.top_k]]
    if args.rag_mode in ['dialog', 'observation']:
        query_context = '\n'.join(
            [date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])
    else:
        query_context = '\n\n'.join(
            [date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])

    return query_context, sorted_context_ids
#*****************

#*********Updated
def get_rag_context(context_database, query_vector, args):

    matches, latency, dist = embedding_utils.vector_search(query_vector, item_list=context_database['dia_id'],
                                                                     embeddings_file=context_database['embeddings'],
                                                                     k=args.top_k)  # todo change to keys list

    matches=list(set(matches))

    query_context=""

    for idx in matches:
        date=context_database['date_time'].get(idx)

        ret_diag=context_database['context'].get(idx)

        if ret_diag is None or ret_diag=='':
            print('Skipping a match')
            continue
        query_context+= date + ': ' + ret_diag+'\n'

    return query_context, matches,sum(dist)/len(dist)
#******************

#************Added
def get_annotations_ann_ref(diatext,args):
    tmp_annotations = {}
    final_annotations = {}
    tmp_ann = run_model(args.aug_model, augmentation_prompts.DIALOGUE_ANNOTATION_AR.format(diatext))
    final_ann = run_model(args.aug_model, augmentation_prompts.REFLECT_ANNOTATION_AR.format(tmp_ann, diatext))
    return final_ann

def get_annotation_str(dia_text,args):

    if args.aug_model == 'llama':
        print('lama prompt')
        prompt_ann = augmentation_prompts.DIALOGUE_TURN_ANNOTATION_LLAMA
    elif args.aug_model == 'mistral':
        print("mistral prmpt")
        prompt_ann = augmentation_prompts.DIALOGUE_TURN_ANNOTATION_MISTRAL
    else:
        prompt_ann = augmentation_prompts.DIALOGUE_TURN_ANNOTATION_BASIC_ORDERED

    annotations_str = ''
    while annotations_str == '':
        annotations_str = run_model(args.aug_model, prompt_ann.format(dia_text))#.lower().strip().replace('\n', '')

    if args.aug_model in ['llama','mistral']:
        re_patterns = r'\[(.*?)\]<(.*?)>'
        matches = re.findall(re_patterns,annotations_str)
        annotations_str = ''
        annotations_str += ''.join('[%s]<%s>' % (a, v) for a, v in matches) if matches else ''

    if TYPE == 'ann':
        return annotations_str
    else:
        return annotations_str+'[dialogue_turn]<%s>'%dia_text

def get_annotation_attributes(dia_text,args):

    if args.aug_model == 'llama':
        prompt_ann = augmentation_prompts.DIALOGUE_TURN_ANNOTATION_LLAMA

    elif args.aug_model == 'mistral':
        prompt_ann = augmentation_prompts.DIALOGUE_TURN_ANNOTATION_MISTRAL

    else:
        prompt_ann = augmentation_prompts.DIALOGUE_TURN_ANNOTATION_BASIC_V3

    annotations_str = ''

    while annotations_str == '':

        if TYPE == 'ann':
            annotations_str = run_model(args.aug_model, prompt_ann.format(dia_text))#.lower().strip().replace('\n', '')

        else:
            annotations_str = run_model(args.aug_model, prompt_ann.format(dia_text)) + dia_text

    att_val={}

    if args.aug_model in ['llama','mistral']:
        re_patterns = r'\[(.*?)\]<(.*?)>'
        matches = re.findall(re_patterns,annotations_str)
        for a, v in matches:
            if a in att_val.keys():
                att_val[a].append(v)
            else:
                att_val[a]=[v]

    return att_val

def prepare_augmentation_database_models(data, dataset_prefix, args,):

    if not os.path.exists(os.path.join(args.emb_dir, '%s_dialog_ann_%s.json' % (dataset_prefix, data['sample_id']))):
        dialogs_ann = []
        date_times = {}
        context_ids = []
        dialogs = {}
        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        filename_index = os.path.join(args.emb_dir, data['sample_id'] + '_DB.index')

        # go over all sessions
        for i in range(min(session_nums), max(session_nums) + 1):

            date_time = data['conversation']['session_%s_date_time' % i]  # session data and time

            #dialog level (in session)
            for dialog in data['conversation']['session_%s' % i]:

                context_ids.append(dialog['dia_id'])
                date_times[dialog['dia_id']] = date_time

                dia_text=''
                if 'blip_caption' in dialog.keys():

                    dia_text= dialog['speaker'] + ' said, \"' + dialog[
                        'text'] + '\"' + ' and shared ' + dialog['blip_caption']
                    dialogs[dialog['dia_id']] = dia_text
                else:
                    dia_text=dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'
                    dialogs[dialog['dia_id']] = dia_text

                annotations_str =get_annotation_str(dia_text, args)# get_annotation_str(dialog['text'], args)

                if annotations_str == '':
                    #if model failed to generate annotations
                    annotations_str='[person]<%s> '%dialog['speaker']+'[text]<%s>'%dialog['text']
                else:
                    annotations_str = '[person]<%s> '%dialog['speaker']+annotations_str

                emb = ''
                while len(emb) == 0: #to ensure something is returned
                    emb = embedding_utils.embed_str(annotations_str)
                    if emb == 'Error' and args.aug_model == 'mistral':
                        print('Long embedding query: ',annotations_str)
                        annotations_str=annotations_str[:50000] #truncation to fulfil # of token limits

                dialogs_ann.append(emb)

        # assert embeddings.shape[0] == len(dialogs_), "Lengths of embeddings and dialogs do not match"
        diag_embeddings = np.array(dialogs_ann)
        d = diag_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(diag_embeddings)

        print("Saving")
        write_index(index, filename_index)

        database = {'embeddings': filename_index,
                    'date_time': date_times,
                    'dia_id': context_ids,
                    'context': dialogs
                    }

        with open(os.path.join(args.emb_dir, '%s_dialog_ann_%s.json' % (dataset_prefix, data['sample_id'])), 'w') as f:
            json.dump(database, f)

    else:
        database = json.load(
            open(os.path.join(args.emb_dir, '%s_dialog_ann_%s.json' % (dataset_prefix, data['sample_id'])), 'rb'))

    return database

#TODO update this and the above method to be one, for now there are 2 different usecases uasge
def prepare_augmentation_database(data,dataset_prefix,args): #args.annotation_model

    print('in prepare augmentation database')

    if not os.path.exists(os.path.join(args.emb_dir, '%s_dialog_ann_%s.json' % (dataset_prefix, data['sample_id']))):
        dialogs_ann = []
        date_times = {}
        context_ids = []
        dialogs = {}
        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]

        filename_index = os.path.join(args.emb_dir, data['sample_id'] + '_DB.index')

        with open(ANNOTATIONS_DIR + '/' + data['sample_id'] + '_sample.json', 'r') as fs:
                session_annotations = json.load(fs)

        # go over all sessions
        for i in range(min(session_nums), max(session_nums) + 1):
            print('in session_%s' % i)
            date_time = data['conversation']['session_%s_date_time' % i]  # session data and time
            for p, diag_att in session_annotations['session_%s' % i].items():
                for id, att in diag_att.items():
                    context_ids.append(id)  # add dialogue ids
                    date_times[id] = date_time  # adding date and time for every turn
                    str_embed = '[person]<' + p + '>'
                    for k, v in att.items():
                        if k == 'date': continue
                        str_embed += '[' + str(k) + ']<' + str(v) + '>'
                    # print('str to embed',str_embed)
                    emb = ''
                    while emb == '' or len(emb) == 0:
                        emb = embedding_utils.embed_str(str_embed)
                    dialogs_ann.append(emb)

            # packing dialogue
            for dialog in data['conversation']['session_%s' % i]:

                if 'blip_caption' in dialog.keys():
                    dialogs[dialog['dia_id']] = dialog['speaker'] + ' said, \"' + dialog[
                        'text'] + '\"' + ' and shared ' + dialog['blip_caption']
                else:
                    dialogs[dialog['dia_id']] = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

        print("Got embeddings for %s dialogs" % len(dialogs_ann))

        diag_embeddings = np.array(dialogs_ann)

        d = diag_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(diag_embeddings)
        print("Saving")
        write_index(index, filename_index)

        database = {'embeddings': filename_index,
                    'date_time': date_times,
                    'dia_id': context_ids,
                    'context': dialogs
                    }

        with open(os.path.join(args.emb_dir, '%s_dialog_ann_%s.json' % (dataset_prefix, data['sample_id'])), 'w') as f:
            json.dump(database, f)

    else:
        database = json.load(
            open(os.path.join(args.emb_dir, '%s_dialog_ann_%s.json' % (dataset_prefix, data['sample_id'])), 'rb'))
    return database
#***************

#*********Updated
def prepare_for_rag(args, data):

    dataset_prefix = os.path.splitext(os.path.split(args.data_file)[-1])[0]

    if args.rag_mode == "summary":

        #load
        # check if embeddings exist
        assert os.path.exists(os.path.join(args.emb_dir, '%s_session_summary_%s.pkl' % (
        dataset_prefix, data['sample_id']))), "Summaries and embeddings do not exist for %s" % data['sample_id']
        database = pickle.load(
            open(os.path.join(args.emb_dir, '%s_session_summary_%s.pkl' % (dataset_prefix, data['sample_id'])), 'rb'))

    elif args.rag_mode == 'dialog':
        # check if embeddings exist
        if not os.path.exists(os.path.join(args.emb_dir, '%s_dialog_%s.pkl' % (dataset_prefix, data['sample_id']))):

            dialogs = []
            date_times = []
            context_ids = []
            session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                            'session' in k and 'date_time' not in k]
            for i in range(min(session_nums), max(session_nums) + 1):

                date_time = data['conversation']['session_%s_date_time' % i]
                for dialog in data['conversation']['session_%s' % i]:
                    context_ids.append(dialog['dia_id'])
                    date_times.append(date_time)
                    if 'blip_caption' in dialog:
                        dialogs.append(
                            dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + ' and shared ' + dialog[
                                'blip_caption'])
                    else:
                        dialogs.append(dialog['speaker'] + ' said, \"' + dialog['text'] + '\"')

            print("Getting embeddings for %s dialogs" % len(dialogs))
            embeddings = get_embeddings(args.retriever, dialogs, 'context')
            assert embeddings.shape[0] == len(dialogs), "Lengths of embeddings and dialogs do not match"
            database = {'embeddings': embeddings,
                        'date_time': date_times,
                        'dia_id': context_ids,
                        'context': dialogs}

            with open(os.path.join(args.emb_dir, '%s_dialog_%s.pkl' % (dataset_prefix, data['sample_id'])), 'wb') as f:
                pickle.dump(database, f)

        else:
            database = pickle.load(
                open(os.path.join(args.emb_dir, '%s_dialog_%s.pkl' % (dataset_prefix, data['sample_id'])), 'rb'))

    elif args.rag_mode == 'aug':

        database = prepare_augmentation_database(data,dataset_prefix,args)

    elif args.rag_mode == 'aug_models':
        database = prepare_augmentation_database_models(data,dataset_prefix,args)#type='diag_ann'

    elif args.rag_mode == 'observation':

        # check if embeddings exist
        assert os.path.exists(os.path.join(args.emb_dir, '%s_observation_%s.pkl' % (
        dataset_prefix, data['sample_id']))), "Observations and embeddings do not exist for %s" % data['sample_id']
        database = pickle.load(
            open(os.path.join(args.emb_dir, '%s_observation_%s.pkl' % (dataset_prefix, data['sample_id'])), 'rb'))


    else:
        raise ValueError

    if args.rag_mode == 'dialog' or args.rag_mode == 'observation':
        print("Getting embeddings for %s questions" % len(data['qa']))
        question_embeddings = get_embeddings(args.retriever, [q['question'] for q in data['qa']], 'query')
        return database, question_embeddings

    #TODO fix return for other types
    print("Getting embeddings for %s questions" % len(data['qa']))

    question_embeddings=[embedding_utils.embed_str(q['question']) for q in data['qa']]

    return database, question_embeddings



import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from tqdm import tqdm
import argparse
import os, json
#from generative_agents.memory_utils import get_session_facts
#from locomo.global_methods import set_openai_key, set_anthropic_key,run_claude, run_claude_with_examples, run_claude_for_annotations
from ..memory import  global_methods as glbl# import run_claude, run_claude_for_annotations
#from task_eval.rag_utils import get_embeddings
import pickle
import re
import random
import pandas as pd
import glob
import ast
import numpy as np
import movie_embedding_retrieval
import faiss
from faiss import write_index, read_index
#from augmentation_prompts import DIALOGUE_ANNOTATION, REFLECT_ANNOTATION
from ..tasks.evaluation import rougel_score
DIALOGUE_ANNOTATION=(
   "You are a dialogue annotator who generates the most relevant attributes in a conversation. Given the conversation below"
   "identify relevant key attributes and their values that describe most important information in the conversation like events, emotions, intent, etc."
   "Attributes should be specific with most relevant and specific values only."
   "For every attribute mention the speaker name."
   "mention attributes with order of relevance from left to right"
   "Important: the response format should be in json format like {speaker name:{[attribute]<value>}. Make sure the attribuite name is between [ ]  and the value between < >."
   "Don't include anything else. Don't include special characters or new lines"
   "Dialogue:"
)


REFLECT_ANNOTATION=(
   "You are aa dialogue generator and evaluator for conversation annotations."
   "Given a list of attributes and their values generated for the dialogue below."
   "Evaluate the efficacy of these annotations to regenerate back the dialogue. Update the annotations if needed to represent the most important information like events, emotions and intent, etc.."
   "Update these attributes and their values to be more descriptive of the dialogue, specific and contain relevant information and their values."
   "Important: the response format should be in json format like {speaker name:{[attribute]<value>}. Make sure the attribuite name is between [ ]  and the value between < >."
   "Don't include anything else. Don't include special characters or new lines."
)

EVENTS_SUMMARY_LABEL=("Generate an event summary paragraph for the events listed below for the given speaker name and dates. Input:")
EVENTS_SUMMARY_AUG=("Given a list of attributes and values listed below in the format [attribute name]<value>. Your job is to generate an event summary paragraph for the included events. Include speaker name and dates"
                    "Include only text and no special characters or brackets. Input:")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--prompt-dir', type=str, default="")
    parser.add_argument('--use-date', action="store_true")
    parser.add_argument('--overwrite', action="store_true", help="set flag to overwrite existing outputs")
    parser.add_argument('--retriever', type=str, default="dragon")

    args = parser.parse_args()
    return args

user_annotations={}
session_annotations={}
attributes_users={}
session_level={}
from retry import retry


sample_file=open('sample_annotations2.txt','w')

def get_attributes(ann):
    attributes={}
    pattern = pattern = r'\[(.*?)\]<(.*?)>'
    matches = re.findall(pattern, ann)
    for match in matches:
        att, value = match
        if value == 'none' or value == 'unspecified' or value=='':
            continue
        if att in attributes.keys():
            attributes[att].append(value)
        else:
            attributes[att] = [value]
    return attributes


def get_annotations_v2(agent_a, agent_b,session_idx):
    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for i, dialog in enumerate(agent_a['session_%s' % session_idx]):
        d_id = dialog["dia_id"]
        sample_file.write(d_id + ':\n')
        d_speaker = dialog['speaker']
        try:
            d_text = dialog['clean_text']
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
        except KeyError:
            d_text = dialog['text']
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

        if 'blip_caption' in dialog:
            d_blip = dialog['blip_caption']
            conversation += ' and shared ' + dialog['blip_caption']

        conversation += '\n'  # keep annotation to files
        # generate annotations
        # print("to annotate",d_text)
    ann = run_claude_for_annotations(conversation).strip()
    #parse and add atrributes
    '''
    pattern_all=r'\{(.*?)\}'
    matches_all=
    pattern_att=r'\[(.*?)\]<(.*?)>'
    '''
    print(ann)


def get_question_annotations(question):
    ann = run_claude_for_annotations(question,type='question')

    return ann#get_attributes(ann)

def get_annotations(agent_a, agent_b,session_idx):
    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for i, dialog in enumerate(agent_a['session_%s' % session_idx]):
        d_id=dialog["dia_id"]
        sample_file.write(d_id+':\n')
        d_speaker= dialog['speaker'].lower()
        try:
            d_text=dialog['clean_text'].lower()
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
        except KeyError:
            d_text = dialog['text'].lower()
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

        if 'blip_caption' in dialog:
            d_blip=dialog['blip_caption']
            conversation += ' and shared ' + dialog['blip_caption']
            #ann = run_claude_for_annotations(dialog['blip_caption'])

        conversation += '\n' #keep annotation to files
        #generate annotations
        #print("to annotate",d_text)
        sample_file.write('Dialogue:' + d_text + '\n')
        ann=run_claude_for_annotations(d_text).lower().strip()
        attributes={}
        pattern = pattern = r'\[(.*?)\]<(.*?)>'
        matches = re.findall(pattern, ann)
        #print('matches', matches)

        for match in matches:
            att, value = match
            att=att.lower()
            value=value.lower()
            # attribure-value dictionary
            if value == 'none' or value == 'unspecified' or value == '':
                continue
            if att in attributes.keys():
                attributes[att].append(value)
            else:
                attributes[att] = [value]
        #############################################
        #attribute to all attributes
            if att in attributes_users:
                if d_speaker in attributes[att]:
                    attributes_users[att][d_speaker].append(d_id+'@@'+value)
                else:
                    attributes_users[att][d_speaker] = [d_id+'@@'+value]
            else:
                attributes_users[att]={d_speaker:d_id+'@@'+value}

        attributes['date'] = agent_a['session_%s_date_time' % session_idx]
        #####################################################
        #adding to user-attributes view
        if d_speaker in user_annotations.keys():
            user_annotations[d_speaker][d_id]=attributes
        else:
            print('adding new user',d_speaker)
            user_annotations[d_speaker]={d_id:attributes}


    sample_file.write('User<->Annotations: ' + str(user_annotations) + '\n')
    print(user_annotations)
    sample_file.write('Attributes<->Users: ' + str(attributes_users) + '\n')
    print(attributes_users)
    #session level attributes
    ann_session = run_claude_for_annotations(conversation,type='session').lower().strip()
    session_attributes = get_attributes(ann)
    session_level[session_idx]=session_attributes
    print(session_level)
    return user_annotations


def find_memory(data, annotations):
    # use keywords = [k.lower() for k in query.split()]
    evidence={}
    questions = data['qa']
    speakers_names = list(set([d['speaker'] for d in data['conversation']['session_1']]))
    print('speakers',speakers_names)

    for q in questions:
        print('Question', q)
        q_ann = get_question_annotations(q['question'])
        for session in annotations:
            name=q_ann['person'][0].replace('\'','').lower()
            print('name',name)
            if name not in annotations:
                print('skipping the name',name)
                continue
            name_att = annotations[name]
            for d,d_atts in name_att.items():
                for att,val in d_atts.items():
                    print('att,val, q keys',att, val, q_ann.keys())
                    print('values',q_ann.values())
                    if att in q_ann.keys():
                        print('att in q_ann keys',att,q_ann.keys())
                        evidence.update({q['question']:d})
                    elif att in q_ann.values():
                        print('att in q_ann values',att,q_ann.values)
                        evidence.update({q['question']:d})
                    elif any(item in q_ann.keys() for item in val):
                        print('val in q_ann keys',val,q_ann.keys())
                        evidence.update({q['question']:d})
                    elif any(item in q_ann.values() for item in val):
                        print('val in q_ann values',val,q_ann.keys())
                        evidence.update({q['question']:d})

                    #if att_value in ann.values():
                    #    print('ev found', att, att_value, ann[att])
                    #    evidence[q]=d
        print('evidence',evidence)
    print('final evidence',evidence)
    return evidence

import csv
def generate_events_file(ann_file,out_file):
    events={}

    with open(ann_file,'r') as f:
        all_annotations=json.load(f)

    for conv_id,con_val in all_annotations.items():
        for session,session_att in con_val.items():
            for speaker,att in session_att.items():
                for diag_id, diag_att in att.items():
                    for a,v in diag_att.items():
                        if 'event' in a:
                            if conv_id not in events.keys():
                                events[conv_id]={session: diag_att['date']+': '+speaker+':'+str(v)}
                            elif session in events[conv_id].keys() :
                                events[conv_id][session] += '\n'+diag_id + ': ' + speaker + ':' + str(v)
                            else:
                                events[conv_id].update({session: diag_id + ': ' + speaker + ':' + str(v)})



    with open(out_file, "w") as f:
        json.dump(events,f,indent=4)
    return events

def evaluate_events(event_ann,data):
    with open(event_ann,'r') as f:
        events_ann = json.load(f)
    with open(data,'r') as f2:
        sample=json.load(f2)
    labels=[]
    aug=[]
    for data in sample:
        print('**********Evaluating**************')
        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]
        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Evaluating events for %s' % data['sample_id']):

            label_session=data['event_summary']['events_session_%s'%i]
            aug_events = events_ann[data['sample_id']]['session_%s'%i]
            labels.append(glbl.run_claude(EVENTS_SUMMARY_LABEL+str(label_session)))
            aug.append(glbl.run_claude(EVENTS_SUMMARY_AUG+str(aug_events)))

    print(labels)
    print(aug)
    print(rougel_score(aug,labels))

def main():

    # get arguments
    args = parse_args()

    # load conversations
    samples = json.load(open(args.data_file))

    out_samples = {}
    all_samples_annotations={}

    for data in samples:

        all_annotations = {}
        date_times = []
        context_ids = []

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if
                        'session' in k and 'date_time' not in k]
        print("annotating sessions")
        for i in tqdm(range(min(session_nums), max(session_nums) + 1),
                      desc='Generating Annotations for %s' % data['sample_id']):

            annotations = get_annotations(data['conversation'], data['conversation'], i)# return_embeddings=False)
            date_time = data['conversation'][('session_%s_date_time') % i]
            all_annotations['session_%s' % i]=annotations

        with open(data['sample_id']+'_sample.json','w') as fs:
            json.dump(all_annotations,fs,indent=4)

        print("Getting questions and memories")
        questions = data['qa']

        print('Sessions annotations',annotations)
        print(find_memory(data,annotations))

        all_samples_annotations[data['sample_id']]=all_annotations
        all_annotations={}

        #check_questions()
    with open("locomo_annotations.txt", 'w') as f:
        json.dump(all_samples_annotations, f,indent=4)


CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def embed_sessions():
    args = parse_args()

    # load conversations
    samples = json.load(open(args.data_file))

    out_samples = {}
    out_samples = {}
    all_samples_annotations = {}

    for data in samples:
        if '50' not in data['sample_id']:
            continue
        print('conv_50')
        with open(data['sample_id'] + '_sample.json', 'r') as fs:
            session_annotations = json.load(fs)
            for session_id,person_att in session_annotations.items():
                filename_index = data['sample_id']+'_'+session_id + '_index.index'
                filename_keys = data['sample_id']+'_'+session_id + '_keys.txt'
                if os.path.exists(filename_index):
                    continue
                dialog_ids=[]
                embeded = []

                if os.path.exists(filename_index): continue
                for p,diag_att in person_att.items():
                     for id,att in diag_att.items():
                         str_embed = '[person]<' + p + '>'
                         dialog_ids.append(id)
                         for k,v in att.items():
                             if k == 'date': continue
                             str_embed+='['+str(k)+']<'+str(v)+'>'
                         embeded.append(movie_embedding_retrieval.embed_str(str_embed))

                with open(filename_keys,'w') as f:
                    f.write(str(dialog_ids)+'\n')
                diag_embeddings = np.array(embeded)
                d = diag_embeddings.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(diag_embeddings)
                print("Saving")
                write_index(index, filename_index)


def evaluate_sessions_qa():

    # get arguments
    args = parse_args()

    # load conversations
    samples = json.load(open(args.data_file))

    out_samples = {}
    all_samples_annotations={}

    results_file = 'results_ann_qa_session_if_values_in_diag_v.csv'
    df = pd.DataFrame(columns=['qa', 'ann_evidence', 'label_ev'])

    for data in samples:

        with open(data['sample_id']+'_sample.json','r') as fs:
            session_annotations = json.load(fs)

        print("Evaluating")

        #pred key (args.model, args.rag_mode, args.top_k)
        #answers = get_claude_answers(data,'')

        questions = data['qa']
        for q in questions:
            #get annotations of the question
            #find most rlevant attributes from all session in this sample
            q_ann = get_question_annotations(q['question'])
            print(q['question'])

            ev_list = get_evidence(q_ann, session_annotations)
            new_row={'qa':[q['question']], 'ann_evidence':[ev_list], 'label_ev':[q['evidence']]}
            df = pd.concat([df,pd.DataFrame(new_row)], ignore_index=True)

        df.to_csv(results_file, mode='a', header=False, index=False)

def evaluate_sessions_qa_embedding(top_k):

    # get arguments
    args = parse_args()

    # load conversations
    samples = json.load(open(args.data_file))

    out_samples = {}
    all_samples_annotations={}

    results_file = f'results_ann_qa_session_embedding_{top_k}_2.csv'
    df = pd.DataFrame({'qa':[], 'ann_evidence':[], 'label_ev':[],'dist':[]})

    for data in samples:
        for k in data['conversation'].keys():
            if 'session' in k and 'date_time' not in k:
                index_file=data['sample_id']+'_'+k+'_index.index'
                print('index file',index_file)
                with open(data['sample_id']+'_'+k+'_keys.txt','r') as f:
                    keys_file =ast.literal_eval(f.readline())


                print("Evaluating")
                questions = data['qa']
                for q in questions:
                    q_ann = get_question_annotations(q['question'])
                    embed_q=movie_embedding_retrieval.embed_str(q_ann)
                    if embed_q == '':
                        print('skipping question',q)
                        continue
                    matches,latency,dist=movie_embedding_retrieval.vector_search(embed_q,movie_list=keys_file,embeddings_file=index_file,k=top_k)#todo change to keys list
                    new_row={'qa':[str(q['question'])], 'ann_evidence':[str(matches)], 'label_ev':[str(q['evidence'])],'dist':[dist]}
                    df = pd.concat([df,pd.DataFrame(new_row)], ignore_index=True)

            df.to_csv(results_file, mode='a', header=False, index=False)

def get_evidence(q_ann,session_ann):

    if 'person' not in q_ann.keys() or q_ann['person'] is None: #no person then return empty list
        return []
    evidence=[]
    person_name_list=[] #to solve the and issue
    person_name=q_ann['person'][0]
    if 'and' in person_name:
        person_name_list = person_name.split('and')  # todo update to chekc for 2
        person_name = person_name_list[0]
    print('person',person_name)
    for session_id,person_att in session_ann.items():
        att_to_check={}
        if person_name not in person_att.keys():
            if len(person_name_list) > 1 and person_name_list[1] in person_att.keys():
                person_name = person_name_list[1]
            else:
                continue #check when embedd possibly check in session attribute
        for diag_id, diag_att in person_att[person_name].items():
            for k,v in q_ann.items():
                for d_k,d_v in diag_att.values():
                    if v in d_v:
                        evidence.append(diag_id)

                #if k in diag_att.keys() and any(item in diag_att.values() for item in v):#v in diag_att.values():
                #    evidence.append(diag_id)
                #if v in diag_att.values():
                #    evidence.append(diag_id)

    return set(evidence)

def get_list(x):
    return ast.literal_eval(x)


def evaluate_files(fname=''):
    if fname == '':
        files= glob.glob('evaluation/*')
    files=[fname]
    df_ev = pd.DataFrame(columns=['Description','Exact Match','Includes','Avg Length of Annoted','Avg Length of Label'])
    incl = 0
    includes = 0
    for file in files:
        print('reading file',file)
        df = pd.read_csv(file,sep=',')#,index_col=['question','evd_embed','evd_label'])
        evd=df['evd_embed']#df.iloc[:, 1]
        all_items=len(evd)
        evd_label=df['evd_label']#.iloc[:, 2]
        eq=0
        for emb,label in zip(evd,evd_label):
            print(emb[:len(label)],'=========',label)
            if emb[:len(label)] == label:
                eq+=1
            '''
            for l in evd_label:
                if l in emb:
                    incl += 1
            if incl >= 1:
                 includes+=1 #todo update to include # of overlap
            incl=0
            '''
        includes = evd_label.isin(evd).sum()

        exact_match=eq #evd_label.eq(evd).sum()
        sum_ev =0
        sum_ev_lbl =0
        for e in evd: sum_ev+=len(e)
        for e in evd_label: sum_ev_lbl += len(e)

        new_row = {'Description':file,
                   'Exact Match': exact_match / all_items,
                   'Includes': includes / all_items,
                   'Avg Length of Annoted': sum_ev/all_items,
                   'Avg Length of Label': sum_ev_lbl/all_items
                   }
        print(new_row)
        df_ev = pd.concat([df_ev, pd.DataFrame([new_row])], ignore_index=True)
        print(includes/all_items,exact_match/all_items)
    df_ev.to_csv("evaluation_result.csv", mode='a', header=False, index=False)


def evaluate_file(fname=''):
    df_ev = pd.DataFrame(
        columns=['Description', 'Exact Match', 'Includes', 'Avg Length of Annoted', 'Avg Length of Label'])
    f = open(fname,'r').readlines()
    eq=0
    incl=0
    includes=0
    sum_e=0
    sum_l=0
    for line in f[1:]:
        sp= line.split('\t')
        em=sp[1]
        label=sp[2]
        em = get_list(em)
        label = get_list(label)
        sum_e+=len(em)
        sum_l+=len(label)
        if em[:len(label)] == label: eq += 1
        for l in label:
            if l in em:
                    incl += 1
        if incl >= 1:
                 includes+=1 #todo update to include # of overlap
        incl=0
    all_items=len(f)-1
    print('total items',all_items)
    new_row = {'Description': fname,
               'Exact Match': eq / all_items,
               'Includes': includes / all_items,
               'Avg Length of Annoted': sum_e / all_items,
               'Avg Length of Label': sum_l/ all_items
               }
    print(new_row)
    df_ev = pd.concat([df_ev, pd.DataFrame([new_row])], ignore_index=True)
    print(includes / all_items, eq / all_items)
    df_ev.to_csv("evaluation_result_embedding.csv", mode='a', header=False, index=False)


def get_questions():

    args = parse_args()
    samples = json.load(open(args.data_file))
    questions=[]
    for data in samples:
        questions.append(data['qa'])
        for ques in data['qa']:
            print(ques['question'])
            print(get_question_annotations(ques['question']))
    return questions


def annotate_and_reflect(datafile):
    with open(datafile,'r') as f:
        samples=json.load(f)

    for data in samples:

        print('in sample',data['sample_id'])
        tmp_annotations = {}
        final_annotations = {}

        if data['sample_id'] not in tmp_annotations.keys(): #add new sample id
            tmp_annotations[data['sample_id']]={}
            final_annotations[data['sample_id']]={}

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
                    conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog[
                        'clean_text'] + '\"'
                except KeyError:
                    d_text = dialog['text'].lower()
                    conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

                if 'blip_caption' in dialog:
                    d_blip = dialog['blip_caption']
                    conversation += ' and shared ' + dialog['blip_caption']
                    # ann = run_claude_for_annotations(dialog['blip_caption'])

                conversation += '\n'  # keep annotation to files

            #session annotations
            conversation=conversation.lower().strip()
            tmp_ann = glbl.run_claude(str(DIALOGUE_ANNOTATION) + conversation)
            tmp_annotations['session_%s' % i]=tmp_ann
            final_ann = glbl.run_claude(str(REFLECT_ANNOTATION) + f" Attribute and values{tmp_ann}" + f"Dialogue:{conversation}")
            final_annotations['session_%s' % i] = final_ann
        #write to cs
        tmp_file = 'tmp_locomo_annotations_all.json'
        final_file = 'final_reflect_locomo_annotations_all.json'
        with open(tmp_file,'a') as f:
            json.dump(tmp_annotations,f,indent=4)
        with open(final_file,'a') as f:
            json.dump(final_annotations,f,indent=4)
    return tmp_file,final_file
def get_event_summaries(ann_file):
    prompt=("Given the following attributes and values that annotate a dialogue for every speaker in the format  {speaker name:{[attribute]<value>},"
            "generate a summary for the event attributes only to describe the main and important events represented in these annotations. Refrain from mentionaing any minimal event. Include any event related details and speaker. "
            "Format: a bullet paraghraph for major life events for every speaker with no special characters."
            "Don't include anything else in your response or extra text or lines."
            "Don't inlcude bullets."
            "Input annotations:"
            )

    with open(ann_file, 'r') as f:
        ann=json.load(f)

    events={}
    for sample_id,sessions in ann.items():
        events[sample_id]={}
        for session,atts in sessions.items():
            summary=glbl.run_claude(prompt+str(atts))
            events[sample_id].update({session:summary})

    return events

def evaluate_summaries(events,datafile):
    prompt = (
        "Given the following list of events for every speaker,"
        "generate a summary of the represented events."
        "Format: a bullet paraghraph for major life events for every speaker with no special characters."
        "Don't include anything else in your response or extra text or lines."
        "Don't inlcude bullets."
        "Input:"
        )

    with open(datafile,'r') as f:
        samples=json.load(f)
    label_summary=[]
    aug_summary=[]
    for data in samples:
        for session, label_events in data['event_summary'].items():
            label_summary.append(glbl.run_claude(prompt+str(label_events)))
            print('session_'+session.split('_')[-1])
            print(data['sample_id'])
            for id,txt in events[data['sample_id']].items():
                print(id,txt)
            aug_summary.append(events[data['sample_id']]['session_'+session.split('_')[-1]])

    scores=[]
    for aug, label in zip(aug_summary, label_summary):
        score = rougel_score(aug,label)
        scores.append(score)

    print(scores)
    scores_1 =scores["rouge-1"]["f"]
    print('Rouge-1:',sum(scores_1) / len(scores_1))
    scores_2 = scores["rouge-2"]["f"]
    print('Rouge-2:',sum(scores_2) / len(scores_2))
    scores_l = scores["rouge-l"]["f"]
    print('Rouge-l:',sum(scores_l) / len(scores_l))
    return scores

if __name__ == "__main__":


    #main()
    #evaluate_sessions_qa()
    #evaluate_file('results_ann_qa_session_embedding.txt')

    #embed_sessions()
    #tmp_file,final_file = annotate_and_reflect('/local/home/ranasal/Mem_Auig_v2/locomo/data/locomo10.json')
    tmp_file='tmp_locomo_annotations_all.json'
    final_file='final_reflect_locomo_annotations_all.json'
    events_tmp=get_event_summaries(tmp_file)
    events_final=get_event_summaries(final_file)
    print("***********Evaluating Tmp File*************")
    evaluate_summaries(events_tmp,'/local/home/ranasal/Mem_Auig_v2/locomo/data/locomo10.json')
    print("***********Evaluating Final File*************")
    evaluate_summaries(events_final, '/local/home/ranasal/Mem_Auig_v2/locomo/data/locomo10.json')

    #evaluate_sessions_qa_embedding(10)

    #evaluate_events('/local/home/ranasal/Mem_Auig_v2/locomo/task_eval/annotations_events.json','/local/home/ranasal/Mem_Auig_v2/locomo/data/locomo10.json')

    #annotate_and_reflect('/local/home/ranasal/Mem_Auig_v2/locomo/data/locomo10_mini.json')

    #if __name__=='main':
    #check_questions()
    #main()
    #get_questions()
    #find_memory(get_questions(), annotations)

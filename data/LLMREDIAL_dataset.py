import getpass
import os
import json
from Tools import read_dialogue, read_user_data, read_jsonl, read_json, get_conversation_by_id
from tqdm import tqdm
import re
#load dataset into turns

DATASET="/local/home/ranasal/Mem_Auig_v2/datasets/LLM-REDIAL/Movie"
all_conversations= read_dialogue(DATASET+'/Conversation.txt') #TODO update
USER_CONVERSATIONS=DATASET+'/user_conversations.json'
CONV_LABELS=DATASET+'/conversation_label.json'
CONV=DATASET+'/Conversation.txt'

class Dialogue:
    def __init__(self,dialogue_id):
        self.text=[]
        self.text_str=''
        self.dialogue_id=dialogue_id
        self.user_turns=[]
        self.agent_turns=[]
        self.turns=[]

class Conversation:

    def __init__(self,conversation_id=None,rec_item_id=None,user_likes_id=None,user_dislikes_id=None,dialogue=None): #TODO update constructor
        self.conversation_id = conversation_id
        self.rec_item_id =rec_item_id
        self.user_likes_id = user_likes_id
        self.user_dislikes_id = user_dislikes_id
        self.dialogue=dialogue
        #self.annotations=[]

class User:
    def __init__(self,user_id,user_information,history_interaction,user_might_likes,conversations):
        self.persona= ""
        self.user_id = user_id
        self.user_information = user_information
        self.history_interaction = history_interaction
        self.user_might_likes = user_might_likes
        self.conversations=conversations

def load_conversations(filename):
    conversations = []
    current_conversation = Conversation()
    with open(filename,'r') as f:
        for line in f:
            line=line.strip()
            if line.isdigit():
                if current_conversation and int(line)>0:
                    #print('appending conversation',current_conversation.conversation_id,current_conversation.dialogue)
                    conversations.append(current_conversation)
                    current_conversation=Conversation()
                current_conversation.conversation_id=line
                current_conversation.dialogue=[]
            elif line.startswith('User:') or line.startswith('Agent:'):
                current_conversation.dialogue.append(line)
        if current_conversation:
            conversations.append(current_conversation)
    return conversations

'''
 dialogs = []
    with open(filename,'r') as f:
        for line in f:
            line=line.strip()
            if line.isdigit() and line in ids:
                if current_conversation:
                    print('appending conversation',current_conversation.conversation_id,current_conversation.dialog)
                    conversations.append(current_conversation)
                current_conversation.conversation_id=line
                current_conversation.dialog=[]
            elif line.startswith('User:') or line.startswith('Agent:'):
                current_conversation.dialog.append(line)
        if current_conversation:
            conversations.append(current_conversation)
    return conversations
'''

'''
def load_dialogue_ids(filename,ids):
    content = 
    dialogues = []
    current_dialogue=None
    dialogue_text=[]
    with open(filename,'r') as f:
        for line in f:
            line=line.strip()
            if line.isdigit():
                if current_id is not None and conversation:
                    if current_id == conversation_id:

                if current_dialogue:
                    dialogues.append(current_dialogue)
                current_dialogue=Dialogue(line)
                current_dialogue.text=[]
            elif line.startswith('User:') or line.startswith('Agent:'):
                current_dialogue.text.append(line)
        if current_dialogue:
            dialogues.append(current_dialogue)
    print(dialogues)
    return dialogues
'''
def load_dialogue_ids(ids):
    dialogues=[]
    for id in ids:
        d=Dialogue(id)
        d.text_str = get_conversation_by_id(all_conversations,id)
        dialogues.append(d)
    return dialogues

def load_items(filename=DATASET+"/item_map.json"):
    items={}
    with open(filename,'r') as file:
        data=json.load(file)
    for item_id,item_name in data.items():
        items[item_id]=item_name
    return items

def load_users(filename=DATASET+"/user_ids.json"):
    users=[]
    with open(filename,'r') as file:
        data=json.load(file)
    for user_id,user_num in data.items():
        print(user_id,user_num)
        users.append(User(user_id,user_num))
    return users

all_conversations= read_dialogue(DATASET+'/Conversation.txt') #TODO update
def load_users_data(filename=DATASET+"/final_data.jsonl"):
    users=[]
    print("Loading user data")
    final_data = read_jsonl(filename)
    Total_len = len(final_data)
    for i in tqdm(range(Total_len), desc='Processing'):
        Per_data = json.loads(final_data[i])
        user_id, user_information = next(iter(Per_data.items()))
        #read user's history_interaction
        history_interaction = user_information['history_interaction']
        # read user_might_likes
        user_might_likes = user_information['user_might_like']
        #read Conversation_info
        Conversation_info = user_information['Conversation']
        # read Conversation Detail Information
        # read Conversation Detail Information
        conversations=[]
        for j in range(len(Conversation_info)):
            per_conversation_info = Conversation_info[j]['conversation_{}'.format(j+1)]
            user_likes_id = per_conversation_info['user_likes']
            user_dislikes_id = per_conversation_info['user_dislikes']
            rec_item_id = per_conversation_info['rec_item']
            # # turn item id into item name, for example:
            #for k in range(len(rec_item_id)):
            #    rec_name = item_map[rec_item_id[k]]
            # Conversation_id could locate the dialogue
            conversation_id = per_conversation_info['conversation_id']
            # Dialogue
            dialogue = get_conversation_by_id(all_conversations, conversation_id)
            conversations.append(Conversation(conversation_id,rec_item_id,user_likes_id,user_dislikes_id,dialogue))
        user=User(user_id,user_information,history_interaction,user_might_likes,conversations)
        users.append(user)
    print('Done')
    return users

def getUser(dialogue_id):#TODO find a better way to locate users from dialogues
    with open(USER_CONVERSATIONS, 'r') as file:
        df = json.load(file)
    for user,conversations in df.items():
        for j in range(len(conversations)):
            per_conversation_info = conversations[j]['conversation_{}'.format(j+1)]
            if (per_conversation_info['conversation_id'] == dialogue_id):
                return user
            if (per_conversation_info['conversation_id'] > dialogue_id):
                print("invalid id")
                return
def conversation_labels():#TODO find a better way to locate users from dialogues
    with open(USER_CONVERSATIONS, 'r') as file:
        df = json.load(file)
    conv_label={}
    for user,conversations in df.items():
        for j in range(len(conversations)):
            per_conversation_info = conversations[j]['conversation_{}'.format(j+1)]
            conv_label[per_conversation_info['conversation_id']]=per_conversation_info['rec_item']
    with open(CONV_LABELS, 'w') as f:
        json.dump(conv_label, f, indent=4)
    return

def mask_text(label,dialogue,mask='[MASK]'):#dialogue list of turns
    masked=[]
    label=label.lower()
    label=label.replace('vhs','')
    label = re.sub(r'[^a-zA-Z0-9\s:]','',label)
    label=label.strip()
    for text in dialogue:
        text=text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s:]','',text)
        text = re.sub(re.escape(label),'[MASK]',text)
        text = text.strip().replace(label,'[MASK]')#in case one was missed
        try:
            matches=[match.group(0) for match in re.finditer(label,text)]
            if matches:
                greatest_overlap=max(matches,key=lambda x:(text.index(x),len(x)))
                print("\ngreatest overlap",greatest_overlap)
                text=text.replace(greatest_overlap,"[MASK]",1)
        except:
            text = re.sub(re.escape(label),'[MASK]',text)
        masked.append(text)
    print('label',label,'dialogue',dialogue,'\n>>>>>>>text',masked)
    return masked

def cutt_conversations():
    conversations=load_conversations(CONV)
    conv_cutts={}
    with open(DATASET+'/item_map.json', 'r') as file:
        item_map = json.load(file)

    with open(DATASET+'/conversation_label.json', 'r') as file:
        conv_label_map = json.load(file)
    for conv in conversations:
        print('convvvvvv',conv.conversation_id,'\n',conv.dialogue)
        try:
            label= conv_label_map[conv.conversation_id]
        except:
            continue
        masked = mask_text(item_map[label[0]],conv.dialogue,'[MASK]')
        cut=[]
        for i in range(len(masked) - 1, -1, -1):
            if '[MASK]' in masked[i]:
                cut=masked[:i-1]
                conv_cutts[conv.conversation_id]={'Cut':cut,'label':masked[i:]}
        if cut == []:
            conv_cutts[conv.conversation_id]={'Cut':masked[0],'label':masked[1:]}

    with open(DATASET+'/conversations_cut.json', 'w') as f:
        print('dumping',len(conv_cutts))
        json.dump(conv_cutts, f, indent=4)

if __name__ == '__main__':
    #load_conversations("/home/ranasal/Mem_Auig/datasets/LLM-REDIAL/Books/Conversation.txt")
    #print(load_items("/home/ranasal/Mem_Auig_v2/datasets/LLM-REDIAL/Books/item_map.json"))
    conversation_labels()
    cutt_conversations()
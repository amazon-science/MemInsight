from tqdm import tqdm


#Returns a list of session conversations from a given sample
def get_conversation(data):
    conversations=[]
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
                conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + d_text+ '\"'

            if 'blip_caption' in dialog:
                d_blip = dialog['blip_caption']
                conversation += ' and shared ' + dialog['blip_caption']

            conversation += '\n'

        conversation = conversation.lower().strip()
        conversations.append(conversation)

    return conversations

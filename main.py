
import argparse
import json
from memory import MemoryAugmentation
from utils import claude_utils
from tasks import evaluate_qa,evaluate_recomm,evaluate_event_summ

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--mode', required=True, type=str)
    parser.add_argument('--annotations', required=False, type=str)
    parser.add_argument('--task', required=False, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get arguments
    args = parse_args()

    if args.dataset == 'llm_redial'and args.mode == 'annotate':
        with open(args.dataset_path, 'r') as file:
            data = json.load(file)
        mem = MemoryAugmentation()
        mem.annotate_item(data, type='Movie')

    elif args.datasset == 'locomo' and args.mode == 'annotate':
        claude_utils.prepare_augmentation_database()
    else:
        print('Invalid dataset')

    if args.mode == 'embed' and args.datasset == 'locomo':
        claude_utils.prepare_augmentation_database()

    if args.mode == 'embed' and args.datasset == 'llm_redial' and args.annotations!='':
       #dataset path, annotation_path
        mem.embedd_user_memories(args.dataset_path,args.annotations)

    if args.mode == 'evaluate' and args.task!='':
        if args.task == 'qa':
            evaluate_qa.main()
        elif args.task == 'recomm':
            evaluate_recomm.main()
        elif args.taks == 'es':
            evaluate_event_summ.main()
        else:
            print('task unkown')
#'data/LLM-REDIAL/Movie/item_map.json'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
import argparse
from ..memory.global_methods import set_openai_key, set_anthropic_key, set_gemini_key
from evaluation import eval_question_answering
from evaluation_stats import analyze_aggr_acc
from ..utils.claude_utils import get_claude_answers
from ..utils.hf_llm_utils import init_hf_model, get_hf_answers


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--aug_model', type=str)
    args = parser.parse_args()
    return args


def main():

    set_anthropic_key()
    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    if 'gpt' in args.model:
        # set openai API key
        set_openai_key()

    #****************** Updated **********************
    #Claude is the model for answer generation for all annotation models
    elif 'claude' in args.model or 'aug' in args.model  or 'aug_att' in args.model:
        # set openai API key
        set_anthropic_key()
    #*************************************************

    elif 'gemini' in args.model:
        # set openai API key
        set_gemini_key()

    elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
        hf_pipeline, hf_model_name = init_hf_model(args)

    else:
        raise NotImplementedError


    # load conversations
    samples = json.load(open(args.data_file))
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)

    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:

        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()
        if 'claude' in args.model or 'aug' in args.model or 'aug_att' in args.model or 'aug_models' in args.model :
            answers = get_claude_answers(data, out_data, prediction_key, args)
        else:
            raise NotImplementedError

        # evaluate individual QA samples and save the score

        exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
        for i in range(0, len(answers['qa'])):
            answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers['qa'][i][model_key + '_recall'] = round(recall[i], 3)

        out_samples[data['sample_id']] = answers

    try:
        with open(args.out_file, 'w') as f:
            json.dump(out_samples.values().tolist(), f, indent=2)
    except:
        with open(args.out_file, 'w') as f:
            json.dump(list(out_samples.values()), f, indent=2)
    
    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace('.json', '_stats.json'),
                model_key, model_key + '_f1', rag=args.use_rag)


main()


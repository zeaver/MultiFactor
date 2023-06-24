import json
import os
import nltk
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')

from bert_score import BERTScorer
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

def read_json(file_path):
    with open(file_path, "r") as f:
        the_data = json.load(f)
    return the_data[1:]

def read_paths(path_txt):
    with open(path_txt, "r") as f:
        data_path_list = f.readlines()
    return [i.strip() for i in data_path_list]

def evaluate_score(preds_list, gold_list):
    from rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor

    rouge = Rouge()
    meteor = Meteor()
    preds = []
    golds = []
    gts = {}
    res = {}
    my_weights = [
            (1.0, 0.0),
            (0.5, 0.5),
            (0.333, 0.333, 0.334),
            (0.25, 0.25, 0.25, 0.25),
            ]
    rouge_score = 0
    for i, (p, g) in enumerate(zip(preds_list, gold_list)):
        rouge_score += rouge.get_scores([p], [g])[0]['rouge-l']['f']
        pred = nltk.word_tokenize(p)
        gold = nltk.word_tokenize(g)
        preds.append(pred)
        golds.append([gold])
        gts[i] = [p]
        res[i] = [g]

    meteor_score, _ = meteor.compute_score(gts, res)
    corpus_rougel = rouge_score/len(preds_list)
    bleu_score = nltk.translate.bleu_score.corpus_bleu(golds, preds, weights=my_weights)
    p, r, f1 = bert_scorer.score(preds_list, gold_list)
    corpus_bert_score = np.mean(f1.tolist())
    return {"BLEU": np.array(bleu_score), "Rouge-L":np.array([corpus_rougel]), "METEOR": np.array([meteor_score]), \
        "BERTScore": np.array(corpus_bert_score)}

def coco_score(preds_list, gold_list):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    scorers = {
        "Bleu": Bleu(4),
        "Meteor": Meteor(),
        "Rouge": Rouge()
    }
    gts = {}
    res = {}
    for i, (p, g) in enumerate(zip(preds_list, gold_list)):
        gts[i] = [p]
        res[i] = [g]
    scores = {}
    for name, scorer in scorers.items():
        score, all_scores = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for i, sc in enumerate(score, 1):
                scores[name + str(i)] = sc
        else:
            scores[name] = score
    
    return scores

def run():
    result_paths = [
        r"demo_result.json"
        ]
    result_paths = [x.strip() for x in result_paths][:]
    print(result_paths)
    for result_path in result_paths:
        experiment_name = os.path.split(result_path)[-1]
        logging.info(f"evaluating {experiment_name}:")
        my_result = read_json(result_path)
        my_preds = [d["pred"].strip() for d in my_result]
        my_golds = [d["gold"].strip() for d in my_result]
        a = evaluate_score(my_preds, my_golds)
        print(a)
        print("=============================")

if __name__ == "__main__":
    run()
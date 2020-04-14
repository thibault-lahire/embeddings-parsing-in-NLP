import argparse
import pcfg 
import parsing
from evaluate import evaluation
import get


def run(args):

    corpus_train, corpus_val, corpus_test = get.compute_corpora()
    vocabulary, embeddings = get.compute_polyglot_words_embeddings()

    rules, non_terms, proba_unary, proba_binary, transition_rules = pcfg.compute_proba_and_rules(corpus_train)

    if args.do_inference:
        get.compute_ground_truth(corpus_test, filename='evaluation_data.ground_truth')
        get.compute_predictions(rules, non_terms, proba_unary, proba_binary, transition_rules, vocabulary, embeddings, corpus_test, filename='evaluation_data.parser_output')

    if args.evaluate:
        evaluation('evaluation_data.ground_truth', 'evaluation_data.parser_output')

    if args.parse:
        parsing.parse_from_txt(args.txt_path, rules, non_terms, proba_unary, proba_binary, transition_rules, vocabulary, embeddings)



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--do_inference', action='store_true')
    argparser.add_argument('--evaluate', action='store_true')
    argparser.add_argument('--parse', action='store_true')
    argparser.add_argument('txt_path', nargs="?", type=lambda d:d)

    run(argparser.parse_args())


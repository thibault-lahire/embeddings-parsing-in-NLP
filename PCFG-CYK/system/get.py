import pickle
import os
import pcfg
import parsing



def compute_polyglot_words_embeddings(path='data'):
    """ 
        Returns vocabulary and embeddings extracted from polyglot lexicon
    """
    full_path = os.path.join(path, 'polyglot-fr.pkl')
    with open(full_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        vocabulary, embeddings = u.load()

    return vocabulary, embeddings



def compute_corpora(path='data'):
    """ 
        Returns train corpus, validation corpus, and test corpus
    """
    corpus = []
    full_path = os.path.join(path, 'sequoia-corpus+fct.mrg_strict')
    with open(full_path) as f:
        for i, l in enumerate(f):
            corpus.append(l)

    idx_end_train = int(len(corpus)*0.8)
    idx_start_test = int(len(corpus)*0.9)
    corpus_train = corpus[:idx_end_train]
    corpus_val = corpus[idx_end_train:idx_start_test]
    corpus_test = corpus[idx_start_test:]

    return corpus_train, corpus_val, corpus_test



def compute_ground_truth(test_corpus, filename='evaluation_data.ground_truth'):
    """ 
        Creates ground_truth file from test corpus (for pyevalb)
    """
    with open(filename, 'w') as f:
        for elem in test_corpus:
            truth = ' '.join(str(pcfg.tree_from_str(elem)).split()) # tree to str
            f.write("%s\n" % truth)



def compute_predictions(rules, non_terminals, unary_rules_proba, binary_rules_proba, transition_rules_list, vocabulary, embeddings, test_corpus, filename='evaluation_data.parser_output'):
    """ 
        Creates prediction file from test corpus (for pyevalb)
    """
    n_test = len(test_corpus)
    with open(filename, 'w') as f:
        for i, elem in enumerate(test_corpus):
            print('{} elements have been processed out of {}'.format(i+1, n_test))
            parsed = parsing.parse(rules, non_terminals, unary_rules_proba, binary_rules_proba, transition_rules_list, vocabulary, embeddings, tree=elem)
            f.write("%s\n" % parsed)


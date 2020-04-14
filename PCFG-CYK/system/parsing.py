from nltk import word_tokenize
from nltk.grammar import Production
import numpy as np
import oov
import pcfg




def parse_from_txt(txt_path, rules, non_terms, proba_unary, proba_binary, transition_rules, vocabulary, embeddings):
    """ 
        Parses each line from text file
        
    Agrgs:
        txt_path (str): path to text file
        rules (dict): stores the rules
        non_terms (dict): stores the non-terminals
        proba_unary (dict): stores the probabilities associated to unary rules
        proba_binary (dict): stores the probabilities associated to binary rules
        transition_rules (dict): stores the transition rules
        vocabulary (list): vocabulary available
        embeddings (list): embeddings associated to the vocabulary available
    """
    file = []
    with open(txt_path) as f:
        for l in f:
            file.append(l)

    for line in file:
        print(parse(rules, non_terms, proba_unary, proba_binary, transition_rules, vocabulary, embeddings, sentence=line))






def parse(rules, non_terms, proba_unary, proba_binary, transition_rules, vocabulary, embeddings, sentence=None, tree=None):
    """ 
        Parses sentence or tree

    Args:
        rules (dict): stores the rules
        non_terms (dict): stores the non-terminals
        proba_unary (dict): stores the probabilities associated to unary rules
        proba_binary (dict): stores the probabilities associated to binary rules
        transition_rules (dict): stores the transition rules
        vocabulary (list): vocabulary available
        embeddings (list): embeddings associated to the vocabulary available
        sentence (str): natural language format to be parsed
        tree (str): bracket format to be parsed

    Same returns as CYK function

    """
    if sentence:
        tokens = word_tokenize(sentence)
    elif tree:
        tokens = pcfg.tree_from_str(tree).leaves()
    else:
        print('Parsing impossible')
        raise ValueError
    return CYK(tokens, rules, non_terms, proba_unary, proba_binary, transition_rules, vocabulary, embeddings)





def CYK(tokens, rules, non_terms, proba_unary, proba_binary, transition_rules, vocabulary, embeddings):
    """ 
        CYK algorithm

    Args:
        tokens (list): list of str tokens
        rules (dict): stores the rules
        non_terms (dict): stores the non-terminals
        proba_unary (dict): stores the probabilities associated to unary rules
        proba_binary (dict): stores the probabilities associated to binary rules
        transition_rules (dict): stores the transition rules
        vocabulary (list): vocabulary available
        embeddings (list): embeddings associated to the vocabulary available

    Returns the most likely parsing tree

    """
    n = len(tokens)
    pi = {}
    backpointer = {}
    non_terms_set = non_terms.keys()
    retrieved_lexicon = [symbol.rhs()[0] for symbol in rules.keys() if type(symbol.rhs()[0])==str]

    intersection = set(retrieved_lexicon).intersection(set(vocabulary))
    word2idx = {w: i for (i, w) in enumerate(vocabulary)}
    mask_indexes = [word2idx[word] for word in intersection]
    vocabulary = np.array(vocabulary)[mask_indexes].tolist()
    embeddings = embeddings[mask_indexes]

    # Initialization of the probability table
    for i in range(n):
        token = tokens[i]
        if token in retrieved_lexicon:
            w = token
        else:
            w = oov.get_neighbor(token, vocabulary, embeddings)
            tokens[i] = w


        for X in non_terms_set:
            if Production(X, (w,)) in rules:
                pi[i, i, X] = proba_unary[Production(X, tuple((w,)))]
            else:
                pi[i, i, X] = 0

    # Algorithm
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            for X in non_terms_set:
                max_score = 0
                args = None
                for R in transition_rules[X]:
                    Y, Z = R[0], R[1]
                    for s in range(i, j):
                        if pi[i, s, Y] > 0 and pi[s + 1, j, Z] > 0:
                            score = proba_binary[Production(X, R)] * pi[i, s, Y] * pi[s + 1, j, Z]
                            if max_score < score:
                                max_score = score
                                args = Y, Z, s

                if max_score > 0:
                    backpointer[i, j, X] = args

                pi[i, j, X] = max_score

    # Retrieve the most probable parsed tree from backpointers and argmax of probability table
    max_score = 0
    args = None
    for X in non_terms_set:

        if max_score < pi[0, n - 1, X]:
            max_score = pi[0, n - 1, X]
            args = 0, n - 1, X

    if args == None:
        return '(SENT (UNK))'
    else:
        return '(SENT (' + retrieve_tree(tokens, backpointer, *args) + '))'





def retrieve_tree(tokens, backpointer, i, j, X):
    """ 
        Computes the parsed tree

    Args:
        tokens (list): list of str tokens
        backpointer (dict): dictionary of back pointers
        i (int): i-th element of the token list
        j (int): j-th element of the token list

    Returns:
        (str): parsed tree in str form
    """
    if i == j:
        return "".join([str(X), ' ', str(tokens[i])])
    else:
        Y, Z, s = backpointer[i, j, X]
        return "".join([str(X), ' (', retrieve_tree(tokens, backpointer, i, s, Y), ') (', retrieve_tree(tokens, backpointer, s + 1, j, Z), ')'])



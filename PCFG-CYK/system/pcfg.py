from nltk import Tree



def remove_functional_labels(tree):
    """ Ignoring functional labels and removing hyphen in a non-terminal name
    """
    for index, subtree in enumerate(tree):
        if not type(subtree) == str:
            # Checks whether the considered subtree is a leaf
            label = subtree.label().split('-')[0]
            subtree.set_label(label)
            tree[index] = subtree
            remove_functional_labels(subtree)
    return tree



def tree_from_str(line):
    """ Preprocesses the tree to have Chomsky Normal Form

    Args:
        line (str): current line read from corpus of trees

    Returns:
        tree (nltk.Tree): preprocessed tree
    """
    tree = Tree.fromstring(line, remove_empty_top_bracketing=True)
    tree = remove_functional_labels(tree)
    tree.collapse_unary(collapsePOS=True)
    tree.chomsky_normal_form()

    return tree




def compute_proba_and_rules(corpus):
    """ Learns dictionary of unary and binary probabilities from training corpus

    Args:
        corpus (list): list of training trees

    Returns:
        rules (dict): stores the rules
        non_terms (dict): stores the non-terminals
        proba_unary (dict): stores the probabilities associated to unary rules
        proba_binary (dict): stores the probabilities associated to binary rules
        transition_rules (dict): stores the transition rules
    """
    rules, non_terms, proba_unary, proba_binary, transition_rules = {}, {}, {}, {}, {}      
    
    
    for line in corpus:
        tree = tree_from_str(line)
        for prod in tree.productions():
            if prod in rules.keys():
                rules[prod] += 1
            else:
                rules[prod] = 1
            if prod.lhs() in non_terms.keys():
                non_terms[prod.lhs()] += 1
            else:
                non_terms[prod.lhs()] = 1
            

    for symbol in rules.keys():
        lhs, rhs = symbol.lhs(), symbol.rhs() 
        
        if len(rhs)==1:
            proba_unary[symbol] = float(rules[symbol]/non_terms[lhs])
        else:
            proba_binary[symbol] = float(rules[symbol]/non_terms[lhs])

        if lhs not in transition_rules.keys():
            transition_rules[lhs] = []
        if len(rhs)==2:
            transition_rules[lhs].append(rhs)

    return rules, non_terms, proba_unary, proba_binary, transition_rules





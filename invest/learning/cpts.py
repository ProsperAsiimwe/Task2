import tempfile
import pyAgrum as gum

def learn_cpts(data, bn, algorithm='MLE'):
    if data is None or len(data) == 0:
        raise Exception("An error occured")

    with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as temp_file:
        data.to_csv(temp_file.name, index=False)
        
        # Create a BNLearner with just the filename
        learner = gum.BNLearner(temp_file.name)
        
        # Learn the BN structure and parameters
        if algorithm == 'MLE':
            learned_bn = learner.learnBN()
        elif algorithm == 'BPE':
            learner.useBDeuPrior()  # Use BDeu prior for Bayesian estimation
            learned_bn = learner.learnBN()
        elif algorithm == 'EM':
            learner.useEM()
            learned_bn = learner.learnBN()
        else:
            return bn  # Use original BN if algorithm is not recognized
        
        # Copy the structure from the original BN to the learned BN
        for arc in bn.arcs():
            if not learned_bn.existsArc(arc[0], arc[1]):
                learned_bn.addArc(arc[0], arc[1])
        
    return learned_bn
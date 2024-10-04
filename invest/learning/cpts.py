import tempfile
import pyAgrum as gum

def learn_cpts(data, bn, algorithm='MLE'):
    if data is None or len(data) == 0:
        return bn  # Return the original BN if no data is provided

    with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as temp_file:
        data.to_csv(temp_file.name, index=False)
        
        # Create a BNLearner with just the filename
        learner = gum.BNLearner(temp_file.name)
        
       # Set the DAG structure
        learner.setDag(bn.dag())

        # Use BDeu score as a form of smoothing
        learner.useScoreAprioriBDeu()
        
        if algorithm == 'MLE':
            learned_model = learner.learnParameters(bn.dag())
        elif algorithm == 'BPE':
            learned_model = learner.learnBNParameters(bn.dag())
        elif algorithm == 'EM':
            learner.useEM()
            learned_model = learner.learnParameters(bn.dag())
        
    return learned_model
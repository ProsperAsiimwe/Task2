import os

import numpy as np
import pyAgrum as gum

def learn_cpts(data, bn, algorithm='MLE'):
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as temp_file:
        data.to_csv(temp_file.name, index=False)
        
        # Create a BNLearner with just the filename
        learner = gum.BNLearner(temp_file.name)
        
        # Set the structure of the Bayesian network
        learner.useAprioriSmoothing()
        learner.setStructureConstraint(bn)
        
        if algorithm == 'MLE':
            learned_model = learner.learnParameters(bn.dag())
        elif algorithm == 'BPE':
            learned_model = learner.learnBNParameters(bn.dag())
        elif algorithm == 'EM':
            learner.useEM()
            learned_model = learner.learnParameters(bn.dag())
        
    return learned_model

def investment_recommendation(data, value_decision, quality_decision, algorithm='MLE'):
    ir_model = gum.InfluenceDiagram()

    investable = gum.LabelizedVariable('Investable', 'Investable share', 2)
    investable.changeLabel(0, 'Yes')
    investable.changeLabel(1, 'No')
    ir_model.addDecisionNode(investable)

    share_performance = gum.LabelizedVariable('Performance', '', 3)
    share_performance.changeLabel(0, 'Positive')
    share_performance.changeLabel(1, 'Stagnant')
    share_performance.changeLabel(2, 'Negative')
    ir_model.addChanceNode(share_performance)

    value = gum.LabelizedVariable('Value', 'Value', 3)
    value.changeLabel(0, 'Cheap')
    value.changeLabel(1, 'FairValue')
    value.changeLabel(2, 'Expensive')
    ir_model.addChanceNode(value)

    quality = gum.LabelizedVariable('Quality', 'Quality', 3)
    quality.changeLabel(0, 'High')
    quality.changeLabel(1, 'Medium')
    quality.changeLabel(2, 'Low')
    ir_model.addChanceNode(quality)

    investment_utility = gum.LabelizedVariable('I_Utility', '', 1)
    ir_model.addUtilityNode(investment_utility)

    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('Quality'))
    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('Value'))
    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('I_Utility'))

    ir_model.addArc(ir_model.idFromName('Value'), ir_model.idFromName('Investable'))
    ir_model.addArc(ir_model.idFromName('Quality'), ir_model.idFromName('Investable'))
    ir_model.addArc(ir_model.idFromName('Investable'), ir_model.idFromName('I_Utility'))

    # Learn CPTs
    learned_model = learn_cpts(data, ir_model, algorithm)

    ie = gum.ShaferShenoyLIMIDInference(learned_model)

    if value_decision == "Cheap":
        ie.addEvidence('Value', [1, 0, 0])
    elif value_decision == "FairValue":
        ie.addEvidence('Value', [0, 1, 0])
    else:
        ie.addEvidence('Value', [0, 0, 1])

    if quality_decision == "High":
        ie.addEvidence('Quality', [1, 0, 0])
    elif quality_decision == "Medium":
        ie.addEvidence('Quality', [0, 1, 0])
    else:
        ie.addEvidence('Quality', [0, 0, 1])

    ie.makeInference()
    var = ie.posteriorUtility('Investable').variable('Investable')

    decision_index = np.argmax(ie.posteriorUtility('Investable').toarray())
    decision = var.label(int(decision_index))

    return format(decision)


def value_network(data, pe_relative_market_state, pe_relative_sector_state, forward_pe_current_vs_history_state,
                  future_performance_state=None, algorithm='MLE'):
    ve_model = gum.InfluenceDiagram()

    # Decision node for Expensive_E
    expensive_decision = gum.LabelizedVariable('Expensive_E', '', 2)
    expensive_decision.changeLabel(0, 'No')
    expensive_decision.changeLabel(1, 'Yes')
    ve_model.addDecisionNode(expensive_decision)

    # Decision node for ValueRelativeToPrice
    value_relative_to_price_decision = gum.LabelizedVariable('ValueRelativeToPrice', '', 3)
    value_relative_to_price_decision.changeLabel(0, 'Cheap')
    value_relative_to_price_decision.changeLabel(1, 'FairValue')
    value_relative_to_price_decision.changeLabel(2, 'Expensive')
    ve_model.addDecisionNode(value_relative_to_price_decision)

    # Add a chance node FutureSharePerformance
    future_share_performance = gum.LabelizedVariable('FutureSharePerformance', '', 3)
    future_share_performance.changeLabel(0, 'Positive')
    future_share_performance.changeLabel(1, 'Stagnant')
    future_share_performance.changeLabel(2, 'Negative')
    ve_model.addChanceNode(future_share_performance)

    # Add a chance node PERelative_ShareMarket
    pe_relative_market = gum.LabelizedVariable('PERelative_ShareMarket', '', 3)
    pe_relative_market.changeLabel(0, 'Cheap')
    pe_relative_market.changeLabel(1, 'FairValue')
    pe_relative_market.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(pe_relative_market)

    # Add a chance node PERelative_ShareSector
    pe_relative_sector = gum.LabelizedVariable('PERelative_ShareSector', '', 3)
    pe_relative_sector.changeLabel(0, 'Cheap')
    pe_relative_sector.changeLabel(1, 'FairValue')
    pe_relative_sector.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(pe_relative_sector)

    # Add a chance node ForwardPE_CurrentVsHistory
    forward_pe_current_vs_history = gum.LabelizedVariable('ForwardPE_CurrentVsHistory', '', 3)
    forward_pe_current_vs_history.changeLabel(0, 'Cheap')
    forward_pe_current_vs_history.changeLabel(1, 'FairValue')
    forward_pe_current_vs_history.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(forward_pe_current_vs_history)

    # Utility node for utility_expensive
    utility_expensive = gum.LabelizedVariable('Expensive_Utility', '', 1)
    ve_model.addUtilityNode(utility_expensive)

    # Utility node for utility_value_relative_to_price
    utility_value_relative_to_price = gum.LabelizedVariable('VRP_Utility', '', 1)
    ve_model.addUtilityNode(utility_value_relative_to_price)

    # Arcs
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('PERelative_ShareMarket'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('PERelative_ShareSector'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('ForwardPE_CurrentVsHistory'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('Expensive_Utility'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('VRP_Utility'))

    ve_model.addArc(ve_model.idFromName('PERelative_ShareMarket'), ve_model.idFromName('Expensive_E'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareMarket'), ve_model.idFromName('ValueRelativeToPrice'))

    ve_model.addArc(ve_model.idFromName('PERelative_ShareSector'), ve_model.idFromName('Expensive_E'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareSector'), ve_model.idFromName('ValueRelativeToPrice'))

    ve_model.addArc(ve_model.idFromName('ForwardPE_CurrentVsHistory'), ve_model.idFromName('ValueRelativeToPrice'))

    ve_model.addArc(ve_model.idFromName('Expensive_E'), ve_model.idFromName('ForwardPE_CurrentVsHistory'))
    ve_model.addArc(ve_model.idFromName('Expensive_E'), ve_model.idFromName('ValueRelativeToPrice'))
    ve_model.addArc(ve_model.idFromName('Expensive_E'), ve_model.idFromName('Expensive_Utility'))

    ve_model.addArc(ve_model.idFromName('ValueRelativeToPrice'), ve_model.idFromName('VRP_Utility'))

    # Learn CPTs
    learned_model = learn_cpts(data, ve_model, algorithm)

    ie = gum.ShaferShenoyLIMIDInference(learned_model)
    ie.addNoForgettingAssumption(['Expensive_E', 'ValueRelativeToPrice'])

    if pe_relative_market_state == "cheap":
        ie.addEvidence('PERelative_ShareMarket', [1, 0, 0])
    elif pe_relative_market_state == "fairValue":
        ie.addEvidence('PERelative_ShareMarket', [0, 1, 0])
    else:
        ie.addEvidence('PERelative_ShareMarket', [0, 0, 1])

    if pe_relative_sector_state == "cheap":
        ie.addEvidence('PERelative_ShareSector', [1, 0, 0])
    elif pe_relative_sector_state == "fairValue":
        ie.addEvidence('PERelative_ShareSector', [0, 1, 0])
    else:
        ie.addEvidence('PERelative_ShareSector', [0, 0, 1])

    if forward_pe_current_vs_history_state == "cheap":
        ie.addEvidence('ForwardPE_CurrentVsHistory', [1, 0, 0])
    elif forward_pe_current_vs_history_state == "fairValue":
        ie.addEvidence('ForwardPE_CurrentVsHistory', [0, 1, 0])
    else:
        ie.addEvidence('ForwardPE_CurrentVsHistory', [0, 0, 1])

    if future_performance_state:
        if future_performance_state == 1 or future_performance_state == "positive":
            ie.addEvidence('FutureSharePerformance', [0.8, 0.1, 0.1])
        elif future_performance_state == 0 or future_performance_state == "stagnant":
            ie.addEvidence('FutureSharePerformance', [0.1, 0.2, 0.1])
        else:
            ie.addEvidence('FutureSharePerformance', [0.1, 0.1, 0.8])

    ie.makeInference()
    var = ie.posteriorUtility('ValueRelativeToPrice').variable('ValueRelativeToPrice')

    decision_index = np.argmax(ie.posteriorUtility('ValueRelativeToPrice').toarray())
    decision = var.label(int(decision_index))

    # Forced Decisions
    if decision == 'Cheap':
        pass
    if decision == 'Expensive':
        if pe_relative_market_state == "cheap" and pe_relative_sector_state == "expensive":
            return 'FairValue'
        elif pe_relative_market_state == "expensive" and pe_relative_sector_state == "cheap":
            return 'FairValue'
        elif pe_relative_market_state == "fairValue" and pe_relative_sector_state == "fairValue" and \
                forward_pe_current_vs_history_state == "fairValue":
            return 'FairValue'

    return format(decision)

def quality_network(data, roe_vs_coe_state, relative_debt_equity_state, cagr_vs_inflation_state, 
                    systematic_risk_state=None, extension=False, algorithm='MLE'):
    qe_model = gum.InfluenceDiagram()

    # Decision node
    quality_decision = gum.LabelizedVariable('Quality', '', 3)
    quality_decision.changeLabel(0, 'High')
    quality_decision.changeLabel(1, 'Medium')
    quality_decision.changeLabel(2, 'Low')
    qe_model.addDecisionNode(quality_decision)

    # FutureSharePerformance node
    future_share_performance = gum.LabelizedVariable('FutureSharePerformance', '', 3)
    future_share_performance.changeLabel(0, 'Positive')
    future_share_performance.changeLabel(1, 'Stagnant')
    future_share_performance.changeLabel(2, 'Negative')
    qe_model.addChanceNode(future_share_performance)

    # CAGR vs Inflation node
    cagr_vs_inflation = gum.LabelizedVariable('CAGRvsInflation', '', 3)
    cagr_vs_inflation.changeLabel(0, 'InflationPlus')
    cagr_vs_inflation.changeLabel(1, 'Inflation')
    cagr_vs_inflation.changeLabel(2, 'InflationMinus')
    qe_model.addChanceNode(cagr_vs_inflation)

    # ROE vs COE node
    roe_vs_coe = gum.LabelizedVariable('ROEvsCOE', '', 3)
    roe_vs_coe.changeLabel(0, 'Above')
    roe_vs_coe.changeLabel(1, 'EqualTo')
    roe_vs_coe.changeLabel(2, 'Below')
    qe_model.addChanceNode(roe_vs_coe)

    # Relative debt to equity node
    relative_debt_equity = gum.LabelizedVariable('RelDE', '', 3)
    relative_debt_equity.changeLabel(0, 'Above')
    relative_debt_equity.changeLabel(1, 'EqualTo')
    relative_debt_equity.changeLabel(2, 'Below')
    qe_model.addChanceNode(relative_debt_equity)

    quality_utility = gum.LabelizedVariable('Q_Utility', '', 1)
    qe_model.addUtilityNode(quality_utility)

    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('CAGRvsInflation'))
    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('ROEvsCOE'))
    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('RelDE'))
    qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('Q_Utility'))

    qe_model.addArc(qe_model.idFromName('CAGRvsInflation'), qe_model.idFromName('Quality'))
    qe_model.addArc(qe_model.idFromName('ROEvsCOE'), qe_model.idFromName('Quality'))
    qe_model.addArc(qe_model.idFromName('RelDE'), qe_model.idFromName('Quality'))
    qe_model.addArc(qe_model.idFromName('Quality'), qe_model.idFromName('Q_Utility'))

    # Extension
    if extension:
        systematic_risk = gum.LabelizedVariable('SystematicRisk', '', 3)
        systematic_risk.changeLabel(0, 'greater')  # Greater than Market
        systematic_risk.changeLabel(1, 'EqualTo')
        systematic_risk.changeLabel(2, 'lower')
        qe_model.addChanceNode(systematic_risk)

        qe_model.addArc(qe_model.idFromName('FutureSharePerformance'), qe_model.idFromName('SystematicRisk'))
        qe_model.addArc(qe_model.idFromName('SystematicRisk'), qe_model.idFromName('Quality'))

    # Learn CPTs
    learned_model = learn_cpts(data, qe_model, algorithm)

    ie = gum.ShaferShenoyLIMIDInference(learned_model)

    if relative_debt_equity_state == "above":
        ie.addEvidence('RelDE', [1, 0, 0])
    elif relative_debt_equity_state == "EqualTo":
        ie.addEvidence('RelDE', [0, 1, 0])
    else:
        ie.addEvidence('RelDE', [0, 0, 1])

    if roe_vs_coe_state == "above":
        ie.addEvidence('ROEvsCOE', [1, 0, 0])
    elif roe_vs_coe_state == "EqualTo":
        ie.addEvidence('ROEvsCOE', [0, 1, 0])
    else:
        ie.addEvidence('ROEvsCOE', [0, 0, 1])

    if cagr_vs_inflation_state == "above":
        ie.addEvidence('CAGRvsInflation', [1, 0, 0])
    elif cagr_vs_inflation_state == "EqualTo":
        ie.addEvidence('CAGRvsInflation', [0, 1, 0])
    else:
        ie.addEvidence('CAGRvsInflation', [0, 0, 1])

    if extension and systematic_risk_state:
        if systematic_risk_state == "greater":
            ie.addEvidence('SystematicRisk', [1, 0, 0])
        elif systematic_risk_state == "EqualTo":
            ie.addEvidence('SystematicRisk', [0, 1, 0])
        else:
            ie.addEvidence('SystematicRisk', [0, 0, 1])

    ie.makeInference()
    var = ie.posteriorUtility('Quality').variable('Quality')

    decision_index = np.argmax(ie.posteriorUtility('Quality').toarray())
    decision = var.label(int(decision_index))
    return format(decision)
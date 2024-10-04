import os

import numpy as np
import pyAgrum as gum

from invest.learning.cpts import learn_cpts


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
import json

import pandas as pd

import invest.evaluation.validation as validation
from invest.networks.invest_recommendation import investment_recommendation
from invest.networks.quality_evaluation import quality_network
from invest.networks.value_evaluation import value_network
#from invest.prediction.main import future_share_price_performance
from invest.preprocessing.simulation import simulate
from invest.store import Store

companies_jcsev = json.load(open('data/jcsev.json'))['names']
companies_jgind = json.load(open('data/jgind.json'))['names']
companies = companies_jcsev + companies_jgind
companies_dict = {"JCSEV": companies_jcsev, "JGIND": companies_jgind}


def investment_portfolio(df_, params, index_code, verbose=False, algorithm='MLE'):
    if params.noise:
        df = simulate(df_)
    else:
        df = df_

    prices_initial = {}
    prices_current = {}
    betas = {}
    investable_shares = {}

    for year in range(params.start, params.end):
        store = Store(df, companies, companies_jcsev, companies_jgind,
                      params.margin_of_safety, params.beta, year, False)
        investable_shares[str(year)] = []
        prices_initial[str(year)] = []
        prices_current[str(year)] = []
        betas[str(year)] = []
        df_future_performance = pd.DataFrame()
        for company in companies_dict[index_code]:
            if store.get_acceptable_stock(company):
                if not df_future_performance.empty:
                    future_performance = df_future_performance[company][0]
                else:
                    future_performance = None
                if investment_decision(store, company, future_performance, params.extension, params.ablation,
                                       params.network, df_, algorithm) == "Yes":
                    mask = (df_['Date'] >= str(year) + '-01-01') & (
                            df_['Date'] <= str(year) + '-12-31') & (df_['Name'] == company)
                    df_year = df_[mask]

                    investable_shares[str(year)].append(company)
                    prices_initial[str(year)].append(df_year.iloc[0]['Price'])
                    prices_current[str(year)].append(df_year.iloc[params.holding_period]['Price'])
                    betas[str(year)].append(df_year.iloc[params.holding_period]["ShareBeta"])

    if verbose:
        print("\n{} {} - {}".format(index_code, params.start, params.end))
        print("-" * 50)
        print("\nInvestable Shares")
        for year in range(params.start, params.end):
            print(year, "IP." + index_code, len(investable_shares[str(year)]), investable_shares[str(year)])

    ip_ar, ip_cr, ip_aar, ip_treynor, ip_sharpe = validation.process_metrics(df_,
                                                                             prices_initial,
                                                                             prices_current,
                                                                             betas,
                                                                             params.start,
                                                                             params.end,
                                                                             index_code)
    benchmark_ar, benchmark_cr, benchmark_aar, benchmark_treynor, benchmark_sharpe = \
        validation.process_benchmark_metrics(params.start, params.end, index_code, params.holding_period)

    portfolio = {
        "ip": {
            "shares": investable_shares,
            "annualReturns": ip_ar,
            "compoundReturn": ip_cr,
            "averageAnnualReturn": ip_aar,
            "treynor": ip_treynor,
            "sharpe": ip_sharpe,
        },
        "benchmark": {
            "annualReturns": benchmark_ar,
            "compoundReturn": benchmark_cr,
            "averageAnnualReturn": benchmark_aar,
            "treynor": benchmark_treynor,
            "sharpe": benchmark_sharpe,
        }
    }
    return portfolio


def investment_decision(store, company, future_performance=None, extension=False, ablation=False, network='v', data=None, algorithm='MLE'):
    pe_relative_market = store.get_pe_relative_market(company)
    pe_relative_sector = store.get_pe_relative_sector(company)
    forward_pe = store.get_forward_pe(company)

    roe_vs_coe = store.get_roe_vs_coe(company)
    relative_debt_equity = store.get_relative_debt_equity(company)
    cagr_vs_inflation = store.get_cagr_vs_inflation(company)
    systematic_risk = store.get_systematic_risk(company)

    # Prepare data for learning if it's not None and contains the company
    if data is not None and company in data['Name'].values:
        learn_data = data[data['Name'] == company].copy()
        learn_data['PERelative_Market'] = pe_relative_market
        learn_data['PERelative_Sector'] = pe_relative_sector
        learn_data['ForwardPE'] = forward_pe
        learn_data['ROEvsCOE'] = roe_vs_coe
        learn_data['RelativeDebtEquity'] = relative_debt_equity
        learn_data['CAGRvsInflation'] = cagr_vs_inflation
        learn_data['SystematicRisk'] = systematic_risk
    else:
        learn_data = None
        algorithm = 'fixed'  # Use fixed CPTs if no learning data is available

    value_decision = value_network(learn_data, pe_relative_market, pe_relative_sector, forward_pe, future_performance, algorithm)
    quality_decision = quality_network(learn_data, roe_vs_coe, relative_debt_equity, cagr_vs_inflation,
                                       systematic_risk, extension, algorithm)
    if ablation and network == 'v':
        if value_decision in ["Cheap", "FairValue"]:
            return "Yes"
        else:
            return "No"
    if ablation and network == 'q':
        if quality_decision in ["High", "Medium"]:
            return "Yes"
        else:
            return "No"
    return investment_recommendation(learn_data, value_decision, quality_decision, algorithm)
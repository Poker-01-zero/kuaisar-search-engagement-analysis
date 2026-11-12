"""Explicitly synthetic A/B case; never evidence of KuaiSAR treatment effects."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest, confint_proportions_2indep


def run(root, seed=20261111):
    rng = np.random.default_rng(seed)
    p0, p1, alpha, power = .10, .11, .05, .80
    n = math.ceil(NormalIndPower().solve_power(proportion_effectsize(p1, p0), power=power, alpha=alpha, ratio=1))
    arm = rng.integers(0, 2, 2*n)
    converted = rng.binomial(1, np.where(arm == 0, p0, p1))
    revenue = converted * rng.lognormal(4, .6, len(arm))
    data = pd.DataFrame({'user_id':np.arange(len(arm)), 'arm':arm,
                         'converted':converted, 'synthetic_revenue':revenue})
    out = root / 'data' / 'interim'; out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / 'synthetic_ab_users.csv', index=False)
    grouped = data.groupby('arm').agg(users=('user_id','size'), conversions=('converted','sum'),
                                     revenue_mean=('synthetic_revenue','mean'))
    grouped['conversion_rate'] = grouped.conversions / grouped.users
    count, sizes = grouped.loc[[1,0],'conversions'].to_numpy(), grouped.loc[[1,0],'users'].to_numpy()
    z, p = proportions_ztest(count, sizes, alternative='two-sided')
    low, high = confint_proportions_2indep(count[0], sizes[0], count[1], sizes[1], method='newcomb', compare='diff')
    srm = stats.chisquare(grouped.users.to_numpy(), f_exp=[n,n])
    welch = stats.ttest_ind(data.loc[data.arm.eq(1),'synthetic_revenue'],
                            data.loc[data.arm.eq(0),'synthetic_revenue'], equal_var=False)
    simulations = []
    for label, treatment_probability in [('null',p0),('preset_effect',p1)]:
        ca, cb = rng.binomial(n, p0, 2000), rng.binomial(n, treatment_probability, 2000)
        pooled = (ca+cb)/(2*n)
        z_values = (cb/n-ca/n)/np.sqrt(pooled*(1-pooled)*2/n)
        p_values = 2*stats.norm.sf(np.abs(z_values))
        simulations.append({'scenario':label,'runs':len(p_values),'rejection_fraction':float((p_values<alpha).mean())})
    diagnostics=[]
    for group,z in data.groupby('arm'):
        v=z.synthetic_revenue
        diagnostics.append({'arm':int(group),'zero_fraction':float(v.eq(0).mean()),'median':float(v.median()),'p99':float(v.quantile(.99)),'max':float(v.max()),'top_one_percent_revenue_share':float(v.nlargest(max(1,len(v)//100)).sum()/v.sum())})
    result = {
        'data_kind':'SYNTHETIC — no real users, no deployed strategy', 'seed':seed,
        'baseline_probability':p0,'preset_treatment_probability':p1,'mde_absolute':p1-p0,
        'alpha':alpha,'target_power':power,'planned_users_per_arm':n,
        'observed_users':len(data),'srm_pvalue':float(srm.pvalue),
        'difference':float(count[0]/sizes[0]-count[1]/sizes[1]),
        'relative_change':float((count[0]/sizes[0])/(count[1]/sizes[1])-1),
        'ci95_newcombe':[float(low),float(high)],'ztest_pvalue':float(p),
        'welch_revenue_pvalue_exploratory':float(welch.pvalue),
        'simulations':simulations,
        'quality':{'unique_user':bool(data.user_id.is_unique),'missing_cells':int(data.isna().sum().sum()),'cross_group_users':0},
    }
    tables = root / 'outputs' / 'tables'; tables.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(tables/'synthetic_ab_revenue_diagnostics.csv',index=False)
    grouped.to_csv(tables / 'synthetic_ab_metrics.csv', encoding='utf-8-sig')
    (tables / 'synthetic_ab_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(simulations).to_csv(tables / 'synthetic_ab_simulation.csv',index=False)
    return result


if __name__ == '__main__':
    print(json.dumps(run(Path(__file__).resolve().parents[1]),ensure_ascii=False,indent=2))

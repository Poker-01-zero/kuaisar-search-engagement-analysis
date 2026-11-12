"""Full-session search diagnostics; SQLite provides an independent aggregation."""
import sqlite3
import numpy as np
import pandas as pd
from .metrics import build_sessions, group_metrics, source_metrics


def run(rows, users, root, cfg, quality, audits):
    from .prepare import audited_join
    table = root/'outputs/tables'
    sessions, audit = build_sessions(rows, cfg['timezone'])
    quality['search_sessions'] = audit
    sessions.to_parquet(root/'data/interim/search_sessions_all.parquet',index=False)
    rows.loc[rows[['user_id','search_session_id']].isna().any(axis=1)].to_csv(root/'outputs/audit/search_missing_key_samples.csv',index=False)
    sessions.loc[~sessions.valid].to_csv(root/'outputs/audit/search_excluded_sessions.csv',index=False)
    s = sessions.loc[sessions.valid].copy()
    assert int(s.result_rows.sum()) + audit['invalid_session_rows'] + audit['missing_key_rows'] == len(rows)
    s = audited_join(s, users[['user_id','search_active_level']], 'user_id','search_user',audits,['result_rows','clicked_rows','has_click']).drop(columns='_merge')
    s.to_parquet(root/'data/interim/search_sessions.parquet',index=False)
    quality['search_users'] = {'dimension_users':len(users),'valid_search_users':s.user_id.nunique(),
                               'no_valid_search_users':len(set(users.user_id)-set(s.user_id))}
    complete=s.loc[s.date.between(cfg['complete_date_start'],cfg['complete_date_end'])].copy()
    tables={
      'search_source_all_dates':source_metrics(s,cfg['minimum_sessions_for_comparison'],cfg['minimum_users_for_comparison']),
      'search_source_metrics':source_metrics(complete,cfg['minimum_sessions_for_comparison'],cfg['minimum_users_for_comparison']),
      'search_source_itemtype':group_metrics(complete,['source','type_mix']),
      'search_daily':group_metrics(s,['date','source']),
      'search_activity':group_metrics(complete,['source','search_active_level'])}
    # Official binary description conflicts with observed >1 values. Do not silently recode.
    # A separately labelled scenario asks whether any positive feedback implies a click.
    alternative=rows.copy()
    legal=alternative.click_cnt.notna() & alternative.click_cnt.ge(0) & alternative.click_cnt.mod(1).eq(0)
    alternative['click_cnt']=alternative.click_cnt.gt(0).astype(float).where(legal)
    alt,alt_audit=build_sessions(alternative,cfg['timezone'])
    alt=alt.loc[alt.valid & alt.date.between(cfg['complete_date_start'],cfg['complete_date_end'])]
    sensitivity=source_metrics(alt,cfg['minimum_sessions_for_comparison'],cfg['minimum_users_for_comparison'])
    sensitivity['scenario']='ASSUMPTION: nonnegative integer positive value means clicked; not official binary contract'
    tables['search_positive_click_sensitivity']=sensitivity
    exclusions=sessions.groupby('source',dropna=False).agg(total=('valid','size'),valid=('valid','sum')).reset_index()
    exclusions['excluded_fraction']=1-exclusions.valid/exclusions.total
    tables['search_exclusions_by_source']=exclusions
    quality['click_contract']={'nonbinary_rows':int((~rows.click_cnt.isin([0,1])).sum()),
       'strict_excluded_sessions':audit['invalid_sessions'],'alternative_valid_complete_sessions':len(alt),
       'decision':'strict binary official contract primary; positive-value assumption sensitivity only; raw magnitudes not summed as clicks'}
    tables['search_weighting_sensitivity']=tables['search_source_metrics'][['source','sessions','users','session_click_rate','user_day_equal_click_rate','user_equal_click_rate']]
    ud=complete.groupby(['user_id','date','source'],observed=True).agg(sessions=('has_click','size'),clicked_sessions=('has_click','sum')).reset_index()
    ud.to_parquet(root/'data/interim/search_user_day.parquet',index=False)
    top=tables['search_source_metrics'].query('display_eligible').head(2).source.tolist()
    compare=complete.loc[complete.source.isin(top)]
    # Common date x result-mix x public activity strata. Same pooled weights for both sources.
    keys=['date','type_mix','search_active_level']
    strata=compare.groupby(keys+['source'],dropna=False,observed=True).has_click.agg(['size','mean']).reset_index()
    rate=strata.pivot(index=keys,columns='source',values='mean').dropna()
    counts=strata.pivot(index=keys,columns='source',values='size').reindex(rate.index).fillna(0)
    weights=counts.sum(axis=1); weights=weights/weights.sum()
    standard=[]
    for source in top:
        common_n=int(counts[source].sum())
        standard.append({'source':source,'common_strata':len(rate),'common_sessions':common_n,
                         'common_support_fraction':common_n/int(compare.source.eq(source).sum()),
                         'common_raw_rate':float((rate[source]*counts[source]).sum()/common_n),
                         'standardized_click_rate':float((rate[source]*weights).sum())})
    tables['search_standardized']=pd.DataFrame(standard)
    tables['search_common_strata']=strata
    sql=(root/'sql/search_metrics_check.sql').read_text(encoding='utf-8-sig')
    with sqlite3.connect(':memory:') as db:
        complete[['source','user_id','has_click','result_rows','clicked_rows']].to_sql('sessions',db,index=False)
        check=pd.read_sql_query(sql,db)
    expected=tables['search_source_metrics'][check.columns].sort_values('source').reset_index(drop=True)
    pd.testing.assert_frame_equal(check.sort_values('source').reset_index(drop=True),expected,check_dtype=False,atol=1e-12)
    quality['sql_check']={'engine':sqlite3.sqlite_version,'passed':True,'sources':len(check),'columns':list(check.columns)}
    # Verify chunk filtering preserves whole selected users, including cross-chunk sessions.
    ids=set(sorted(users.user_id)[:200])
    dev=rows.loc[rows.user_id.isin(ids)]
    chunks=[rows.iloc[i:i+cfg['chunksize']].loc[lambda z:z.user_id.isin(ids)] for i in range(0,len(rows),cfg['chunksize'])]
    a,_=build_sessions(dev,cfg['timezone']); b,_=build_sessions(pd.concat(chunks),cfg['timezone'])
    pd.testing.assert_frame_equal(a,b)
    quality['development_check']={'users_selected':200,'rows':len(dev),'chunk_filter_equals_whole':True}
    for name,value in tables.items(): value.to_csv(table/f'{name}.csv',index=False,encoding='utf-8-sig')
    return s, complete, tables

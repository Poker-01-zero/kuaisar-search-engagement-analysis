"""Reproducible full-small workflow. Run: python -m src.run --config config.json."""
import argparse, hashlib, json, time, platform
from pathlib import Path
import pandas as pd
import psutil
from . import prepare, search_analysis, content_analysis, ab_analysis


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.json');args=parser.parse_args()
    config_path=Path(args.config).resolve();root=config_path.parent
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    for d in ['outputs/tables','outputs/audit','outputs/figures','reports','data/interim','notebooks']:(root/d).mkdir(parents=True,exist_ok=True)
    start=time.monotonic();quality={};inventory=[];profiles=[];audits=[]
    archive=root/'data/raw/KuaiSAR.zip'
    with archive.open('rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
    if md5!=cfg['archive_md5']:raise ValueError('Archive checksum mismatch; run download_data.py')
    def load(name,cols):return prepare.load(root,name,cols,inventory,profiles,quality,cfg['chunksize'])
    users=load('user_features.csv',['user_id','search_active_level','reco_active_level'])
    if not users.user_id.is_unique or users.user_id.isna().any():raise ValueError('Invalid user dimension')
    users.user_id.to_csv(root/'data/interim/selected_users.csv',index=False)
    search=load('src_inter.csv',['user_id','item_id','search_session_id','search_source','item_type','click_cnt','search_session_timestamp','search_session_time'])
    prepare.time_fields(search,'search_session_timestamp','search_session_time',cfg['timezone'],quality,'src_inter.csv')
    quality['src_inter.csv']['click_values']=search.click_cnt.value_counts(dropna=False).to_dict()
    quality['src_inter.csv']['result_types']=search.item_type.value_counts(dropna=False).to_dict()
    # Keep complete user histories; all-small is the formal configuration.
    s,complete,st=search_analysis.run(search,users,root,cfg,quality,audits)
    print('Search and SQL checks complete',flush=True)
    items=load('item_features.csv',['item_id','item_type','first_level_category_id','first_level_category_name_en'])
    if not items.item_id.is_unique or items.item_id.isna().any():raise ValueError('Invalid item dimension')
    items=items.rename(columns={'item_type':'content_item_type'})
    sj=prepare.audited_join(search[['user_id','item_id','item_type','click_cnt']],items,'item_id','search_item',audits,['click_cnt'])
    sj.groupby('item_type',dropna=False,observed=True)._merge.apply(lambda z:z.eq('both').mean()).reset_index(name='coverage').to_csv(root/'outputs/audit/search_item_type_coverage.csv',index=False)
    sj.loc[sj._merge.eq('left_only')].head(20).to_csv(root/'outputs/audit/search_unmatched_samples.csv',index=False)
    del sj,search
    rec=load('rec_inter.csv',['user_id','item_id','duration_ms','playing_time','timestamp','time','click','like','follow'])
    prepare.time_fields(rec,'timestamp','time',cfg['timezone'],quality,'rec_inter.csv')
    quality['rec_inter.csv']['click_values']=rec.click.value_counts(dropna=False).to_dict()
    ct=content_analysis.run(rec,items,root,cfg,quality,audits)
    print('Content complete',flush=True)
    del rec,items
    ab=ab_analysis.run(root, cfg['seed'])
    pd.DataFrame(inventory).to_csv(root/'outputs/audit/data_inventory.csv',index=False)
    pd.DataFrame(profiles).to_csv(root/'outputs/audit/field_profile.csv',index=False)
    pd.DataFrame(audits).to_csv(root/'outputs/audit/join_audit.csv',index=False)
    manifest={'dataset':cfg['dataset'],'archive_bytes':archive.stat().st_size,'archive_md5':md5,
       'source':'https://zenodo.org/records/8181109','scope':cfg['scope'],'sampling':'none; all published rows',
       'selected_users':len(users),'complete_dates':[cfg['complete_date_start'],cfg['complete_date_end']],
       'actual_time_ranges':{name:quality[name]['time'] for name in ['src_inter.csv','rec_inter.csv']},
       'files':inventory,'python':platform.python_version(),'config':cfg}
    prepare.dump(root/'data/manifest.json',manifest)
    elapsed=time.monotonic()-start
    quality['runtime']={'seconds_before_reporting':elapsed,'process_peak_working_set_bytes':psutil.Process().memory_info().peak_wset,
       'memory_note':'Windows process peak working set; not a performance improvement benchmark'}
    prepare.dump(root/'outputs/audit/quality_report.json',quality)
    from .deliver import deliver
    deliver(root,cfg,quality,manifest,st,ct,ab)
    print(f'P1 pipeline complete: {time.monotonic()-start:.1f}s',flush=True)

if __name__=='__main__':main()

"""Content with valid durations; media type is deliberately not inferred."""
import numpy as np
import pandas as pd
from .metrics import playback_fields
from .prepare import audited_join


def summarize(x, keys):
    m=x.groupby(keys,observed=True,dropna=False).agg(
      rows=('user_id','size'),users=('user_id','nunique'),duration_median_ms=('duration_ms','median'),
      mean_play_ratio=('play_ratio','mean'),median_play_ratio=('play_ratio','median'),
      p90_play_ratio=('play_ratio',lambda z:z.quantile(.9)),
      capped_play_ratio=('capped_play_ratio','mean'),reaches_duration_rate=('reaches_duration','mean'),
      exceeds_duration_rate=('exceeds_duration','mean'),like_rate=('like','mean'),follow_rate=('follow','mean'),
      played_rows=('playing_time',lambda z:z.gt(0).sum()))
    ud=x.groupby(keys+['user_id','date'],observed=True,dropna=False).agg(
      reaches=('reaches_duration','mean'),capped=('capped_play_ratio','mean'),likes=('like','mean'))
    equal=ud.groupby(level=list(range(len(keys))),observed=True,dropna=False).mean().rename(columns={
      'reaches':'user_day_equal_reaches_rate','capped':'user_day_equal_capped_ratio','likes':'user_day_equal_like_rate'})
    user_like=x.groupby(keys+['user_id'],observed=True,dropna=False)['like'].max().groupby(level=list(range(len(keys))),observed=True,dropna=False).mean()
    m['user_like_rate']=user_like
    return m.join(equal).reset_index()


def run(rec, items, root, cfg, quality, audits):
    joined=audited_join(rec,items,'item_id','recommendation_item',audits,['playing_time','duration_ms','like','follow'])
    quality['content_join_comparison']={}
    for label,z in [('all',joined),('matched',joined.loc[joined._merge.eq('both')]),('unmatched',joined.loc[joined._merge.eq('left_only')])]:
        quality['content_join_comparison'][label]={'rows':len(z),'users':z.user_id.nunique(),'like_rate':float(z.like.mean())}
    joined.loc[joined._merge.eq('left_only')].head(20).to_csv(root/'outputs/audit/content_unmatched_samples.csv',index=False)
    joined['category']=joined['first_level_category_name_en'].fillna('UNMATCHED')
    unexpected=~joined.content_item_type.isin(['NORMAL','AD','UNKNOW','LIVEGAME_PLAYBACK'])
    quality['item_attribute_warning']={'unexpected_type_records':int(unexpected.sum()),'policy':'unexpected content type kept as unverified; category ID and English label retained; no media inference'}
    x=playback_fields(joined)
    quality['content']={'all_rows':len(x),'invalid_duration_rows':int((~x.valid_duration).sum()),
       'zero_duration_rows':int(x.duration_ms.eq(0).sum()),'negative_duration_rows':int(x.duration_ms.lt(0).sum()),
       'negative_play_rows':int(x.playing_time.lt(0).sum()),'invalid_feedback_rows':int((~x['like'].isin([0,1])|~x.follow.isin([0,1])).sum()),
       'type_scope':'legal duration content; no complete video/image mapping available',
       'content_item_types':x.content_item_type.value_counts(dropna=False).to_dict()}
    valid=x.loc[x.valid_duration & x.date.between(cfg['complete_date_start'],cfg['complete_date_end']) & x['like'].isin([0,1]) & x.follow.isin([0,1])].copy()
    quality['content']['complete_valid_rows']=len(valid)
    quality['content']['complete_valid_users']=valid.user_id.nunique()
    tables={'content_duration_metrics':summarize(valid,['duration_band']),
      'content_category_duration':summarize(valid,['category','duration_band']),
      'content_played_duration_metrics':summarize(valid.loc[valid.playing_time.gt(0)],['duration_band'])}
    distribution=valid.play_ratio.quantile([0,.25,.5,.75,.9,.95,.99,1]).rename_axis('quantile').reset_index(name='play_ratio')
    tables['playback_distribution']=distribution
    for name,value in tables.items():value.to_csv(root/'outputs/tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    return tables

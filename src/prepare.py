"""Chunked ingestion, original-field profiling and auditable joins."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(z) for k,z in v.items()}
        if isinstance(v,(list,tuple)):return [clean(z) for z in v]
        if isinstance(v,(float,np.floating)) and not math.isfinite(v):return None
        if isinstance(v,np.integer):return int(v)
        return v
    path.write_text(json.dumps(clean(obj),allow_nan=False, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load(root, name, columns, inventory, profiles, quality, chunksize=200000):
    path = next((root/'data/raw').rglob(name))
    parts, hashes, rows, nulls, lo, hi, dtypes = [], [], 0, {}, {}, {}, {}
    first_block_memory=None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if first_block_memory is None:first_block_memory={'rows':len(chunk),'selected_column_bytes':int(chunk[columns].memory_usage(deep=True).sum())}
        rows += len(chunk)
        hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        for c in chunk:
            nulls[c] = nulls.get(c, 0) + int(chunk[c].isna().sum())
            dtypes[c] = str(chunk[c].dtype)
            if pd.api.types.is_numeric_dtype(chunk[c]):
                a, b = chunk[c].min(), chunk[c].max()
                if pd.notna(a): lo[c] = min(lo.get(c, a), a); hi[c] = max(hi.get(c, b), b)
        parts.append(chunk[columns].copy())
    x = pd.concat(parts, ignore_index=True)
    h = np.concatenate(hashes)
    quality[name] = {'duplicate_full_row_hashes':int(len(h)-len(np.unique(h))),
                     'duplicate_policy':'preserved; 64-bit all-original-field hash diagnostic',
                     'first_block_memory':first_block_memory, 'loaded_columns': columns, 'loaded_memory_bytes':int(x.memory_usage(deep=True).sum())}
    inventory.append({'file':name,'bytes':path.stat().st_size,'rows':rows,'columns':len(nulls),'loaded_columns':len(columns)})
    profiles.extend({'file':name,'field':c,'dtype':dtypes[c],'nulls':nulls[c], 'min':lo.get(c),'max':hi.get(c),'scope':'full file'} for c in nulls)
    print(f'Loaded {name}: {rows:,} rows', flush=True)
    return x


def time_fields(x, timestamp, human, timezone, quality, label):
    dt = pd.to_datetime(x[timestamp], unit='ms', utc=True, errors='coerce').dt.tz_convert(timezone)
    expected = pd.to_datetime(x[human], errors='coerce')
    mismatch = dt.dt.tz_localize(None).dt.floor('s').ne(expected)
    quality[label]['time'] = {'min':str(dt.min()),'max':str(dt.max()),'invalid':int(dt.isna().sum()),
                             'human_time_mismatches':int(mismatch.sum()),'unit':'milliseconds','timezone':timezone}
    x['date'] = dt.dt.strftime('%Y-%m-%d')
    return x


def audited_join(left, right, key, label, audits, sums=()):
    if right[key].isna().any() or not right[key].is_unique:
        raise ValueError(f'{label}: dimension key must be nonnull and unique')
    result = left.merge(right, on=key, how='left', validate='many_to_one', indicator=True)
    assert len(result) == len(left)
    assert result.user_id.nunique() == left.user_id.nunique() if 'user_id' in left else True
    for c in sums: assert np.isclose(result[c].sum(),left[c].sum())
    audits.append({'join':label,'input_rows':len(left),'output_rows':len(result),
                   'matched_rows':int(result._merge.eq('both').sum()),'coverage':float(result._merge.eq('both').mean()),
                   'users_before':left.user_id.nunique() if 'user_id' in left else None,
                   'users_after':result.user_id.nunique() if 'user_id' in left else None,
                   'checked_sums':','.join(sums)})
    return result

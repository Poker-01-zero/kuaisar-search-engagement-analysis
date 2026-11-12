"""Core metrics. Observational differences are not treatment effects."""
import numpy as np
import pandas as pd

KEY = ['user_id', 'search_session_id']


def build_sessions(rows, timezone="Asia/Shanghai"):
    """Aggregate complete result rows; quarantine a whole session if invalid."""
    x = rows.copy()
    missing_key = x[KEY].isna().any(axis=1)
    audit = {'input_rows': len(x), 'missing_key_rows': int(missing_key.sum())}
    x = x.loc[~missing_key].copy()
    x['bad_row'] = (~x['click_cnt'].isin([0, 1]) |
                    x['search_source'].isna() |
                    x['search_session_timestamp'].isna())
    x['clicked'] = x['click_cnt'].eq(1).astype('int8')
    x['video'] = x['item_type'].eq('VIDEO').astype('int8')
    x['unknown_type'] = (x['item_type'].isna() | x['item_type'].isin(['UNKNOWN', 'UNKNOW'])).astype('int8')
    grouped = x.groupby(KEY, sort=False, observed=True)
    sessions = grouped.agg(
        timestamp=('search_session_timestamp', 'min'),
        timestamp_max=('search_session_timestamp', 'max'),
        source=('search_source', 'first'),
        source_values=('search_source', 'nunique'),
        result_rows=('clicked', 'size'),
        clicked_rows=('clicked', 'sum'),
        has_click=('clicked', 'max'),
        video_rows=('video', 'sum'),
        unknown_type_rows=('unknown_type', 'sum'),
        bad_rows=('bad_row', 'sum'),
    ).reset_index()
    if 'item_id' in x:
        sessions['distinct_items'] = grouped.item_id.nunique().to_numpy()
    sessions['valid'] = (sessions['bad_rows'].eq(0) &
                         sessions['source_values'].eq(1) &
                         sessions['timestamp'].eq(sessions['timestamp_max']))
    sessions['type_mix'] = np.select(
        [sessions.unknown_type_rows.gt(0),
         sessions.video_rows.eq(sessions.result_rows),
         sessions.video_rows.gt(0)],
        ['unknown', 'video_only', 'video_mixed'], default='non_video')
    dt = pd.to_datetime(sessions.timestamp, unit='ms', utc=True, errors='coerce').dt.tz_convert(timezone)
    sessions['date'] = dt.dt.strftime('%Y-%m-%d')
    sessions['hour'] = dt.dt.hour
    sessions['valid'] &= dt.notna()
    audit.update(total_sessions=len(sessions),
                 invalid_sessions=int((~sessions.valid).sum()),
                 invalid_session_rows=int(sessions.loc[~sessions.valid, 'result_rows'].sum()),
                 source_conflicts=int(sessions.source_values.gt(1).sum()),
                 timestamp_conflicts=int(sessions.timestamp.ne(sessions.timestamp_max).sum()))
    return sessions, audit


def group_metrics(sessions, keys):
    m = sessions.groupby(keys, observed=True, dropna=False).agg(
        users=('user_id', 'nunique'), sessions=('has_click', 'size'),
        clicked_sessions=('has_click', 'sum'), result_rows=('result_rows', 'sum'),
        clicked_rows=('clicked_rows', 'sum'))
    m['session_click_rate'] = m.clicked_sessions / m.sessions
    m['no_click_rate'] = 1 - m.session_click_rate
    m['result_click_ratio'] = m.clicked_rows / m.result_rows
    return m.reset_index()


def source_metrics(sessions, minimum_sessions=100, minimum_users=30):
    m = group_metrics(sessions, ['source']).set_index('source')
    ud = sessions.groupby(['source', 'user_id', 'date'], observed=True).has_click.mean()
    u = sessions.groupby(['source', 'user_id'], observed=True).has_click.mean()
    m['user_day_equal_click_rate'] = ud.groupby(level='source').mean()
    m['user_equal_click_rate'] = u.groupby(level='source').mean()
    m['session_share'] = m.sessions / m.sessions.sum()
    m['display_eligible'] = m.sessions.ge(minimum_sessions) & m.users.ge(minimum_users)
    return m.reset_index().sort_values('sessions', ascending=False)


def playback_fields(frame):
    x = frame.copy()
    x['valid_duration'] = (np.isfinite(x.duration_ms) & x.duration_ms.gt(0) &
                           np.isfinite(x.playing_time) & x.playing_time.ge(0))
    x['play_ratio'] = np.where(x.valid_duration, x.playing_time / x.duration_ms.where(x.duration_ms.gt(0)), np.nan)
    x['capped_play_ratio'] = x.play_ratio.clip(0, 1)
    x['reaches_duration'] = x.play_ratio.ge(1)
    x['exceeds_duration'] = x.play_ratio.gt(1)
    x['duration_band'] = pd.cut(x.duration_ms / 1000, [0, 15, 30, 60, np.inf],
                                labels=['0–15s', '15–30s', '30–60s', '>60s'])
    return x

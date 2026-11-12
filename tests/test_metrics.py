import unittest
import numpy as np
import pandas as pd
from src.metrics import build_sessions, source_metrics, playback_fields


def fixture():
    x = pd.DataFrame([(1,101,'A',0,'VIDEO'),(1,101,'A',0,'VIDEO'),
                      (1,102,'B',1,'VIDEO'),(1,102,'B',0,'USER'),
                      (2,101,'A',1,'VIDEO'),(3,103,'A',0,'USER'),(1,104,'A',0,'VIDEO')],
                     columns=['user_id','search_session_id','search_source','click_cnt','item_type'])
    x['search_session_timestamp'] = 1684800000000
    return x


class MetricTests(unittest.TestCase):
    def test_hand_calculation_and_weighting(self):
        s, audit = build_sessions(fixture())
        self.assertEqual(len(s), 5)
        self.assertEqual(s.has_click.sum(), 2)
        self.assertEqual(audit['invalid_sessions'], 0)
        a = source_metrics(s).set_index('source').loc['A']
        self.assertAlmostEqual(a.session_click_rate, .25)
        self.assertAlmostEqual(a.user_day_equal_click_rate, 1/3)
        self.assertEqual(s.type_mix.value_counts().to_dict(), {'video_only':3,'video_mixed':1,'non_video':1})

    def test_bad_click_quarantines_whole_session(self):
        x = fixture(); x.loc[0,'click_cnt'] = np.nan
        s, audit = build_sessions(x)
        self.assertEqual(audit['invalid_sessions'], 1)
        self.assertEqual(audit['invalid_session_rows'], 2)

    def test_source_conflict(self):
        x = fixture(); x.loc[0,'search_source'] = 'B'
        _, audit = build_sessions(x)
        self.assertEqual(audit['source_conflicts'], 1)

    def test_missing_key_not_grouped(self):
        x = fixture(); x.loc[0,'search_session_id'] = np.nan
        _, audit = build_sessions(x)
        self.assertEqual(audit['missing_key_rows'], 1)

    def test_chunks_reassembled_before_aggregation(self):
        x = fixture()
        actual, _ = build_sessions(pd.concat([x.iloc[:1], x.iloc[1:]], ignore_index=True))
        expected, _ = build_sessions(x)
        pd.testing.assert_frame_equal(actual, expected)

    def test_nonbinary_click_and_time_conflict(self):
        x=fixture();x.loc[0,'click_cnt']=2
        s,audit=build_sessions(x)
        self.assertEqual(audit['invalid_sessions'],1)
        self.assertEqual(audit['invalid_session_rows'],2)
        x=fixture();x.loc[0,'search_session_timestamp']+=1000
        _,audit=build_sessions(x)
        self.assertEqual(audit['timestamp_conflicts'],1)

    def test_unknown_literal_is_not_known_nonvideo(self):
        x=fixture();x.loc[0,'item_type']='UNKNOWN'
        s,_=build_sessions(x)
        self.assertEqual(s.iloc[0].type_mix,'unknown')

    def test_duration_boundaries(self):
        x = playback_fields(pd.DataFrame({'duration_ms':[60000,0,-1,15000,30000],
                                         'playing_time':[120000,1,1,0,30000]}))
        self.assertEqual(x.loc[0,'play_ratio'], 2)
        self.assertEqual(x.loc[0,'capped_play_ratio'], 1)
        self.assertTrue(x.loc[0,'exceeds_duration'])
        self.assertFalse(x.loc[1,'valid_duration'])
        self.assertFalse(x.loc[2,'valid_duration'])
        self.assertEqual(x.loc[3,'duration_band'], '0–15s')

    def test_duplicate_dimension_rejected(self):
        with self.assertRaises(pd.errors.MergeError):
            pd.DataFrame({'item_id':[1]}).merge(pd.DataFrame({'item_id':[1,1]}), validate='many_to_one')


if __name__ == '__main__':
    unittest.main()

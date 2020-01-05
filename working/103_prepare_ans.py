# -*- coding: utf-8 -*-

from util import *

"""
for c in ans.columns:
    t = ans[c].value_counts(1)
    if t.iloc[0]>0.98:
        print(c)
        
is_good
is_rec
is_dest
has_video
reci_xxx
reci_no_help

"""

ans = open_gz('answer_info_0926')
ans.columns = ['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest', 'has_img',
               'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
               'reci_xxx', 'reci_no_help', 'reci_dis']

logging.info("ans %s", ans.shape)
ans = ans.drop_duplicates(['qid', 'uid'])
ans['a_day'] = extract_day(ans['ans_dt'])
ans['a_hour'] = extract_hour(ans['ans_dt'])
del ans['ans_dt']
ans = ans.sort_values(['qid', 'a_day', 'a_hour'])

for col in ['aid', 'is_good', 'is_rec', 'is_dest', 'has_video', 'reci_xxx', 'reci_no_help']:
    del ans[col]

ans = reduce(ans)

to_pkl(ans, 'ans.pkl')

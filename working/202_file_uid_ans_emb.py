# -*- coding: utf-8 -*-
from util import *


# 基于用户回答的内容，生成用户的emb
# 不依赖新数据


# uid_ans_t1_mean_0.h5
# uid_ans_t2_mean_0.h5
def cal_user_ans_emb(param):
    col_name, index = param
    end_day = get_label_end_day(index)
    # (ans['a_day'] > end_day - 37) &
    df = load_ans()
    df = df[(df['a_day'] <= get_feature_end_day(end_day))][['uid', col_name]]
    df = df.groupby('uid')[col_name].agg(lambda x: ','.join(x)).reset_index()
    if col_name == 'ans_t2':
        vec = load_emb_dict('word_vectors_64d')
    elif col_name == 'ans_t1':
        vec = load_emb_dict('single_word_vectors_64d')
    for pooling in get_pooling():
        emb = get_emb(df, 'uid', col_name, vec, pool=pooling)
        dump_h5(emb, f'uid_{col_name}_{pooling}_{index}.h5')


p = []
for x in ['ans_t1', 'ans_t2']:
    for y in get_index():
        p.append((x, y))

logger.info("param %s", p)
go_process(cal_user_ans_emb, p)

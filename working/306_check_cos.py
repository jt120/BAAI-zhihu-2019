# -*- coding: utf-8 -*-
from util import *

# 这里只需要关注，其他的在下面
cos_files = [
    'user_follow_topic_emb_cos',  # 用户关注的主题
    'user_inter_topic_emb_cos',  # 用户感兴趣的主题
]


def check(index):
    # 用户关注和兴趣
    for file in cos_files:
        filename = f'{file}_{index}.h5'
        feature_exists(filename)

    for col_name in ['ans_t1', 'ans_t2']:
        filename = f'cos_{col_name}_mean_{index}.h5'
        feature_exists(filename)

    filename = f'lag_inv_cos_{index}.h5'
    feature_exists(filename)

    # 更多的相似性
    for pos_type in ['title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']:
        for feature_type in ['ans', 'inv', 'invans']:
            for pt in get_pooling():
                user_emb_key = f'uid_{pos_type}_{pt}_{feature_type}_{index}'
                filename = f'cos_{user_emb_key}.h5'
                feature_exists(filename)


for index in get_index():
    check(index)

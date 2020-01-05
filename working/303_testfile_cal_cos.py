# -*- coding: utf-8 -*-
from util import *


# 用户行为和当前问题的cos
# f'../feature/user_ques_title_cos_{index}.h5'
# f'../feature/user_ques_desc_cos_{index}.h5'
# 用户关注topic的cos user_follow_topic_emb_cos_0.h5
# 用户感兴趣topic的cos user_inter_topic_emb_cos_0.h5


def dump_user_ques_cos(user_emb_key, ques_emb, index):
    # for index in [0, 2]:
    name = f'cos_{user_emb_key}.h5'
    if feature_exists(name):
        return None
    user_emb = load_h5(f'{user_emb_key}.h5')
    user_emb.columns = ['key', 'key_emb']
    # M1000000382
    user_emb = user_emb.set_index('key').to_dict()['key_emb']

    label_end_day = get_label_end_day(index)
    data = load_data()

    label = data[(data['day'] > get_feature_end_day(label_end_day)) & (data['day'] <= label_end_day)][['uid', 'qid']]
    label = label[['uid', 'qid']].drop_duplicates()
    df = cal_sim(user_emb, ques_emb, label)
    col_key = f'cos_{user_emb_key}'[:-2]
    df.columns = ['uid', 'qid', col_key]
    logger.info('cos null rate %s', df[col_key].isnull().sum() / len(label))
    # cos_uid_topic_mean_inv.h5
    # ../feature/cos_uid_title_t1_max_ans_2.h5
    dump_h5(df, name)
    logger.info("dump file %s", name)


for pos_type in ['title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']:
    for pt in get_pooling():
        ques_df = load_h5(f'qid_{pos_type}_{pt}.h5')
        ques_df.columns = ['key', 'key_emb']
        ques_df = ques_df.set_index('key').to_dict()['key_emb']
        for index in get_index():
            # user_emb_key, ques_emb, index, data
            with multiprocessing.Pool() as pool:
                pool.starmap(dump_user_ques_cos, [
                    (f'uid_{pos_type}_{pt}_ans_{index}', ques_df, index),
                    (f'uid_{pos_type}_{pt}_inv_{index}', ques_df, index),
                    (f'uid_{pos_type}_{pt}_invans_{index}', ques_df, index),
                ])

# -*- coding: utf-8 -*-
from util import *


# 基于用户的行为，和用户相关的问题，生成user emb

# uid_desc_t1_mean_ans_0.h5
# uid_desc_t1_mean_ans_2.h5
# uid_desc_t1_mean_inv_0.h5
# uid_desc_t1_mean_inv_2.h5
# uid_desc_t1_mean_invans_0.h5
# uid_desc_t1_mean_invans_2.h5
# uid_desc_t2_mean_ans_0.h5
# uid_desc_t2_mean_ans_2.h5
# uid_desc_t2_mean_inv_0.h5
# uid_desc_t2_mean_inv_2.h5
# uid_desc_t2_mean_invans_0.h5
# uid_desc_t2_mean_invans_2.h5
# uid_title_t1_mean_ans_0.h5
# uid_title_t1_mean_ans_2.h5
# uid_title_t1_mean_inv_0.h5
# uid_title_t1_mean_inv_2.h5
# uid_title_t1_mean_invans_0.h5
# uid_title_t1_mean_invans_2.h5
# uid_title_t2_mean_ans_0.h5
# uid_title_t2_mean_ans_2.h5
# uid_title_t2_mean_inv_0.h5
# uid_title_t2_mean_inv_2.h5
# uid_title_t2_mean_invans_0.h5
# uid_title_t2_mean_invans_2.h5
# uid_topic_mean_ans_0.h5
# uid_topic_mean_ans_2.h5
# uid_topic_mean_inv_0.h5
# uid_topic_mean_inv_2.h5
# uid_topic_mean_invans_0.h5
# uid_topic_mean_invans_2.h5
def process_user_1(df, vec, index, feature_type, pos_type):
    """

    :param df:
    :param vec:
    :param index: 0 2
    :param feature_type: ans inv invans
    :param vec_type: single word topic
    :return:
    """
    pos_list = get_pos_list(pos_type)
    for pos in pos_list:
        t = group_related(df, pos)
        pooling = get_pooling()
        logger.info("pooling %s", pooling)
        for pool in pooling:
            cal1(t, pos, pool, feature_type, vec, index)


def cal1(df, pos, pool, feature_type, vec, index):
    filename = f'uid_{pos}_{pool}_{feature_type}_{index}.h5'
    if feature_exists(filename):
        logger.info("exist %s", filename)
        return
    emb = get_emb(df, 'uid', pos, vec, pool=pool)
    dump_h5(emb, filename)


def group_related(df, key):
    df = df[df[key] != '-1']
    t = df.groupby('uid')[key].agg(lambda x: ','.join(x)).reset_index()
    t[key] = t[key].map(lambda x: ','.join([a for a in x.split(',') if a != '-1']))
    return t


# 用户字的emb
def process_user(param):
    """
    :param index: 0, 2
    :param vec:
    :param vec_type: single, word, topic
    :return:
    """
    data = load_data()
    ans = load_ans()
    ques = load_org_ques()

    ans = pd.merge(ans[['uid', 'qid', 'a_day']], ques, on='qid')
    data = pd.merge(data[['uid', 'qid', 'day', 'label']], ques, on='qid')

    index, vec_type = param
    end_day = get_label_end_day(index)

    if vec_type == pos_type_sing:
        vec = load_emb_dict('single_word_vectors_64d')
    elif vec_type == pos_type_word:
        vec = load_emb_dict('word_vectors_64d')
    elif vec_type == pos_type_topic:
        vec = load_emb_dict('topic_vectors_64d')
    # 回答的emb
    if 1:
        df = ans[ans['a_day'] <= get_feature_end_day(end_day)]
        # df = pd.merge(df, ques, on='qid')
        process_user_1(df, vec, index, 'ans', vec_type)

    # 邀请回答
    if 1:
        df = data[(data['day'] <= get_feature_end_day(end_day)) & (data['label'] == 1)]
        # df = pd.merge(df, ques, on='qid')
        process_user_1(df, vec, index, 'invans', vec_type)


# 字的emb
# with multiprocessing.Pool() as pool:
#     single_vec_dict, word_vec_dict, topic_vec_dict = pool.map(load_emb_dict,
#                                                               ['single_word_vectors_64d', 'word_vectors_64d',
#                                                                'topic_vectors_64d'])
# = load_emb_dict('')
# = load_emb_dict('')
# = load_emb_dict('')

# 用户问题，字emb
if 1:
    req = []
    for index in get_index():
        req.append((index, pos_type_sing))
        req.append((index, pos_type_word))
        req.append((index, pos_type_topic))

    with multiprocessing.Pool() as pool:
        pool.map(process_user, req)

#

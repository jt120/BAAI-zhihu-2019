# -*- coding: utf-8 -*-
from util import *
import argparse
import gc


# last n3
def cal_1(label, key='uid', diff_key='diff_day_inv_ques'):
    t = label[[key, diff_key]].copy()
    t['flag'] = 1
    a = t.groupby([key, diff_key])['flag'].sum().reset_index()
    a.columns = [key, diff_key, 'each_count']
    b = t.groupby(key)[diff_key].nunique().reset_index()
    b.columns = [key, 'diff_day_nunq']
    a = pd.merge(a, b, on=key, how='left')
    a['choose'] = np.where(a['each_count'] > 1, 1, 0)
    a['key'] = a.groupby(key)['choose'].transform('sum') / a['diff_day_nunq']
    a = a[[key, 'key']].drop_duplicates(key)
    return a


def cal_2(label):
    a = cal_1(label, 'uid', 'diff_day_inv_ques')
    a = a.set_index('uid').to_dict()['key']
    label['uid_activity_prda'] = label['uid'].map(a)

    a = cal_1(label, 'qid', 'diff_day_inv_ques')
    a = a.set_index('qid').to_dict()['key']
    label['qid_activity_prda'] = label['qid'].map(a)
    return label


def merge_emb(index):
    label_name = config.label_name
    filename = f'{label_name}2_{index}.h5'

    label = load_h5(filename)
    logger.info("label %s shape %s", index, label.shape)
    label = cal_2(label)
    label = extract_label_feature(label)

    t = load_h5(f'uid_ans_t2_mean_{index}.h5')
    for i in range(64):
        t['uid_ans_emb_' + str(i)] = t['ans_t2_emb'].map(lambda x: np.float32(x[i]))
    del t['ans_t2_emb']
    label = pd.merge(label, t, on='uid', how='left')

    ques_emb = load_h5('qid_topic_mean.h5')
    user_inter_emb = load_h5('uid_inter_topic_mean.h5')
    user_follow_emb = load_h5('uid_follow_topic_mean.h5')
    for i in range(64):
        ques_emb['ques_emb_' + str(i)] = ques_emb['topic_emb'].map(lambda x: np.float32(x[i]))
        user_inter_emb['user_int_' + str(i)] = user_inter_emb['inter_topic_emb'].map(lambda x: np.float32(x[i]))
        user_follow_emb['user_fol_' + str(i)] = user_follow_emb['follow_topic_emb'].map(lambda x: np.float32(x[i]))
    del ques_emb['topic_emb'], user_inter_emb['inter_topic_emb'], user_follow_emb['follow_topic_emb']

    label = pd.merge(label, ques_emb, on='qid', how='left')
    label = pd.merge(label, user_inter_emb, on='uid', how='left')
    label = pd.merge(label, user_follow_emb, on='uid', how='left')
    del ques_emb, user_inter_emb, user_follow_emb
    gc.collect()
    # label = reduce(label)
    label_name = config.label_name
    filename = f'{label_name}3_{index}.h5'
    dump_h5(label, filename)


go_process(merge_emb, get_index())

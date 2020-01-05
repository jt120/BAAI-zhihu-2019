# -*- coding: utf-8 -*-
from util import *


# 近3次邀请和当前邀请的cos
# 2019-12-10 14:24:34
# 特征 'recent_q0_topic', 'recent_q1_topic', 'recent_q2_topic', 'recent_q0_title', 'recent_q1_title', 'recent_q2_title',
# 'recent_q0_desc', 'recent_q1_desc', 'recent_q2_desc'
# 文件 lag_inv_cos_0.h5, lag_inv_cos_2.h5


def to_dict(df):
    df.columns = ['key', 'key_emb']
    return df.set_index('key').to_dict()['key_emb']


def cal_cos(qid, recent, vec_dict):
    if pd.notnull(recent):
        v1 = vec_dict.get(qid, np.nan)
        v2 = vec_dict.get(recent, np.nan)
        if isinstance(v1, np.float) or isinstance(v2, np.float):
            return np.nan
        else:
            score = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)).item()
        return score
    return np.nan


def cal1(label, vec):
    vec = load_h5(f'{vec}.h5')
    vec = to_dict(vec)

    ret = []
    label = label.drop_duplicates()
    for row in tqdm(label.itertuples()):
        qid = row[1]
        recent_qid = row[2]
        s = cal_cos(qid, recent_qid, vec)
        ret.append((qid, recent_qid, s))
    ret = pd.DataFrame(ret)
    return ret


# @dump_feature
def cal_sim(label):
    with multiprocessing.Pool() as pool:
        a1, b1, c1 = pool.starmap(cal1, [
            (label[['qid', 'qid_1']], 'qid_topic_mean'),
            (label[['qid', 'qid_2']], 'qid_topic_mean'),
            (label[['qid', 'qid_-1']], 'qid_topic_mean'),
        ])

    a1.columns = ['qid', 'qid_1', 'recent_q0_topic']
    label = pd.merge(label, a1, on=['qid', 'qid_1'], how='left')

    b1.columns = ['qid', 'qid_2', 'recent_q1_topic']
    label = pd.merge(label, b1, on=['qid', 'qid_2'], how='left')

    c1.columns = ['qid', 'qid_-1', 'recent_q2_topic']
    label = pd.merge(label, c1, on=['qid', 'qid_-1'], how='left')
    del a1, b1, c1

    with multiprocessing.Pool() as pool:
        a2, b2, c2, = pool.starmap(cal1, [
            (label[['qid', 'qid_1']], 'qid_title_t2_mean'),
            (label[['qid', 'qid_2']], 'qid_title_t2_mean'),
            (label[['qid', 'qid_-1']], 'qid_title_t2_mean'),
        ])

    # 没用
    a2.columns = ['qid', 'qid_1', 'recent_q0_title']
    label = pd.merge(label, a2, on=['qid', 'qid_1'], how='left')

    b2.columns = ['qid', 'qid_2', 'recent_q1_title']
    label = pd.merge(label, b2, on=['qid', 'qid_2'], how='left')

    c2.columns = ['qid', 'qid_-1', 'recent_q2_title']
    label = pd.merge(label, c2, on=['qid', 'qid_-1'], how='left')

    del a2, b2, c2

    with multiprocessing.Pool() as pool:
        a3, b3, c3 = pool.starmap(cal1, [
            (label[['qid', 'qid_1']], 'qid_desc_t2_mean'),
            (label[['qid', 'qid_2']], 'qid_desc_t2_mean'),
            (label[['qid', 'qid_-1']], 'qid_desc_t2_mean'),
        ])

    a3.columns = ['qid', 'qid_1', 'recent_q0_desc']
    label = pd.merge(label, a3, on=['qid', 'qid_1'], how='left')

    b3.columns = ['qid', 'qid_2', 'recent_q1_desc']
    label = pd.merge(label, b3, on=['qid', 'qid_2'], how='left')
    c3.columns = ['qid', 'qid_-1', 'recent_q2_desc']
    label = pd.merge(label, c3, on=['qid', 'qid_-1'], how='left')

    del a3, b3, c3

    return label


def shift(data, key, offset):
    data[f'qid_{offset}'] = data[key].shift(offset)
    data.loc[data['uid'] != data['uid'].shift(offset), f'qid_{offset}'] = np.nan
    return data


def process1(index):
    label_end_day = get_label_end_day(index)

    filename = f'lag_inv_cos_{index}.h5'

    if feature_exists(filename):
        logger.info("file exists %s", filename)
        return
    data = load_data()

    # data = load_pkl('data.pkl')
    data = data.sort_values(['uid', 'day', 'hour'], ascending=False)

    data = shift(data, 'qid', 1)
    data = shift(data, 'qid', -1)
    data = shift(data, 'qid', 2)

    label = data[(data['day'] > get_feature_end_day(label_end_day)) & (data['day'] <= label_end_day)][
        ['index', 'uid', 'qid', 'qid_1', 'qid_2', 'qid_-1']]

    label = cal_sim(label)
    for col in ['uid', 'qid', 'qid_1', 'qid_2', 'qid_-1']:
        del label[col]
    dump_h5(label, filename)


if 1:
    for index in get_index():
        process1(index)

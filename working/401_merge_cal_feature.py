# -*- coding: utf-8 -*-
from util import *

"""
这里都是计算的特征
"""


def cal_union_count(row, uid_topic_dict):
    uid = row['uid']
    topic = row['topic']
    t2 = uid_topic_dict.get(uid, set())
    return len(topic & t2)


def to_dict(df):
    return df.set_index('uid').to_dict()['topic']


def group_topic(df):
    uid_topic_dict = df.groupby(['uid'])['topic'].agg(
        lambda x: set(itertools.chain.from_iterable([a for a in x]))).reset_index()
    uid_topic_dict = to_dict(uid_topic_dict)
    return uid_topic_dict


def cal_topic_feature(index):
    label_end_day = get_label_end_day(index)
    feature_end_day = get_feature_end_day(label_end_day)
    topic_inv = data[(data['day'] <= label_end_day)][['topic']]
    topic_inv_ans = data[(data['day'] <= feature_end_day)][['label', 'topic']]
    topic_ans = ans[ans['a_day'] <= feature_end_day][['topic']]

    # 邀请量
    topic_inv_dict = collections.defaultdict(int)
    topic_inv_ans_inv_dict = collections.defaultdict(int)
    topic_inv_ans_ans_dict = collections.defaultdict(int)
    topic_ans_dict = collections.defaultdict(int)

    for r in tqdm(topic_inv.itertuples()):
        topic = r[1]
        for t in topic:
            topic_inv_dict[t] += 1

    topic_inv = dict2df(topic_inv_dict)
    topic_inv.columns = ['topic', 'topic_label_inv_count']
    logger.info("topic inv %s", topic_inv.head())

    for r in tqdm(topic_inv_ans.itertuples()):
        label = r[1]
        topic = r[2]
        for t in topic:
            topic_inv_ans_inv_dict[t] += 1
            if label == 1:
                topic_inv_ans_ans_dict[t] += 1

    topic_inv_ans1 = dict2df(topic_inv_ans_inv_dict)
    topic_inv_ans1.columns = ['topic', 'topic_inv_ans_inv_count']
    topic_inv_ans2 = dict2df(topic_inv_ans_ans_dict)
    topic_inv_ans2.columns = ['topic', 'topic_inv_ans_ans_count']
    df = pd.merge(topic_inv_ans1, topic_inv_ans2, on='topic', how='left')
    logger.info("topic inv ans %s", df.head())

    for r in tqdm(topic_ans.itertuples()):
        topic = r[1]
        for t in topic:
            topic_ans_dict[t] += 1

    topic_ans = dict2df(topic_ans_dict)
    topic_ans.columns = ['topic', 'topic_ans_count']
    logger.info("topic_ans %s", topic_ans.head())

    df = pd.merge(df, topic_inv, on='topic', how='left')
    df = pd.merge(df, topic_ans, on='topic', how='left')
    df.fillna(0, inplace=True)
    df['topic_inv_ans_ans_rate'] = (df['topic_inv_ans_ans_count'] + 1) / (df['topic_inv_ans_inv_count'] + 10)
    logger.info("final topic %s", df.head())
    df = reduce(df)
    return df


def flatten_qid_topic(target):
    ret = []
    df = target[['qid', 'topic']].drop_duplicates(['qid'])
    for row in tqdm(df.itertuples()):
        kid = row[1]
        topic = row[2]
        for t in topic:
            ret.append((kid, t))

    df = pd.DataFrame.from_records(ret)
    df.columns = ['qid', 'topic']
    return df


def flatten_uid_qid_topic(target):
    ret = []
    df = target[['uid', 'qid', 'topic']].drop_duplicates(['uid', 'qid'])
    for row in tqdm(df.itertuples()):
        uid = row[1]
        qid = row[2]
        topic = row[3]
        for t in topic:
            ret.append((uid, qid, t))

    df = pd.DataFrame.from_records(ret)
    df.columns = ['uid', 'qid', 'topic']
    return df


def cal_user_topic_feature(feature_end):
    feature = ans[(ans['a_day'] <= feature_end)]
    user_topic_cheer = collections.defaultdict(int)
    user_topic_word = collections.defaultdict(int)

    for row in tqdm(feature[['uid', 'reci_cheer', 'topic', 'word_count']].itertuples()):
        uid = row[1]
        reci_cheer = row[2]
        topic = row[3]
        word_count = row[4]

        for tp in topic:
            if topic == '-1':
                continue
            user_topic_cheer[(uid, tp)] += reci_cheer
            user_topic_word[(uid, tp)] += word_count

    df = dict2df(user_topic_cheer)
    df.columns = ['uid_topic', 'user_topic_cheer_count']
    df['uid'] = df['uid_topic'].map(lambda x: x[0])
    df['topic'] = df['uid_topic'].map(lambda x: x[1])
    del df['uid_topic']

    df1 = dict2df(user_topic_cheer)
    df1.columns = ['uid_topic', 'user_topic_word_count']
    df1['uid'] = df1['uid_topic'].map(lambda x: x[0])
    df1['topic'] = df1['uid_topic'].map(lambda x: x[1])
    del df1['uid_topic']

    df = pd.merge(df, df1, on=['uid', 'topic'], how='left')

    return df


# new 从dump迁移过来
def merge_topic_feature(target, index):
    topic_feature = cal_topic_feature(index)
    df = flatten_qid_topic(target)
    df = pd.merge(df, topic_feature, on='topic', how='left')

    target = get_merge_feature(target, df, 'qid', 'topic_inv_ans_inv_count', ['sum', 'mean', 'max', 'std'])
    target = get_merge_feature(target, df, 'qid', 'topic_inv_ans_ans_count', ['sum', 'mean', 'max', 'std'])
    target = get_merge_feature(target, df, 'qid', 'topic_label_inv_count', ['sum', 'mean', 'max', 'std'])
    target = get_merge_feature(target, df, 'qid', 'topic_ans_count', ['sum', 'mean', 'max', 'std'])
    target = get_merge_feature(target, df, 'qid', 'topic_inv_ans_ans_rate', ['mean', 'max', 'std'])

    # 用户和话题

    # label下uid qid topic
    df = flatten_uid_qid_topic(target)

    label_end_day = get_label_end_day(index)
    feature_end_day = get_feature_end_day(label_end_day)

    # uid topic
    feature = cal_user_topic_feature(feature_end_day)
    df = pd.merge(df, feature, on=['uid', 'topic'], how='left')
    target = get_merge_feature2(target, df, 'uid', 'qid', 'user_topic_cheer_count', ['max', 'mean', 'std', 'sum'])
    # user在这个主题下的字数
    target = get_merge_feature2(target, df, 'uid', 'qid', 'user_topic_word_count', ['max', 'mean', 'std', 'sum'])

    feature = data[(data['day'] <= feature_end_day) & (data['label'] == 1)]
    uid_topic_dict = group_topic(feature)
    target['topic_union_label_ans_count'] = target[['uid', 'topic']].apply(lambda x: cal_union_count(x, uid_topic_dict),
                                                                           axis=1)

    feature = ans[ans['a_day'] <= feature_end_day]
    uid_topic_dict = group_topic(feature)
    target['topic_union_ans_count'] = target[['uid', 'topic']].apply(lambda x: cal_union_count(x, uid_topic_dict),
                                                                     axis=1)

    feature = data[data['day'] <= label_end_day]
    uid_topic_dict = group_topic(feature)
    target['topic_union_label_inv_count'] = target[['uid', 'topic']].apply(lambda x: cal_union_count(x, uid_topic_dict),
                                                                           axis=1)

    t = load_h5('topic_new_1.h5')
    bins = [-1, 4.0, 7.0, 9.0, 12.0, 15.0, 18.0, 22.0, 28.0, 35.0, 44.0, 57.0, 75.0, 107.0, 158.0, 244.0, 385.0, 626.0,
            1124.0, 2457.0, 24191.0]
    ques = load_org_ques()[['qid', 'topic']]
    ques.index = ques['qid'].values
    ques = ques['topic'].to_dict()

    label_day = get_label_end_day(index)
    feature_day = get_feature_end_day(label_day)
    target['topic'] = target['qid'].map(ques)
    target = get_merge_feature(target, t, 'topic', 'qid', ['count'])
    target['topic_qid_w_count_bin'] = pd.cut(target['topic_qid_w_count'], bins=bins).cat.codes
    feature = t[t['day'] <= feature_day]
    target = get_merge_feature(target, feature, 'topic_qid_w_count_bin', 'label', ['mean', 'sum', 'std'])

    return target


# 'uid_unq_day', 用户邀请day unq count
# 'gap_min_day', 用户当前距label最小的天数
# 'gap_max_day'  用户当前距label最大的天数差
#
# 邀请特征
def merge_inv_feature(target, df, index):
    label_end_day = get_label_end_day(index)
    feature_end = get_feature_end_day(label_end_day)
    # 邀请特征
    feature = df[(df['day'] <= feature_end)].copy()

    # 时间特征
    target = get_merge_feature(target, feature, 'hour_bin', 'label', ['mean', 'sum'])
    target = get_merge_feature(target, feature, 'weekend_hour_bin', 'label', ['mean', 'sum'])
    target = get_merge_feature2(target, feature, 'uid', 'hour_bin', 'label', ['count', 'sum', 'mean'])
    target = get_merge_feature2(target, feature, 'uid', 'weekend_hour_bin', 'label', ['count', 'sum', 'mean'])

    # 2019-12-16 18:15:11 新加uf_b_new
    target = get_merge_feature(target, feature, 'uf_b_new', 'label', ['mean', 'sum'])

    for feat in ['gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4']:
        target = get_merge_feature2(target, feature, 'qid', feat, 'label', ['count', 'sum', 'mean', 'std'])
    # todo 这些用户属性，对话题
    # uid day lag
    target = merge_lag(target, feature, 'uid', 'day', sub='label')
    target = merge_lag(target, feature, 'qid', 'day', sub='label')
    target = get_merge_feature(target, feature, 'uid', 'diff_day_inv_ques', ['max', 'mean', 'std'])

    target = get_merge_feature(target, feature, 'score_bin', 'label', agg=['mean', 'std', 'count', 'sum'])

    # 邀请回答时间差
    t = ans[ans['a_day'] <= feature_end][['uid', 'qid', 'a_day']]
    feature = feature.merge(t, on=['uid', 'qid'], how='left')
    feature['diff_day_inv_ans'] = feature['a_day'] - feature['day']
    feature.loc[feature['diff_day_inv_ans'] < 0, 'diff_day_inv_ans'] = 0
    target = get_merge_feature(target, feature, 'uid', 'diff_day_inv_ans', ['mean', 'std', 'max'])

    # feature = feature.merge(user, on='uid', how='left')
    logger.info("extract inv feature start %s, end %s", feature['day'].min(), feature['day'].max())
    # 直接提取的时间足够快，那么为什么要先保存文件？
    target = get_merge_feature(target, feature, 'freq', 'label', ['count'])
    target = get_merge_feature(target, feature, 'uf_c1', 'label', ['count', 'sum', 'mean', 'std'])
    target = get_merge_feature(target, feature, 'uf_c2', 'label', ['count', 'mean'])
    target = get_merge_feature(target, feature, 'uf_c3', 'label', ['count', 'sum', 'mean'])
    target = get_merge_feature(target, feature, 'uf_c4', 'label', ['count', 'sum', 'mean'])
    target = get_merge_feature(target, feature, 'weekday_hour', 'label', ['count', 'sum', 'mean', 'std'])

    target = get_unq_count(target, feature, 'uid', 'day')
    target = get_unq_count(target, feature, 'uid', 'weekday_hour')
    target = get_unq_count(target, feature, 'qid', 'day')
    target = get_diff_day(target, feature, 'day')

    # label 1
    # uid_uid_label_diff_lag1_day_mean_wlabel
    feature = df[(df['day'] <= feature_end) & (df['label'] == 1)].copy()
    target = merge_lag(target, feature, 'uid', 'day', sub='label1')
    target = merge_lag(target, feature, 'qid', 'day', sub='label1')

    # 7天 label 1
    feature_end2 = get_feature_end_day(feature_end)
    feature = df[(df['day'] > feature_end2) & (df['day'] <= feature_end) & (df['label'] == 1)].copy()
    target = get_diff_day(target, feature, sub='label1_7d')
    # uid_day_wlabel1_7d_max
    # uid_day_wlabel1_7d_min
    target = merge_lag(target, feature, 'uid', 'day', sub='label1_7d')
    target = merge_lag(target, feature, 'qid', 'day', sub='label1_7d')

    feature = data[(data['day'] > feature_end) & (data['day'] <= label_end_day)]
    target = get_merge_feature2(target, feature, 'uid', 'hour_bin', 'label', ['count'])
    target = get_merge_feature2(target, feature, 'uid', 'weekend_hour_bin', 'label', ['count'])

    return target


def merge_rank_feature(target, index):
    label_end_day = get_label_end_day(index)
    feature_day = get_feature_end_day(label_end_day)
    df = data[(data['day'] > feature_day) & (data['day'] <= label_end_day)]
    df = df.sort_values(['uid', 'day', 'hour'])
    df['week'] = df['day'] // 7

    df['day_rank'] = 1
    df['day_rank'] = df.groupby(['uid', 'day'])['day_rank'].cumsum()
    t = df[['index', 'day_rank']]
    t.index = df['index'].values
    t = t['day_rank'].to_dict()
    target['day_rank'] = target['index'].map(t)

    df['week_rank'] = 1
    df['week_rank'] = df.groupby(['uid', 'week'])['week_rank'].cumsum()
    t = df[['index', 'week_rank']]
    t.index = df['index'].values
    t = t['week_rank'].to_dict()
    target['week_rank'] = target['index'].map(t)

    df = load_h5('new_topic_fea.h5')
    target = get_merge_feature(target, df, 'qid', 'topic_rel_size', ['mean', 'max', 'std', 'min'])
    target = get_merge_feature(target, df, 'qid', 'topic_qid_size', ['mean', 'max', 'std', 'min'])
    return target


# 回答特征
def merge_ans_feature(target, feature_end):
    """
    :param target: label
    :param ans: ans
    :param feature_end:
    :return:
    """
    feature = ans[(ans['a_day'] <= feature_end)].copy()
    feature['ans'] = 1
    # ans lag
    # 用户每次回答距上一次的天数
    target = merge_lag(target, feature, 'uid', 'a_day', sub='ans')
    target = merge_lag(target, feature, 'qid', 'a_day', sub='ans')

    # 用户第一次回答和最后一次回答
    target = get_diff_day(target, feature, dt_col='a_day', sub='ans')

    key = 'hour'
    # 24小时中每小时的ans量
    target = get_merge_feature(target, feature, key, 'ans', ['count'])

    key = 'weekday'
    # 7天内，每天的ans量
    target = get_merge_feature(target, feature, key, 'ans', ['count'])

    key = 'weekday_hour'
    # weekday x hour count
    target = get_merge_feature(target, feature, key, 'ans', ['count'])

    # weekday x uid count
    target['weekday_hour_uid'] = make_key(target['weekday_hour'], target['uid'])
    feature['weekday_hour_uid'] = make_key(feature['weekday_hour'], feature['uid'])
    target = get_merge_feature(target, feature, 'weekday_hour_uid', 'ans', ['count'])
    del target['weekday_hour_uid']

    target['hour_uid'] = make_key(target['hour'], target['uid'])
    feature['hour_uid'] = make_key(feature['hour'], feature['uid'])
    # uid x hour count
    target = get_merge_feature(target, feature, 'hour_uid', 'ans', ['count'])
    del target['hour_uid']

    # weekday x uid count
    target['weekday_uid'] = make_key(target['weekday'], target['uid'])
    feature['weekday_uid'] = make_key(feature['weekday'], feature['uid'])
    target = get_merge_feature(target, feature, 'weekday_uid', 'ans', ['count'])
    del target['weekday_uid']

    # # uid回答多少问题
    key = 'uid'
    target = get_merge_feature(target, feature, key, 'ans', ['count'])
    target = get_merge_feature(target, feature, key, 'ans_t1_count', ['mean', 'sum', 'max', 'std'])
    target = get_merge_feature(target, feature, key, 'ans_t2_count', ['mean', 'sum', 'max', 'std'])
    target = get_merge_feature(target, feature, key, 'has_img', ['mean', 'sum'])
    target = get_merge_feature(target, feature, key, 'word_count', ['mean', 'sum', 'max'])
    target = get_merge_feature(target, feature, key, 'reci_cheer', ['mean', 'sum', 'max', 'std'])
    target = get_merge_feature(target, feature, key, 'reci_mark', ['mean', 'sum', 'max', 'std'])
    target = get_merge_feature(target, feature, key, 'diff_day_ques_ans', ['mean', 'sum', 'max', 'std'])
    target = get_merge_feature(target, feature, key, 'reci_tks', ['mean', 'sum'])
    target = get_merge_feature(target, feature, key, 'reci_uncheer', ['mean', 'sum'])
    target = get_merge_feature(target, feature, key, 'reci_comment', ['mean', 'sum', 'max', 'std'])

    # 近2天的特征
    feature_win = feature[feature['a_day'] >= feature_end - 2]
    target = get_merge_feature(target, feature_win, key, 'reci_cheer', ['mean', 'sum'], sub=2)

    feature_win = feature[feature['a_day'] >= feature_end - 2]
    target = get_merge_feature(target, feature_win, key, 'reci_comment', ['mean', 'sum'], sub=2)

    feature_win = feature[feature['a_day'] >= feature_end - 2]
    target = get_merge_feature(target, feature_win, key, 'word_count', ['mean', 'sum', 'max'], sub=1)

    feature_win = feature[feature['a_day'] >= feature_end - 2]
    target = get_merge_feature(target, feature_win, key, 'word_count', ['mean', 'sum', 'max'], sub=2)

    # # qid被多少用户回答
    key = 'qid'
    target = get_merge_feature(target, feature, key, 'ans', ['count'])
    target = get_merge_feature(target, feature, key, 'ans_t2_count', ['mean', 'sum', 'max', 'std'])
    target = get_merge_feature(target, feature, key, 'has_img', ['mean'])
    target = get_merge_feature(target, feature, key, 'word_count', ['mean', 'sum', 'max'])
    target = get_merge_feature(target, feature, key, 'diff_day_ques_ans', ['mean', 'std', 'sum', 'max'])
    target = get_merge_feature(target, feature, key, 'reci_tks', ['mean'])
    target = get_merge_feature(target, feature, key, 'reci_mark', ['mean'])
    target = get_merge_feature(target, feature, key, 'reci_comment', ['mean', 'sum'])
    target = get_merge_feature(target, feature, key, 'reci_cheer', ['mean', 'sum'])
    target = get_merge_feature(target, feature, key, 'reci_uncheer', ['mean'])

    # 加窗口
    feature_win = feature[feature['a_day'] >= feature_end - 1]
    target = get_merge_feature(target, feature_win, key, 'word_count', ['mean', 'sum', 'max'], sub=1)

    feature_win = feature[feature['a_day'] >= feature_end - 2]
    target = get_merge_feature(target, feature_win, key, 'word_count', ['mean', 'sum', 'max'], sub=2)

    # 上一次的字数，点赞数，感谢数#

    feature = feature[['uid', 'qid', 'a_day', 'reci_cheer', 'reci_tks', 'word_count']]
    t = feature.groupby(['uid', 'a_day'])['reci_cheer', 'reci_tks', 'word_count'].max().reset_index()
    t = t.sort_values(['uid', 'a_day'])

    for col in ['a_day', 'word_count', 'reci_cheer', 'reci_tks']:
        t[f'lag_{col}'] = t[col].shift(1)
        t.loc[t['uid'] != t['uid'].shift(1), f'lag_{col}'] = None

    t = t.drop_duplicates(['uid'], keep='last')
    target = merge_dict(target, t, 'uid')
    target = cal_ans_lag_feature(target)

    # todo 此处可以加入qid的lag

    return target


#  标签特征
def merge_label_feature(df):
    org_size = len(df)

    # uid时间差
    df = cal_time_gaps(df, 'uid', 'day', sub='label')
    df = cal_time_gaps(df, 'qid', 'day', sub='label')

    # label侧每天的邀请量
    df['label_inv'] = 1
    feature_df = df
    df = get_merge_feature(df, feature_df, 'hour', 'label_inv', ['count'], scale=True)

    assert org_size == len(df)
    # 时间+用户
    df['weekday_uid'] = df['weekday'].astype(str) + df['uid']
    feature_df['weekday_uid'] = feature_df['weekday'].astype(str) + feature_df['uid']
    df = get_merge_feature(df, feature_df, 'weekday_uid', 'label_inv', ['count'], scale=True)
    del df['weekday_uid']

    df['hour_uid'] = df['hour'].astype(str) + df['uid']
    feature_df['hour_uid'] = feature_df['hour'].astype(str) + feature_df['uid']
    df = get_merge_feature(df, feature_df, 'hour_uid', 'label_inv', ['count'], scale=True)
    del df['hour_uid']

    df['weekday_hour_uid'] = df['weekday_hour'].astype(str) + df['uid']
    feature_df['weekday_hour_uid'] = feature_df['weekday_hour'].astype(str) + feature_df['uid']
    df = get_merge_feature(df, feature_df, 'weekday_hour_uid', 'label_inv', ['count'], scale=True)
    del df['weekday_hour_uid']

    del df['label_inv']

    # unq day
    df = get_unq_count(df, feature_df, 'uid', 'day', 'x')
    df = get_unq_count(df, feature_df, 'qid', 'day', 'x')
    # 距上一次邀请，距下一次邀请
    df = get_diff_day(df, feature_df, 'day', sub='x')

    # todo 是否应该按照q_day排序
    key = 'uid'
    feature_df = df[[key, 'q_day', 'q_hour']].copy()
    t = get_inv_rank(feature_df, key)
    t = t.set_index('key').to_dict()[f'{key}_inv_rank']
    df['key'] = df[key].astype(str) + df['q_day'].astype(str) + df['q_hour'].astype(str)
    df[f'inv_{key}_rank'] = df['key'].map(t)
    del df['key']

    df = get_merge_feature(df, df[['uid', 'diff_day_inv_ques']], 'uid', 'diff_day_inv_ques',
                           ['max', 'mean', 'std', 'sum'], sub='label')

    return df


def get_inv_rank(a, key):
    a = a.sort_values([key, 'q_day', 'q_hour'])
    a['inv_rank'] = 1
    a['inv_rank'] = a.groupby([key])['inv_rank'].cumsum()
    a['key'] = a[key].astype(str) + a['q_day'].astype(str) + a['q_hour'].astype(str)
    a['inv_rank'] = a['inv_rank'] / a.groupby([key])['inv_rank'].transform('max')
    a = a[['key', 'inv_rank']]
    a.columns = ['key', f'{key}_inv_rank']
    return a


# del user

# {'uid_uid_label_diff_lag1_day_max',
#  'uid_uid_label_diff_lag1_day_mean',
#  'uid_uid_label_diff_lag1_day_std',
#  'uid_uid_label_diff_lag1_day_sum'}
def merge_lag(target, feature, key, dt_col, sub=''):
    feature = feature.sort_values([key, dt_col])
    df = cal_lag_feature(feature, dt_col, key, agg=['mean', 'max', 'std', 'sum'], lag=1, scale=False, sub=sub)
    target = pd.merge(target, df, on=key, how='left')
    return target


def cal_feature(index):
    logger.info('merge index %s', index)
    label_end_day = get_label_end_day(index)
    # 标签
    feature_end_day = get_feature_end_day(label_end_day)
    target = data[(data['day'] > feature_end_day) & (data['day'] <= label_end_day)]
    org_size = len(target)
    logger.info("label shape %s, start %s, end %s", target.shape, target['day'].min(), target['day'].max())

    # 基本特征
    # target = target.merge(user, on='uid', how='left', copy=False)
    # target = target.merge(ques, on='qid', how='left', copy=False)
    # 邀请和问题的时间差

    assert org_size == len(target)

    target['qid_enc'] = q_lb.transform(target['qid'])
    target['uid_enc'] = u_lb.transform(target['uid'])

    target = merge_rank_feature(target, index)

    # merge topic特征
    target = merge_topic_feature(target, index)

    # 回答特征
    target = merge_ans_feature(target, feature_end_day)

    # 标签特征
    target = merge_label_feature(target)
    assert org_size == len(target)

    # 邀请特征
    target = merge_inv_feature(target, data, index)
    assert org_size == len(target)

    """
    diff_first_inv_day                   537.0
    diff_last_inv_day                    533.5
    diff_first_last_inv_day              590.5
    """
    # 用户距第一次和最后一次邀请的时间差
    # t1 = feature.groupby(['uid'])['day'].first().to_dict()
    # t2 = feature.groupby(['uid'])['day'].last().to_dict()
    # target = cal_diff_inv(target, t1, t2)

    # 用过去全量数据，做scale
    # todo 去掉归一看看
    # t[f'{feat}_label_count'] = t[f'{feat}_label_count'] / t.groupby(feat)[f'{feat}_label_count'].transform('max')
    # t[f'{feat}_label_sum'] = t[f'{feat}_label_sum'] / t.groupby(feat)[f'{feat}_label_sum'].transform('max')
    assert org_size == len(target)
    label_name = config.label_name
    filename = f'{label_name}1_{index}.h5'
    dump_h5(target, filename)
    # return target


def dump_topic_feature():
    ques = load_ques()
    ques = ques[ques['topic_count'] > 0][['qid', 'topic']]

    # topic的相关topic数
    topic_rel_dict = collections.defaultdict(set)
    t = ques[ques['topic'] != '-1'][['topic']]
    for row in t.itertuples():
        topic = row[1]
        for tp in topic:
            for tp2 in topic:
                if tp != tp2:
                    topic_rel_dict[tp].add(tp2)

    df_rel = pd.DataFrame.from_records(list(topic_rel_dict.items()))
    df_rel.columns = ['topic', 'rel']
    df_rel['topic_rel_size'] = df_rel['rel'].map(lambda x: len(x))
    del df_rel['rel']

    # topic的问题数
    topic_dict = collections.defaultdict(list)
    for row in tqdm(ques.itertuples()):
        qid = row[1]
        topic = row[2]
        for tp in topic:
            topic_dict[tp].append(qid)

    # topic当做node，qid当做属性，入度
    df = pd.DataFrame.from_records(list(topic_dict.items()))
    df.columns = ['topic', 'qid']
    df['topic_qid_size'] = df['qid'].map(lambda x: len(x))
    del df['qid']

    data = load_data()
    data = pd.merge(data, ques, on='qid')

    t = flatten_qid_topic(data)

    t = pd.merge(t, df_rel, on='topic', how='left')
    t = pd.merge(t, df, on='topic', how='left')

    dump_h5(t, 'new_topic_fea.h5')

    # new topic
    ques = load_org_ques()[['qid', 'topic']]
    ques.index = ques['qid'].values
    ques = ques['topic'].to_dict()

    data = load_data()
    data['topic'] = data['qid'].map(ques)

    data = get_merge_feature(data, data, 'topic', 'qid', ['count'])
    bins = get_bins(data['topic_qid_w_count'])
    data['topic_qid_w_count_bin'] = pd.cut(data['topic_qid_w_count'], bins=bins).cat.codes

    t = data[['qid', 'uid', 'topic_qid_w_count_bin', 'label', 'day', 'topic']]
    dump_h5(t, 'topic_new_1.h5')


def cal_dt_feature(train):
    bins = [-1, 8., 11., 15., 19., 23.]
    train['hour_bin'] = pd.cut(train['hour'], bins=bins).cat.codes
    train['weekend'] = (train['day'] - 3819) % 7 + 1
    train['weekend'] = np.where(train['weekend'].isin([6, 7]), 1, 0)
    train['weekend_hour_bin'] = train['hour_bin'] * 10 + train['weekend']


def prepare_data(ques_df):
    bins = [94.0, 294.0, 304.0, 314.0, 324.0, 334.0, 345.0, 356.0, 368.0, 381.0, 395.0, 410.0, 427.0, 447.0,
            471.0, 499.0, 532.0, 566.0, 599.0, 641.0, 890.0]

    df = load_data()
    user_df = load_user()

    user_df['score_bin'] = pd.cut(user_df['score'], bins=bins).cat.codes

    user_df['uf_b_new'] = user_df['uf_b1'].astype(str) + user_df['uf_b2'].astype(str) + user_df['uf_b3'].astype(str) + \
                          user_df['uf_b4'].astype(str) + user_df['uf_b5'].astype(str)
    lb = LabelEncoder()
    user_df['uf_b_new'] = lb.fit_transform(user_df['uf_b_new'])

    df['weekday'] = df['day'] % 7
    df['weekday_hour'] = df['weekday'] * 100 + df['hour']

    cal_dt_feature(df)

    df = pd.merge(df, user_df, on='uid', how='left')
    df = pd.merge(df, ques_df, on='qid', how='left')

    # 'uid_diff_day_inv_ques_w_max',
    # 'uid_diff_day_inv_ques_w_mean',
    # 'uid_diff_day_inv_ques_w_std',
    # 'uid_diff_day_inv_ques_w_sum',
    df['diff_day_inv_ques'] = df['day'] - df['q_day']
    df.loc[df['diff_day_inv_ques'] < 0, 'diff_day_inv_ques'] = 0
    # del user_df['follow_topic'], user_df['inter_topic']

    u_lb = LabelEncoder()
    u_lb.fit(user_df['uid'])
    return df, user_df, u_lb


def prepare_ques():
    df = load_ques()
    # del df['topic']
    q_lb = LabelEncoder()
    q_lb.fit(df['qid'])
    return df, q_lb


def prepare_ans(ques_df):
    df = load_ans()
    df['ans_t1_count'] = df['ans_t1'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))
    df['ans_t2_count'] = df['ans_t2'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))
    df['ans_t2_word_rate'] = (df['ans_t2_count'] + 4) / (df['word_count'] + 30)
    df['ans_t2_word_diff'] = df['word_count'] - df['ans_t1_count']
    del df['ans_t1'], df['ans_t2']

    df = pd.merge(df, ques_df[['qid', 'topic']], on='qid', how='left')

    t = ques_df[['qid', 'q_day']].set_index('qid').to_dict()['q_day']
    df['q_day'] = df['qid'].map(t)

    df['diff_day_ques_ans'] = df['a_day'] - df['q_day']
    df.loc[df['diff_day_ques_ans'] < 0, 'diff_day_ques_ans'] = 0

    df['weekday'] = df['a_day'] % 7
    df['weekday_hour'] = df['weekday'] * 100 + df['a_hour']

    df.rename(columns={'a_hour': 'hour'}, inplace=True)
    return df


# config._parse({"debug": True})


# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--index', type=int, default=-1)
#
# args = parser.parse_args()
# index = args.index
ques, q_lb = prepare_ques()
ans = prepare_ans(ques)

data, user, u_lb = prepare_data(ques)
dump_topic_feature()

for index in get_index():
    cal_feature(index)
#
# if 1:
#     with multiprocessing.Pool() as pool:
#         pool.map(cal_feature, get_index())

# dump_h5(label, f'label_n1_0.h5')
# dump_h5(test, f'label_n1_2.h5')

# if 0:
#     # test
#     index = 0
#     feature_start, feature_end, label_start, label_end = time_index_dict[f'label{index}']
#     # 标签
#     target = data[(data['day'] >= label_end - 6) & (data['day'] <= label_end)]
#     org_size = len(target)
#     logger.info("label shape %s, start %s, end %s", target.shape, target['day'].min(), target['day'].max())
#
#     target = merge_topic_feature(target, index)
#
#     # df = cal_topic_feature(index)
#     # target = merge_ans_feature(target, feature_end)
#
#     # target = merge_label_feature(target)
#
#     target = merge_inv_feature(target, data[data['day'] <= feature_end], feature_end)
#
#     # feature = data[(data['day'] <= label_end)]
#     # target = cal_inv_feature(target, feature)
#
#     # df = target
#     # df = cal_lag_feature(df, 'day', 'uid', lag=1, scale=False)

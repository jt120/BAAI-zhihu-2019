# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import logging
import joblib
import datetime
import sys
import collections
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import time
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn import metrics
from functools import wraps
import wrapt
import random
import tarfile
import zipfile

SEED = 47
warnings.filterwarnings('ignore')

offset_22 = 22
offset_29 = 29
offset_53 = 53
pos_type_topic = 'topic'
pos_type_word = 'word'
pos_type_sing = 'sing'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)


class Config(object):
    debug = False
    round = 5000
    frac = 0.3
    label_name = 'label_v'
    model_name = 'recommend.pkl'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


drop_feature = ['index', 'is_test', 'label', 'uid', 'qid', 'dt', 'day', 'q_day', 'uid_enc', 'qid_enc',
                'inv_min_day', 'topic', 'follow_topic', 'inter_topic', 'key', 'day_hour', 'diff_day_inv_ans',
                'inv_max_day', 'week', 'uf_b_new', 'cos_ans_t1']

config = Config()


# log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
# logging.basicConfig(format=log_fmt, level=logging.INFO)


# def timeit(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         tic = time.time()
#         print(func.__name__, 'start')
#         ret = func(*args, **kwargs)
#         toc = time.time() - tic
#         print(func.__name__, 'cost', toc)
#         return ret
#
#     return wrapper

def no_use(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return wrapper(*args, **kwargs)

    return wrapper


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.time()
        file = [x for x in args if isinstance(x, str)]
        if len(file) > 0:
            file = file[0]
        ret = func(*args, **kwargs)
        toc = time.time() - tic
        print('%s %s cost [%.3f]' % (func.__name__, file, toc))
        return ret

    return wrapper


def log_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if isinstance(ret, pd.DataFrame):
            logger.info("df %s", ret.shape)
        return ret

    return wrapper


@wrapt.decorator
def good(wrapped, inst, args, kwargs):
    return wrapped(*args, **kwargs)


@wrapt.decorator
def checklist(wrapped, inst, args, kwargs):
    return wrapped(*args, **kwargs)


@wrapt.decorator
def new_feature(wrapped, inst, args, kwargs):
    return wrapped(*args, **kwargs)


@wrapt.decorator
def bad(wrapped, inst, args, kwargs):
    return wrapped(*args, **kwargs)


@wrapt.decorator
def feature(wrapped, inst, args, kwargs):
    return wrapped(*args, **kwargs)


@wrapt.decorator
def dump_feature(wrapped, inst, args, kwargs):
    return wrapped(*args, **kwargs)


@wrapt.decorator
def pb(wrapped, inst, args, kwargs):
    return wrapped(*args, **kwargs)


@timeit
@log_it
def merge_left(df1, df2, key):
    return df1.merge(df2, on=key, how='left', copy=False)


def configure_logging(filename):
    if len(filename) < 10:
        log_filename = f'{filename}_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
    else:
        log_filename = filename

    log_fmt = "[%(asctime)s] %(funcName)s: %(message)s"
    formatter = logging.Formatter(log_fmt)

    fh = logging.FileHandler(filename=os.path.join('../logs', log_filename))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


logger = configure_logging('info')


# answer_info_0926.tar.gz  invite_info_evaluate_1_0926.txt  question_info_0926.tar.gz	  topic_vectors_64d.tar.gz
# invite_info_0926.tar.gz  member_info_0926.tar.gz	  single_word_vectors_64d.tar.gz  word_vectors_64d.tar.gz

def open_gz(file):
    with tarfile.open(f'{base_path}/{file}.tar.gz', "r:*") as tar:
        csv_path = tar.getnames()[0]
        df = pd.read_csv(tar.extractfile(csv_path), header=None, sep='\t')
        return df


def load1(filename, index):
    # return load_h5(f'label_final_{index}.h5')
    return load_h5(f'{filename}_{index}.h5')


def load_invite(index):
    if index == 2:
        df = pd.read_csv(f'{base_path}/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
        df.columns = ['qid', 'uid', 'dt']
        df['is_test'] = 1
    elif index == 1:
        df = open_gz('invite_info_0926')
        # df = pd.read_csv(f'{base_path}/invite_info_0926.tar.gz', skiprows=1, sep='\t', header=None)
        df.columns = ['qid', 'uid', 'dt', 'label']
        df['is_test'] = 0
    elif index == 3:
        df = pd.read_csv(f'{base_path}/invite_info_evaluate_2_0926.txt', sep='\t', header=None)
        df.columns = ['qid', 'uid', 'dt']
        df['is_test'] = 2
    logging.info("invite shape %s, index %s", df.shape, index)

    df['day'] = extract_day(df['dt'])
    df['hour'] = extract_hour(df['dt'])
    df = df.reset_index()
    # index_1 = 9489162
    # index_2 = 1141683
    # index1+index2=10630845
    if index == 2:
        df['index'] = df['index'] + 9489162
    elif index == 3:
        df['index'] = df['index'] + 10630845
    return df


def load_emb_dict(name='topic_vectors_64d', to_dict=True):
    # name = 'word_vectors_64d'
    # name2 = 'topic_vectors_64d'
    df = open_gz(name)

    # df = pd.read_csv(f'{base_path}/{name}.tar.gz', header=None, sep='\t')
    df.columns = ['key', 'key_vec']
    df['key_vec'] = df['key_vec'].map(lambda x: np.asarray([float(a) for a in x.split(' ')]))
    logger.info('name %s, \n%s', name, df.head())
    if to_dict:
        df = df.set_index('key').to_dict()
        df = df['key_vec']
    return df


def load_ques_txt():
    ques = load_org_ques()
    ques['q_day'] = extract_day(ques['q_dt'])
    ques['q_hour'] = extract_hour(ques['q_dt'])
    del ques['q_dt']
    ques['topic'] = ques['topic'].map(lambda x: set([a for a in x.split(',') if a != '-1']))
    return ques


@timeit
def load_h5(filename):
    return pd.read_hdf(f'../feature/{filename}')


@timeit
def dump_h5(df, name):
    logger.info('file %s, shape %s \n%s', name, df.shape, df.isnull().sum() / len(df))
    return df.to_hdf(f'../feature/{name}', 'df', mode='w')


@timeit
def dump_pkl(df, filename):
    logger.info("dump model %s", filename)
    return joblib.dump(df, f'../feature/{filename}')


@timeit
def load_pkl(filename):
    return joblib.load(f'../feature/{filename}')


@timeit
def read_pkl(filename):
    return pd.read_pickle(f'../feature/{filename}')


@timeit
def to_pkl(df, filename):
    df.to_pickle(f'../feature/{filename}')


def load_user():
    return load_h5('user.h5')


def load_ques():
    return load_h5('ques.h5')


def load_ans():
    return read_pkl('ans.pkl')


def sample_label(label, frac=0.2):
    logger.info("start %s", label.shape)
    label1 = label[label['label'] == 1]
    label0 = label[label['label'] == 0].sample(frac=frac)
    label = pd.concat((label0, label1))
    logger.info("after %s", label.shape)
    return label


def load_org_ques():
    # ques = pd.read_csv(f'{base_path}/question_info_0926.txt', header=None, sep='\t')
    ques = open_gz('question_info_0926')
    # ques = pd.read_csv(f'{base_path}/question_info_0926.tar.gz', header=None, sep='\t')
    ques.columns = ['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']
    ques = ques.drop_duplicates(subset=['qid'])
    logger.info("ques %s", ques.shape)
    return ques


def load_ques_word():
    ques = load_org_ques()
    ques['title_t2'] = ques['title_t2'].map(lambda x: set(x.split(',')))
    ques['desc_t2'] = ques['desc_t2'].map(lambda x: set(x.split(',')))
    ques = ques[['qid', 'title_t2', 'desc_t2']]
    return ques


def load_ques_topic():
    ques = load_org_ques()
    ques = ques[['qid', 'topic']]
    ques['topic'] = ques['topic'].map(lambda x: set([a for a in x.split(',') if a != '-1']))
    ques['topic_size'] = ques['topic'].map(lambda x: len(x))
    ques = ques[ques['topic_size'] > 0]
    del ques['topic_size']
    return ques


def dict2df(d):
    df = pd.DataFrame.from_records(list(d.items()))
    return df


def go_process(func, param):
    with multiprocessing.Pool() as pool:
        pool.map(func, param)


def load_data():
    data = load_h5('data2.h5')
    if config.debug:
        logger.info("debug mode, sample 0.01 data")
        data = data.sample(frac=0.01)
    return data


def get_pos_list(pos_type):
    if pos_type == pos_type_topic:
        return ['topic']
    elif pos_type == pos_type_word:
        return ['title_t2', 'desc_t2']
    elif pos_type == pos_type_sing:
        return ['title_t1', 'desc_t1']


def get_emb(df, col_key, col_name, vec_dict, pool='mean'):
    logger.info("get_emb %s, vec_dict %s", col_name, len(vec_dict))
    has_weight = col_name == 'inter_topic'
    df = df[[col_key, col_name]]
    emb_v = {}

    for row in tqdm(df.itertuples()):
        key = row[1]
        key_list = row[2]
        if '-1' == key_list:
            continue
        key_score_list = key_list.split(',')
        ret = []
        if has_weight:
            for key_score in key_score_list:
                split = key_score.split(':')
                try:
                    k = split[0]
                    s = float(split[1])
                except:
                    print(key, key_score)
                    return None
                if np.isinf(s):
                    continue
                ret.append(vec_dict[k] * s)
        else:
            for item in key_score_list:
                if item == '-1':
                    continue
                ret.append(vec_dict[item])

        if len(ret) > 0:
            if pool == 'mean':
                emb_v[key] = np.mean(ret, axis=0)
            elif pool == 'sum':
                emb_v[key] = np.sum(ret, axis=0)
            else:
                raise Exception('not support ' + pool)

    df = pd.DataFrame.from_records(list(emb_v.items()))
    df.columns = [col_key, f'{col_name}_emb']
    return df


def feature_exists(filename, check=True):
    return False
    # if not check:
    #     return False
    # exists = os.path.exists(f'../feature/{filename}')
    # if not exists:
    #     print("file %s not found" % (filename))
    # return exists


# dump user的向量
def dump_user_v(df, index, key, word_vec):
    # 不去重，总回答的问题，提高权重
    t = df.groupby(['uid'])[key].agg(lambda x: list(itertools.chain.from_iterable([a for a in x]))).reset_index()

    user_v = {}
    for row in tqdm(t.itertuples()):
        uid = row[1]
        topic = row[2]
        ret = []
        for x in topic:
            if '-1' == x:
                continue
            ret.append(word_vec[x])
        if len(ret) > 0:
            user_v[uid] = np.mean(ret, axis=0)
        else:
            user_v[uid] = np.nan
    return user_v


@no_use
def dum_ques_v(ques, vec, name='title'):
    ques_topic_v = {}

    for row in tqdm(ques.itertuples()):
        key = row[1]
        key_list = row[2]
        if '-1' in key_list:
            continue
        ret = [vec[x] for x in key_list]
        ques_topic_v[key] = np.mean(ret, axis=0)

    joblib.dump(ques_topic_v, f'../feature/ques_{name}_emb.dump')


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


def get_unq_count(target, feature, key, ukey, sub=''):
    t = feature.groupby(key)[ukey].nunique().to_dict()
    target[f'{key}_unq_{ukey}_w{sub}_count'] = target[key].map(t)
    return target


@good
def get_diff_day(target, feature, dt_col='day', sub=''):
    for key in ['uid', 'qid']:
        t = get_key1_feature(feature, key, dt_col, ['min', 'max'], sub=sub, ret_dict=True)
        key1, key2 = list(t.keys())
        for dk in t:
            target[dk] = target[key].map(t[dk])
        target[f'{key1}_{key2}'] = target[key2] - target[key1]
        # 时间区间不同，需要scal
        target[f'{key1}_{key2}'] = target[f'{key1}_{key2}'] / target[f'{key1}_{key2}'].max()
        for dk in t:
            if 'min' in dk or sub == 'ans':
                target[dk] = target['day'] - target[dk]
            else:
                target[dk] = target[dk] - target['day']
    return target


def cal_key_rate(df, key, windows):
    end_day = df['day'].max()
    start_day = end_day - windows + 1
    t = df[(df['day'] >= start_day) & (df['day'] <= end_day)]
    t = t.groupby(key)['label'].agg(['sum', 'count', 'std', 'mean']).reset_index()
    t.columns = [key, f'{key}_{windows}d_inv_sum', f'{key}_{windows}d_inv_count', f'{key}_{windows}d_inv_std',
                 f'{key}_{windows}d_inv_mean']
    t[f'{key}_{windows}d_inv_gmean'] = (t[f'{key}_{windows}d_inv_sum'] + 1) / (t[f'{key}_{windows}d_inv_count'] + 5)
    return t


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


@timeit
def get_key1_feature(df, key1, target, agg, sub='', ret_dict=False, scale=False):
    t = df.groupby([key1])[target].agg(agg).reset_index()
    cols = []
    for a in agg:
        cols.append(f'{key1}_{target}_w{sub}_{a}')
    t.columns = [key1] + cols
    logger.info("columns %s", t.columns.values.tolist())
    if scale:
        scaler = MinMaxScaler()
        for col in t.columns:
            if key1 == col:
                continue
            t[col] = scaler.fit_transform(t[col].values.reshape(-1, 1))
    if ret_dict:
        t = t.set_index(key1).to_dict()
    return t


def make_key(s1, s2):
    return s1.astype(str) + s2.astype(str)


def get_merge_feature(df, feature_df, col_key, col_target, agg, sub='', scale=False):
    logger.info("df %s, col_key %s, col_target %s, agg %s, sub %s", df.shape, col_key, col_target, agg, sub)
    # t1 = get_key1_feature(feature, key, target_col, agg, sub, scale=scale)
    # target = pd.merge(target, t1, on=key, how='left')

    for ag in agg:
        new_col_names = f'{col_key}_{col_target}_w{sub}_{ag}'
        tmp_df = feature_df.groupby(col_key)[col_target].agg([ag]).reset_index().rename(columns={ag: new_col_names})
        tmp_df.index = list(tmp_df[col_key])
        if scale:
            scaler = MinMaxScaler()
            tmp_df[new_col_names] = scaler.fit_transform(tmp_df[new_col_names].values.reshape(-1, 1))
        tmp_df = tmp_df[new_col_names].to_dict()
        df[new_col_names] = df[col_key].map(tmp_df).astype('float32')
        print(new_col_names, ', ', end='')
    # t1 = get_key1_feature(feature, key, target_col, agg, sub, ret_dict=True, scale=scale)
    # target = merge_dict(target, t1, key)
    return df


def get_merge_feature2(target, feature_df, key1, key2, target_col, agg, sub='', scale=False):
    logger.info("key1 %s, key2 %s, target_col %s, agg %s", key1, key2, target_col, agg)
    t = get_key2_feature(feature_df, key1, key2, target_col, agg, sub=sub, scale=scale)
    target = target.merge(t, on=[key1, key2], how='left')
    return target


@timeit
def get_key2_feature(df, key1, key2, target_col, agg, sub='', scale=False):
    t = df.groupby([key1, key2])[target_col].agg(agg).reset_index()
    cols = []
    for a in agg:
        cols.append(f'{key1}_{key2}_{target_col}_w{sub}_{a}')
    t.columns = [key1, key2] + cols

    if scale:
        scaler = MinMaxScaler()
        for col in t.columns:
            if key1 == col:
                continue
            t[col] = scaler.fit_transform(t[col].values.reshape(-1, 1))
    return t


# 问题特征
def extract_inv_win_feature(feature, index):
    inv_feature = {}
    for key in ['uid', 'qid']:
        for win in [1, 3, 7, 14]:
            logging.info("key %s, win %s", key, win)
            rate = cal_key_rate(feature, key, win)
            inv_feature[f'{key}_win_{win}'] = rate
            # target = pd.merge(target, rate, on=key, how='left')
    filename = f'../feature/inv_feature_{index}.dump'
    joblib.dump(inv_feature, filename)
    logging.info("dump %s", filename)


def extract_label_feature(data):
    # data['uid_enc'] = u_lb.transform(data['uid'])
    # data['qid_enc'] = q_lb.transform(data['qid'])
    for feat in ['uid_enc', 'qid_enc', 'gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5']:
        col_name = '{}_target_count'.format(feat)
        data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
        # data.loc[data[col_name] < 2, feat] = -1
        # data[feat] += 1
        data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
        data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())
        logger.info("cal %s", feat)
    return data


def reduce(df, safe=True, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if safe:
                    df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            else:
                if safe:
                    df[col] = df[col].astype(np.float32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def cal_lag_feature(df, dt_col, key, agg=None, lag=1, scale=False, sub=''):
    df['shift_day'] = df[dt_col].shift(lag)
    pk = f'{key}_label_diff_lag{lag}_{dt_col}'
    df[pk] = df[dt_col] - df['shift_day']
    df.loc[(df[key] != df[key].shift(lag)), pk] = np.nan
    del df['shift_day']
    if scale:
        df[pk] = df[pk] / df[pk].max()
    if agg:
        df = df.groupby([key])[pk].agg(agg).reset_index()
        df.columns = [key] + [f'{key}_{pk}_{x}_w{sub}' for x in agg]
    return df


def cal_time_gaps(df, key, dt_col, scale=False, sub=''):
    df = df.sort_values([key, dt_col])
    df = cal_lag_feature(df, dt_col, key, lag=1, scale=scale, sub=sub)
    df = cal_lag_feature(df, dt_col, key, lag=-1, scale=scale, sub=sub)
    return df


# 时间间隔
# def group_time_gaps(df, key, dt_col):
#     df = cal_time_gaps(df, key, dt_col, 1, False)
#
#     return t
# 距上次回答的时间
# 距上上次
def cal_ans_lag_feature(label):
    label['diff_a_day'] = label['day'] - label['a_day']
    label['diff_lag_a_day'] = label['day'] - label['lag_a_day']
    del label['a_day']
    del label['lag_a_day']
    return label


# 最小，最大时间
def get_time_min_max(df, key, dt_col):
    t = df.groupby(key)[dt_col].agg(['min', 'max']).reset_index()
    t.columns = [key, 'label_day_min', 'label_day_max']
    t[f'{key}_{dt_col}_max_min'] = t['label_day_max'] - t['label_day_min']
    return t[[key, f'{key}_{dt_col}_max_min']]


def df2dic(df):
    df.columns = ['uid', 'qid', 'cos']
    df['key'] = df['uid'] + df['qid']
    df = df[['key', 'cos']]
    df = df.set_index('key').to_dict()['cos']
    return df


def cal_sim(user_v_dict, target_v_dict, target):
    ret = []
    for row in tqdm(target.itertuples()):
        uid = row[1]
        qid = row[2]
        cur = [uid, qid]
        user_v = user_v_dict.get(uid, np.nan)
        ques_v = target_v_dict.get(qid, np.nan)
        if isinstance(user_v, np.float) or isinstance(ques_v, np.float):
            cur.append(np.nan)
        else:
            try:
                similarity = cosine_similarity(user_v.reshape(1, -1), ques_v.reshape(1, -1))
                cur.append(similarity.item())
            except:
                cur.append(np.nan)
                logger.info("error user: %s, user: %s, ques: %s", uid, user_v, ques_v)
                break
        ret.append(cur)
    df = pd.DataFrame(ret)
    return df


base_path = '../input/data_set_0926'

time_index_dict = {
    'label0': [3838, 3860, 3861, 3867],  # 3807
    'label1': [3837, 3859, 3860, 3866],  # 3807
    'label2': [3845, 3867, 3868, 3874]
}


def get_index(t=3, num=0):
    if t == 3:
        return [0, 1, 2]
        # return [3]
        # return [0, 2]
    elif t == 2:
        return [0, 2]
    elif t == 1:
        return [num]


end_day_dict = {
    2: 3874,
    1: 3860,
    0: 3867,
    3: 3853,
}


def get_feature_end_day(label_end_day):
    return label_end_day - 7


def get_label_end_day(index):
    return end_day_dict[index]


def get_pooling():
    return ['mean']
    # return ['mean']
    # return ['sum']


def get_bins(s):
    bins = s.quantile(np.arange(0, 1.05, 0.05)).values
    bins = np.unique(bins)
    bins[0] = bins[0] - 1
    print(bins)
    return bins


def check_result(index, sub, result_name):
    result = load_invite(index)
    test_size = len(result)
    # assert len(sub) == test_size
    result = pd.merge(result, sub, on='index')
    # assert len(test) == test_size
    logger.info(result.isnull().sum() / len(result))
    logger.info(result.head())
    logger.info(result['label'].quantile(np.arange(1, 0.1, -0.05)))
    result = result[['qid', 'uid', 'dt', 'label']]
    submit(result, result_name + str(index))


def submit(sub, filename):
    sub.to_csv("result.txt", index=None, header=None, sep='\t')

    with zipfile.ZipFile(f'{filename}.zip', 'w', zipfile.ZIP_DEFLATED) as zp:
        zp.write("result.txt")


@timeit
@log_it
def merge_dict(target, df, key):
    if not isinstance(df, dict):
        df = df.set_index(key).to_dict()
    for dk in df:
        target[dk] = target[key].map(df[dk])
    return target

seed_everything(SEED)

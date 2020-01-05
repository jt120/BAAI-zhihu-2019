# -*- coding: utf-8 -*-
from util import *


def cal_union_count(row, uid_topic_dict):
    uid = row['uid']
    topic = row['topic']
    t2 = uid_topic_dict.get(uid, set())
    return len(topic & t2)


# 这里只需要关注，其他的在下面
cos_files = [
    'user_follow_topic_emb_cos',  # 用户关注的主题
    'user_inter_topic_emb_cos',  # 用户感兴趣的主题
]


def merge_cos(label, index):
    old_size = len(label)

    # 用户关注和兴趣
    label['key'] = label['uid'] + label['qid']
    for file in cos_files:
        filename = f'{file}_{index}.h5'
        logger.info("file %s", filename)
        df = load_h5(filename)
        df = df2dic(df)
        label[file] = label['key'].map(df)
    for col_name in ['ans_t1', 'ans_t2']:
        df = load_h5(f'cos_{col_name}_mean_{index}.h5')
        df = df2dic(df)
        label[f'cos_{col_name}'] = label['key'].map(df)
    del label['key']
    assert len(label) == old_size

    cos0 = load_h5(f'lag_inv_cos_{index}.h5')
    for col in ['uid', 'qid', 'qid_1', 'qid_2', 'qid_3']:
        try:
            del cos0[col]
        except:
            pass

    old_size = len(label)
    label = label.merge(cos0, on='index', how='left')
    assert len(label) == old_size

    # 更多的相似性
    for pos_type in ['title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']:
        for feature_type in ['ans', 'inv', 'invans']:
            for pt in get_pooling():
                user_emb_key = f'uid_{pos_type}_{pt}_{feature_type}_{index}'
                col_key = f'cos_{user_emb_key}'[:-2]
                if col_key in label:
                    print('in label', col_key)
                    continue
                name = f'cos_{user_emb_key}.h5'
                df = load_h5(name)
                key2 = df.columns.values.tolist()[2]
                df = df.set_index('uid').to_dict()[key2]
                label[key2] = label['uid'].map(df)
                print('done', user_emb_key, key2)
    assert len(label) == old_size

    return label


def merge_dump(index):
    logger.info('merge index %s', index)
    label_name = config.label_name
    filename = f'{label_name}1_{index}.h5'

    label = load_h5(filename)
    # label = pd.merge(label, ques[['qid', 'topic']], on='qid', how='left')

    logger.info("label shape %s, start %s, end %s", label.shape, label['day'].min(), label['day'].max())

    """
    用户和问题的cos相似度
    """
    label = merge_cos(label, index)

    label_name = config.label_name
    filename = f'{label_name}2_{index}.h5'
    dump_h5(label, filename)
    # return label


for index in get_index():
    merge_dump(index)

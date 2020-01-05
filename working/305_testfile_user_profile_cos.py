# -*- coding: utf-8 -*-
from util import *


# 用户profile和当前问题的cos
# f'../feature/user_ques_title_cos_{index}.h5'
# f'../feature/user_ques_desc_cos_{index}.h5'
# 用户关注topic的cos user_follow_topic_emb_cos_0.h5
# 用户感兴趣topic的cos user_inter_topic_emb_cos_0.h5

def cal_sim(user_v_dict, target_v_dict, target):
    ret = []
    for row in tqdm(target.itertuples()):
        uid = row[1]
        qid = row[2]
        # 这里顺序不能乱
        cur = [uid, qid]
        user_v = user_v_dict.get(uid, np.nan)
        ques_v = target_v_dict.get(qid, np.nan)
        if isinstance(user_v, np.float) or isinstance(ques_v, np.float):
            cur.append(np.nan)
        else:
            try:
                similarity = cosine_similarity(user_v.reshape(1, -1), ques_v.reshape(1, -1))
                cur.append(similarity.item())
            except Exception as e:
                cur.append(np.nan)
                logger.info("error user: %s, user: %s, ques: %s", uid, user_v, ques_v)
                print(e)
                break
        ret.append(cur)
    df = pd.DataFrame(ret)
    return df


def to_dict(df):
    df.columns = ['key', 'value']
    return df.set_index('key').to_dict()['value']


@checklist
def hello():
    pass


def process1(index):
    data = load_data()

    ques_emb = load_h5('qid_topic_mean.h5')
    ques_emb = to_dict(ques_emb)

    user_inter_emb = load_h5('uid_inter_topic_mean.h5')
    user_inter_emb = to_dict(user_inter_emb)

    user_follow_emb = load_h5('uid_follow_topic_mean.h5')
    user_follow_emb = to_dict(user_follow_emb)

    label_end_day = get_label_end_day(index)
    label = data[(data['day'] > get_feature_end_day(label_end_day)) & (data['day'] <= label_end_day)][['uid', 'qid']]
    label = label[['uid', 'qid']].drop_duplicates()
    print(label.shape)
    filename = f'user_follow_topic_emb_cos_{index}.h5'
    if not feature_exists(filename, False):
        df = cal_sim(user_follow_emb, ques_emb, label)
        dump_h5(df, filename)

    filename = f'user_inter_topic_emb_cos_{index}.h5'
    if not feature_exists(filename, False):
        df = cal_sim(user_inter_emb, ques_emb, label)
        dump_h5(df, filename)


# file user_inter_topic_emb_cos_0.h5 not found
# file lag_inv_cos_0.h5 not found
# file user_inter_topic_emb_cos_1.h5 not found
# file lag_inv_cos_1.h5 not found
# file lag_inv_cos_2.h5 not found

if 1:
    go_process(process1, get_index())

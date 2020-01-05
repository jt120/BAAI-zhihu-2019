from util import *

##########################
# 用户特征
# 用户画像特征的标签 count sum mean std
# ../feature/user.h5
# f'../feature/user_feat_{index}.dump'
##########################

# gender_label_count
# 用户画像的各类target encode
"""
uf_c1                                538.0
uf_c1_label_mean                     360.0
uf_c1_label_count                    280.5

uf_c3_label_mean                     531.5
uf_c3                                515.0
uf_c3_label_count                    449.5

uf_c4                                616.5
uf_c4_label_mean                     467.0
uf_c4_label_count                    343.5

qid_gender_label_std                 215.0
qid_gender_label_mean                185.0
qid_gender_label_count               177.0
gender                               158.5
qid_gender_label_sum                  93.5
gender_label_mean                     46.0
gender_target_count                   10.5
gender_label_std                       0.0
gender_label_sum                       0.0
gender_label_count                     0.0

uf_c5没贡献
"""

"""
freq_label_w_count

"""

# user = pd.read_csv(f'{base_path}/member_info_0926.txt', header=None, sep='\t')
user = open_gz('member_info_0926')
user.columns = ['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1', 'uf_b2',
                'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score', 'follow_topic',
                'inter_topic']
logger.info("user %s", user.shape)

unq = user.nunique()
logger.info("user unq %s", unq)

for x in unq[unq == 1].index:
    del user[x]
    logger.info('del unq==1 %s', x)

user['follow_topic_count'] = user['follow_topic'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))
user['inter_topic_count'] = user['inter_topic'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))

# del user['follow_topic'], user['inter_topic']

t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['follow_topic', 'inter_topic', 'uid']]
logger.info("user cat %s", cats)

# score_bin = [0, 260, 272, 285, 290, 295, 299, 303, 308, 313, 320, 328, 337, 347, 359, 373, 392, 417, 455, 526, 890,
#              1000]
# user['score_bin'] = pd.cut(user['score'], bins=score_bin)
# user['score_bin'] = user['score_bin'].cat.codes

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logger.info('encode %s', d)

user = reduce(user)

dump_h5(user, 'user.h5')

# 用户画像的各类target encode

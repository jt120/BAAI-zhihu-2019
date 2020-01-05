# -*- coding: utf-8 -*-
from util import *


# 问题title single, title word
# desc single, desc word
# topic 的emb


# (base) -bash-4.1$ ls | grep qid | grep mean
# qid_desc_t1_mean.h5
# qid_desc_t2_mean.h5
# qid_title_t1_mean.h5
# qid_title_t2_mean.h5
# qid_topic_mean.h5
# (base) -bash-4.1$ ls | grep qid | grep max
# qid_desc_t1_max.h5
# qid_desc_t2_max.h5
# qid_title_t1_max.h5
# qid_title_t2_max.h5
# qid_topic_max.h5

# qid_topic_mean
def process_ques(vec_type):
    dump_type = 'h5'
    ques = load_org_ques()[['qid', 'topic', 'title_t2', 'desc_t2', 'title_t1', 'desc_t1']]
    if vec_type == pos_type_topic:
        vec = load_emb_dict('topic_vectors_64d')
    elif vec_type == pos_type_word:
        vec = load_emb_dict('word_vectors_64d')
    elif vec_type == pos_type_sing:
        vec = load_emb_dict('single_word_vectors_64d')
    pos_list = get_pos_list(vec_type)
    logger.info("ques vec %s", vec_type)
    for pos in pos_list:
        for pt in get_pooling():
            emb = get_emb(ques, 'qid', pos, vec, pool=pt)
            if dump_type == 'h5':
                dump_h5(emb, f'qid_{pos}_{pt}.h5')
            else:
                dump_pkl(emb, f'qid_{pos}_{pt}.pkl')


# 邀请数据提取的用户emb


# 字的emb

# process_ques(pos_type_topic)

# qid_topic_mean, qid_title_t2_mean, qid_desc_t2_mean
if 1:
    with multiprocessing.Pool() as pool:
        pool.map(process_ques, [
            pos_type_sing,
            pos_type_word,
            pos_type_topic])

# -*- coding: utf-8 -*-
from util import *


# 用户关注和感兴趣的topic，生成emb

# uid_follow_topic_mean.h5
# uid_inter_topic_mean.h5
#


def process1(param):
    key, pool_type = param
    user = load_user()
    user = user[['uid', 'follow_topic', 'inter_topic']]
    topic_vec_dict = load_emb_dict('topic_vectors_64d')

    emb = get_emb(user, 'uid', key, topic_vec_dict, pool=pool_type)
    filename = f'uid_{key}_{pool_type}.h5'
    dump_h5(emb, filename)
    logger.info("dump file %s", filename)


req = []
for key in ['follow_topic', 'inter_topic']:
    for pt in get_pooling():
        req.append((key, pt))

go_process(process1, req)

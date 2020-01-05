# -*- coding: utf-8 -*-

from util import *

##########################
# 问题特征
# '../feature/ques.h5'
##########################

ques = load_ques_txt()
#
ques['topic_count'] = ques['topic'].apply(lambda x: len(x))
ques['title_t1_count'] = ques['title_t1'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))
ques['title_t2_count'] = ques['title_t2'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))
ques['desc_t1_count'] = ques['desc_t1'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))
ques['desc_t2_count'] = ques['desc_t2'].apply(lambda x: 0 if x == '-1' else len(x.split(',')))

del ques['title_t1'], ques['title_t2'], ques['desc_t1'], ques['desc_t2']

ques = reduce(ques)

dump_h5(ques, "ques.h5")

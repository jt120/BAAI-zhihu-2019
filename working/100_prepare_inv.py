# -*- coding: utf-8 -*-
from util import *

train = load_invite(1)
del train['dt']

test = load_invite(2)
test['label'] = 0
del test['dt']

# 新数据
test2 = load_invite(3)
test['label'] = 0
del test2['dt']

data = pd.concat((train, test, test2), axis=0, sort=False, ignore_index=True)
del train, test, test2

data = reduce(data)

# dump_pkl(data, 'data.pkl')

dump_h5(data, 'data2.h5')

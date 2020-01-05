#!/usr/bin/env bash

echo "start"
python 100_prepare_inv.py
python 101_prepare_user.py
python 102_prepare_ques.py
python 103_prepare_ans.py

python 201_file_qid_emb.py
python 202_file_uid_ans_emb.py
python 203_file_uid_behavior_cos.py
python 204_file_uid_follow_inter_emb.py
python 205_testfile_uid_behavior_cos.py

python 302_testfile_cal_ans_ques_cos.py
python 303_testfile_cal_cos.py
python 304_testfile_lag_inv_cos.py
python 305_testfile_user_profile_cos.py
python 306_check_cos.py

python 401_merge_cal_feature.py
python 402_merge_dump_feature.py
python 403_prepare_train.py


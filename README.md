# zhihu_2019
https://www.biendata.com/competition/zhihu2019/

# 数据

链接: https://pan.baidu.com/s/1Ttc3TUnW_C2p_Sa6i5d11Q  
密码: slrl

# 目标

邀请回答预测，可以理解为转化率预估问题，不同与其他问题的点是邀请是召回算法的结果。

其中50%数据，作为初赛数据。剩下50%，最后一天释放，用于最终排名

# 使用

```bash
1. 请把比赛数据放到 input/data_set_0926 下
2. cd 到workinng目录
3. 执行 sh run.sh

```

> 1. 请把比赛数据放到 input/data_set_0926 下
> 2. models目录下的模型特征为

所有特征
```bash
['hour', 'weekday', 'weekday_hour', 'gender', 'freq', 'uf_b1', 'uf_b2', 'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score', 'follow_topic_count', 'inter_topic_count', 'score_bin', 'q_hour', 'topic_count', 'title_t1_count', 'title_t2_count', 'desc_t1_count', 'desc_t2_count', 'diff_day_inv_ques', 'qid_topic_inv_ans_inv_count_w_sum', 'qid_topic_inv_ans_inv_count_w_mean', 'qid_topic_inv_ans_inv_count_w_max', 'qid_topic_inv_ans_inv_count_w_std', 'qid_topic_inv_ans_ans_count_w_sum', 'qid_topic_inv_ans_ans_count_w_mean', 'qid_topic_inv_ans_ans_count_w_max', 'qid_topic_inv_ans_ans_count_w_std', 'qid_topic_label_inv_count_w_sum', 'qid_topic_label_inv_count_w_mean', 'qid_topic_label_inv_count_w_max', 'qid_topic_label_inv_count_w_std', 'qid_topic_ans_count_w_sum', 'qid_topic_ans_count_w_mean', 'qid_topic_ans_count_w_max', 'qid_topic_ans_count_w_std', 'qid_topic_inv_ans_ans_rate_w_mean', 'qid_topic_inv_ans_ans_rate_w_max', 'qid_topic_inv_ans_ans_rate_w_std', 'uid_qid_user_topic_cheer_count_w_max', 'uid_qid_user_topic_cheer_count_w_mean', 'uid_qid_user_topic_cheer_count_w_std', 'uid_qid_user_topic_cheer_count_w_sum', 'uid_qid_user_topic_word_count_w_max', 'uid_qid_user_topic_word_count_w_mean', 'uid_qid_user_topic_word_count_w_std', 'uid_qid_user_topic_word_count_w_sum', 'topic_union_label_ans_count', 'topic_union_ans_count', 'topic_union_label_inv_count', 'uid_uid_label_diff_lag1_a_day_mean_wans', 'uid_uid_label_diff_lag1_a_day_max_wans', 'uid_uid_label_diff_lag1_a_day_std_wans', 'uid_uid_label_diff_lag1_a_day_sum_wans', 'qid_qid_label_diff_lag1_a_day_mean_wans', 'qid_qid_label_diff_lag1_a_day_max_wans', 'qid_qid_label_diff_lag1_a_day_std_wans', 'qid_qid_label_diff_lag1_a_day_sum_wans', 'uid_a_day_wans_min', 'uid_a_day_wans_max', 'uid_a_day_wans_min_uid_a_day_wans_max', 'qid_a_day_wans_min', 'qid_a_day_wans_max', 'qid_a_day_wans_min_qid_a_day_wans_max', 'hour_ans_w_count', 'weekday_ans_w_count', 'weekday_hour_ans_w_count', 'weekday_hour_uid_ans_w_count', 'hour_uid_ans_w_count', 'weekday_uid_ans_w_count', 'uid_ans_w_count', 'uid_ans_t1_count_w_mean', 'uid_ans_t1_count_w_sum', 'uid_ans_t1_count_w_max', 'uid_ans_t1_count_w_std', 'uid_ans_t2_count_w_mean', 'uid_ans_t2_count_w_sum', 'uid_ans_t2_count_w_max', 'uid_ans_t2_count_w_std', 'uid_has_img_w_mean', 'uid_has_img_w_sum', 'uid_word_count_w_mean', 'uid_word_count_w_sum', 'uid_word_count_w_max', 'uid_reci_cheer_w_mean', 'uid_reci_cheer_w_sum', 'uid_reci_cheer_w_max', 'uid_reci_cheer_w_std', 'uid_reci_mark_w_mean', 'uid_reci_mark_w_sum', 'uid_reci_mark_w_max', 'uid_reci_mark_w_std', 'uid_diff_day_ques_ans_w_mean', 'uid_diff_day_ques_ans_w_sum', 'uid_diff_day_ques_ans_w_max', 'uid_diff_day_ques_ans_w_std', 'uid_reci_tks_w_mean', 'uid_reci_tks_w_sum', 'uid_reci_uncheer_w_mean', 'uid_reci_uncheer_w_sum', 'uid_reci_comment_w_mean', 'uid_reci_comment_w_sum', 'uid_reci_comment_w_max', 'uid_reci_comment_w_std', 'uid_reci_cheer_w2_mean', 'uid_reci_cheer_w2_sum', 'uid_reci_comment_w2_mean', 'uid_reci_comment_w2_sum', 'uid_word_count_w1_mean', 'uid_word_count_w1_sum', 'uid_word_count_w1_max', 'uid_word_count_w2_mean', 'uid_word_count_w2_sum', 'uid_word_count_w2_max', 'qid_ans_w_count', 'qid_ans_t2_count_w_mean', 'qid_ans_t2_count_w_sum', 'qid_ans_t2_count_w_max', 'qid_ans_t2_count_w_std', 'qid_has_img_w_mean', 'qid_word_count_w_mean', 'qid_word_count_w_sum', 'qid_word_count_w_max', 'qid_diff_day_ques_ans_w_mean', 'qid_diff_day_ques_ans_w_std', 'qid_diff_day_ques_ans_w_sum', 'qid_diff_day_ques_ans_w_max', 'qid_reci_tks_w_mean', 'qid_reci_mark_w_mean', 'qid_reci_comment_w_mean', 'qid_reci_comment_w_sum', 'qid_reci_cheer_w_mean', 'qid_reci_cheer_w_sum', 'qid_reci_uncheer_w_mean', 'qid_word_count_w1_mean', 'qid_word_count_w1_sum', 'qid_word_count_w1_max', 'qid_word_count_w2_mean', 'qid_word_count_w2_sum', 'qid_word_count_w2_max', 'reci_cheer', 'reci_tks', 'word_count', 'lag_word_count', 'lag_reci_cheer', 'lag_reci_tks', 'diff_a_day', 'diff_lag_a_day', 'uid_label_diff_lag1_day', 'uid_label_diff_lag-1_day', 'qid_label_diff_lag1_day', 'qid_label_diff_lag-1_day', 'hour_label_inv_w_count', 'weekday_uid_label_inv_w_count', 'hour_uid_label_inv_w_count', 'weekday_hour_uid_label_inv_w_count', 'hour_qid_label_inv_w_count', 'weekday_hour_qid_label_inv_w_count', 'uid_unq_day_wx_count', 'qid_unq_day_wx_count', 'uid_day_wx_min', 'uid_day_wx_max', 'uid_day_wx_min_uid_day_wx_max', 'qid_day_wx_min', 'qid_day_wx_max', 'qid_day_wx_min_qid_day_wx_max', 'inv_uid_rank', 'uid_diff_day_inv_ques_wlabel_max', 'uid_diff_day_inv_ques_wlabel_mean', 'uid_diff_day_inv_ques_wlabel_std', 'uid_diff_day_inv_ques_wlabel_sum', 'uf_b_new_label_w_mean', 'uf_b_new_label_w_sum', 'qid_gender_label_w_count', 'qid_gender_label_w_sum', 'qid_gender_label_w_mean', 'qid_gender_label_w_std', 'qid_freq_label_w_count', 'qid_freq_label_w_sum', 'qid_freq_label_w_mean', 'qid_freq_label_w_std', 'qid_uf_c1_label_w_count', 'qid_uf_c1_label_w_sum', 'qid_uf_c1_label_w_mean', 'qid_uf_c1_label_w_std', 'qid_uf_c2_label_w_count', 'qid_uf_c2_label_w_sum', 'qid_uf_c2_label_w_mean', 'qid_uf_c2_label_w_std', 'qid_uf_c3_label_w_count', 'qid_uf_c3_label_w_sum', 'qid_uf_c3_label_w_mean', 'qid_uf_c3_label_w_std', 'qid_uf_c4_label_w_count', 'qid_uf_c4_label_w_sum', 'qid_uf_c4_label_w_mean', 'qid_uf_c4_label_w_std', 'uid_uid_label_diff_lag1_day_mean_wlabel', 'uid_uid_label_diff_lag1_day_max_wlabel', 'uid_uid_label_diff_lag1_day_std_wlabel', 'uid_uid_label_diff_lag1_day_sum_wlabel', 'qid_qid_label_diff_lag1_day_mean_wlabel', 'qid_qid_label_diff_lag1_day_max_wlabel', 'qid_qid_label_diff_lag1_day_std_wlabel', 'qid_qid_label_diff_lag1_day_sum_wlabel', 'uid_diff_day_inv_ques_w_max', 'uid_diff_day_inv_ques_w_mean', 'uid_diff_day_inv_ques_w_std', 'score_bin_label_w_mean', 'score_bin_label_w_std', 'score_bin_label_w_count', 'score_bin_label_w_sum', 'uid_diff_day_inv_ans_w_mean', 'uid_diff_day_inv_ans_w_std', 'uid_diff_day_inv_ans_w_max', 'freq_label_w_count', 'uf_c1_label_w_count', 'uf_c1_label_w_sum', 'uf_c1_label_w_mean', 'uf_c1_label_w_std', 'uf_c2_label_w_count', 'uf_c2_label_w_mean', 'uf_c3_label_w_count', 'uf_c3_label_w_sum', 'uf_c3_label_w_mean', 'uf_c4_label_w_count', 'uf_c4_label_w_sum', 'uf_c4_label_w_mean', 'weekday_hour_label_w_count', 'weekday_hour_label_w_sum', 'weekday_hour_label_w_mean', 'weekday_hour_label_w_std', 'uid_unq_day_w_count', 'uid_unq_weekday_hour_w_count', 'qid_unq_day_w_count', 'uid_day_w_min', 'uid_day_w_max', 'uid_day_w_min_uid_day_w_max', 'qid_day_w_min', 'qid_day_w_max', 'qid_day_w_min_qid_day_w_max', 'uid_uid_label_diff_lag1_day_mean_wlabel1', 'uid_uid_label_diff_lag1_day_max_wlabel1', 'uid_uid_label_diff_lag1_day_std_wlabel1', 'uid_uid_label_diff_lag1_day_sum_wlabel1', 'qid_qid_label_diff_lag1_day_mean_wlabel1', 'qid_qid_label_diff_lag1_day_max_wlabel1', 'qid_qid_label_diff_lag1_day_std_wlabel1', 'qid_qid_label_diff_lag1_day_sum_wlabel1', 'uid_day_wlabel1_7d_min', 'uid_day_wlabel1_7d_max', 'uid_day_wlabel1_7d_min_uid_day_wlabel1_7d_max', 'qid_day_wlabel1_7d_min', 'qid_day_wlabel1_7d_max', 'qid_day_wlabel1_7d_min_qid_day_wlabel1_7d_max', 'uid_uid_label_diff_lag1_day_mean_wlabel1_7d', 'uid_uid_label_diff_lag1_day_max_wlabel1_7d', 'uid_uid_label_diff_lag1_day_std_wlabel1_7d', 'uid_uid_label_diff_lag1_day_sum_wlabel1_7d', 'qid_qid_label_diff_lag1_day_mean_wlabel1_7d', 'qid_qid_label_diff_lag1_day_max_wlabel1_7d', 'qid_qid_label_diff_lag1_day_std_wlabel1_7d', 'qid_qid_label_diff_lag1_day_sum_wlabel1_7d', 'user_follow_topic_emb_cos', 'user_inter_topic_emb_cos', 'cos_ans_t1', 'cos_ans_t2', 'recent_q0_topic', 'recent_q1_topic', 'recent_q2_topic', 'recent_q0_title', 'recent_q1_title', 'recent_q2_title', 'recent_q0_desc', 'recent_q1_desc', 'recent_q2_desc', 'cos_uid_title_t1_mean_ans', 'cos_uid_title_t1_mean_inv', 'cos_uid_title_t1_mean_invans', 'cos_uid_title_t2_mean_ans', 'cos_uid_title_t2_mean_inv', 'cos_uid_title_t2_mean_invans', 'cos_uid_desc_t1_mean_ans', 'cos_uid_desc_t1_mean_inv', 'cos_uid_desc_t1_mean_invans', 'cos_uid_desc_t2_mean_ans', 'cos_uid_desc_t2_mean_inv', 'cos_uid_desc_t2_mean_invans', 'cos_uid_topic_mean_ans', 'cos_uid_topic_mean_inv', 'cos_uid_topic_mean_invans', 'uid_activity_prda', 'qid_activity_prda', 'uid_enc_target_count', 'qid_enc_target_count', 'gender_target_count', 'freq_target_count', 'uf_c1_target_count', 'uf_c2_target_count', 'uf_c3_target_count', 'uf_c4_target_count', 'uf_c5_target_count', 'uid_ans_emb_0', 'uid_ans_emb_1', 'uid_ans_emb_2', 'uid_ans_emb_3', 'uid_ans_emb_4', 'uid_ans_emb_5', 'uid_ans_emb_6', 'uid_ans_emb_7', 'uid_ans_emb_8', 'uid_ans_emb_9', 'uid_ans_emb_10', 'uid_ans_emb_11', 'uid_ans_emb_12', 'uid_ans_emb_13', 'uid_ans_emb_14', 'uid_ans_emb_15', 'uid_ans_emb_16', 'uid_ans_emb_17', 'uid_ans_emb_18', 'uid_ans_emb_19', 'uid_ans_emb_20', 'uid_ans_emb_21', 'uid_ans_emb_22', 'uid_ans_emb_23', 'uid_ans_emb_24', 'uid_ans_emb_25', 'uid_ans_emb_26', 'uid_ans_emb_27', 'uid_ans_emb_28', 'uid_ans_emb_29', 'uid_ans_emb_30', 'uid_ans_emb_31', 'uid_ans_emb_32', 'uid_ans_emb_33', 'uid_ans_emb_34', 'uid_ans_emb_35', 'uid_ans_emb_36', 'uid_ans_emb_37', 'uid_ans_emb_38', 'uid_ans_emb_39', 'uid_ans_emb_40', 'uid_ans_emb_41', 'uid_ans_emb_42', 'uid_ans_emb_43', 'uid_ans_emb_44', 'uid_ans_emb_45', 'uid_ans_emb_46', 'uid_ans_emb_47', 'uid_ans_emb_48', 'uid_ans_emb_49', 'uid_ans_emb_50', 'uid_ans_emb_51', 'uid_ans_emb_52', 'uid_ans_emb_53', 'uid_ans_emb_54', 'uid_ans_emb_55', 'uid_ans_emb_56', 'uid_ans_emb_57', 'uid_ans_emb_58', 'uid_ans_emb_59', 'uid_ans_emb_60', 'uid_ans_emb_61', 'uid_ans_emb_62', 'uid_ans_emb_63', 'ques_emb_0', 'ques_emb_1', 'ques_emb_2', 'ques_emb_3', 'ques_emb_4', 'ques_emb_5', 'ques_emb_6', 'ques_emb_7', 'ques_emb_8', 'ques_emb_9', 'ques_emb_10', 'ques_emb_11', 'ques_emb_12', 'ques_emb_13', 'ques_emb_14', 'ques_emb_15', 'ques_emb_16', 'ques_emb_17', 'ques_emb_18', 'ques_emb_19', 'ques_emb_20', 'ques_emb_21', 'ques_emb_22', 'ques_emb_23', 'ques_emb_24', 'ques_emb_25', 'ques_emb_26', 'ques_emb_27', 'ques_emb_28', 'ques_emb_29', 'ques_emb_30', 'ques_emb_31', 'ques_emb_32', 'ques_emb_33', 'ques_emb_34', 'ques_emb_35', 'ques_emb_36', 'ques_emb_37', 'ques_emb_38', 'ques_emb_39', 'ques_emb_40', 'ques_emb_41', 'ques_emb_42', 'ques_emb_43', 'ques_emb_44', 'ques_emb_45', 'ques_emb_46', 'ques_emb_47', 'ques_emb_48', 'ques_emb_49', 'ques_emb_50', 'ques_emb_51', 'ques_emb_52', 'ques_emb_53', 'ques_emb_54', 'ques_emb_55', 'ques_emb_56', 'ques_emb_57', 'ques_emb_58', 'ques_emb_59', 'ques_emb_60', 'ques_emb_61', 'ques_emb_62', 'ques_emb_63', 'user_int_0', 'user_int_1', 'user_int_2', 'user_int_3', 'user_int_4', 'user_int_5', 'user_int_6', 'user_int_7', 'user_int_8', 'user_int_9', 'user_int_10', 'user_int_11', 'user_int_12', 'user_int_13', 'user_int_14', 'user_int_15', 'user_int_16', 'user_int_17', 'user_int_18', 'user_int_19', 'user_int_20', 'user_int_21', 'user_int_22', 'user_int_23', 'user_int_24', 'user_int_25', 'user_int_26', 'user_int_27', 'user_int_28', 'user_int_29', 'user_int_30', 'user_int_31', 'user_int_32', 'user_int_33', 'user_int_34', 'user_int_35', 'user_int_36', 'user_int_37', 'user_int_38', 'user_int_39', 'user_int_40', 'user_int_41', 'user_int_42', 'user_int_43', 'user_int_44', 'user_int_45', 'user_int_46', 'user_int_47', 'user_int_48', 'user_int_49', 'user_int_50', 'user_int_51', 'user_int_52', 'user_int_53', 'user_int_54', 'user_int_55', 'user_int_56', 'user_int_57', 'user_int_58', 'user_int_59', 'user_int_60', 'user_int_61', 'user_int_62', 'user_int_63', 'user_fol_0', 'user_fol_1', 'user_fol_2', 'user_fol_3', 'user_fol_4', 'user_fol_5', 'user_fol_6', 'user_fol_7', 'user_fol_8', 'user_fol_9', 'user_fol_10', 'user_fol_11', 'user_fol_12', 'user_fol_13', 'user_fol_14', 'user_fol_15', 'user_fol_16', 'user_fol_17', 'user_fol_18', 'user_fol_19', 'user_fol_20', 'user_fol_21', 'user_fol_22', 'user_fol_23', 'user_fol_24', 'user_fol_25', 'user_fol_26', 'user_fol_27', 'user_fol_28', 'user_fol_29', 'user_fol_30', 'user_fol_31', 'user_fol_32', 'user_fol_33', 'user_fol_34', 'user_fol_35', 'user_fol_36', 'user_fol_37', 'user_fol_38', 'user_fol_39', 'user_fol_40', 'user_fol_41', 'user_fol_42', 'user_fol_43', 'user_fol_44', 'user_fol_45', 'user_fol_46', 'user_fol_47', 'user_fol_48', 'user_fol_49', 'user_fol_50', 'user_fol_51', 'user_fol_52', 'user_fol_53', 'user_fol_54', 'user_fol_55', 'user_fol_56', 'user_fol_57', 'user_fol_58', 'user_fol_59', 'user_fol_60', 'user_fol_61', 'user_fol_62', 'user_fol_63', 'hour_bin', 'weekend', 'weekend_hour_bin', 'hour_bin_label_w_mean', 'hour_bin_label_w_sum', 'weekend_hour_bin_label_w_mean', 'weekend_hour_bin_label_w_sum', 'uid_hour_bin_label_w_count_x', 'uid_hour_bin_label_w_sum', 'uid_hour_bin_label_w_mean', 'uid_weekend_hour_bin_label_w_count_x', 'uid_weekend_hour_bin_label_w_sum', 'uid_weekend_hour_bin_label_w_mean', 'uid_hour_bin_label_w_count_y', 'uid_weekend_hour_bin_label_w_count_y', 'day_rank', 'week_rank', 'qid_topic_rel_size_w_mean', 'qid_topic_rel_size_w_max', 'qid_topic_rel_size_w_std', 'qid_topic_rel_size_w_min', 'qid_topic_qid_size_w_mean', 'qid_topic_qid_size_w_max', 'qid_topic_qid_size_w_std', 'qid_topic_qid_size_w_min']

```

最终的结果，在working目录会生成result.txt及相应的zip文件

```
cd working
# 生成特征
sh run_data.sh
# 训练和预测
sh run_predict.sh
```

# 文件说明

- 1xx 表示基础数据生成
- 2xx 表示特征抽取
- 3xx 表示更高层次的特征计算
- 4xx merge所有特征
- 5xx 训练
- 6xx 预测

# cv

> 使用train中7天的数据做为label区，划分两个label区，7天前的数据，做为feature区

# 融合

最终的分数结果，使用了多个版本模型的融合，包含

1. fold 3训练模型
2. 全量数据训练模型
3. 加入预赛结果数据取概率大于95%和小于1%的作为伪标签

融合方法，所有预测结果，做mean，能提升0.005，因为在实际工作中不会使用，故此处不包括这部分了内容

# 楼上的大佬

* 4th: https://github.com/VoldeMortzzz/2019Baai-zhihu-Cup-findexp-4th
* 6th: https://github.com/liuchenailq/zhihu-findexp

# 知乎文章

里面比赛过程的一些记录

https://zhuanlan.zhihu.com/p/101161233

# 拿胡渊鸣大佬的文章镇楼

https://zhuanlan.zhihu.com/p/97700605


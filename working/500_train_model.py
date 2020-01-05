# -*- coding: utf-8 -*-
from util import *
import argparse
import xgboost as xgb

try:
    from lightgbm import LGBMClassifier
except:
    print('import lgb fail')
from catboost import CatBoostClassifier

pd.set_option('display.max_rows', 1000)
import gc


# f1 = load_feature('inv_feature_1.dump')


##########################
# 拼接
##########################


def show_impt(model_lgb):
    df = pd.DataFrame()
    df['name'] = label.columns
    df['score'] = model_lgb.feature_importances_
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df


SEED = 47


def get_model():
    logger.info("model type %s", model_type)
    if model_type == 'lgb':
        params = {'num_leaves': 445,
                  'n_estimators': total_round,
                  'min_child_weight': 0.14743644917196846,
                  'feature_fraction': 0.5613611767691925,
                  'bagging_fraction': 0.9417122177047442,
                  'min_data_in_leaf': 21,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.05197017180724227,
                  "boosting_type": "gbdt",
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "verbosity": -1,
                  'reg_alpha': 1.3471802314345476,
                  'reg_lambda': 1.8902863480298606,
                  'random_state': SEED
                  }
        model = LGBMClassifier(**params)
        return model
    elif model_type == 'xgb':
        params = {
            "n_estimators": total_round,
            "max_depth": 12,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "min_child_weight": 2.5,
            "tree_method": 'hist',
            'n_jobs': -1,
            'eval_metric': 'auc',
            'random_state': SEED,
            'reg_alpha': 0.5,
            'reg_lambda': 0.4,
            # 'missing': -999,
        }
        #
        model = xgb.XGBClassifier(
            **params
        )
        return model
    elif model_type == 'cat':

        cat_params = {
            'n_estimators': total_round,
            # 'learning_rate': 0.07,
            'thread_count': 24,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'random_seed': SEED,
            'metric_period': 100,
            'od_wait': 50,
            # 'task_type': 'GPU',
            'depth': 12,
            'l2_leaf_reg': 3,
            # 'colsample_bylevel':0.7,
        }
        model = CatBoostClassifier(**cat_params)
        return model


# nohup python 500_train_model.py --round 1 --frac 0.05 &

# nohup python 500_train_model.py --round 5000 --frac 0.3 &

# nohup python 500_train_model.py --round 5000 --frac 0.3 --train_data label_v7 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', type=float, default=config.frac)
    parser.add_argument('--round', type=int, default=config.round)
    parser.add_argument('--train_data', type=str, default=f'{config.label_name}3')
    seed_everything(2019)

    args = parser.parse_args()
    frac = args.frac
    total_round = args.round
    train_data = args.train_data
    model_type = 'lgb'
    day = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    logger.info("day %s params %s", day, args)

    logger = configure_logging('train_' + day)

    label = sample_label(load1(train_data, 0), frac)
    label1 = sample_label(load1(train_data, 1), frac)

    logger.info("label %s, label1 %s", len(label), len(label1))
    label = pd.concat((label, label1))
    del label1
    logger.info("final train data %s", label.shape)
    gc.collect()

    org_label_size = len(label)

    cat_cols = ['gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'weekday', 'hour']

    for col in ['topic', 'follow_topic', 'inter_topic']:
        if col in label:
            del label[col]
    gc.collect()

    feature_cols = [x for x in label.columns if
                    x not in drop_feature]
    # ,'uid_enc','qid_enc'
    # target编码
    logger.info("feature size %s, %s", len(feature_cols), feature_cols)

    y_train_all = label['label']

    n_fold = 5
    fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    feature_df = pd.DataFrame()
    label = label[feature_cols]
    y_train = y_train_all
    for index, (train_idx, val_idx) in enumerate(fold.split(X=label, y=y_train_all)):

        if index > 0:
            break
        X_val, y_val = label.iloc[val_idx], \
                       y_train_all.iloc[val_idx]
        # index = X_train[X_train]
        # logger.info("fold %s train: %s, val: %s", index, X_train.shape, X_val.shape)
        model = get_model()

        model.fit(label, y_train,
                  eval_metric=['logloss', 'auc'],
                  eval_set=[(X_val, y_val)],
                  verbose=50,
                  early_stopping_rounds=50,
                  )

        # h = clf.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT],
        #             eval_set=[(X_train[cols].iloc[idxV], y_train.iloc[idxV])],
        #             verbose=100, early_stopping_rounds=200)

        y_pred = model.predict_proba(X_val)[:, 1]

        # filename = f'model_{index}_{day}_{model_type}.pkl'
        filename = config.model_name
        dump_pkl(model, filename)
        auc = metrics.roc_auc_score(y_val, y_pred)
        logger.info("model %s, fold %s, auc %s ", filename, index, auc)
        fold_feature_df = pd.DataFrame()
        fold_feature_df['feature'] = feature_cols
        fold_feature_df['importance'] = model.feature_importances_
        fold_feature_df['fold'] = index
        feature_df = pd.concat([feature_df, fold_feature_df], axis=0)

        gc.collect()
    # joblib.dump(model_lgb, f'../submit/model_{day}.dump')

    t = feature_df[['feature', 'importance']].groupby(['feature'])['importance'].mean().sort_values(ascending=False)
    logger.info('%s', t)

    # df['index'] = np.arange(1, len(df) + 1)
    logger.info("feature size %s", len(t))

    logger.info(t.head(100))

    t.to_csv(f'../feature/importance_{day}_{model_type}.csv')

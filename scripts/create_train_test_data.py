"""特徴量生成スクリプト
    実行コマンド
        docker exec -it 3caa389926c5 /bin/bash -c "cd ./atmaCup_vol10/scripts && python3 create_train_test_data.py"

"""

import os
import time
import warnings
from functools import wraps
import math
import psutil

import numpy as np
import pandas as pd
import yaml
from PIL import ImageColor
from tqdm import tqdm as tqdm

from takaggle.feature import category_encoder, feature_engineering
from takaggle.training import util

tqdm.pandas()
warnings.filterwarnings("ignore")

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file, Loader=yaml.FullLoader)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']
TRAIN_FILE_NAME = yml['SETTING']['TRAIN_FILE_NAME']
TEST_FILE_NAME = yml['SETTING']['TEST_FILE_NAME']
TARGET_COL = yml['SETTING']['TARGET_COL']


def elapsed_time(f):
    """関数の処理時間を計測してlogに出力するデコレータ"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        p = psutil.Process(os.getpid())
        m0 = p.memory_info()[0] / 2. ** 30
        logger.info(f'Begin:\t{f.__name__}')
        v = f(*args, **kwargs)
        m1 = p.memory_info()[0] / 2. ** 30
        delta = m1 - m0
        sign = '+' if delta >= 0 else '-'
        delta = math.fabs(delta)
        logger.info(f'End:\t{f.__name__} {time.time() - start:.2f}sec [{m1:.1f}GB({sign}{delta:.1f}GB)]')
        return v
    return wrapper


# seedの固定
def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


@elapsed_time
def load_data():
    """データをロードする"""
    files = {
        'train': 'train.csv',
        'test': 'test.csv',
        'color': 'color.csv',
        'palette': 'palette.csv',
        'material': 'material.csv',
        'historical_person': 'historical_person.csv',
        'object_collection': 'object_collection.csv',
        'production_place': 'production_place.csv',
        'technique': 'technique.csv',
        'maker': 'maker.csv',
        'principal_maker': 'principal_maker.csv',
        'principal_maker_occupation': 'principal_maker_occupation.csv',
    }
    dfs = {}
    for k, v in files.items():
        dfs[k] = pd.read_csv(RAW_DATA_DIR_NAME + v)

    return dfs


def hex_to_rgb(input_df, col):
    """hexコードを対応するRGBへ変換する"""
    rbg_df = pd.DataFrame(input_df[col].str.strip().map(ImageColor.getrgb).values.tolist(), columns=['color_R', 'color_G', 'color_B'])
    return pd.concat([input_df, rbg_df], axis=1)


@elapsed_time
def preprocessing_color(input_df):
    """colorの前処理
    col:
        object_id
        percentage: 全体にしめる割合 (%)
        hex: 色
    """

    # hexをrgbに変換
    df = hex_to_rgb(input_df, 'hex')
    # RGBに対してpercentageで比率を計算
    rgb_df = df[['color_R', 'color_G', 'color_B']].apply(lambda x: x * (df.loc[:, 'percentage'] * 0.01))
    # マージ
    df = pd.concat([df[['object_id']], rgb_df], axis=1)
    # RGBの合計値を取得
    rgf_sum_df = df.groupby('object_id')[['color_R', 'color_G', 'color_B']].sum().reset_index()

    # percentageの分散（どのくらいの種類の色が平均的に使われているか）
    # これが小さいということは、多くの色が均等に使われている、ということになる
    percentage_std_df = input_df.groupby('object_id')['percentage'].std().reset_index().rename(columns={'percentage': 'percentage_std'})
    output_df = pd.merge(rgf_sum_df, percentage_std_df, how='left', on='object_id')
    return output_df


@elapsed_time
def preprocessing_material(input_df):
    """materialデータの前処理を行う
    作品の材料が紐付いたテーブルです。ひとつの作品に対して複数対応する可能性があります。

    col:
        object_id
        name: 材料名
    """
    # カウントエンコーディング
    count_df = category_encoder.count_encoder(input_df, ['material_name'])
    # object_idごとにどのくらいレアな材料を使っているか
    rare_material_df = count_df.groupby('object_id')[['material_name_count_enc']].sum().reset_index().rename(columns={'material_name_count_enc': 'material_count_enc_sum'})

    # object_idごとにいくつの材料があるか
    material_sum_df = input_df.groupby('object_id')[['material_name']].count().reset_index().rename(columns={'material_name': 'material_sum_by_object'})
    output_df = pd.merge(rare_material_df, material_sum_df, how='left', on='object_id')

    return output_df


@elapsed_time
def preprocessing_art(input_df):
    """art(train/test)の前処理"""

    # とりあえず学習できるカラムだけ取得
    num_cols = feature_engineering.get_num_col(input_df)
    cat_cols = ['principal_maker', 'principal_or_first_maker', 'copyright_holder', 'acquisition_method', 'acquisition_credit_line']  # ラベルエンコードするカテゴリカラム
    output_df = input_df[['object_id'] + num_cols + cat_cols].copy()
    output_df = category_encoder.sklearn_label_encoder(output_df, cat_cols, drop_col=True)
    return output_df


@elapsed_time
def merge_data(art, color, material):
    """データをマージする"""
    outout_df = pd.merge(art, color, how='left', on='object_id')
    outout_df = pd.merge(outout_df, material, how='left', on='object_id')
    return outout_df


@elapsed_time
def save_data(input_df, train_len):
    """pkl形式でデータを保存する"""

    # trainとtestを再分割
    train = input_df.iloc[:train_len, :]
    test = input_df.iloc[train_len:, :]
    test = test.drop(TARGET_COL, axis=1)
    # indexの初期化
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    assert train.shape[1]-1 == test.shape[1], 'trainとtestのデータ形式が異なります'

    # 保存
    train.to_pickle(FEATURE_DIR_NAME + TRAIN_FILE_NAME)
    test.to_pickle(FEATURE_DIR_NAME + TEST_FILE_NAME)
    logger.info_log(f'train shape: {train.shape}, test shape, {test.shape}')

    # 学習に使用できるカラムを出力
    category_cols = feature_engineering.get_category_col(input_df)
    # 学習に不要なカラム出力から除外
    features_list = list(set(input_df.columns) - {TARGET_COL} - set(category_cols))

    # 特徴量リストの保存
    features_list = sorted(features_list)
    with open(FEATURE_DIR_NAME + 'features_list.txt', 'wt') as f:
        for i in range(len(features_list)):
            f.write('  - ' + str(features_list[i]) + '\n')

    return None


def main():

    # データのロード
    dfs = load_data()

    # id毎にデータを集約する
    color = preprocessing_color(dfs['color'])
    material = preprocessing_material(dfs['material'].rename(columns={'name': 'material_name'}))

    # train / testの前処理
    art = pd.concat([dfs['train'], dfs['test']], axis=0, sort=False).reset_index(drop=True)
    art = preprocessing_art(art)

    # データをマージしてtrainとtestを生成する
    df = merge_data(art, color, material)

    # データの保存
    save_data(df, len(dfs['train']))

    return 'main done'


if __name__ == "__main__":

    # loggerの設定
    global logger, run_name, ts
    logger = util.Logger()
    logger.info_log('★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆')

    # seedの固定
    seed_everything()

    main()

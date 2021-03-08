"""特徴量生成スクリプト
    実行コマンド
        docker exec -it 962f598235e9 /bin/bash -c "cd ./atmaCup_vol10/scripts && python3 create_train_test_data.py"

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
from takaggle.training.util import Logger

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


def aggregation(df, target_col, agg_target_col):
    """集計特徴量の生成処理

    Args:
        df (pd.DataFrame): 対象のDF
        target_col (list of str): 集計元カラム（多くの場合カテゴリ変数のカラム名リスト）
        agg_target_col (str): 集計対象のカラム（多くの場合連続変数）

    Returns:
        pd.DataFrame: データフレーム
    """

    # カラム名を定義
    target_col_name = ''
    for col in target_col:
        target_col_name += str(col)
        target_col_name += '_'

    gr = df.groupby(target_col)[agg_target_col]
    df[f'{target_col_name}{agg_target_col}_mean'] = gr.transform('mean').astype('float16')
    df[f'{target_col_name}{agg_target_col}_max'] = gr.transform('max').astype('float16')
    df[f'{target_col_name}{agg_target_col}_min'] = gr.transform('min').astype('float16')
    df[f'{target_col_name}{agg_target_col}_std'] = gr.transform('std').astype('float16')
    df[f'{target_col_name}{agg_target_col}_median'] = gr.transform('median').astype('float16')

    # 自身の値との差分
    df[f'{target_col_name}{agg_target_col}_mean_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_mean']
    df[f'{target_col_name}{agg_target_col}_max_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_max']
    df[f'{target_col_name}{agg_target_col}_min_diff'] = df[agg_target_col] - df[f'{target_col_name}{agg_target_col}_min']

    return df


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

    def get_material_cumcount_df(input_df, prefix):
        """input_dfに存在するカテゴリ変数を横に展開する
        """
        _df = input_df.copy()
        max_size = _df.groupby('object_id').size().max()
        _df['cumcount'] = _df.groupby('object_id').cumcount()
        output_df = pd.DataFrame({'object_id': _df['object_id'].unique()})

        for i in range(max_size):
            temp_df = _df[_df['cumcount']==i].reset_index(drop=True)
            output_df = output_df.merge(temp_df[['object_id', 'material_name']], on='object_id', how='left').rename(columns={'material_name':f'{prefix}_{i}'})

        # ラベルエンコーディング
        material_cols = [col for col in output_df.columns if col.startswith('material_')]
        output_df = category_encoder.sklearn_label_encoder(output_df, material_cols, drop_col=True)
        return output_df

    # カウントエンコーディング
    count_df = category_encoder.count_encoder(input_df, ['material_name'])
    # object_idごとにどのくらいレアな材料を使っているか
    rare_material_df = count_df.groupby('object_id')[['material_name_count_enc']].sum().reset_index().rename(columns={'material_name_count_enc': 'material_count_enc_sum'})

    # object_idごとにいくつの材料があるか
    material_sum_df = input_df.groupby('object_id')[['material_name']].count().reset_index().rename(columns={'material_name': 'material_sum_by_object'})
    output_df = pd.merge(rare_material_df, material_sum_df, how='left', on='object_id')

    # 素材を横に並べたDFを取得
    material_cumcount_df = get_material_cumcount_df(input_df, 'material')
    output_df = pd.merge(output_df, material_cumcount_df, how='left', on='object_id')

    return output_df


@elapsed_time
def preprocessing_art(input_df):
    """art(train/test)の前処理"""

    def get_size_from_subtitle(input_df):
        """subtitleからサイズを抽出"""
        output_df = input_df.copy()
        for axis in ['h', 'w', 't', 'd']:
            column_name = f'size_{axis}'
            size_info = output_df['sub_title'].str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis))  # 正規表現を使ってサイズを抽出
            size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
            size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float)  # dtypeがobjectになってるのでfloatに直す
            size_info[column_name] = size_info.apply(lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1) # 　単位をmmに統一する
            output_df[column_name] = size_info[column_name]
        return output_df

    def create_day_feature(df, col, prefix, change_utc2asia=False,
                           attrs=['year', 'quarter', 'month', 'week', 'day', 'dayofweek', 'hour', 'minute']):
        """日時特徴量の生成処理

        Args:
            df (pd.DataFrame): 日時特徴量を含むDF
            col (str)): 日時特徴量のカラム名
            prefix (str): 新しく生成するカラム名に付与するprefix
            attrs (list of str): 生成する日付特徴量. Defaults to ['year', 'quarter', 'month', 'week', 'day', 'dayofweek', 'hour', 'minute']
                                cf. https://qiita.com/Takemura-T/items/79b16313e45576bb6492

        Returns:
            pd.DataFrame: 日時特徴量を付与したDF

        """

        df.loc[:, col] = pd.to_datetime(df[col])
        for attr in attrs:
            dtype = np.float32 if attr == 'year' else np.float16
            df[prefix + '_' + attr] = getattr(df[col].dt, attr).astype(dtype)

        return df

    output_df = input_df.copy()

    # subtitleからサイズの情報を取得する
    output_df = get_size_from_subtitle(input_df)

    # 日付特徴量
    # 製作期間
    output_df['production_period'] = output_df['dating_year_late'] - output_df['dating_year_early']
    # acquisition_date(収集日)から日付特徴量を取得
    output_df = create_day_feature(output_df, col='acquisition_date', prefix='acquisition', attrs=['year', 'month', 'day', 'dayofweek'])
    # 製作期間が終わってから収集されるまでの期間
    output_df['acquisition_period'] = output_df['acquisition_year'] - output_df['dating_year_late']
    # dating_sorting_dateをbin分割する
    output_df['dating_sorting_date'] = output_df['dating_sorting_date'].fillna(output_df['dating_sorting_date'].mean())  # とりあえず平均で埋める
    century = pd.cut(
        output_df['dating_sorting_date'],
        [-float('inf'), 1499, 1549, 1599, 1649, 1699, 1749, 1799, 1849, 1899, float('inf')],
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    output_df['century'] = century.values.astype(np.int8)

    # テキストカラムの特徴量生成
    text_cols = [
        'title',
        'description',
        'long_title',
        'sub_title',
        'more_title'
    ]
    # 文字列の長さ
    for c in text_cols:
        output_df[f'{c}_text_len'] = output_df[c].str.len()

    # bert vecを結合
    # ref: https://colab.research.google.com/drive/1SEpFu6BuKnf-f7WrjBy3PiDCW4uyrB8O?authuser=3#scrollTo=n7RydsVa945l
    # title_bert_vev = pd.read_pickle(FEATURE_DIR_NAME + 'title_bert_vec.pkl')  # title
    # output_df = pd.concat([output_df, title_bert_vev], axis=1)
    # description_bert_vev = pd.read_pickle(FEATURE_DIR_NAME + 'description_bert_vec.pkl')  # description
    # output_df = pd.concat([output_df, description_bert_vev], axis=1)

    # 学習できるカラムだけ取得
    num_cols = feature_engineering.get_num_col(output_df)
    cat_cols = ['title', 'principal_maker', 'principal_or_first_maker', 'copyright_holder',
                'acquisition_method', 'acquisition_credit_line']  # ラベルエンコードするカテゴリカラム

    output_df = output_df[['object_id'] + num_cols + cat_cols]

    # カテゴリ変数をエンコード
    output_df = category_encoder.count_encoder(output_df, cat_cols)
    output_df = category_encoder.sklearn_label_encoder(output_df, cat_cols, drop_col=True)
    return output_df


@elapsed_time
def merge_data(art, color, material):
    """データをマージする"""
    outout_df = pd.merge(art, color, how='left', on='object_id')
    outout_df = pd.merge(outout_df, material, how='left', on='object_id')
    return outout_df


@elapsed_time
def agg_features(input_df):
    """集計特徴量を生成する"""

    output_df = feature_engineering.aggregation(input_df, ['century'], 'description_text_len')
    output_df = feature_engineering.aggregation(input_df, ['century'], 'long_title_text_len')
    output_df = feature_engineering.aggregation(input_df, ['century'], 'more_title_text_len')
    output_df = feature_engineering.aggregation(input_df, ['century'], 'sub_title_text_len')
    output_df = feature_engineering.aggregation(input_df, ['century'], 'title_text_len')
    output_df = feature_engineering.aggregation(input_df, ['century'], 'material_count_enc_sum')

    return output_df


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

    # マージ後のデータで集約特徴量を生成
    # TODO:
    df = agg_features(df)

    # データの保存
    save_data(df, len(dfs['train']))

    return 'main done'


if __name__ == "__main__":

    # loggerの設定
    global logger, run_name, ts
    logger = Logger()
    logger.info_log('★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆')

    # seedの固定
    seed_everything()

    main()

"""学習スクリプト
    実行コマンド
        docker exec -it 962f598235e9 /bin/bash -c "cd ./atmaCup_vol10/scripts && python3 run.py"

    MLFlow起動コマンド


"""

import sys
import os
import datetime
import yaml
import json
import collections as cl
import warnings
import fire
import traceback
import mlflow
import shutil

from takaggle.training.runner import Runner
from takaggle.training.model_lgb import ModelLGB
from takaggle.training.model_cb import ModelCB
# from takaggle.training.util import Submission
from atma10_util import atma10_Submission

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

key_list = ['use_features', 'model_params', 'cv', 'setting']

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
EXPERIMENT_NAME = yml['SETTING']['EXPERIMENT_NAME']
TRACKING_DIR = yml['SETTING']['TRACKING_DIR']
NOTE = '擬似ラベルを追加'  # どんな実験内容かを簡単にメモする

# 使用する特徴量のロード
with open('../configs/feature.yaml') as file:
    feature_yml = yaml.load(file)


def exist_check(path, run_name) -> None:
    """モデルディレクトリの存在チェック

    Args:
        path (str): モデルディレクトリのpath
        run_name (str): チェックするrun名

    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)
    return None


def my_makedirs(path) -> None:
    """引数のpathディレクトリが存在しなければ、新規で作成する

    Args:
        path (str): 作成するディレクトリ名

    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return None


def save_model_config(key_list, value_list, dir_name, run_name) -> None:
    """学習のjsonファイル生成

    どんなパラメータ/特徴量で学習させたモデルかを管理するjsonファイルを出力する

    """
    def set_default(obj):
        """json出力の際にset型のオブジェクトをリストに変更する"""
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

    ys = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        ys[v] = data
        mlflow.log_param(v, data)
    fw = open(dir_name + run_name + '_param.json', 'w')
    json.dump(ys, fw, indent=4, default=set_default)
    return None


def get_cv_info() -> dict:
    """CVの情報を設定する

    methodは[KFold, StratifiedKFold ,GroupKFold, StratifiedGroupKFold, CustomTimeSeriesSplitter, TrainTestSplit]から選択可能
    CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    StratifiedKFold or GroupKFold or StratifiedGroupKFold の場合はcv_target_gr, cv_target_sfに対象カラム名を設定する

    Returns:
        dict: cvの辞書

    """
    return yml['SETTING']['CV']


def get_run_name(cv, model_type):
    """run名を設定する
    """
    run_name = model_type
    suffix = '_' + datetime.datetime.now().strftime("%m%d%H%M")
    model_info = ''
    run_name = run_name + '_' + cv.get('method') + model_info + suffix
    return run_name


def get_setting_info():
    """setting情報を設定する
    """
    setting = {
        'experiment_note': NOTE,
        'feature_directory': yml['SETTING']['FEATURE_DIR_NAME'],  # 特徴量の読み込み先ディレクトリ
        'model_directory': MODEL_DIR_NAME,  # モデルの保存先ディレクトリ
        'train_file_name': yml['SETTING']['TRAIN_FILE_NAME'],
        'test_file_name': yml['SETTING']['TEST_FILE_NAME'],
        'target': yml['SETTING']['TARGET_COL'],  # 目的変数
        'metrics': yml['SETTING']['METRICS'],  # 評価指標
        'calc_shap': yml['SETTING']['CALC_SHAP'],  # shap値を計算するか否か
        'save_train_pred': yml['SETTING']['SAVE_TRAIN_PRED']  # trainデータでの推論値を特徴量として加えたい場合はTrueに設定する
    }
    return setting


def main(model_type='lgb') -> str:
    """トレーニングのmain関数

    model_typeによって学習するモデルを変更する
    → lgb, cb, xgb, nnが標準で用意されている

    Args:
        model_type (str, optional): どのモデルで学習させるかを指定. Defaults to 'lgb'.

    Returns:
        str: [description]

    Examples:
        >>> python hoge.py --model_type="lgb"
        >>> python hoge.py lgb

    """

    cv = get_cv_info()  # CVの情報辞書
    run_name = get_run_name(cv, model_type)  # run名
    dir_name = MODEL_DIR_NAME + run_name + '/'  # 学習に使用するディレクトリ
    setting = get_setting_info()  # 諸々の設定ファイル辞書

    # すでに実行済みのrun名がないかチェックし、ディレクトリを作成する
    exist_check(MODEL_DIR_NAME, run_name)
    my_makedirs(dir_name)

    # configファイルをコピーする
    shutil.copy("../configs/config.yaml", dir_name + run_name + "_config.yaml")
    shutil.copy("../configs/feature.yaml", dir_name + run_name + "_feature.yaml")

    # モデルに合わせてパラメータを読み込む
    model_cls = None
    model_params = None
    if model_type == 'lgb':
        model_params = yml['MODEL_LGB']['PARAM']
        model_cls = ModelLGB
    elif model_type == 'cb':
        model_params = yml['MODEL_CB']['PARAM']
        model_cls = ModelCB
    elif model_type == 'xgb':
        pass
    elif model_type == 'nn':
        pass
    else:
        print('model_typeが不正なため終了します')
        sys.exit(0)

    features = feature_yml['FEATURES']

    # mlflowトラッキングの設定
    mlflow.set_tracking_uri(TRACKING_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name=run_name)

    category_col = ['principal_maker_lbl_enc', 'principal_or_first_maker_lbl_enc']

    try:
        # インスタンス生成
        runner = Runner(run_name, model_cls, features, setting, model_params, cv, category_col,
                        is_add_pseudo=True, pseudo_label_file='pseudo_lb_0p9826.pkl')
        use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

        # モデルのconfigをjsonで保存
        value_list = [use_feature_name, model_params, cv, setting]
        save_model_config(key_list, value_list, dir_name, run_name)

        # 学習・推論
        runner.run_train_cv()
        runner.run_predict_cv()

        # submit作成
        atma10_Submission.create_submission(run_name, dir_name, 'atmacup10__sample_submission.csv', setting.get('target'))

        if model_type == 'lgb':
            # feature_importanceを計算
            ModelLGB.calc_feature_importance(dir_name, run_name, use_feature_name, cv.get('n_splits'), type='gain')

        print('Done')

    except Exception as e:
        print(traceback.format_exc())
        print(f'ERROR:{e}')


if __name__ == '__main__':
    fire.Fire(main)

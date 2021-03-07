import pandas as pd
import yaml
from takaggle.training.util import Submission, Logger, Util

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file, Loader=yaml.FullLoader)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']


class atma10_Submission(Submission):

    @classmethod
    def create_submission(cls, run_name, pred_file_path, sample_sub_file_name, sub_y_column):
        """submission用のCSVを生成する関数

        Args:
            run_name (str)): 実験名
            pred_file_path (str)): 予測結果のpklファイルが出力されたディレクトリ
            sample_sub_file_name (str): sample_submission用のcsvファイル. sub_y_columnをカラムに持つデータ. サンプル数はtestと等価でないといけない.
            sub_y_column (str): 目的変数のカラム名. ここに設定したカラム名でcsvファイルが出力される

        Return:
            None

        """

        logger = Logger(pred_file_path)
        logger.info(f'{run_name} - start create submission')

        submission = pd.read_csv(RAW_DATA_DIR_NAME + sample_sub_file_name)
        pred = Util.load_df_pickle(pred_file_path + f'{run_name}_pred.pkl')
        submission[sub_y_column] = pred
        submission.loc[submission[sub_y_column] <= 0, sub_y_column] = 0  # add atma10 0以下を0に変換
        submission.to_csv(pred_file_path + f'{run_name}_submission.csv', index=False)

        logger.info(f'{run_name} - end create submission')

        return None

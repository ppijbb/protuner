import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

class outlier:
    def __init__(self):
        self._ = 0

    def get_outlier(self,df=None, column=None, weight=1.5):
        # target 값과 상관관계가 높은 열을 우선적으로 진행
        quantile_25 = np.percentile(df[column].values, 25)
        quantile_75 = np.percentile(df[column].values, 75)

        IQR = quantile_75 - quantile_25
        IQR_weight = IQR * weight

        lowest = quantile_25 - IQR_weight
        highest = quantile_75 + IQR_weight

        outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index

        return outlier_idx

class matplot_fonts:
    def __init__(self):
        self.sys = platform.system()

    def kor_font(self):
        if self.sys == 'Darwin':  # 맥
            plt.rc('font', family='AppleGothic')
        elif self.sys == 'Windows':  # 윈도우
            plt.rc('font', family='Malgun Gothic')
        elif self.sys == 'Linux':  # 리눅스 (구글 콜랩)
            # !wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
            # !mv malgun.ttf /usr/share/fonts/truetype/
            # import matplotlib.font_manager as fm
            # fm._rebuild()
            plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결

class data_processing:
    def __init__(self,data,b):
        self.data = data
        self.b = b
    def binding(self, x):
        if type(x)== pd.Series:
            return pd.cut(x, bins=np.histogram(x, bins=self.b)[1], labels=False)
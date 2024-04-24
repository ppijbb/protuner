from ProTuner.src import utility as u, analysis
from ProTuner.src import pro
import pandas as pd

data = pd.read_excel("./Data/DepressPeptideData.xlsx",index_col='Unnamed: 0')

# 데이터 수정
data = data.drop(columns=['나이','성별'])
data['연구실코드'] = data['연구실코드'].replace({'SEC': 0, 'ART': 1, 'ADT': 1}).copy()

if __name__=='__main__':
    util = u.matplot_fonts()
    util.kor_font()
    # 데이터 전처리 필요
   #bind_data = cont_data.apply(u.data_processing(cont_data,b=12).binding, axis=0).fillna(0).copy()
   #new_data = pd.concat([cate_data, bind_data], axis=1).copy()

    PT = pro.tuner(data, '연구실코드')

    importance_mdi = PT.mdi()
    importance_pi = PT.pi()
    importance_shap = PT.shap()

    summary = analysis.summary(data, '연구실코드','Depress')
    summary.report(importance_mdi,importance_pi,importance_shap)

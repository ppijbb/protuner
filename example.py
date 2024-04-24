from ProTuner.src import utility as u, analysis
from ProTuner.src import pro
import pandas as pd

data = pd.read_excel("./Data/Dataset_TuringBio.xlsx")

# 데이터 수정
data['(성인) 현재 흡연 여부'] = data['(성인) 현재 흡연 여부'].replace({3: 1, 1: 3}).copy()
data['현재 복용 치료제 유무'] = data['현재 복용 치료제 유무'].replace({2: 0, 1: 2, 3: 1}).copy()
data['보조제 복용 유무'] = data['보조제 복용 유무'].replace({2: 1, 1: 2}).copy()

# 필요없는 columns 제거
data = data.drop(columns=['NO', 'Unnamed: 0', '현재 복용 치료제 유무', '고혈압 의사진단', '고지혈증 의사진단', '여가_중강도 신체활동 시간(분)',
                          '현재 복용 치료제 종류']).copy()

# 자가문진 가능한 11개 columns(현재 복용 치료제 유무 제거시)
cate_data = data.loc[:, ['규칙적 운동 여부', '보조제 복용 유무', '스트레스3_무기력감', '스트레스2_신경질',  # '현재 복용 치료제 유무',
                         '여가_중강도 신체활동 여부', '자신의 건강', '스트레스5_피로', '음주 여부 및 음주량',
                         '스트레스1_긴장, 불안', '스트레스8_대면어려움', '스트레스9_시선어려움']].copy()

# 연속형 변수 columns 22개
cont_data = data.loc[:,
            ['SBP 2차', 'DBP 1차', 'SBP 1차', 'Vit E(mg)', 'HDL', 'LDL', 'LDL-c', 'HCT', 'CHOL', '회분(g)', '식물성 Fe(mg)',
             'HGB', '비만진단-복부지방률', 'Mo(ug)', 'RBC', 'MONO', 'VitB2(mg)', '동물성 단백질(g)', 'Cu(ug)', 'Vit C(mg)',
             'Protein(g)', 'WBC']].copy()

# 자가문진 가능한 11개 columns(현재 복용 치료제 유무 제거시)
cate_data = data.loc[:, ['규칙적 운동 여부', '보조제 복용 유무', '스트레스3_무기력감', '스트레스2_신경질',  # '현재 복용 치료제 유무',
                         '여가_중강도 신체활동 여부', '자신의 건강', '스트레스5_피로', '음주 여부 및 음주량',
                         '스트레스1_긴장, 불안', '스트레스8_대면어려움', '스트레스9_시선어려움']].copy()

# 연속형 변수 columns 22개
cont_data = data.loc[:,
            ['SBP 2차', 'DBP 1차', 'SBP 1차', 'Vit E(mg)', 'HDL', 'LDL', 'LDL-c', 'HCT', 'CHOL', '회분(g)', '식물성 Fe(mg)',
             'HGB', '비만진단-복부지방률', 'Mo(ug)', 'RBC', 'MONO', 'VitB2(mg)', '동물성 단백질(g)', 'Cu(ug)', 'Vit C(mg)',
             'Protein(g)', 'WBC']].copy()

# 범주형 데이터 columns
cate_columns = ['Sex', '보조제 복용 유무', '자신의 건강', '현재 복용 치료제 유무', '현재 복용 치료제 종류',
                '규칙적 운동 여부', '고혈압 의사진단', '고지혈증 의사진단',
                '스트레스1_긴장, 불안', '스트레스2_신경질', '스트레스3_무기력감', '스트레스4_집중저하', '스트레스5_피로',
                '스트레스6_곤욕', '스트레스7_고민', '스트레스8_대면어려움', '스트레스9_시선어려움', '스트레스10_대면불편감',
                '여가_중강도 신체활동 여부', '여가_중강도 신체활동 일수', '여가_중강도 신체활동 시간(시간)', '여가_중강도 신체활동 시간(분)',
                '평소 하루 앉아서 보내는 시간(시간)', '(성인) 현재 흡연 여부', '(성인) 현재흡연자 하루 평균 흡연량', '(성인) 가끔흡연자 1달간 흡연일수',
                '3.과거흡연현재금연_끊은년수', '3.과거흡연 현재금연_과거흡연기간(년)', '3.과거흡연 현재금연_하루평균흡연량(개비)', '음주 여부 및 음주량',
                '음주 시 1회 음주량(잔)', '음주 시 1회 음주 종류']

if __name__=='__main__':
    util = u.matplot_fonts()
    util.kor_font()
    # 데이터 전처리 필요


   #bind_data = cont_data.apply(u.data_processing(cont_data,b=12).binding, axis=0).fillna(0).copy()
   #new_data = pd.concat([cate_data, bind_data], axis=1).copy()

    PT = pro.tuner(data, '현질환')

    importance_mdi = PT.mdi()
    importance_pi = PT.pi()
    importance_shap = PT.shap()

    summary = analysis.summary(data, '현질환','Cardio')
    summary.report(importance_mdi,importance_pi,importance_shap)
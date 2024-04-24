from ProTuner.src import utility as u
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import *
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from matplotlib.backends.backend_pdf import PdfPages


class summary:
    def __init__(self,data,target,name):
        self.data = data
        self.target = target
        self.name = name

    def report(self,mdi,pi,shap):
        importance_shap = shap
        importance_mdi = mdi.iloc[:10]
        importance_pi = pi.iloc[:10]
        data = self.data
        target = self.target
        props = dict(linewidth=3)
        pdf = PdfPages(self.name+'Report.pdf',)

        importance_shap.name = 'importance'
        importance_mdi = importance_mdi.set_index('feature')
        importance_pi = importance_pi.set_index('feature')
        importance_shap = pd.DataFrame(importance_shap)
        importance_shap.index.name = 'feature'
        importance_mdi['Score'], importance_pi['Score'], importance_shap['Score'] = 0, 0, 0

        for i in range(10):
            importance_mdi['Score'][i] = (10 - i)*.8
            importance_pi['Score'][i] = (10 - i)*.9
            importance_shap['Score'][i] = (10 - i)

        result = pd.concat([importance_mdi['Score'], importance_pi['Score'], importance_shap['Score']], axis=1).fillna(0)
        result.columns = ['MDI', 'PI', 'SHAP']
        total = pd.eval("total = result.MDI+result.PI+result.SHAP", target=result)
        total.sort_values(by=['total'], ascending=False, inplace=True)

        total.plot.bar(figsize=(33, 5), grid=True)

        for column in total.index:
            if data[column].nunique() > 20:  # 항목 수가 20개 이상을 연속형 변수로 추정
                fig, ax = plt.subplots(1, 5, figsize=(35, 7))
                plt.suptitle('[{}] {} Analysis Result'.format(total.index.to_list().index(column), column))
                X = data.drop(u.outlier().get_outlier(data, column))[column].copy()
                y = data.loc[X.index, target].values
                X = X.values.reshape(-1, 1)

                clf = LogisticRegression(penalty='l2', C=0.01, random_state=0).fit(X, y)
                LDA = LinearDiscriminantAnalysis().fit(X, y)
                QDA = QuadraticDiscriminantAnalysis().fit(X, y)

                y_score = clf.predict(X)
                text1 = "Mann-Whitney-statistic : %f \n p-value : %f" % mannwhitneyu(data[column][data[target] == 0],
                                                                                     data[column][
                                                                                         data[target] == 1])  # 순위합
                text2 = "T-test-statistic : %f \n p-value : %f" % ttest_ind(data[column][data[target] == 0],
                                                                            data[column][data[target] == 1])  # 평균차이
                text3 = "Shapiro-Wilk-statistic : %f \n p-value : %f" % shapiro(data[column])  # 정규성
                if len(data[column][data[target] == 0]) == len(data[column][data[target] == 1]):
                    text4 = "Paired-T-test-statistic : %f \n p-value : %f" % ttest_rel(data[column][data[target] == 0],
                                                                                       data[column][data[target] == 1])  # paired ttest
                    text5 = "Wilcoxon-statistic : %f \n p-value : %f" % wilcoxon(data[column][data[target] == 0],
                                                                                 data[column][data[target] == 1])  # 윌콕슨
                else:
                    text4 = "Paired-T-test-statistic : not paired Data"
                    text5 = "Wilcoxon-statistic : not paired Data"

                ax[0].boxplot((data[column][data[target] == 0], data[column][data[target] == 1]), sym="o",
                              labels=["대조군", "질환군"],
                              widths=.6, boxprops=props, whiskerprops=props, capprops=props, medianprops=props)
                metrics.plot_roc_curve(clf, X, y, ax=ax[1], lw=3)
                metrics.plot_roc_curve(LDA, X, y, ax=ax[1], ls='--', lw=3)
                metrics.plot_roc_curve(QDA, X, y, ax=ax[1], ls=':', lw=3)
                ax[1].plot([0, 1], [0, 1], linestyle='--', markersize=0.01, color='black')
                ax[1].grid(True)
                ConfusionMatrixDisplay(confusion_matrix(y, y_score)).plot(ax=ax[2])
                sns.distplot(data[column][data[target] == 0], ax=ax[3], label='대조군',kde_kws=props)
                sns.distplot(data[column][data[target] == 1], ax=ax[3], label='질환군',kde_kws=props)
                ax[3].legend()
                ax[4].text(0.1, .9, text1)
                ax[4].text(0.1, .75, text2)
                ax[4].text(0.1, .6, text3)
                ax[4].text(0.1, .45, text4)
                ax[4].text(0.1, .3, text5)
                ax[4].axes.xaxis.set_visible(False)
                ax[4].axes.yaxis.set_visible(False)
                pdf.savefig(fig)

            else:  # 항목 수 20개 이하를 이산형(범주형) 변수로 추정
                fig, ax = plt.subplots(1, 5, figsize=(35, 7))
                plt.suptitle('[{}] {} Analysis Result'.format(total.index.to_list().index(column), column))
                X = data[column].values.reshape(-1, 1)
                y = data[target].values

                clf = LogisticRegression(penalty='l2', C=0.01, random_state=0).fit(X, y)
                LDA = LinearDiscriminantAnalysis().fit(X, y)
                QDA = QuadraticDiscriminantAnalysis().fit(X, y)

                y_score = clf.predict(X)
                text1 = "Mann-Whitney-statistic : %f \n p-value : %f" % mannwhitneyu(data[column][data[target] == 0],
                                                                                     data[column][data[target] == 1])
                text2 = "Chi-Square-statistic : %f \n p-value : %f" % chisquare(data[column], 1)
                text3 = "Kruskal-statistic : %f \n p-value : %f" % kruskal(data[column][data[target] == 0],
                                                                           data[column][data[target] == 1])
                ax[0].boxplot((data[column][data[target] == 0], data[column][data[target] == 1]), sym="o",
                              labels=["대조군", "질환군"],
                              widths=.6, boxprops=props, whiskerprops=props, capprops=props, medianprops=props)
                metrics.plot_roc_curve(clf, X, y, ax=ax[1])
                metrics.plot_roc_curve(LDA, X, y, ax=ax[1])
                metrics.plot_roc_curve(QDA, X, y, ax=ax[1])
                ax[1].plot([0, 1], [0, 1], linestyle='--', markersize=0.01, color='black')
                ConfusionMatrixDisplay(confusion_matrix(y, y_score)).plot(ax=ax[2])
                sns.countplot(x=column, hue=target, data=data, ax=ax[3])
                ax[4].text(0.1, .9, text1)
                ax[4].text(0.1, .75, text2)
                ax[4].text(0.1, .6, text3)
                ax[4].axes.xaxis.set_visible(False)
                ax[4].axes.yaxis.set_visible(False)
                pdf.savefig(fig)

        pdf.close()

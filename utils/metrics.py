from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
import os

def save_scores_labels(scores, labels,data_name1,data_name2, path):
    data = {'scores':scores,'labels':labels}
    name = f'{data_name1}_{data_name2}'
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    f1_list = 2 * precision * recall / (precision + recall + 1e-6)
    f1_idx = f1_list.argmax()
    best_threshold = thresholds[f1_idx]
    # 根据最佳阈值进行预测
    y_pred = np.where(scores >= best_threshold, 1, 0)
    # 计算最佳 F1 分数
    best_f1 = f1_score(labels, y_pred)
    
    if path is not None:
        np.savez(f'{path}/{name}{best_f1:.4f}.npz',**data)
        print('save done in ',f'{path}/{name}{best_f1:.4f}.npz')
    else:
        print('no save done')

def get_score_list(y_true, score,data_name1,data_name2,pth):
    auc = roc_auc_score(y_true, score)
    if auc <= 0.5:
        score = 1-score

    precision, recall, thresholds = precision_recall_curve(y_true, score)

    f1_list = 2 * precision * recall / (precision + recall + 1e-6)
    f1_idx = f1_list.argmax()
    best_threshold = thresholds[f1_idx]
    # 根据最佳阈值进行预测
    y_pred = np.where(score >= best_threshold, 1, 0)
    # 计算最佳 F1 分数
    best_f1 = f1_score(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, score)
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 计算灵敏度 (Sensitivity) / 真正例率 (TPR)
    sensitivity = tp / (tp + fn)
    # 计算特异性 (Specificity) / 真负例率 (TNR)
    specificity = tn / (tn + fp)
    # 计算 Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)
    # 计算 G-mean (几何平均数)
    g_mean = (sensitivity * specificity) ** 0.5
    mcc = matthews_corrcoef(y_true, y_pred)
    # 计算 ROC 曲线和 PR 曲线
    fpr, tpr, thresholds = roc_curve(y_true, score)
    
    scores_list = {
        "accuracy":accuracy,
        'best_f1':best_f1,
        "precision":precision,
        "recall":recall,
        "auc":auc,
        "sensitivity":sensitivity,
        "specificity":specificity,
        "mcc":mcc,
        "g_mean":g_mean,
    }

    if pth is not None:
        df = pd.DataFrame([scores_list])
        name = f'{data_name1}_{data_name2}{best_f1:.4f}.csv'
        df.to_csv(os.path.join(pth,name), index=False)
        print(f'save metrics scores')
    else:
        print('no save metrics scores')
    return scores_list, score
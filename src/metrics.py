import pandas as pd
import numpy as np


# функции являющиеся метриками:


#hit_rate: вычисляет показатель попадания.
#Возвращает 1, если хотя бы один элемент из рекомендованного списка присутствует в списке покупок, и 0 в противном случае.
def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1


#hit_rate_at_k: вычисляет показатель попадания на первых k позициях.
#Использует функцию hit_rate для оценки только первых k элементов из рекомендованного списка.
def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)


#precision: вычисляет точность.
#Определяет долю элементов из рекомендованного списка, которые присутствуют в списке покупок.
def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)


#precision_at_k: вычисляет точность на первых k позициях.
#Использует функцию precision для оценки только первых k элементов из рекомендованного списка.
def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)


#money_precision_at_k: вычисляет точность с учетом стоимости элементов.
#Умножает бинарные флаги на цены рекомендованных элементов и делит сумму на общую стоимость рекомендаций.
def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()


#recall: вычисляет полноту. Определяет долю элементов из списка покупок, которые присутствуют в рекомендованном списке.
def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)


#recall_at_k: вычисляет полноту на первых k позициях. Использует функцию recall для оценки только первых k элементов из рекомендованного списка.
def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


#money_recall_at_k: вычисляет полноту с учетом стоимости элементов.
#Умножает бинарные флаги на цены рекомендованных элементов и делит сумму на общую стоимость покупок.
def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()


#ap_k: вычисляет среднюю точность на первых k позициях.
#Вычисляет точность на каждой позиции из рекомендованного списка до k и возвращает среднее значение.
def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[recommended_list <= k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)

    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant
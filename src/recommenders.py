import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
            
        return model

    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        # Получаем идентификатор пользователя из его имени
        user_id = self.userid_to_id[user]
        
        # Получаем список похожих пользователей
        similar_users = self.own_recommender.similar_users(user_id, N+1)
        
        # Удаляем исходного пользователя из списка похожих пользователей
        similar_users = similar_users[1:]
        
        # Получаем список товаров, купленных похожими пользователями
        items = []
        for similar_user_id in similar_users:
            recs = self.own_recommender.recommend(similar_user_id, self.user_item_matrix.T.tocsr(), N=1)
            items.append(self.id_to_itemid[recs[0][0]])
        
        # Удаляем дубликаты и возвращаем список рекомендаций
        res = list(set(items))
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    
    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # Получаем идентификатор пользователя из его имени
        user_id = self.userid_to_id[user]
        
        # Получаем топ-N товаров, купленных пользователем
        top_items = self.user_item_matrix.loc[user_id].sort_values(ascending=False).head(N)
        
        # Получаем похожие товары для каждого из топ-N товаров
        similar_items = []
        for item_id, score in top_items.iteritems():
            recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
            similar_items.extend([self.id_to_itemid[rec[0]] for rec in recs[1:]])
        
        # Удаляем дубликаты и возвращаем список рекомендаций
        res = list(set(similar_items))
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
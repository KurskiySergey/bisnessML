import numpy as np
import pandas as pd
from collections import namedtuple

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

def random_recommendation(items, n=5):
    """Случайные рекоммендации"""

    items = np.array(items)
    recs = np.random.choice(items, size=n, replace=False)

    return recs.tolist()


def get_weights(data):
    weight_function = lambda x: np.log(x + 1)
    items_weights = data.groupby("item_id")["sales_value"].sum().reset_index()
    items_weights["sales_value"] = items_weights["sales_value"].apply(weight_function)
    total_weight = items_weights["sales_value"].sum()
    items_weights = items_weights.rename(columns={"sales_value": "weights"})
    items_weights["weights"] = items_weights["weights"].apply(lambda x: x / total_weight)
    return items_weights


def weighted_random_recommendation(items_weights, n=5):
    """Случайные рекоммендации

    Input
    -----
    items_weights: pd.DataFrame
        Датафрейм со столбцами item_id, weight. Сумма weight по всем товарам = 1
    """

    items = np.array(items_weights["item_id"])
    weights = np.array(items_weights["weights"])

    recs = np.random.choice(items, size=n, replace=False, p=weights)

    return recs.tolist()


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    MODEL_TYPES = namedtuple("M_TYPES", ["ALS", "BPR", "ItemItem"])(0, 1, 2)

    def __init__(self, user_info):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        # pd.DataFrame
        self.model_type = MainRecommender.MODEL_TYPES.ALS
        self.fit_params = {"factors": 20,
                           "regularization": 0.001,
                           "iterations": 15,
                           "calculate_training_loss": True,
                           "num_threads": 4}
        self.user_info = user_info

        self.id_to_itemid = {}
        self.id_to_userid = {}
        self.itemid_to_id = {}
        self.userid_to_id = {}

        self.user_item_matrix = None
        self.model = None

    def set_model_type(self, model_type, **kwargs):
        self.model_type = model_type
        self.fit_params = kwargs

    def fit(self, user_item_matrix, weighting=True):
        """Обучает ALS"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))

        if weighting:
            user_item_matrix = bm25_weight(user_item_matrix.T).T

        self.user_item_matrix = csr_matrix(user_item_matrix)

        if self.model_type == MainRecommender.MODEL_TYPES.ALS:
            model = AlternatingLeastSquares(**self.fit_params)

        elif self.model_type == MainRecommender.MODEL_TYPES.BPR:
            model = BayesianPersonalizedRanking(**self.fit_params)

        elif self.model_type == MainRecommender.MODEL_TYPES.ItemItem:
            model = ItemItemRecommender(**self.fit_params)

        else:
            model = None

        model.fit(self.user_item_matrix.T.tocsr())
        self.model = model

    def get_model(self):
        return self.model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        N_items = self.user_info.loc[self.user_info["user_id"] == user]["actual"][:N]
        N_id = []
        for item in N_items:
            for item_id in item:
                N_id.append(self.itemid_to_id[item_id])
        similar_items = self.model.similar_items(N_id,
                                                 N=5,
                                                 # item_users=self.user_item_matrix.T.tocsr()
                                                 )
        recommendations = []
        for similar_item in similar_items[0]:
            for item_id in similar_item:
                real_item_id = self.id_to_itemid.get(item_id)
                if real_item_id is not None:
                    recommendations.append(real_item_id)
        return recommendations

    # def get_similar_users_recommendation(self, user, N=5):
    #     """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    #
    #     # your_code
    #
    #     assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
    #     return res


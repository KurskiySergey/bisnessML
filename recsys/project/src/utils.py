import pandas as pd
import numpy as np
import os
from .config import DATASETS_DIR


def load_csv_dataset(dataset_name=None):
    try:
        if dataset_name is not None:
            dataset = pd.read_csv(f"{os.path.join(DATASETS_DIR, dataset_name)}.csv")
        else:
            dataset = pd.DataFrame()
    except (FileNotFoundError, FileExistsError):
        print(f"file {dataset_name} not exist")
        dataset = pd.DataFrame()

    return dataset

def split_dataset(dataset, test_size_weeks = 3):
    data_train = dataset[dataset['week_no'] < dataset['week_no'].max() - test_size_weeks]
    data_test = dataset[dataset['week_no'] >= dataset['week_no'].max() - test_size_weeks]
    return data_train, data_test


class Preprocess:

    def __init__(self, top_filter=0.5, non_top_filter=0.01, week_filter=12, price_filter=84, low_price_filter=10,
                 department_filter: list = []):
        self.top_filter = top_filter
        self.non_top_filter = non_top_filter
        self.week_filter = week_filter
        self.price_filter = price_filter
        self.low_price_filter = low_price_filter
        self.department_filter = department_filter

    def fit(self, data: pd.DataFrame, copy_input=True):
        result_data = data.copy() if copy_input else data
        pipeline = [
            self.__find_item_sale,
            self.__find_popularity,
            self.__filter_top,
            self.__filter_not_top,
            self.__filter_by_week,
            self.__filter_by_price,
            self.__filter_by_low_price,
            self.__filter_by_department
        ]

        for pipe in pipeline:
            result_data = pipe(result_data)

        return result_data

    def __find_item_sale(self, data: pd.DataFrame):
        # quantity can be zero
        data["price"] = data["sales_value"] / np.maximum(data["quantity"], 1)
        return data

    def __find_popularity(self, data: pd.DataFrame):
        # popularity of item -> count of item-users / total_count
        users_count = data["user_id"].nunique()
        popularity = (data.groupby("item_id")["user_id"].nunique() / users_count).to_dict()
        data["popularity"] = data["item_id"].apply(lambda item_id: popularity[item_id])
        return data

    def __filter_top(self, data: pd.DataFrame):
        data = data.loc[data["popularity"] < self.top_filter]
        return data

    def __filter_not_top(self, data: pd.DataFrame):
        data = data.loc[data["popularity"] > self.non_top_filter]
        return data

    def __filter_by_department(self, data: pd.DataFrame):
        data = data.loc[~data["department"].isin(self.department_filter)]
        return data

    def __filter_by_week(self, data: pd.DataFrame):
        max_week = data["week_no"].max()
        data = data.loc[data["week_no"] > max_week - self.week_filter]
        return data

    def __filter_by_price(self, data: pd.DataFrame):
        data = data.loc[data["price"] < self.price_filter]
        return data

    def __filter_by_low_price(self, data: pd.DataFrame):
        data = data.loc[data["price"] > self.low_price_filter]
        return data
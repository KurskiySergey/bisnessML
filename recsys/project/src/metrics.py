import numpy as np

def indicate_at_k(recommended_list: list, bought_list: list, k=-1):
    if len(recommended_list) != 0:
        recommended_list = np.asarray(recommended_list) if k == -1 else np.asarray(recommended_list)[:k]
    else:
        recommended_list = np.asarray([0])
    bought_list = np.asarray(bought_list)

    return np.isin(recommended_list, bought_list)

def recall_at_k(recommended_list: list, bought_list: list, k =-1):
    if len(bought_list) == 0:
        result = 0
    else:
        indication = indicate_at_k(recommended_list, bought_list, k=k)
        result = indication.sum() / len(bought_list)
    return result


def money_recall_at_k(recommended_list: list, bought_list: list, recommended_prices: list, bought_prices: list, k=-1):
    if len(bought_list) == 0:
        result = 0
    else:
        rec_prices = np.asarray(recommended_prices) if k == -1 else np.asarray(recommended_prices)[:k]
        buy_prices = np.asarray(bought_prices)
        indication = indicate_at_k(recommended_list, bought_list, k=k)

        result = np.sum(indication * rec_prices) / buy_prices.sum()

    return result


def preccision_at_k(recommended_list, bought_list, k=-1):
    indication = indicate_at_k(recommended_list, bought_list, k=k)
    if k != -1:
        recommended_list = recommended_list[:k]

    precision = indication.sum() / len(recommended_list)

    return precision


def mrr_at_k(recommended_list, bought_list, k=-1):
    indication = indicate_at_k(recommended_list, bought_list, k=k)
    r_k = np.argmax(indication)
    if r_k == 0 and not indication[0]:
        result = 0
    else:
        result = 1 / (r_k + 1)

    return result


def nDCG_at_k(recommended_list, bought_list, k=-1):

    def discount(j):
        return 1 / np.log2(j + 1)

    vec_disc = np.vectorize(discount)

    indication = indicate_at_k(recommended_list, bought_list, k=k)
    bought_id = range(1, len(bought_list) + 1)
    indication_id = range(1, indication.shape[0] + 1)

    dcg_at_k = indication * vec_disc(indication_id)
    i_dcg_at_k = vec_disc(bought_id)
    if k != -1:
        i_dcg_at_k = i_dcg_at_k[:k]

    nDCG_at_k = dcg_at_k.sum() / i_dcg_at_k.sum()

    return nDCG_at_k
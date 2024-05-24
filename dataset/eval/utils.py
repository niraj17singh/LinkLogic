import pandas as pd


def stringify_path(path):
    return '--'.join(path)

def res_to_df(res):
    result_list = [record for record in res]
    return pd.DataFrame(result_list, columns=res.keys())

import pandas as pd
import requests
import zlib
import os
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple, Any

# HTTP 요청에 사용될 헤더 설정
headers: dict = {
    'User-Agent': 'Googlebot-Image/1.0',
    'X-Forwarded-For': '64.18.15.200'
}


def check_download(row: pd.Series) -> bool:
    try:
        response: requests.Response = requests.head(row['url'], stream=False, timeout=5, allow_redirects=True,
                                                    headers=headers)
        row['status'] = response.status_code
    except:
        row['status'] = 408
        return False

    return response.ok


def df_multiprocess(df: pd.DataFrame, processes: int, chunk_size: int, func: callable) -> pd.DataFrame:
    print("Generating parts...")
    pbar: tqdm = tqdm(total=len(df), position=0)

    pool_data: Any = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)))
    results: list = []

    pbar.desc = "Checking Downloads"
    with Pool(processes) as pool:
        for result in pool.imap_unordered(_df_split_apply, pool_data, 2):
            results.append(result)
            pbar.update(len(result[1]))

    pbar.close()
    print("Finished Checking.")
    return pd.concat([res[1] for res in results], sort=True)


def _df_split_apply(tup_arg: Tuple[int, pd.DataFrame, callable]) -> Tuple[int, pd.DataFrame]:
    split_ind, subset, func = tup_arg
    # func(row) 함수가 True인 행만 반환
    return split_ind, subset[subset.apply(func, axis=1)]


def open_tsv(file_name: str) -> pd.DataFrame:
    print("Opening %s Data File..." % file_name)
    df: pd.DataFrame = pd.read_csv(file_name, sep='\t', names=["caption", "url"], usecols=range(1, 2))
    print("Processing", len(df), " Images:")
    return df

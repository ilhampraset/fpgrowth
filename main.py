import time

from mlxtend.frequent_patterns import fpgrowth, fpmax
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

if __name__ == '__main__':
    url = "https://docs.google.com/spreadsheets/d/1VqTaGSr5vGu8qSwk2O0IIIjsPMCFyVvS/gviz/tq?tqx=out:csv&sheet=Sheet1"
    groceries = pd.read_csv(url)
    transactions = groceries['barang'].apply(lambda t: t.split(', '))

    transactions = list(transactions)
    start = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(df)
    print(df.mean())
    result = fpgrowth(df, min_support=0.4, use_colnames=True, verbose=0)
    print(result)
    end = time.time()
    print("time execution : %f" % (end - start))




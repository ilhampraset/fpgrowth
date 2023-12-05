import string
import time

from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
from fpgrowth_py import fpgrowth as fpg2
from fpgrowth_py.utils import *
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder


def to_frozenset(x):
    return frozenset(map(any, x.split("{")[1].split("}")[0].split(",")))


def to_string_format(x):
    return "->".join(x)


def execfpgrowth(min_support, min_confidence):
    url = ("https://docs.google.com/spreadsheets/d/1VqTaGSr5vGu8qSwk2O0IIIjsPMCFyVvS/gviz/tq?tqx=out:csv&sheet"
           "=Sheet1")
    groceries = pd.read_csv(url)
    transactions = groceries['barang'].apply(lambda t: t.split(', '))

    transactions = list(transactions)

    start = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    df = pd.DataFrame(te_ary, columns=te.columns_)
    result = fpgrowth(df, min_support=min_support, use_colnames=True, verbose=1)
    rules = association_rules(result, metric="confidence", min_threshold=min_confidence)
    result['itemsets'] = result['itemsets'].apply(lambda x: to_string_format(x))
    rules['antecedents'] = rules['antecedents'].apply(lambda x: to_string_format(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: to_string_format(x))

    st.write(result)
    st.write(rules)
    end = time.time()
    print("time execution : %f" % (end - start))


if __name__ == '__main__':
    min_support = st.number_input('Insert Minimum Support')
    min_confidence = st.number_input('Insert Minimum Confidence')
    if st.button('Process'):
        execfpgrowth(min_support, min_confidence)


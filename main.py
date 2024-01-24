import json
import math
import string
import time
import pyfpgrowth
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
import fptree
import fp
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder


def to_frozenset(x):
    return frozenset(map(any, x.split("{")[1].split("}")[0].split(",")))


def frozenset_to_str(fset):
    return ", ".join(sorted(list(fset)))


def to_string_format(x):
    return "->".join(x)


def execfpgrowth(min_support, min_confidence):
    url = ("https://docs.google.com/spreadsheets/d/1VqTaGSr5vGu8qSwk2O0IIIjsPMCFyVvS/gviz/tq?tqx=out:csv&sheet"
           "=Sheet1")
    groceries = pd.read_csv(url)
    transactions = groceries['barang'].apply(lambda t: t.split(', '))
    min_support = math.floor(min_support * len(transactions))
    print(min_support)
    transactions = list(transactions)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(df)
    st.table(df)
    transformed_data = {frozenset(item): 1 for item in transactions}

    start = time.time()

    fp_tree, header_table = fptree.build_fp_tree(transactions, min_support)
    print(transactions)

    js = list(fptree.fp_tree_to_json(fp_tree).items())
    st.title("FP Tree")
    st.json(js[2][1])
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)

    rules2 = pyfpgrowth.generate_association_rules(patterns, min_confidence)
    table_data = []
    for consequent, (antecedent, confidence) in rules2.items():
        antecedent_str = frozenset_to_str(antecedent)
        consequent_str = frozenset_to_str(consequent)
        support = patterns[consequent]
        lift_ratio = confidence/support
        table_data.append(
            {'Rule': antecedent_str+' => '+consequent_str, 'Support': support/len(transactions), 'Confidence': confidence, 'Lift Ratio':lift_ratio})
    st.title("Association Rules")
    st.table(table_data)
    end = time.time()
    print("time execution : %f" % (end - start))


if __name__ == '__main__':
    min_support = st.number_input('Insert Minimum Support')
    min_confidence = st.number_input('Insert Minimum Confidence')
    if st.button('Process'):
        execfpgrowth(min_support, min_confidence)

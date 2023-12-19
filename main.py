import json
import string
import time
import pyfpgrowth
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
import fptree
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder


def to_frozenset(x):
    return frozenset(map(any, x.split("{")[1].split("}")[0].split(",")))


def frozenset_to_str(fset):
    return ", ".join(sorted(list(fset)))


def to_string_format(x):
    return "->".join(x)


def execfpgrowth(min_support, min_confidence, discount):
    url = ("https://docs.google.com/spreadsheets/d/1VqTaGSr5vGu8qSwk2O0IIIjsPMCFyVvS/gviz/tq?tqx=out:csv&sheet"
           "=Sheet1")
    groceries = pd.read_csv(url)
    transactions = groceries['barang'].apply(lambda t: t.split(', '))

    transactions = list(transactions)
    print(len(transactions))
    transformed_data = {frozenset(item): 1 for item in transactions}
    start = time.time()
    fp_tree, header_table = fptree.build_fp_tree(transformed_data, min_support)
    js = list(fptree.fp_tree_to_json(fp_tree).items())
    st.title("FP Tree")
    st.json(js[2][1])
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)  # 2 is the minimum support count

    # st.write(patterns)
    # Generate association rules from the frequent patterns
    rules2 = pyfpgrowth.generate_association_rules(patterns, min_confidence)  # 0.7 is the minimu
    table_data = []
    for consequent, (antecedent, confidence) in rules2.items():
        antecedent_str = frozenset_to_str(antecedent)
        consequent_str = frozenset_to_str(consequent)
        support = patterns[consequent]  # Access support directly from patterns
        table_data.append(
            {'Rule': antecedent_str+' => '+consequent_str, 'Support': support/len(transactions), 'Confidence': confidence, 'Discount':discount})
    st.title("Association Rules")
    st.table(table_data)
    end = time.time()
    print("time execution : %f" % (end - start))


if __name__ == '__main__':
    min_support = st.number_input('Insert Minimum Support')
    min_confidence = st.number_input('Insert Minimum Confidence')
    discount = st.number_input('Insert Discount')
    if st.button('Process'):
        execfpgrowth(min_support, min_confidence, discount)

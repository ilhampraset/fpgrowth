import math
import time
import pyfpgrowth
import pandas as pd
import fptree
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder


def frozenset_to_str(fset):
    return ", ".join(sorted(list(fset)))


def execfpgrowth(min_support, min_confidence):

    groceries = pd.read_csv('dataset.csv')
    transactions = groceries['barang'].apply(lambda t: t.split(', '))
    msupport = math.floor(min_support * len(transactions))
    print(msupport)
    transactions = list(transactions)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    st.dataframe(df, hide_index=True)
    start = time.time()

    fp_tree, header_table = fptree.build_fp_tree(transactions, msupport)

    js = list(fptree.fp_tree_to_json(fp_tree).items())
    st.title("FP Tree")
    st.json(js[2][1])
    patterns = pyfpgrowth.find_frequent_patterns(transactions, msupport)
    rules2 = pyfpgrowth.generate_association_rules(patterns, min_confidence)
    table_data = []
    count = 0
    for consequent, (antecedent, confidence) in rules2.items():
        antecedent_str = frozenset_to_str(antecedent)
        consequent_str = frozenset_to_str(consequent)
        supports = patterns[consequent] / len(transactions)
        lift_ratio = confidence / support
        count = count + 1
        table_data.append(
            {'No': count, 'Rule': antecedent_str + ' => ' + consequent_str, 'Support': supports,
             'Confidence': confidence,
             'Lift Ratio': lift_ratio})

    st.title("Association Rules")

    st.dataframe(table_data, hide_index=True)
    end = time.time()
    print("time execution : %f" % (end - start))


if __name__ == '__main__':
    support = st.number_input('Insert Minimum Support')
    confidences = st.number_input('Insert Minimum Confidence')
    if st.button('Process'):
        execfpgrowth(support, confidences)

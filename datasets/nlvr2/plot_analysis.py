import pdb
import random

from matplotlib import pyplot as plt

import streamlit as st
import numpy as np

random.seed(1337)

data = [
    {
        "result": random.choice(range(2)),
        "score_avg_mins": random.random(),
    }
    for _ in range(1000)
]

results = np.array([datum["result"] for datum in data])
scores = np.array([datum["score_avg_mins"] for datum in data])

num_bins = 10
_, bins = np.histogram(scores, bins=num_bins)
idxs = np.digitize(scores, bins)

def get_frac_correct(i):
    r = results[idxs == i]
    return 100 * r.sum() / len(r)

frac_correct = [get_frac_correct(i) for i in range(1, num_bins + 1)]

# bins
# frac_correct

fig, ax = plt.subplots()
width = 0.8 * (bins[1] - bins[0])
ax.bar(bins[1:], frac_correct, align="edge", width=width)
ax.set_xlabel("similarity to train data")
ax.set_ylabel("fraction correct (%)")
st.pyplot(fig)

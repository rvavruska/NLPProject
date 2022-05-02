import argparse
from lib2to3.pgen2 import token
import logging
import math
import os
import random
from functools import partial
import streamlit as st
import pandas as pd
import numpy as np

from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = None
model = None

def Run_Summary(text):
    if tokenizer == None:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    if model == None:
        model = BartForConditionalGeneration.from_pretrained("summ_output")
    
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(
    input_ids,
    max_length=1024,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    )

    return tokenizer.decode(output_ids[0])

st.title('Article Summurization')
text = st.text_area("Article to summarize")
st.write("Summary: ", Run_Summary(text))
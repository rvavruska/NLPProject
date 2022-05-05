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

import transformers
from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("summ_output")

st.title('Article Summurization')
summary = ""

def Run_Summary(text):    
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(
    input_ids,
    max_length=128,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    )
    st.write("Summary: ", tokenizer.decode(output_ids[0]))

text = st.text_area("Article to summarize", "")
if st.button("Summarize"):
    Run_Summary(text)
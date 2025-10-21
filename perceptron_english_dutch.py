import numpy as np

"""
labels[1 = english, -1 = dutch]

Creates 2 separate .csv file with different features from english and dutch  
1. For training set (uni declaration) 
2. For dev/test set (40 translations from additional text)

Feature extractions:
1. Char patterns : 
    a. Dutch : ij, sch, aa, oo, ee, ijk, cht
    b. English : th, sh, ing, ion, ough
2. Whole words :
    a. Dutch: van, de, en
    b. English: the, and, has
3. Average Word Length : Dutch longer words than English

"""
with open("universal-declaration/english.txt", "r", encoding="utf-8") as f:
    english_test_text = [line.strip() for line in f if line.strip()]

with open("universal-declaration/dutch.txt", "r", encoding="utf-8") as f:
    dutch_test_text = [line.strip() for line in f if line.strip()]

with open("bible-translation/english.txt", "r", encoding="utf-8") as f:
    english_other_text = [line.strip() for line in f if line.strip()]

with open("bible-translation/dutch.txt", "r", encoding="utf-8") as f:
    dutch_other_text = [line.strip() for line in f if line.strip()]



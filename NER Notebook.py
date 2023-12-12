#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
#nlp = spacy.blank("en") # load a new spacy model
nlp = spacy.load("en_core_web_lg") # load other spacy model
Training_Data = ("Your file path")

db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

os.chdir(r'your path')
db.to_disk("./train.spacy") # save the docbin object

nlp = spacy.load("en_core_web_sm")  # load the spacy model

db = DocBin()  # create a DocBin object
Test_data = ("Your file path")

for text, annot in tqdm(TEST_DATA):
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents  # label the text with the ents
    db.add(doc)

os.chdir(r'your path')
db.to_disk("./test.spacy")  # save the docbin object

python -m spacy init fill-config base_config.cfg config.cfg # for configuration

python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy # For training

nlp1 = spacy.load(r"yout path") #load the best model
doc = nlp1("Ella spent $100 on a new shoes using his cash.") # input sample text

spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter

Ella PERSON spent $100 AMOUNT on a new shoes PRODUCT using his cash ENTITY .


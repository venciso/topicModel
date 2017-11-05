# Topic Modelling in python
This repository contains:
* topic_model.py file which contains code for a topic modelling object.
* The pickled files directory contains all the files needed by topic_model.py
* Topic_modelling.ipynb is a notebook with the details behind the creation of the model in topic_model.py

Note: The user needs to have the spacy and gensim modules installed for this code to work. 

## Usage
### Class: topic_model

Make a new topic model object by instatiating the topic_model

``` 
import topic_model as t
model = t.topicMod()
```

The 'model' object contains a trained LDA model that will find topics for a given document. 

The object also has pre-loaded documents in the `model.test_documents` list. The list consists of 2042 documets from the 20newsgroups dataset.

The `model.get_document()` method gets a random document from the test documents.

The `model.lda_description` method takes a text document and finds the topics relevant to it. 


Some examples of how to get topics for a document: 
```
model.lda_description(model.get_document())

model.lda_description(model.test_documents[1230])

model.lda_description("Some text I want to get topics from. This text will be preferably at least 100-150 words long")

```

# Language Classifier
## Character Level Recurrent Neural Network to predict European languages.

Based on the PyTorch tutorial found [here](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#sphx-glr-intermediate-char-rnn-classification-tutorial-py).

The model uses data from the [European Parliament Proceedings Parallel Corpus](http://www.statmt.org/europarl/).
The corpora are segmented by word, and the model uses only one instance of each unique word, per language.

Current Languages included are:
- Bulgarian
- Czech
- Dutch
- English
- French
- German
- Italian
- Polish
- Spanish
- Swedish

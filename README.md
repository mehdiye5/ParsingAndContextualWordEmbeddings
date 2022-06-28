# Parsing And Contextual Word Embeddings

## Analysis

The first fix was to the attention calculation. This was done by passing the encoder's hidden state through a linear layer and passing the decoder's hidden state through a separate linear layer. The outputs were then added together to get the attention score for each token in the sentence. Then, the softmax function was applied to the attention score vector. This resulted in the correct implementation of the attention module and a BLEU score of 17.11.
</br> </br>

### Fixed Attention </br> </br>

when when i was in my 20s , i had my first <unk> . . </br>
i was a and i was particularly in berkeley . berkeley . </br>
she was a a woman woman named alex . </br>
when the came came up the first first , she came up and she a a a , </br>
and she fell into the couch in my office , the her her and and told me </br>
, about about about . </br>
and when i heard this , i was was . </br> </br>

After implementing the attention module, we noticed that there were still numerous "UNK" tokens in our output file. As this token could only hurt our BLEU score, we decided to experiment with replacing this token with a common word. We tested "a", "of" and "the", and found that "the" worked best. This improved our BLEU score to 17.15. </br> </br>

### Unkown Word Replacement </br> </br>

when when i was in my 20s , i had my first the . . </br>
i was a and i was particularly in berkeley . berkeley . </br>
she was a a woman woman named alex . </br>
when the came came up the first first , she came up and she a a a , </br>
and she fell into the couch in my office , the her her and and told me </br>
, about about about . </br>
and when i heard this , i was was . </br> </br>

The next thing that we noticed was that the translated sentence was including the same word in succession. For example, in the first 5 rows of the output.txt file, we can see the "a" and "about" are repeated 3 times in succession in the same sentence. This is not common in English so we added a condition to remove duplicated words in succession. This increased our BLEU score to 17.25. Finally, we tested the 5 neural machine translation models available in Google Drive. We found that the 'seq2seq_E047.pt' model worked best, and improved our score to 17.49.
</br> </br>

when i was in my 20s , i had my first the .
i was a and i was particularly in berkeley .
she was a woman named alex .
as a came in the first meeting , she was sitting and a , and she fell 
into the office in my office , her and she was me to talk about .
and when i heard that , i was .
</br> </br>

### Model Iterations </br> </br>

Fixed Attention BLEU = 17.11 54.0/24.1/12.0/6.5 (BP = 0.957 ratio = 0.958 hyp_len = 23858 ref_len = 24902)
</br> </br>

Unknown Word Replacement BLEU = 17.14 54.1/24.1/12.0/6.5 (BP = 0.957 ratio = 0.958 hyp_len = 23858 ref_len = 24902)
</br> </br>

3x word removals BLEU = 17.15 55.7/24.9/12.5/6.8 (BP = 0.926 ratio = 0.929 hyp_len = 23127 ref_len = 24902)
</br> </br>

2x word removals BLEU = 17.25 61.8/28.2/14.6/8.1 (BP = 0.809 ratio = 0.825 hyp_len = 20545 ref_len = 24902)
</br> </br>

047 Model BLEU = 17.49 62.3/28.9/15.1/8.4 (BP = 0.799 ratio = 0.817 hyp_len = 20342 ref_len = 24902)
</br> </br>

### I. Instructions:

`default.py` contains the default solution, which assigns all alpha in
attention to be equal, and the context vector takes simply the average of all
encoder states.

Before you can run the default solution, make sure you either download the
pre-trained models from `https://jetic.org/cloud/s/YdofIN0CvuCAgux`, or access
it on CSIL machine.

Once you've done that, create a symbolic link to all files in that zip file
from `data/` using something like this (you don't need to unzip on CSIL
machines):

    > unzip ~/Downloads/Archiv.zip -d ~/Downloads/Archiv
    > ln -s ~/Downloads/Archiv/*.pt data/

There should be 5 `.pt` files:

    seq2seq_E045.pt
    seq2seq_E046.pt
    seq2seq_E047.pt
    seq2seq_E048.pt
    seq2seq_E049.pt

By default, the model being used is `seq2seq_E049.pt`, but you can change that
in `default.py`.
When you are implementing ensemble, you can choose which ones to use as well.

#### 1. Baseline: Fixing attention

Attention is defined as follows:

$$\mathrm{score}_i = W_{enc}( h^{enc}_i ) + W_{dec}( h^{dec} )$$

Define $\alpha_i$ for each source side token $i$ as follows:

$$\alpha_i = \mathrm{softmax}(V_{att} \mathrm{tanh} (\mathrm{score}_i))$$

The we define the context vector using the $\alpha$ weights:

$$c = \sum_i \alpha_i \times h^{enc}_i$$

The context vector $c$ is combined with the current decoder hidden
state $h^{dec}$ and this representation is used to compute the
softmax over the target language vocabulary at the current decoder
time step. We then move to the next time step and repeat this process
until we produce an end of sentence marker.

#### 2. Extensions

We fixed the interface in a specific way that allows you to implement at least:

1. UNK replacement: https://www.aclweb.org/anthology/P15-1002/

2. BeamSearch

3. Ensemble

Original training data is also provided (tokenised). You may use it whichever
way you want to augment the provided Seq2Seq model.

### II. Useful Tool

For visualisation, one could easily use the included functions in `utils.py`:

    from utils import alphaPlot

    # Since alpha is batched, alpha[0] refers to the first item in the batch
    alpha_plot = alphaPlot(alpha[0], output, source)

This converts the alpha values into a nice attention graph.
Example code in combination with `tensorboard` is provided in `validator.py`.
This can help you visualise an entire `test_iter`.

In addition, `default.py` has an additional parameter `-n`.
If your inference is taking too long and you'd like to test your implementation
with a subset of dev (say first 100 samples), you can do that.
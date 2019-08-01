import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

%matplotlib inline

"""

Welcome to the second assignment of Week 2. You are going to use word vector representations to build an Emojifier

Have you ever wanted to make your text messages more expressive?
Your emojifier app will help you do that


GO FROM THIS --> So rather than writing "Congratulations on the promotion! Lets get coffee and talk. Love you!"

the emojifier can automatically turn this into...

TO THIS --> "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️"


You will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!")
and finds the most appropriate emoji to be used with this sentence (⚾)

In many emoji interfaces, you need to remember that ❤ is the "heart" symbol rather than the "love" symbol

But using word vectors, you'll see that even if your training set explicitly relates only a few words to a particular..
.. emoji, your algorithm will be able to generalize and associate words in the test set to the same emoji
    --> This allows you to build an accurate classifier mapping from sentences to emojis even using a small training set

In this exercise, you'll start with a baseline model (Emojifier-V1) using word embeddings
    -- then build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM

Lets get started! Run the following cell to load the package you are going to use.

"""



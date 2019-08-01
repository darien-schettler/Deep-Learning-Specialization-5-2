import numpy as np
from w2v_utils import *

"""

Because word embeddings are computationally expensive to train most people will load a pre-trained set of embeddings

After this assignment you will be able to:

1. Load pre-trained word vectors, and measure similarity using cosine similarity
2. Use word embeddings to solve word analogy problems such as Man is to Woman as King is to __.
3. Modify word embeddings to reduce their gender bias

"""

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

'''

You've loaded:

words: set of words in the vocabulary
word_to_vec_map: dictionary mapping words to their GloVe vector representation
You've seen that one-hot vectors do not do a good job cpaturing what words are similar
GloVe vectors provide much more useful information about the meaning of individual words
Lets now see how you can use GloVe vectors to decide how similar two words are


COSINE SIMILARITY

To measure how similar two words are we measure the degree of similarity between two embedding vectors for the two words

Given two vectors  u  and  v, cosine similarity is defined as follows:

CosineSimilarity(u,v) = np.dot(u,v) / ( ||u||[2] * ||v||[2] ) = cos(θ)
 
    where   np.dot(u,v)  is the dot product (or inner product) of two vectors
            ||u||[2]  is the norm (or length) of the vector  u
            θ  is the angle between  u  and  v
    
This similarity depends on the angle between  u  and  v
    --> If  u  and  v  are very similar, their cosine similarity will be close to 1; 
    --> If  u  and  v  are dissimilar, the cosine similarity will take a smaller value


EXERCISE

Implement the function cosine_similarity() to evaluate similarity between word vectors
                                                     _______________________
Reminder: The norm of  u  is defined as  ||u||[2] = √ ∑[n][i=1] for (u[i])^2

'''


# GRADED FUNCTION: cosine_similarity

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    distance = 0.0

    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u, v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u ** 2))

    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v ** 2))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]

ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]

france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]

paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("\n\ncosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("\ncosine_similarity(ball, crocodile) = ", cosine_similarity(ball, crocodile))
print("\ncosine_similarity(france - paris, rome - italy) = ", cosine_similarity(france - paris, rome - italy))

'''

Please feel free to modify the inputs and measure the cos similarity between other pairs of words!
Playing around the cosine similarity of other inputs will give you a better sense of how word vectors behave


WORD ANALOGY TASK

In the word analogy task, we complete the sentence "a is to b as c is to __". An example is 'man is to woman as king is to queen'
In detail...
    We want to find a word d, such that the associated word vecs (ea,eb,ec,ed) are related in the following manner:  
                                                
                                                eb−ea≈ed−ec
          
    We will measure the similarity between  eb−ea  and  ed−ec  using cosine similarity

EXERCISE BELOW WILL SHOW THIS...

'''


# GRADED FUNCTION: complete_analogy

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """

    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # Get the word embeddings v_a, v_b and v_c (≈1-3 lines)
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100  # Initialize max_cosine_sim to a large negative number
    best_word = None  # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c]:
            continue

        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        # If the cosine_sim is more than the max_cosine_sim seen so far,
        # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


triads_to_try = [('italy', 'italian', 'spain'),
                 ('india', 'delhi', 'japan'),
                 ('man', 'woman', 'boy'),
                 ('small', 'smaller', 'large')]
print()
for triad in triads_to_try:
    print('\n{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))

'''

You've come to the end of this assignment. Here are the main points you should remember:

1. Cosine similarity a good way to compare similarity between pairs of word vectors. (Though L2 distance works too.)
2. For NLP applications, using a pre-trained set of word vectors from the internet is often a good way to get started


DE-BIASING WORD VECTORS

Let's examine gender biases that can be reflected in a word embedding, and explore algorithms for reducing the bias

In addition to learning about debiasing, this will also help hone your intuition about what word vectors are doing

Lets first see how the GloVe word embeddings relate to gender
You will first compute a vector  g = e[woman] − e[man] in an attempt to roughly encode the idea of gender

'''

g = word_to_vec_map['woman'] - word_to_vec_map['man']
print("\n\nGender Encoding\nWoman - Man:")
print(g)  # Woman - Man

'''

Now, you will consider the cosine similarity of different words with  g
Consider what a positive value of similarity means vs a negative cosine similarity

'''

print('\n\nList of names and their similarities with constructed vector:\n')

# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))

'''

As you can see, female first names tend to have a positive cosine similarity with our constructed vector  g
While male first names tend to have a negative cosine similarity

This is not surpising, and the result seems acceptable

But let's try with some other words...

'''

print('\n\nOther words and their similarities:\n')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior', 'doctor', 'tree', 'receptionist',
             'technology', 'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))

'''

Do you notice anything surprising? It is astonishing how these results reflect certain unhealthy gender stereotypes

For example, "computer" is closer to "man" while "literature" is closer to "woman". Ouch!

We'll see below how to reduce the bias of these vectors, using an algorithm due to Boliukbasi et al., 2016
    
    NOTE  Some word pairs such as "actor"/"actress" or "grandmother"/"grandfather" should remain gender specific
          While other words such as "receptionist" or "technology" should be neutralized, i.e. not be gender-related

You will have to treat these two type of words differently when de-biasing


NEUTRALIZE BIAS FOR NON-GENDER SPECIFIC WORDS
    Observe the image that shows PCA illustrating only one dimension of the embedding is gender
    The other components of the embedding SHOULD OFTEN be unrelated to this gender dimension
    i.e. Receptionist should not be dependant/related to gender
    
Exercise:   Implement neutralize() to remove the bias of words such as "receptionist" or "scientist"
            Given an input embedding  e , you can use the following formulas to compute  edebiased :

        ebias_component = { e⋅g * (||g||[2])^2 } ∗ g

        edebiased = e − ebias_component
 
You may recognize  ebias_component  as the projection of  e  onto the direction  g

'''


def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
    e = word_to_vec_map[word]

    # Compute e_biascomponent using the formula give above. (≈ 1 line)
    e_biascomponent = (np.dot(e, g) / (np.sum(g ** 2))) * g

    # Neutralize e by substracting e_biascomponent from it
    # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
    e_debiased = e - e_biascomponent

    return e_debiased


e = "receptionist"
print("\n\ncosine similarity between " + e + " and g, before neutralizing: ",
      cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("\ncosine similarity between " + e + " and g, after neutralizing: ",
      cosine_similarity(e_debiased, g))

'''
EQUALIZATION ALGORITHM FOR GENDER-SPECIFIC WORDS

Next, lets see how debiasing can also be applied to word pairs such as "actress" and "actor"

Equalization is applied to pairs of words that you might want to have differ only through the gender property

As a concrete example, suppose that "actress" is closer to "babysit" than "actor"
    --> By applying neutralizing to "babysit" we can reduce the gender-stereotype associated with babysitting
    --> However, this still does not guarantee that "actor" and "actress" are equidistant from "babysit"
    --> The equalization algorithm takes care of this

The idea behind equalization is to ensure that a particular pair of words are equi-distant from the 49-dimensional  g⊥

The equalization step also ensures:
    1. The two equalized steps are now the same distance from  e[receptionist][debiased]
    2. The two equalized steps are now the same distance from any other work that has been neutralized

'''


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
    w1, w2 = pair[0], pair[1]
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
    mu = (e_w1 + e_w2) / 2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
    mu_B = (np.dot(mu, bias_axis) / (np.sum(bias_axis ** 2))) * bias_axis
    mu_orth = mu - mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
    e_w1B = (np.dot(e_w1, bias_axis) / (np.sum(bias_axis ** 2))) * bias_axis
    e_w2B = (np.dot(e_w2, bias_axis) / (np.sum(bias_axis ** 2))) * bias_axis

    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e_w1B - mu_B) / (
        np.linalg.norm((e_w1 - mu_orth) - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e_w2B - mu_B) / (
        np.linalg.norm((e_w2 - mu_orth) - mu_B))

    # Step 6: De-bias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2


print("\n\ncosine similarities before equalizing:")
print("\ncosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("\ncosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("\ncosine similarities after equalizing:")
print("\ncosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("\ncosine_similarity(e2, gender) = ", cosine_similarity(e2, g))

'''

These de-biasing algorithms are very helpful for reducing bias, but aren't perfect and don't eliminate all traces of bias
    i.e. one weakness of this implementation is the bias direction  g  was defined using 1 pair of words (woman and man)
        As discussed earlier, if  g  were defined by computing:   g1=ewoman−eman;  g2=emother−efather;  g3=egirl−eboy;
        and so on and averaging over them,
        you would obtain a better estimate of the "gender" dimension in the 50 dimensional word embedding space

Feel free to play with such variants as well

'''
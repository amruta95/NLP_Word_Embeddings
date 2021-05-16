# NLP_Word_Embeddings

In this project, I have implemented a Skip-gram model to learn word representations and computed loss using Cross Entropy and NCE methods. I then used the above model (along with some hyperparameter tuning) to determine relations between word pairs.

For example, if given the below query:

Among these word pairs (Relation R),

(1) pig:mud

(2) politician:votes

(3) dog:bone

(4) bird:worm

The model gives the answer to the below two questions:

Q1. Which word pairs has the MOST illustrative(similar) example of the relation R?

Q2. Which word pairs has the LEAST illustrative(similar) example of the relation R?


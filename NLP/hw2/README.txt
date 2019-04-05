HW2 Parsing
W4705_001 - Natural Language Processing
Prof. Kathleen McKeown
Submission date : October 14th, 2017

This assignment will comprise of tasks in unlabeled dependency parsing. Submit the answers to the below questions along
with checking in your code.

Part 1 - Arc Standard
1. Output of your validate_transitions.py
2. List all the feature types you implemented. [Ex: ‘TOP_STK_TOKEN_’ , ‘...’, ‘....’]
For our reference.
3. Explain any 6 feature types in detail. What does the feature type represent? Why do you think this is useful for
transition predictions?
4. How do you interpret precision and recall in context of this task?
5. What is your final F1 score on the dev set?
You should report your final F1 score on the dev set. Since you will be evaluated on your F1 score on the test set, the
scores you report are a reference. We will be running your code separately for grading.

Part 2 - Domain Adaptation
6. Average F1 score from 10 runs of domain_adaptation_eval.py.
As it is a smaller training set you may notice varying F1 scores. Write the average score over 10 runs. The scores you
report are a reference. We will be running your code separately for grading.
Format for each line-
‘train_genre : <train_genre>, test_genre : <test_genre>, Avg F1 : <F1_score>’
7. Provide an explanation of the performance of the feature types in domain adaptation. What features generalized well?
 What features did not?


PROVIDED CODE -

dataset.py - Module with the load_data function given a web treebank data set filename.
dependency.py - Module with all the functions pertaining to the arc standard algorithm. shift, right_arc, left_arc, oracle_std, make_transition.
domain_adaptation.py - Module with the train function for domain adaptation.
domain_adaptation.pu - Module to test and evaluate domain adaptation.
en-ud-dev.conllu - WebTreebank dev dataset.
en-ud-train.conllu - WebTreebank train dataset.
feature_extraction.py - Module that defines the feature_extractor function.
feature_extraction_da.py - Module that defines the feature_extractor for the domain adaptation experiment.
train.py - Module for training a LinearSVC model for transition predictions.
utils.py - Module with helper functions.
validate_transitions.py - Module to validate the function in dependency.py.

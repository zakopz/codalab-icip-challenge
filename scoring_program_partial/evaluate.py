#!/usr/bin/env python
import sys
import os
import os.path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    truth_file = os.path.join(truth_dir, "truth.txt")
    with open(truth_file) as myfile:
        truth = [next(myfile).rstrip() for x in range(50)]
    
    indices = []
    truth_single = []
    for idx, val in enumerate(truth):
        if "," not in val:
            indices.append(idx)
            truth_single.append(val)	
    #truth_single = [x for x in truth if "," not in x ]

    submission_answer_file = os.path.join(submit_dir, "predict.txt")
    with open(submission_answer_file) as myfile2:
        submission_answer = [next(myfile2).rstrip() for x in range(50)]
    
    submission_single = []
    for idx, val in enumerate(submission_answer):
        if "," not in val and idx in indices:
            submission_single.append(val)
        if "," in val and idx in indices:
            submission_single.append('-1')
	#submission_single = [y for y in submission_answer if "," not in y ]
	
    mlb = MultiLabelBinarizer()
    mlb.fit([['1', '2', '3', '4', '5']])
    truth_sparse = mlb.transform(truth)
    submission_sparse = mlb.transform(submission_answer)
	
    f1_single = f1_score(truth_single,submission_single, average='micro')
    f1 = f1_score(truth_sparse, submission_sparse, average='micro')
    acc = accuracy_score(truth_sparse, submission_sparse)
	
    output_file.write("F1-score (Single-distortion): " + str(f1_single))
    output_file.write("\n")
    output_file.write("F1-score (Single + Multi distortions): " + str(f1))
    output_file.write("\n")
    output_file.write("Accuracy: " + str(acc))
    
    output_file.close()
	

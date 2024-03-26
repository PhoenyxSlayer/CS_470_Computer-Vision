# MIT LICENSE
#
# Copyright 2020 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

import os
import sys
import numpy as np
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.utils as utils
from sklearn.metrics import (accuracy_score, f1_score)
import time
import A05

base_dir = "assign05"
out_dir = base_dir + "/" + "output"

###############################################################################
# CALCULATE METRICS
###############################################################################

def compute_metrics(ground, pred):    
    scores = {}
    scores["accuracy"] = accuracy_score(y_true=ground, y_pred=pred)
    scores["f1"] = f1_score(y_true=ground, y_pred=pred, average="macro")
    return scores

###############################################################################
# PERFORM EVALUATION ON ONE MODEL AND DATASET
###############################################################################

def perform_one_evaluation_on_one_model(approach_name, model, data_type, x, y):
    print("Evaluating " + approach_name + " on " + data_type + " data...")
    
    # Predic1 with model
    pred = model.predict(x)
    
    # Convert to labels if necessary
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=-1)
        
    # Compute metrics
    metrics = compute_metrics(y, pred)
    
    return metrics     

###############################################################################
# PERFORM EVALUATION ON ONE MODEL AND ALL DATASETS
###############################################################################

def perform_alL_evaluation_on_one_model(approach_name, model, all_data):
    one_model_metrics = {}
    
    for data_type in all_data:
        one_model_metrics[data_type] = perform_one_evaluation_on_one_model(
                                                        approach_name, model, 
                                                        data_type, 
                                                        all_data[data_type]["x"],
                                                        all_data[data_type]["y"])                              
    
    return one_model_metrics   

###############################################################################
# PRINTS RESULTS (to STDOUT or file)
###############################################################################
def print_results(approach_name, model, model_metrics, all_times, stream=sys.stdout):
    boundary = "****************************************"
    
    print(boundary, file=stream)
    print("APPROACH: " + approach_name, file=stream)
    print(boundary, file=stream)
    print("", file=stream)
    
    print(boundary, file=stream)
    print("RESULTS:", file=stream)
    print(boundary, file=stream)
    for data_type in model_metrics:
        print("\t" + data_type + ":", file=stream)
        data_metrics = model_metrics[data_type]
        for key in data_metrics:
            print("\t\t", key, "=", data_metrics[key], file=stream)
    print("", file=stream)
     
    print(boundary, file=stream)
    print("TIME TAKEN:", file=stream)     
    print(boundary, file=stream)  
    for one_time in all_times:
        time_string = "\t%s: %.4f seconds" % (one_time, all_times[one_time])
        print(time_string, file=stream)
    print("", file=stream)
            
    if A05.is_keras_model(approach_name):
        def summary_print(s):
            print(s, file=stream)

        print(boundary, file=stream)
        print("MODEL ARCHITECTURE:", file=stream)       
        print(boundary, file=stream)
        model.summary(print_fn=summary_print)  
        print("", file=stream)
        
    print(boundary, file=stream)
    print("RESULTS IN TABLE:", file=stream)     
    print(boundary, file=stream) 
    
    header = ""    
    for data_type in model_metrics:        
        data_metrics = model_metrics[data_type]
        for key in data_metrics:
            header += data_type + "_" + key + "\t"
    
    table_data = header + "\n"
    for data_type in model_metrics:        
        data_metrics = model_metrics[data_type]
        for key in data_metrics:
            cell_string = "%.4f\t" % data_metrics[key]
            table_data += cell_string
    table_data += "\n"
    print(table_data, file=stream)   
    print("", file=stream)           
  
###############################################################################
# MAIN
###############################################################################

def main():   
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load CIFAR10 data
    (x_orig_train, y_train),(x_orig_test, y_test) = cifar10.load_data()
    
    # Reshape labels
    y_train = np.reshape(y_train, (-1,))
    y_test = np.reshape(y_test, (-1,))
    
    # Get number of classes
    class_cnt = len(np.unique(y_train))
    
    # Get one-hot vectors for labels
    y_hot_train = utils.to_categorical(y_train, num_classes=class_cnt)
    y_hot_test = utils.to_categorical(y_test, num_classes=class_cnt)
        
    # Get names of all approaches
    all_names = A05.get_approach_names()
    
    # Which one?
    print("Classifier names:")
    for i in range(len(all_names)):
        print(str(i) + ". " + all_names[i])
    choice = int(input("Enter choice: "))
    approach_name = all_names[choice]
    
    # Set up timing dictionary
    all_times = {}
        
    # Prepare data
    start_time = time.time()
    print("Preparing training data...")
    x_train = A05.prepare_data(approach_name, x_orig_train)
    print("Preparing testing data...")
    x_test = A05.prepare_data(approach_name, x_orig_test)
    all_times["PREPARATION"] = time.time() - start_time
    
    # Set up data dictionary
    all_data = {
        "TRAINING": {
            "x": x_train,
            "y": y_train
        },
        "TESTING": {
            "x": x_test,
            "y": y_test
        }
    }
        
    # Train classifiers
    start_time = time.time()
    print("Training " + approach_name + "...")
    model = A05.train_classifier(approach_name, x_train, y_train, y_hot_train, class_cnt)
    print("Training complete!")
    all_times["TRAINING"] = time.time() - start_time
    
    # Evaluate
    start_time = time.time()
    model_metrics = perform_alL_evaluation_on_one_model(approach_name, model, all_data)
    all_times["EVALUATION"] = time.time() - start_time
             
    # Print and save metrics
    print_results(approach_name, model, model_metrics, all_times)
    with open(out_dir + "/" + approach_name + "_RESULTS.txt", "w") as f:
        print_results(approach_name, model, model_metrics, all_times, stream=f)
    
if __name__ == "__main__": 
    main()
    
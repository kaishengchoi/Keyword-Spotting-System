# -*- coding: utf-8 -*-
"""
This file consists of fuctions that sort out and generate the training, 
testing and validation datasets of require commands. 

Created on Fri Jan 31 14:12:51 2020

@author: KaiShengChoi
"""
import os
import tensorflow as tf
from constant import dataset_path, train_commands, commands
from mfcc import mfcc

def train_list():
    """
    
    Descriptions
    ------------
    This function prepare the list of voices for training.
    
    Arguments
    ---------
    None.

    Returns
    -------
    None.

    """
    testing_list = open(dataset_path + "/" + "testing_list.txt","r")
    validation_list = open(dataset_path + "/" + "validation_list.txt","r")
    train_list = open(dataset_path + "/" + "train_list.txt","w+")
    
    testing_voices_list = testing_list.read().split("\n")
    validation_voices_list =  validation_list.read().split("\n")
    
    for directive in commands:
        voices_list = os.listdir(dataset_path + "/" + directive)
        for voice in voices_list:
            voice = directive + "/" + voice
            if (not (voice in testing_voices_list)) and (not (voice in validation_voices_list)):
                train_list.write(voice + "\n")
                
    testing_list.close()
    validation_list.close()
    train_list.close()

def dataset_sort():
    """
    
    Descriptions
    ------------
    This function sort out the training, validation and training datasets from 
    train_list.txt, testing_list.txt and validation_list.txt to 
    train_dataset.txt, testing_dataset.txt and validation_dataset.txt based 
    on the the train_commands listed in constant.py.
    
    Arguments
    ---------
    None.
    

    Returns
    -------
    None.

    """
    testing_list = open(dataset_path + "/" + "testing_list.txt","r+")
    validation_list = open(dataset_path + "/" + "validation_list.txt","r+")
    train_list = open(dataset_path + "/" + "train_list.txt", "r+")
    
    testing_dataset = open('testing_dataset.txt', 'w+') 
    validation_dataset = open('validation_dataset.txt', 'w+') 
    train_dataset = open('train_dataset.txt', 'w+') 
    
    for files in testing_list.read().split("\n"):
        if files.split("/")[0] in train_commands:
            testing_dataset.write(files + "\n")
            
    for files in validation_list.read().split("\n"):
        if files.split("/")[0] in train_commands:
            validation_dataset.write(files  + "\n")
            
    for files in train_list.read().split("\n"):
        if files.split("/")[0] in train_commands:
            train_dataset.write(files  + "\n")
    
    testing_list.close()
    validation_list.close()
    train_list.close()
    
    testing_dataset.close()
    validation_dataset.close()
    train_dataset.close()


def dataset_gen():
    """
    
    Descriptions
    ------------
    This function generate the mfcc and its label for each voices in training, 
    testing and validation datasets.
    
    Arguments
    ---------
    None.

    Returns
    -------
    train_dataset : Tensor of type tf.float32
        training dataset.
    train_label : Tensor of type tf.uint8
        label for training dataset .
    testing_dataset : Tensor of type tf.float32
        testing dataset.
    testing_label : Tensor of type tf.uint8
        label for testing dataset.
    validation_dataset : Tensor of type tf.float32
        validation dataset.
    validation_label : Tensor of type tf.uint8
        label for validation dataset.

    """
    
    testing_dataset_list = open('testing_dataset.txt', 'r') 
    validation_dataset_list = open('validation_dataset.txt', 'r') 
    train_dataset_list = open('train_dataset.txt', 'r') 
    
    testing_dataset = []
    validation_dataset = []
    train_dataset = []
    
    testing_label = []
    validation_label = []
    train_label = []
    
    for files in testing_dataset_list.read().split("\n"):
        if files != "":
            testing_dataset.append(mfcc(dataset_path + "/" + files));
            label = train_commands.index(files.split("/")[0])
            testing_label.append([label])
            
    testing_dataset = tf.cast(testing_dataset, tf.float32)
    testing_label = tf.cast(testing_label, tf.uint8)          
            
    for files in validation_dataset_list.read().split("\n"):
        if files != "":
            validation_dataset.append(mfcc(dataset_path + "/" + files));
            label = train_commands.index(files.split("/")[0])
            validation_label.append([label])
            
    validation_dataset = tf.cast(validation_dataset, tf.float32)
    validation_label = tf.cast(validation_label, tf.uint8)     
        
    for files in train_dataset_list.read().split("\n"):
        if files != "":
            train_dataset.append(mfcc(dataset_path + "/" + files));
            label = train_commands.index(files.split("/")[0])
            train_label.append([label])
            
    train_dataset = tf.cast(train_dataset, tf.float32)
    train_label = tf.cast(train_label, tf.uint8)    
    
    testing_dataset_list.close()
    validation_dataset_list.close()
    train_dataset_list.close()
    
    return (train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label)

if __name__ == "__main__":
    train_list()
    dataset_sort()
    train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label = dataset_gen()
    print(train_dataset)
    print(train_label)
    print(testing_dataset)
    print(testing_label)
    print(validation_dataset)
    print(validation_label)

    
    
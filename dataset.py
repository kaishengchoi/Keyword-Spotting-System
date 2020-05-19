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
from audio import decode_wav
from numpy import save, load

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
    
    testing_dataset = open('./dataset/testing_dataset.txt', 'w+') 
    validation_dataset = open('./dataset/validation_dataset.txt', 'w+') 
    train_dataset = open('./dataset/train_dataset.txt', 'w+') 
    
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
    def get_mfcc (filename):
        
        samples, sample_rate = decode_wav(filename)
        return mfcc(samples, sample_rate)
    
    def sort_out (dataset_list):
        
        dataset = []
        dataset_label = []
    
        for files in dataset_list.read().split("\n"):
            if files != "":
                dataset.append(get_mfcc(dataset_path + "/" + files));
                label = train_commands.index(files.split("/")[0])
                dataset_label.append([label])
                
        dataset = tf.cast(dataset, tf.float32)
        dataset_label = tf.cast(dataset_label, tf.uint8)          
     
        return dataset, dataset_label
        
    testing_dataset_list = open('./dataset/testing_dataset.txt', 'r') 
    validation_dataset_list = open('./dataset/validation_dataset.txt', 'r') 
    train_dataset_list = open('./dataset/train_dataset.txt', 'r') 
    
    testing_dataset, testing_label = sort_out (testing_dataset_list)
    validation_dataset, validation_label = sort_out (validation_dataset_list)
    train_dataset, train_label = sort_out (train_dataset_list)
    
    testing_dataset_list.close()
    validation_dataset_list.close()
    train_dataset_list.close()
    
    return (train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label)

def get_dataset ():
    
    saved_command = list(load('./dataset/saved_commands.npy', allow_pickle=True))
    if commands != saved_command :
        print("Updating Dataset......")
        train_list()
        dataset_sort()
        train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label = dataset_gen()
        save('./dataset/train_dataset.npy', train_dataset)
        save('./dataset/train_label.npy', train_label)
        save('./dataset/testing_dataset.npy', testing_dataset)
        save('./dataset/testing_label.npy', testing_label)
        save('./dataset/validation_dataset.npy', validation_dataset)
        save('./dataset/validation_label.npy', validation_label)
        save('./dataset/saved_commands.npy', commands)
        print("Dataset Updated!")
    else:
        print("Fetching Dataset......")
        train_dataset = load('./dataset/train_dataset.npy', allow_pickle=True)
        train_label = load('./dataset/train_label.npy', allow_pickle=True)
        testing_dataset = load('./dataset/testing_dataset.npy', allow_pickle=True)
        testing_label = load('./dataset/testing_label.npy', allow_pickle=True)
        validation_dataset = load('./dataset/validation_dataset.npy', allow_pickle=True)
        validation_label = load('./dataset/validation_label.npy', allow_pickle=True)
        print("Done Fetching Dataset.")
        
        train_dataset = tf.reshape(train_dataset, [train_dataset.shape[0], train_dataset.shape[2], train_dataset.shape[3], train_dataset.shape[1]])
        testing_dataset = tf.reshape(testing_dataset, [testing_dataset.shape[0], testing_dataset.shape[2], testing_dataset.shape[3], testing_dataset.shape[1]])
        validation_dataset = tf.reshape(validation_dataset,[validation_dataset.shape[0], validation_dataset.shape[2], validation_dataset.shape[3],validation_dataset.shape[1]])

    return train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label

if __name__ == "__main__":
    train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label = get_dataset ()


    
    
import pytest
from cnn import *

def test_breed_number_check():
    "Checking that the loaded label dataset contains 120 unique breeds"
    labels = import_labels()
    assert len(labels['breed'].unique()) == 120

def test_one_hot_encoding():

def test 

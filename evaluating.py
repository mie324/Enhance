import matplotlib.pyplot as plt
import torch
import tensorflow as tf
start_path = 'results/'


start_path = 'results/'
counter = 0
for filename in os.listdir(start_path):
    if filename.startswith('real'):

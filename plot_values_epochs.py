import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npyname', type=str, default='')

    args = parser.parse_args()
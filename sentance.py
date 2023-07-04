import torch
from torch import nn

import numpy as np

text = ['hey how are you','good i am fine','have a nice day']

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(''.join(text))
print(f'chars is {chars}')

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))
print(f'int2char is {int2char}')

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}
print(f'char2int is {char2int}')


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 14:00:32 2015

@author: admin
"""

# Pseudo code
# Step: Define the range
# Step: Start beginning of range

# Step: Test if beginning number is even
# Step: If Yes: Print
# Step: If No: Move On
# Become Program "Is_Even"

# Step: If end of range: Stop
# Step: If not end of range: Move On
# Step: Test if N+1 is Even
# ...
# Step: 
# END

# Beginning of range

x = 1
y = 11
list_to_test = (range(x,y))
for number in list_to_test:
    print number

print list_to_test

# Example
def is_even(n):
    """
    Arguments: n - an integer
    Returns: Boolean - True or False
    Tests whether n is true or galse, and returns answer
    """
    return n % 2

x = 1
y = 11
list_to_check = range(x,y)

for number in list_to_check:
    if is_even(number) == 0:
        print number


def print_even_numbers_from_list(first_number,last_number):
    x = first_number
    y = last_number
    list_to_check = range(x,y)
    
    for number in list_to_check:
        if is_even(number) == 0:
            print number

print_even_numbers_from_list(1,11)
print_even_numbers_from_list(1000,1005)
print_even_numbers_from_list(1000000,1000005)
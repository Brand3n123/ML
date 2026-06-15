import numpy as np

list_of_nums = [4, 8, 15, 16, 23, 42]
array = np.array(list_of_nums)
print(type(array))
print(type(list_of_nums))


first_value = array[0]
print(first_value)
last_element = array[-1]
print(last_element)


array[0] = 100
array[-1] = 50
print(array)

sequential_numbers = np.arange(17) # vector/array with 1-16
seq_num_2 = np.arange(17, 23, 2.5)
print(sequential_numbers)
print(seq_num_2)


#concatenation
list1 = [1, 2]
list2 = [3, 4]
print(list1 + list2)

#addition
first = np.arange(1, 5) #1, 2, 3, 4
second = np.array([6, 8, 3, 2])
summation = first + second
print(summation)

#subtraction
subtraction = first - second
print(subtraction)

#multiplication
multiplication = first * second
print(multiplication)

#division
division = np.around(first / second, 2) #rounds to 2 decimals
print(division)

#exponentation
exponentation = first ** second
print(exponentation)


#DataTypes
array = np.array([4, 8, 15, 16, 23, 42])
print(array.dtype) #int64
array = np.array([4, 8, 15, 16, 23, 42], dtype = np.int8)
print(array.dtype) #int8
array = array + 100 # no good, due to int8
print(array) #no good

#size
print(array.size)
print(array.nbytes)
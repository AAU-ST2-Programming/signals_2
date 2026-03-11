def my_summer(a=None,b=None,c=None,d=None,e=None):  # pyright: ignore[reportRedeclaration]
    my_sum = 0
    if a is not None:
        my_sum += a 
    if b is not None:
        my_sum += b
    
    return my_sum

print(my_summer(50,4))

def my_summer(A):
    return sum(A)
print(my_summer([50,4]))

import multiprocessing
import time
from functools import partial

N_sentences = 1
N_s = 1
# define the list of strings
strings_list = [
    "The quick brown fox jumps over the lazy dog."*N_sentences,
    "She sells seashells by the seashore."*N_sentences,
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"*N_sentences
]*N_s


# define the dictionary of word replacements
replacements = {
    "quick": "slow",
    "brown": "red",
    "lazy": "energetic",
    "seashells": "pearls",
    "woodchuck": "beaver"
}

# define the function to process each string
def process_string(replacements,old_string):
    # split the string into words

    # loop through each word in the string
    it = 0
    for item in replacements:
        # check if the word is in the dictionary
        if item in old_string:
            # replace the word with its value in the dictionary
            old_string = old_string.replace(item,replacements[item])
            it+=1
    # join the words back into a string
    return old_string,it

if __name__ == '__main__':
    # create a pool of worker processes
    pool = multiprocessing.Pool()
    
    # use the pool to process each string in parallel
    func = partial(process_string,replacements)
    processed_strings = pool.map(func, strings_list)
    
    pool.close()
    pool.join()



    # print the modified list of strings
    print(processed_strings)


# from functools import partial
  
# # A normal function
# def f(a, b, c, x):
#     return 1000*a + 100*b + 10*c + x
  
# # A partial function that calls f with
# # a as 3, b as 1 and c as 4.
# g = partial(f, 3, 1, 4)
  
# # Calling g()
# print(g(5))
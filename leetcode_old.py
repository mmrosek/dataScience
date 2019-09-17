### Hamming distance

class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        return bin(x^y).count('1')

########################################################    

# Perimeter of an island represented as a grid

import operator

grid = [[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]

def islandPerimeter(self, grid):
    return sum(sum(map(operator.ne, [0] + row, row + [0])) for row in grid + map(list, zip(*grid)))
    return(sum(sum(map(operator.ne, [0] + row, row + [0])) for row in grid + list(map(list,zip(*grid)))))

for row in grid + list(map(list, zip(*grid))):
    print(sum(map(operator.ne,row + [0], [0] + row)))
for row in grid + list(map(list, zip(*grid))):
    print(row + [0], [0] + row)

### Checks for number of differences between two lists of numbers --> very useful
sum(map(operator.ne,[1,0,1,0],[1,0,0,0]))

#################################################################

### Using recursion to reverse a string

def reverseString(s):
    l = len(s)
    if l < 2:
        return s
    return reverseString(s[int(l/2):]) + reverseString(s[:int(l/2)])

#################################################################

### Using map, filter and reduce

# Returns a list with each value equal to itself + itself + 1 
#  Function (x --> x + x + 1) is being mapped to each value in the list
list(map(lambda x: x + x + 1, [1,2,3,4]))

# Returns only numbers that are smaller than their squares
list(filter(lambda x: x < x**2, [0,1,2,3,4]))

# Adds 1 + 2**2, stores the value in x (5) and then adds 3**2, stores value (14), then adds 4**2 to get 30
from functools import reduce
reduce(lambda x,y: x+y**2, [1,2,3,4])

####################################################################

### Return only number in a list that appears once

from collections import defaultdict
import collections

test_dict = defaultdict(int)

test = [1,1,2,3,3]

for num in test:
    test_dict[num] += 1

# Gets same result as above without using defaultdict
dic = {}
for num in test:
    dic[num] = dic.get(num,0)+1

answer = [key for key in test_dict if test_dict[key] == 1][0]

### XOR solution
ans = 0
for num in test:
    # Sets ans = the result of XOR with ans and num --> XOR of a number with itself is 0
    #  Therefore, the only number left will be the number without a match
    ans ^= num
 
# More efficient way to get count of elements in a list (or characters in a string)
count_dict = collections.Counter(test)

######################################################################

# Fast way to return missing elements in a list (supposed to have 1-n elements, some elements repeat) 
#  without creating any extra objects(space) and running in O(n)

nums = [1,1,2,3,3,5]

N = len(nums)

for i in range(N):
    x = nums[i]%N
    nums[x-1] = nums[x-1] + N
    
ans =  [i+1 for i in range(N) if nums[i]<=N]

#######################################################################

### maxDepth finds the maximum depth of a binary tree

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
 
tree = TreeNode(14)

tree.right = TreeNode(10)

tree.right.right = TreeNode(69)

tree.right.left = TreeNode(13)


def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

maxDepth(tree)

#######################################################################

### Add two numbers without using "+" or "-"

def getSum(a, b):
    """
    :type a: int
    :type b: int
    :rtype: int
    """
    # 32 bits integer max
    MAX = 0x7FFFFFFF
    # 32 bits interger min
    MIN = 0x80000000
    # mask to get last 32 bits
    mask = 0xFFFFFFFF
    while b != 0:
        print(a,b)
        # ^ get different bits and & gets double 1s, << moves carry
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask
    # if a is negative, get a's 32 bits complement positive first
    # then get 32-bit positive's Python complement negative
    return a if a <= MAX else ~(a ^ mask)

MAX = 0x7FFFFFFF
MIN = 0x80000000
mask = 0xFFFFFFFF

#########################################################################

### Invert a binary tree

def invertTree(root):
if root:
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

#########################################################################

### Add digits of a number together until there is only one number left

from functools import reduce

def addDigits(num):
    """
    :type num: int
    :rtype: int
    """
    while num > 9:
        num = reduce((lambda x, y: int(x) + int(y)), list(str(num)))
    return num

######################################################################

# Calculate optimal length and width given area

import math

area = 210

length = 10000000
width = 0
for i in range(int(math.sqrt(area)),0,-1):
    if (area % i == 0) & (abs(i - area//i) < length-width):
        length = max(i,area//i)
        width = min(i,area//i)
print([length, width])

#########################################################################  
              
### Sort a list after calling a function on each element 

# Sort so that 0s at end of list
nums = [1,2,69,1,0]
nums.sort(key = lambda x: 1 if x == 0 else 0)
nums

# Sort by 2nd letter
test = ['hello','batch','which','rmsprop']
test.sort(key = lambda x: x[1])
test

#########################################################################

### Finds minimum absolute difference between two nodes in a binary search tree

#Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

tree = TreeNode(15)

tree.right = TreeNode(30)

tree.right.right = TreeNode(35)

tree.right.left = TreeNode(20)

tree.left = TreeNode(6)

tree.left.right = TreeNode(9)

tree.right.left.right = TreeNode(25)

# Works by always checking if its possible to go left, starting at the root
#  When it is not possible to go left, append value of the leaf then try to go right 
#   Then check from there if we can go left, rinse and repeat

# 1. Check if you can go left
#   1a. If you can go left, do it and go back to step 1
# 2. If can't go left or already did, append value of leaf
# 3. Go right if possible
#   3a. If you can go right, go right and start back at step 1 
# 4. If you can't go right, cycle starts back where you last went left and you are now at step 2
def getMinimumDifference(root):
    l = []
    def bfs(node):
        print("Node.val" + str(node.val))
        if node.left: bfs(node.left)
        print("Append.val" + str(node.val))
        l.append(node.val)
        if node.right: bfs(node.right)
    bfs(root)
    return min([abs(l[i] - l[i + 1]) for i in range(len(l) - 1)])
 
getMinimumDifference(tree)


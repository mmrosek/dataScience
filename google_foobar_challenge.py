#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:57:39 2019

@author: miller
"""


# 1
def solution(area):
  panel_areas = []
  while area > 0:
    side_length = int(area ** 0.5)
    panel_area = side_length ** 2
    panel_areas.append(panel_area)
    area -= panel_area
  return panel_areas

solution(12)
##############################

from math import log2

def min_ct(lambs):
    return int(log2(lambs + 1))

def fibonacci(ct):
    a,b = 1,1
    for _ in range(0,ct):
        a, b = b, a + b
    return a

def max_ct(lambs):
    n = 1
    while True:
        total = (fibonacci(n + 1) - 1)
        if total <= lambs:
            n += 1
        else:
            n -= 1
            break
    return n

def answer(lambs):
    return max_ct(lambs) - min_ct(lambs)

answer(13)

fibonacci(7)

1,1,2,3,5,8,13,21


from itertools import accumulate
def max_ct(lambs):
    for henchmen, total_pay in enumerate(accumulate(fibonacci())):
        if total_pay > lambs:
            return henchmen

def fibonacci():
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b



max_ct(13)


answer(13)


stingy(13)

def stingy(lambs):
    num = 1
    while True:
        total = (fibonacci(num + 2) - 1)
        if total <= lambs:
            num += 1
        else:
            num -= 1
            break
    return num

def fibonacci(num):
    if num > -1 and num < 3:
        return 1
    else:
        num = fibonacci(num - 1) + fibonacci(num - 2)
        return num

fibonacci(8)


from math import sqrt
from math import log
from math import pow

def answer(total_lambs):
    phi = (1+sqrt(5))/2  # golden search ratio
    tau = (1-sqrt(5))/2  # equal to 1/phi
    eps = pow(10, -10)

    max_hunchmen = int(round(log((total_lambs + 1) * sqrt(5)+eps, phi))) - 2
    Fib_num = int(round((pow(phi, max_hunchmen+2)-pow(tau,max_hunchmen+2))/sqrt(5)))
    if total_lambs+1 < Fib_num:
      max_hunchmen -= 1

    min_hunchmen = int(log((total_lambs + 1), 2))

    return abs(max_hunchmen - min_hunchmen)




def answer(total_lambs):
    
    if total_lambs <= 10**9: return 0
    
    phi = (1+sqrt(5))/2  
    tau = (1-sqrt(5))/2  
    eps = pow(10, -10)
    
    max_hunchmen = int(round(log((total_lambs + 1) * sqrt(5)+eps, phi))) - 2
    Fib_num = int(round((pow(phi, max_hunchmen+2)-pow(tau,max_hunchmen+2))/sqrt(5)))
    if total_lambs+1 < Fib_num:
      max_hunchmen -= 1
    elif total_lambs + 1 == Fib_num:
        total_lambs = Fib_num
    if (total_lambs + 1) % 2 == 0:
         min_hunchmen = int(round(log((total_lambs + 1), 2)))
    else:
        min_hunchmen = int(log((total_lambs + 1), 2))
    
    return abs(max_hunchmen - min_hunchmen)




from math import log2



def min_ct(total_lambs):
    log2 = (total_lambs + 1).bit_length() - 1
    return int(log2)

def fibonacci(n):
    a,b = 1,1
    for _ in range(0,n):
        a, b = b, a + b
    return a

def max_ct(total_lambs):
    n = 1
    while True:
        total = (fibonacci(n + 1) - 1)
        if total <= total_lambs:
            n += 1
        else:
            n -= 1
            break
    return n

def solution(total_lambs):
    return max_ct(total_lambs) - min_ct(total_lambs)

solution(10)

x=8
(7).bit_length() - 1

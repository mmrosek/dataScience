import string
import numpy as np
import math

def new_char():
    
    all_possible_chars = string.printable[:-10] + " "
    idx = np.random.randint(len(all_possible_chars))
    
    return all_possible_chars[idx]

class DNA:
    def __init__(self, num_genes):
        self.genes = []
        self.fitness = 0
        self.num_genes = num_genes
    
        for i in range(self.num_genes): 
            self.genes.append(new_char()) #Pick from range of chars

    # Converts character array to a String
    def get_phrase(self): 
        return "".join(self.genes);

    # Fitness function (returns floating point % of "correct" characters)
    def calc_fitness(self, target):
        score = 0
        for i in range(len(self.genes)):
            if self.genes[i] == target[i]:
                score += 1
        self.fitness = score / len(target)

    # Crossover
    def crossover(self, partner):
        # A new child
        child = DNA(len(self.genes))

        midpoint = min(max(3, np.random.randint(len(self.genes))), len(self.genes)-3) # Pick a midpoint

        for i in range(len(self.genes)):  
            if (i > midpoint): child.genes[i] = self.genes[i]
            else: child.genes[i] = partner.genes[i]

        return child

    # Based on a mutation probability, picks a new random character
    def mutate(self, mut_rate):
        for i in range(len(self.genes)):  
            if (np.random.random() < mut_rate):
                self.genes[i] = new_char()
                
                
if __name__ == "__main__":
    pass

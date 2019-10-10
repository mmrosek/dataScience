import string
import numpy as np
import math

def new_char():
    
    all_possible_chars = string.printable[:-10] + " "
    idx = np.random.randint(len(all_possible_chars))
    
    return all_possible_chars[idx]

class DNA:
    def __init__(self, num_genes, verbose = False):
        self.genes = []
        self.fitness = 0
        self.num_genes = num_genes
        self.verbose = verbose
    
        for i in range(self.num_genes): 
            self.genes.append(new_char()) 

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

    def crossover(self, partner, midpt_bool):

        child = DNA(len(self.genes))
        
        if not midpt_bool:
        
            total_fitness = self.fitness + partner.fitness
            
            # THIS WAS NEW
            # self_prob = prob of taking one of own genes in crossover
            # Weighting self_prob based on fitness, capping at max_self_prob
            max_self_prob = 0.8
            if total_fitness == 0: self_prob = 0.5    
            else: self_prob = min( max_self_prob, max( (1-max_self_prob) , self.fitness / max(total_fitness, 1e-4) ) )
            
            if self.verbose:
                print(f"self.fitness: {self.fitness}")
                print(f"partner.fitness: {partner.fitness}")            
                print(f"self_prob: {self_prob}")
            
            for i in range(len(self.genes)):  
                val = np.random.random()
                if self.verbose: print(f"val: {val}")
                if val < self_prob: 
                    if self.verbose: print("self gene")
                    child.genes[i] = self.genes[i]
                else: child.genes[i] = partner.genes[i]
            
        else:

            midpt = min(max(2, np.random.randint(len(self.genes))), len(self.genes)-2) 

            for i in range(len(self.genes)):  
                if (i > midpt): child.genes[i] = self.genes[i]
                else: child.genes[i] = partner.genes[i]

        return child

    def mutate(self, mut_rate):
        '''Based on a mutation probability, picks a new random character'''
        for i in range(len(self.genes)):  
            if (np.random.random() < mut_rate):
                self.genes[i] = new_char()
                
### Needed for importation of DNA class ###
if __name__ == "__main__":
    pass

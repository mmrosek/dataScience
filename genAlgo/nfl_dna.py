import numpy as np

'''
ToDo:
    Create function that draws from pre-computed player pdfs --> draw_plyr_pdf()
'''

# Somehow need to ensure we dont pick a player already in self.plyrs
def new_plyr():
    idx = np.random.randint(len(plyrs))
    return plyrs[idx]

class DNA:
    def __init__(self, n_plyrs, verbose = False):
        self.plyrs = []
        self.fitness = 0
        self.verbose = verbose
        
        # Intializing dna/plyrs
        for i in range(n_plyrs): self.plyrs.append(new_plyr()) 
    
    # Fitness function (returns floating point % of "correct" characters)
    def calc_fitness(self, n_draws, min_eval_perc, max_eval_perc):
        '''
        n_draws: number of times to draw from the pdf of each player
        min_eval_perc: percentile indicating lower bound of lineup totals to include in fitness
        max_eval_perc: percentile indicating upper bound of lineup totals to include in fitness
        '''
        if min_eval_perc > 1: min_eval_perc /= 100
        if max_eval_perc > 1: max_eval_perc /= 100
        
        # Drawing from each player's pdf n_draws times
        draws=np.zeros(shape=(self.n_plyrs, n_draws))
        for plyr_idx in range(len(self.n_plyrs)):            
            for draw_idx in range(n_draws):
                draws[plyr_idx, draw_idx] = self.draw_plyr_pdf(self.plyrs[plyr_idx])
                
        # Correlating draws between players and summing
        correlated_draws = self.correlate_plyrs(draws)
        correlated_lineup_totals = correlated_draws.sum(axis=0)
        
        # Taking mean of lineup totals b/w min and max eval %
        min_eval_idx = n_draws * min_eval_perc
        max_eval_idx = n_draws * max_eval_perc
        self.fitness = correlated_lineup_totals[min_eval_idx:max_eval_idx].mean()
        
    ### Not sure if these need to be class methods, may be better to have as external methods each instance calls
    def correlate_plyrs(self):
        pass

    def draw_plyr_pdf(self):
        pass

    def crossover(self, partner, midpt_bool):

        child = DNA(len(self.plyrs))
        
        if not midpt_bool:
            total_fitness = self.fitness + partner.fitness
            
            # THIS WAS NEW
            # self_prob = prob of taking one of own plyrs in crossover
            # Weighting self_prob based on fitness, capping at max_self_prob
            max_self_prob = 0.8
            if total_fitness == 0: self_prob = 0.5    
            else: self_prob = min( max_self_prob, max( (1-max_self_prob) , self.fitness / max(total_fitness, 1e-4) ) )
            
            if self.verbose:
                print(f"self.fitness: {self.fitness}")
                print(f"partner.fitness: {partner.fitness}")            
                print(f"self_prob: {self_prob}")
            
            for i in range(len(self.plyrs)):  
                val = np.random.random()
                if self.verbose: print(f"val: {val}")
                if val < self_prob: 
                    if self.verbose: print("self gene")
                    child.plyrs[i] = self.plyrs[i]
                else: child.plyrs[i] = partner.plyrs[i]
            
        else:
            midpt = min(max(2, np.random.randint(len(self.plyrs))), len(self.plyrs)-2) 

            for i in range(len(self.plyrs)):  
                if (i > midpt): child.plyrs[i] = self.plyrs[i]
                else: child.plyrs[i] = partner.plyrs[i]

        return child

    def mutate(self, mut_rate):
        '''Based on a mutation probability, picks a new random character'''
        for i in range(len(self.plyrs)):  
            if (np.random.random() < mut_rate):
                self.plyrs[i] = new_plyr()
                
### Needed for importation of DNA class ###
if __name__ == "__main__":
    pass

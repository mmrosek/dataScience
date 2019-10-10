import string
import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from scipy.stats import pearsonr
from scipy.stats import randint as sp_randint
from sklearn.metrics import r2_score
import pdb

def new_char():
    
    all_possible_chars = string.printable[:-10] + " "
    idx = np.random.randint(len(all_possible_chars))
    
    return all_possible_chars[idx]

def mse(arr1, arr2):
    return ((arr1 - arr2)**2)/arr1.shape[0]



class DNA:
    def __init__(self, data, preproc_algos, models, mutant = False, verbose = False):
        self.genes = {}
        self.genes["data"] = []
        self.genes["preproc"] = []
        self.genes["models"] = []
        self.fitness = 0
        self.verbose = verbose
                
        if not mutant:
            # Allocating genes --> data, preproc and models are lists of strings
            print(f"data: {data}")
            for idx in data: 
                if np.random.random() > 0.01: self.genes["data"].append(idx)
                else: self.genes["data"].append(None)
           
            # Ensuring each DNA instance has at least one dataset
            if len(self.genes["data"]) == 0:
                idx = np.random.randint(0, len(data))
                self.genes["data"].append(data[idx])
            
            print(f"self.genes['data']: {self.genes['data']}")

            for p in preproc_algos: 
                if np.random.random() > 0.01: self.genes["preproc"].append(p)
                else: self.genes["preproc"].append(None)

             
            for m in models: 
                if np.random.random() > 0.01: self.genes["models"].append(m)
                else: self.genes["models"].append(None)
            
            # Ensuring each DNA instance has at least one model
            if len(self.genes["models"]) == 0:
                idx = np.random.randint(0, len(models))
                self.genes["models"].append(models[idx])
               

    def crossover(self, partner, midpt_bool):

        child = DNA( None, None, None, mutant=True)
        
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
                
            for i in range(len(self.genes['data'])):
                val = np.random.random()
                if self.verbose: print(f"val: {val}")
                if val < self_prob: 
                    if self.verbose: print("self gene")
                    child.genes['data'].append(self.genes['data'][i])
                else: child.genes['data'].append(partner.genes['data'][i])
                    
            for i in range(len(self.genes['models'])):
                val = np.random.random()
                if self.verbose: print(f"val: {val}")
                if val < self_prob: 
                    if self.verbose: print("self gene")
                    child.genes['models'].append(self.genes['models'][i])
                else: child.genes['models'].append(partner.genes['models'][i])
            
        else:

            midpt = min(max(2, np.random.randint(len(self.genes))), len(self.genes)-2) 

            for i in range(len(self.genes)):  
                if (i > midpt): child.genes[i] = self.genes[i]
                else: child.genes[i] = partner.genes[i]

        return child

    # NEED TO UPDATE !!!!
    def mutate(self, mut_rate):
        
        '''Based on a mutation probability, picks a new random character'''
        
        for i in range(len(self.genes['data'])):  
            pass
#             if (np.random.random() < mut_rate):
#                 self.genes['data'] = new_char()
                
           
    def calc_fitness(self, df_dict, tgt):
        
        self.genes['preds'] = []
        
        # Perform preprocessing if desired
#         for df in self.genes['data']:
#             if 'preproc_algos' in self.genes.keys(): pass
#             else: continue
        
        # Concatenating subsets into full df
        df_keys = [df_idx for df_idx in self.genes['data'] if df_idx is not None]
        df_tuple = tuple([df_dict[key] for key in df_keys])
        
        df = np.concatenate( df_tuple , axis=1)
#         full_df = pd.concat([df_dict[key] for key in df_keys], axis=1)
        
        X_tr, X_te, y_tr, y_te = self.split_train_test(df,tgt)
        del df
        
        for model in self.genes['models']:
            if model is not None:
                test_preds = self.train_mod_and_predict(model, X_tr, y_tr, X_te)            
                self.genes['preds'].append(test_preds)
                try: print(f"\nR2 for test_preds: {r2_score(y_te, test_preds)}")
                except: pdb.set_trace()
                print(f"\n test_preds head: {test_preds[:5]}")
        
        # Ensembling and final fitness calculation
        if len(self.genes['preds']) == 0: self.fitness = 0
        else: self.fitness = self.ensemble_and_score(self.genes['preds'], y_te)
            
            
    def split_train_test(self, df, tgt, rand_state = 2, test_float = 0.2):
        X_tr, X_te, y_tr, y_te = train_test_split(df, tgt, test_size=test_float, random_state=rand_state)
        return X_tr, X_te, y_tr, y_te
        
        
    def train_mod_and_predict(self, mod, X_tr, y_tr, X_te, num_folds = 5, n_iter = 10):
    
        if mod == 'rf':
            est = RandomForestRegressor(criterion='mse')    

            params = {'max_depth': sp_randint(1,12),
                      'min_samples_leaf': sp_randint(1,50),
                      'n_estimators': sp_randint(1,30),
                      'max_features': sp_randint(X_tr.shape[1]*0.3, X_tr.shape[1])}

            rs = RandomizedSearchCV(est, param_distributions=params, 
                                    n_jobs=24, n_iter=n_iter, cv=num_folds)

            print("\nPerforming randomized search")
            rs.fit(X_tr, y_tr)
            print("Best score: %0.3f" % rs.best_score_)
            print("Best parameters set:")
            best_parameters = rs.best_estimator_.get_params()
            for param_name in sorted(params.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))

            preds = rs.predict(X_te)
            return preds

        elif mod == 'lr':
            print("Linear regression")
            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            preds = lr.predict(X_te)
            return preds       
        
    def ensemble_and_score(self, pred_list, y_te, ens_perc = 0.7):
        
        '''NEED TO SPLIT PREDS TO LEARN WEIGHTS ON TOP HALF AND EVAL ON BOTTOM HALF of test set'''
        
        ### Processing pred_list ###
        pred_array = np.array(pred_list)
        
        print(f"\npred_array.shape: {pred_array.shape}")
#         print(f"pred_array: {pred_array}")
        
#         print(f"\nScore on og_preds: {r2_score(eval_labels, eval_preds[:,0])}")
        
        # Ensuring pred_array has column dimension if only one set of preds
        if pred_array.ndim == 1: pred_array.reshape(-1,1)
            
        # Transposing so pred_array will have samples as rows and predictions by each model as different col
        else: pred_array = pred_array.T
        
        print(f"\npred_array head: {pred_array[:5,:]}")
        
        if pred_array.shape[0] != y_te.shape[0]: raise Exception("Different number of predictions and ground truths.")
        
        ###############################   
        print(f"pred_array.shape post-processing: {pred_array.shape}")
        
        print(f"\ny_te[:5]: {y_te[:5]}")
        
        ens_preds, eval_preds, ens_labels, eval_labels = self.split_train_test(pred_array, y_te, test_float = 1-ens_perc)
        
#         num_ens_samples = int(pred_array.shape[0] * ens_perc)
        
#         # Model predictions and labels used to learn ensemble weights
#         ens_preds = pred_array[ : num_ens_samples, :]
#         ens_labels = y_te[ : num_ens_samples].reshape(-1,1)
        
#         # Model predictions and labels used for evaluation
#         eval_preds = pred_array[ num_ens_samples : , :]
#         eval_labels = y_te[ num_ens_samples : ].reshape(-1,1)
        
#         print(f"\n ens_preds first col: {ens_preds[:,0]}")
#         print(f"\n eval_preds first col: {eval_preds[:,0]}")        
#         print(f"\nens_labels: {ens_labels}")
#         print(f"\neval_labels: {eval_labels}")

        print(f"ens_preds.shape: {ens_preds.shape}")
        print(f"eval_preds.shape: {eval_preds.shape}")
        print(f"ens_labels.shape: {ens_labels.shape}")
        print(f"eval_labels.shape: {eval_labels.shape}")  
        
#         ### Ensembling ###
#         score = -10000
#         for wt in [0.1, 0.3, 0.5, 0.7, 0.9]:
#             wt_score = r2_score(eval_labels, (eval_preds[:,0]*wt + eval_preds[:,1]*(1-wt)))
#             if wt_score > score:
#                 final_wt = wt
#                 score = wt_score
                
#         final_wt_score = r2_score(eval_labels, (eval_preds[:,0]*final_wt + eval_preds[:,1]*(1-final_wt)))
#         print(f"\nScore from simple weighting: {final_wt_score}\n")
#         print(f"final_wt: {final_wt}")
        
        lr = LinearRegression()        
        lr.fit(ens_preds, ens_labels)
        ens_eval_preds = lr.predict(eval_preds)
        
        el_net = self.elastic_net_ensemble(ens_preds, ens_labels)
        print(f"\nel_net coefficients: {el_net.coef_}")
        el_net_ens_eval_preds = el_net.predict(eval_preds)
        
        ###################
        if self.verbose:
            print(f"\nScore on el_net eval: {r2_score(eval_labels, el_net_ens_eval_preds)}")

            print(f"\nScore on training/ensemble samples: {lr.score(ens_preds, ens_labels)}")
            print(f"\nScore on lr.score(eval_preds, eval_labels): {lr.score(eval_preds, eval_labels)}")
            print(f"\nScore on averaging eval samples: {r2_score(eval_labels,np.mean(eval_preds, axis=1))}")
            print(f"\nScore on first col eval_preds: {r2_score(eval_labels, eval_preds[:,0])}")
            print(f"\nScore on full og_preds first col: {r2_score(y_te, pred_array[:,0])}")
            print(f"\nScore on avg og_preds: {r2_score(y_te, np.mean(pred_array, axis=1))}")

            print(f"\nLR coefficients: {lr.coef_}")
            print(f"LR intercept: {lr.intercept_}")

            print(f"ens eval preds shape: {ens_eval_preds.shape}")

            print(f"\neval_preds[:5, :]: {eval_preds[:5, :]}")
            print(f"\nens_eval_preds[:5]: {ens_eval_preds[:5]}")
            print(f"eval_labels[:5]: {eval_labels[:5]}")
        
        # Ensuring ens_eval_preds has column dimension if only one set of preds
        if ens_eval_preds.ndim == 1: ens_eval_preds.reshape(-1,1)
           
        if ens_eval_preds.shape != eval_labels.shape:
            raise Exception("Shape of preds is not the same as the shape of labels")
        
        print(f"ens_eval_preds.shape: {ens_eval_preds.shape}")
        print(f"eval_labels.shape: {eval_labels.shape}")
        
        score = r2_score(eval_labels, ens_eval_preds)
        print(f"\nScore: {score}\n")
        
        return score
    
    def elastic_net_ensemble(self, X_train, y_train):
        
        el = ElasticNet(normalize=True, max_iter=10000)
        parameters = {
                'alpha': (0.2, 0.5, 1, 5),
                 'l1_ratio': (0.5, 0.7, 0.9, 1)
            }

        # find the best parameters for both the feature extraction and classifier
        gs = GridSearchCV(el, parameters, scoring = 'r2', n_jobs=16, cv = 10)

        print("\nPerforming grid search")
        gs.fit(X_train, y_train)
        print("Best score: %0.3f" % gs.best_score_)
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        return gs.best_estimator_
    
                
    def get_genes(self):
        return {'Data':self.genes['data'], 'Preprocessing':self.genes['preproc'], 'Models':self.genes['models']}
                                          
       
        
                
### Needed for importation of DNA class ###
if __name__ == "__main__":
    pass

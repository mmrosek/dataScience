import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from scipy.stats import pearsonr
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.metrics import r2_score
import xgboost as xgb
import pdb

#####################################################################################

def split_train_test(df, tgt, rand_state = 2, test_float = 0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(df, tgt, test_size=test_float, random_state=rand_state)
    return X_tr, X_te, y_tr, y_te

#####################################################################################

def train_rf(X_tr, y_tr, X_te, n_iter = 25, num_folds = 5):
    
    print("Random Forest")
    params = {'max_depth': sp_randint(1,5),
              'min_samples_leaf': sp_randint(5,50),
              'n_estimators': sp_randint(1,30),
              'max_features': sp_randint(max(int(X_tr.shape[1]*0.2),1), X_tr.shape[1])}

    rs = RandomizedSearchCV(estimator = RandomForestRegressor(criterion='mse'), 
                            param_distributions=params, n_jobs=24, n_iter=n_iter, cv=num_folds)

    print("Performing randomized search")
    try: rs.fit(X_tr, y_tr)
    except Exception as e:
        print(e)
        pdb.set_trace()
    print("Best score: %0.3f" % rs.best_score_)
    best_parameters = rs.best_estimator_.get_params()
    preds = rs.predict(X_te)
    return preds, best_parameters, rs.best_estimator_.feature_importances_

#####################################################################################

def train_xgb(X_tr, y_tr, X_te, n_iter = 25, num_folds = 5):
    
    print("XGBoost")
    params = {'max_depth': sp_randint(1,5),
              'min_child_weight': sp_randint(1,35),
              'learning_rate': uniform(0.06,0.03),
              'reg_lambda': uniform(1.5,1.5),
              'subsample': uniform(0.8,0.2),
              'colsample_bytree':uniform(0.8,0.2)}

    rs = RandomizedSearchCV(estimator = xgb.XGBRegressor(),
        param_distributions=params, cv = num_folds, n_jobs = 24, n_iter = n_iter)

    print("Performing randomized search")
    try: rs.fit(X_tr, y_tr)
    except Exception as e:
        print(e)
        pdb.set_trace()
    print("Best score: %0.3f" % rs.best_score_)
    best_parameters = rs.best_estimator_.get_params()
    preds = rs.predict(X_te)
    
#     # Add features whose importance are 0
#     for i in range(x_train.shape[1]):
#         key = 'f' + str(i)
#         if key not in importances_raw.keys(): ## The importance of this feature is 0
#             importances_raw[key] = 0
    
    return preds, best_parameters, rs.best_estimator_.feature_importances_

#####################################################################################

class DNA:
    def __init__(self, data, preproc_algos, models, mutant = False, verbose = False):
        self.genes = {}
        self.genes["data"] = []
        self.genes["preproc"] = []
        self.genes["models"] = []
        self.genes["model_params"] = []
        self.genes["feat_imps"] = []
        self.fitness = 0
        self.verbose = verbose
        
        # Allocating genes --> data, preproc and models are lists of strings
        # For new DNA instances (i.e. not being bred by combining existing instances)
        if not mutant:
            
            for idx in data: 
                if np.random.random() > 0.5: self.genes["data"].append(idx)
                else: self.genes["data"].append(None)
           
            # Ensuring each DNA instance has at least one dataset
            if np.sum([1 for d in self.genes["data"] if d is not None]) == 0:
                idx = np.random.randint(0, len(data))
                self.genes["data"][0] = data[idx]
            
            print(f"self.genes['data']: {self.genes['data']}")

            
            for p in preproc_algos: 
                if np.random.random() > 0.01: self.genes["preproc"].append(p)
                else: self.genes["preproc"].append(None)

                    
            for m in models: 
                if np.random.random() > 0.5: self.genes["models"].append(m)
                else: self.genes["models"].append(None)
                    
                # Initializing
                self.genes["model_params"].append(None)
                self.genes["feat_imps"].append(None)
            
            # Ensuring each DNA instance has at least one model
            if np.sum([1 for m in self.genes["models"] if m is not None]) == 0:
                idx = np.random.randint(0, len(models))
                self.genes["models"][0] = models[idx]
                
            print(f"self.genes['models']: {self.genes['models']}")
            
    #################################################################################               

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
            
            if self.verbose: print(f"self.fitness: {self.fitness}\npartner.fitness: {partner.fitness}\nself_prob: {self_prob}")
            
            # Mixing data genes
            for i in range(len(self.genes['data'])):
                val = np.random.random()
                if self.verbose: print(f"val: {val}")
                if val < self_prob: 
                    if self.verbose: print("self gene")
                    child.genes['data'].append(self.genes['data'][i])
                else: child.genes['data'].append(partner.genes['data'][i])
            
            # Mixing model genes
            for i in range(len(self.genes['models'])):
                val = np.random.random()
                if self.verbose: print(f"val: {val}")
                if val < self_prob: 
                    if self.verbose: print("self gene")
                    child.genes['models'].append(self.genes['models'][i])
                    child.genes['model_params'].append(self.genes['model_params'][i])
                    child.genes['feat_imps'].append(self.genes['feat_imps'][i])
                else: 
                    child.genes['models'].append(partner.genes['models'][i])
                    child.genes['model_params'].append(partner.genes['model_params'][i])
                    child.genes['feat_imps'].append(partner.genes['feat_imps'][i])
            
        else:

            midpt = min(max(2, np.random.randint(len(self.genes))), len(self.genes)-2) 

            for i in range(len(self.genes)):  
                if (i > midpt): child.genes[i] = self.genes[i]
                else: child.genes[i] = partner.genes[i]

        return child

    ###############################################################################
    
    def mutate(self, mut_rate, models):
        
        '''
        Based on a mutation probability, picks a new random character
        '''
        if (np.random.random() < mut_rate):

            # Selecting which gene (data, preproc or models) to mutate and which idx (nucleotide) w/in the gene to mutate
            genes = [key for key in self.genes.keys() if key not in ['model_params', 'feat_imps']]
            gene = genes[np.random.randint(0, len(genes))]
            print(f"Gene being mutated: {gene}")
            
            if gene != 'preproc':
                try: nucleotide = np.random.randint(0, len(self.genes[gene]))
                except: pdb.set_trace()

            if gene == 'data':
                if self.genes[gene][nucleotide] is None:
                    self.genes[gene][nucleotide] = nucleotide
                else:
                    self.genes[gene][nucleotide] = None
                                         
            elif gene == 'models':
                candidates = models + [None]
                self.genes[gene][nucleotide] = models[np.random.randint(0, len(models))]
                
    #################################################################################   
        
    def calc_fitness(self, df_dict, tgt):
        
        self.genes['preds'] = []
        self.genes['model_params'] = []
        self.genes['feat_imps'] = []
        
        # Perform preprocessing if desired
#         for df in self.genes['data']:
#             if 'preproc_algos' in self.genes.keys(): pass
#             else: continue
        
        # Concatenating subsets into full df
        df_keys = [df_idx for df_idx in self.genes['data'] if df_idx is not None]
        try: df_tuple = tuple([df_dict[key] for key in df_keys])
        except: pdb.set_trace()
        
        df = np.concatenate( df_tuple , axis=1)
        
        X_tr, X_te, y_tr, y_te = split_train_test(df,tgt)
        
        if X_tr.shape[0] == 0: pdb.set_trace()
        
        del df
        
        for idx in range(len(self.genes['models'])):
            model = self.genes['models'][idx]
            if model is not None:
                test_preds, params, feat_imps = self.train_predict(model, X_tr, y_tr, X_te)            
                self.genes['preds'].append(test_preds)
                self.genes['model_params'].append(params)
                self.genes['feat_imps'].append(feat_imps)
                try: print(f"\nR2 for test_preds: {r2_score(y_te, test_preds)}")
                except: pdb.set_trace()
            else:
                self.genes['model_params'].append(None)    
                self.genes['feat_imps'].append(None)
        
        # Ensembling and final fitness calculation
        if len(self.genes['preds']) == 0: self.fitness = 0
        else: self.fitness = self.ensemble_and_score(self.genes['preds'], y_te)
            
    ##########################################################################
        
    def train_predict(self, mod, X_tr, y_tr, X_te, num_folds = 5, n_iter = 10):
    
        if mod == 'rf': preds, params, feat_imps = train_rf(X_tr, y_tr, X_te)
            
        elif mod == 'xgb': preds, params, feat_imps = train_xgb(X_tr, y_tr, X_te)

        elif mod == 'lr':
            print("Linear regression")
            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            preds = lr.predict(X_te)
            params = lr.coef_
            feat_imps = lr.coef_
        
        return preds, params, feat_imps    
    
    ##########################################################################
        
    def ensemble_and_score(self, pred_list, y_te, ens_perc = 0.7):
        
        '''
        NEED TO SPLIT PREDS TO LEARN WEIGHTS ON TOP HALF AND EVAL ON BOTTOM HALF of test set.
        Skipping for now for simplicity.
        '''
        
        ### Processing pred_list ###
        pred_array = np.array(pred_list)
        
        if self.verbose: print(f"\npred_array.shape: {pred_array.shape}")
        
        # Ensuring pred_array has column dimension if only one set of preds
        if pred_array.ndim == 1: pred_array.reshape(-1,1)
            
        # Transposing so pred_array will have samples as rows and predictions by each model as different col
        else: pred_array = pred_array.T
        
        if pred_array.shape[0] != y_te.shape[0]: raise Exception("Different number of predictions and ground truths.")
        
        ###############################   
        if self.verbose:
            print(f"\npred_array head: {pred_array[:5,:]}")
            print(f"pred_array.shape post-processing: {pred_array.shape}")
            print(f"\ny_te[:5]: {y_te[:5]}")
        
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
        
#         lr = LinearRegression()        
#         lr.fit(pred_array, y_te)
#         ens_eval_preds = lr.predict(pred_array)
        
#         el_net = self.elastic_net_ensemble(pred_array, y_te)
#         print(f"\nel_net coefficients: {el_net.coef_}")
#         el_net_ens_eval_preds = el_net.predict(pred_array)
        
        score = r2_score( y_te, np.mean(pred_array, axis=1) )
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
    
                
    def get_genes(self, get_all = False):
        if get_all: 
            return {'Data':self.genes['data'], 'Preprocessing':self.genes['preproc'], 'Models':self.genes['models'], 'Params':self.genes['model_params'], 'Feature Importances':self.genes['feat_imps']}
        else: return {'Data':self.genes['data'], 'Preprocessing':self.genes['preproc'], 'Models':self.genes['models']}
                                          

### Needed for importation of DNA class ###
if __name__ == "__main__":
    pass

import sys
import time
import numpy as np
import algorithms as test

### running single algoritm test, instead of pipeline

data_folder = sys.argv[1]
test.load_data(data_folder)

method = sys.argv[2]
method_param = int(sys.argv[3])
outfile = sys.argv[4]

query_idx = np.arange(int(sys.argv[6]))
qn = len(query_idx)
queries = test.queries[query_idx]
queries_modified = [test.queries_modified[i] for i in query_idx]
answers = test.answers[:,query_idx]

sort_flag = True

def call_method_weighted(fnc, q_index, input_idx, nc):
    result = np.zeros(nc, dtype=np.int32)
    score_result = np.zeros(nc, dtype=np.float32)
    fnc(queries_modified[q_index], input_idx, result, score_result, sort_flag)
    return result

def call_method_uniform(fnc, q_index, input_idx, nc, method_param=-1):
    result = np.zeros(nc, dtype=np.int32)
    score_result = np.zeros(nc, dtype=np.float32)
    if method_param == -1:
        fnc(queries[q_index], input_idx, result, score_result, sort_flag)
    else:
        fnc(queries[q_index], input_idx, result, score_result, sort_flag, method_param)
    return result

def call_exact(q_index, input_idx):
    return [test.exact_emd(queries[q_index], input_idx)]

fdic = {}
fdic["mean"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.means_rank, q_index, input_idx, nc)
fdic["overlap"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.overlap_rank, q_index, input_idx, nc)
fdic["quadtree"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.quadtree_rank, q_index,input_idx,  nc)
fdic["flowtree"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.flowtree_rank, q_index, input_idx, nc)
fdic["rwmd"] = lambda q_index, input_idx, nc:call_method_uniform(test.rwmd, q_index, input_idx, nc)
fdic["lcwmd"] = lambda q_index, input_idx, nc:call_method_uniform(test.lc_wmd, q_index, input_idx, nc, method_param)
fdic["sinkhorn"] = lambda q_index, input_idx, nc:call_method_uniform(test.sinkhorn, q_index, input_idx, nc, method_param)
fdic["exact"] = lambda q_index, input_idx, nc:call_exact(q_index, input_idx)

### Main

K_SIZE = int(sys.argv[5])
accs = np.zeros(K_SIZE)
input_idx = None
orig_input_idx = np.zeros(len(test.dataset), dtype=np.int32)

print("qn:", qn)

for i in range(len(test.dataset)):
    orig_input_idx[i] = i

start_time = time.time()
for q in range(qn): # one query
    print(f"query: {q}", end='\r') 
    input_idx = orig_input_idx
    input_idx = fdic[method](q, input_idx, K_SIZE) # calculate top K
    for i, result in enumerate(input_idx):
        if answers[0,q] == result: # check for when correct result appears in 2000
            for j in range(i, K_SIZE):
                accs[j] += 1 # increase accuracy of k-value and all lower indices
            break

print("execution time:", time.time() - start_time)
accs = accs/qn
for i, acc in enumerate(accs):
    if (i % 100 == 0):
        print("", i, " ", acc)

        
accuracyNP = np.array(accs)
np.save(outfile, accuracyNP)
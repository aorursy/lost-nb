#!/usr/bin/env python
# coding: utf-8



# https://www.kaggle.com/vipito/santa-ip
# https://www.kaggle.com/golubev/optimization-preference-cost-mincostflow
# https://www.kaggle.com/nickel/santa-s-2019-fast-pythonic-cost-23-s
# https://www.kaggle.com/inversion/santa-s-2019-starter-notebook




import random
import numpy as np
import pandas as pd

from numba import njit
from ortools.graph.pywrapgraph import SimpleMinCostFlow
from ortools.linear_solver.pywraplp import Solver

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

N_DAYS = 100
N_FAMILIES = 5000
N_CHOICES = 10
MAX_POP = 300
MIN_POP = 125

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

choice_cols = [f'choice_{i}' for i in range(N_CHOICES)]
CHOICES = data[choice_cols].values - 1

COST_PER_FAMILY = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
COST_PER_MEMBER = [0,  0,  9,   9,   9,  18,  18,  36,  36, 235, 434]
F_COUNTS = data['n_people'].astype(int).values

C_COSTS = np.zeros((N_FAMILIES, N_DAYS), dtype=np.int32)
for f in range(N_FAMILIES):
    for d in range(N_DAYS):
        if d in CHOICES[f, :]:
            c = list(CHOICES[f, :]).index(d)
        else:
            c = N_CHOICES
        C_COSTS[f, d] = COST_PER_FAMILY[c] + F_COUNTS[f] * COST_PER_MEMBER[c]




@njit(fastmath=True)
def get_daily_occupancy(schedule):
    daily_occupancy = np.zeros(N_DAYS, np.int32)
    for f, d in enumerate(schedule):
        daily_occupancy[d] += F_COUNTS[f]
    return daily_occupancy


@njit(fastmath=True)
def cost_function(schedule):
    choice_cost = 0
    for f, d in enumerate(schedule):
        choice_cost += C_COSTS[f, d]
    
    daily_occupancy = get_daily_occupancy(schedule)
        
    accounting_cost = 0
    for d0 in range(N_DAYS):
        pop0 = daily_occupancy[d0]
        d1 = min(d0+1, N_DAYS-1)
        pop1 = daily_occupancy[d1]
        accounting_cost += max(0, (pop0-125.0) / 400.0 * pop0**(0.5 + abs(pop0 - pop1) / 50.0))
    
    violations = (np.count_nonzero(daily_occupancy < MIN_POP) + 
                  np.count_nonzero(daily_occupancy > MAX_POP))
    penalty = int(violations * 10e8)
    
    return choice_cost, accounting_cost, penalty


def fix_schedule(schedule):
    daily_occupancy = get_daily_occupancy(schedule)
    
    f_list = np.flip(np.argsort(F_COUNTS))
    
    while (daily_occupancy.min() < MIN_POP) or           (daily_occupancy.max() > MAX_POP):
        
        for c in range(N_CHOICES):
            for f in f_list:
                n = F_COUNTS[f]
                d_old = schedule[f]
                d_new = CHOICES[f, c]

                if (daily_occupancy[d_old] > MAX_POP) and                    ((daily_occupancy[d_new] + n) <= MAX_POP):
                    schedule[f] = d_new
                    daily_occupancy[d_new] += n
                    daily_occupancy[d_old] -= n

        for c in range(N_CHOICES):
            for f in f_list:
                n = F_COUNTS[f]
                d_old = schedule[f]
                d_new = CHOICES[f, c]

                if (daily_occupancy[d_new] < MIN_POP) and                    ((daily_occupancy[d_old] - n) >= MIN_POP):
                    schedule[f] = d_new
                    daily_occupancy[d_new] += n
                    daily_occupancy[d_old] -= n
    
    return schedule




model = Solver('SantaLinear', Solver.GLOP_LINEAR_PROGRAMMING)

set_f = range(N_FAMILIES)
set_d = range(N_DAYS)

x = {(f, d): model.BoolVar(f'x[{f},{d}]') for f in set_f for d in CHOICES[f, :]}
y = {(d): model.IntVar(0, MAX_POP-MIN_POP, f'y[{d}]') for d in set_d}

for f in set_f:
    model.Add(model.Sum(x[f, d] for d in set_d if (f, d) in x.keys()) == 1)

pops = [model.Sum(x[f, d] * F_COUNTS[f] for f in set_f if (f, d) in x.keys()) for d in set_d]

for d0 in set_d:
    pop0 = pops[d0]
    model.Add(pop0 >= MIN_POP)
    model.Add(pop0 <= MAX_POP)

    d1 = min(d0+1, N_DAYS-1)
    pop1 = pops[d1]
    model.Add(pop0 - pop1 <= y[d])
    model.Add(pop1 - pop0 <= y[d])
    
    model.Add(y[d] <= 30)

DELTA_WEIGHT = 500
objective = model.Sum(x[f, d] * C_COSTS[f, d] for f, d in x.keys())
objective += model.Sum(y[d] for d in set_d) * DELTA_WEIGHT

model.Minimize(objective)

model.SetTimeLimit(5 * 60 * 1000)
status = model.Solve()

if status == Solver.OPTIMAL:
    print('Found Optimal Solution')
else:
    print(f'Solver Error. Status = {status}')

schedule = np.full(N_FAMILIES, -1, dtype=np.int8)

x_vals = np.zeros((N_FAMILIES, N_DAYS))
for f, d in x.keys():
    x_vals[f, d] = x[f, d].solution_value()

for f, vals in enumerate(x_vals):
    d = np.argmax(vals)
    schedule[f] = d

score = cost_function(schedule)
print(sum(score), '|', score)

schedule = fix_schedule(schedule)
score = cost_function(schedule)
print(sum(score), '|', score)




def choice_search(schedule):
    best_score = cost_function(schedule)
    
    f_list = np.flip(np.argsort(F_COUNTS))

    for f in f_list:
        d_old = schedule[f]
        for d_new in CHOICES[f, :]:
            schedule[f] = d_new

            score = cost_function(schedule)
                
            if (sum(score) < sum(best_score)) or                (sum(score) == sum(best_score) and np.random.random() < 0.5):
                best_score = score
                d_old = d_new
            else:
                schedule[f] = d_old
    return schedule




def min_cost_flow(schedule):
    MIN_FAMILY = F_COUNTS.min()
    MAX_FAMILY = F_COUNTS.max()
    
    solver = SimpleMinCostFlow()
    
    occupancy = np.zeros((N_DAYS, MAX_FAMILY+1), dtype=np.int32)
    for f, n in enumerate(F_COUNTS):
        f_node = int(f)
        f_demand = -1
        solver.SetNodeSupply(f_node, f_demand)
        
        d = schedule[f]
        occupancy[d, n] += 1
        
    for d in range(N_DAYS):
        for n in range(MIN_FAMILY, MAX_FAMILY):
            occ_node = int(N_FAMILIES + (n-2) * N_DAYS + d)
            occ_supply = int(occupancy[d, n])
            solver.SetNodeSupply(occ_node, occ_supply)

    for f, n in enumerate(F_COUNTS):
        f_node = int(f)
        
        for c in range(N_CHOICES):
            d = CHOICES[f, c]
            c_cost = int(C_COSTS[f, d])
            occ_node = int(N_FAMILIES + (n-2) * N_DAYS + d)
            solver.AddArcWithCapacityAndUnitCost(occ_node, f_node, 1, c_cost)

    status = solver.SolveMaxFlowWithMinCost()

    if status == SimpleMinCostFlow.OPTIMAL:
        for arc in range(solver.NumArcs()):
            if solver.Flow(arc) > 0:
                head = solver.Head(arc)

                if head in range(N_FAMILIES):
                    f = head
                    n = F_COUNTS[f]
                    occ_node = solver.Tail(arc)
                    d = occ_node - N_FAMILIES - (n-2) * N_DAYS
                    schedule[f] = d
    else:
        print(f'Solver Error. Status = {status}')

    return schedule




def swap_search(schedule):
    best_score = cost_function(schedule)
    
    f_list = np.random.permutation(N_FAMILIES)
    
    for f0 in f_list:
        d0 = schedule[f0]
        c0 = list(CHOICES[f0, :]).index(d0)

        swapped = False
        for d1 in CHOICES[f0, 0:c0]:
            f1_set = np.where(schedule == d1)[0]
            for f1 in f1_set:
                if d0 in CHOICES[f1, :]:
                    schedule[f0] = d1
                    schedule[f1] = d0
                    score = cost_function(schedule)
                    
                    if (sum(score) < sum(best_score)) or                        (sum(score) == sum(best_score) and np.random.random() < 0.5):
                        best_score = score
                        swapped = True
                        break
                    else:
                        schedule[f0] = d0
                        schedule[f1] = d1
            if swapped: break
                
    return schedule




def random_climb(schedule, repeats=100000):
    best_score = cost_function(schedule)

    for _ in range(repeats):
        f = np.random.randint(N_FAMILIES)
        c = np.random.randint(N_CHOICES)

        d_old = schedule[f]
        d_new = CHOICES[f, c]

        schedule[f] = d_new
        score = cost_function(schedule)

        if (sum(score) < sum(best_score)) or            (sum(score) == sum(best_score) and np.random.random() < 0.5):
            best_score = score
        else:
            schedule[f] = d_old

    return schedule




best_score = cost_function(schedule)

no_improvement = 0
while no_improvement < 5:
    improved = False
    
    while True:
        schedule = random_climb(schedule)
        score = cost_function(schedule)
        print('Random  :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
        else:
            break
    
    while True:
        schedule = swap_search(schedule)
        score = cost_function(schedule)
        print('Swaps   :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
        else:
            break
    
    while True:
        schedule = choice_search(schedule)
        score = cost_function(schedule)
        print('Choice  :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
        else:
            break
    
    if not improved:
        schedule = min_cost_flow(schedule)
        score = cost_function(schedule)
        print('MinCost :', sum(score), '|', score)
        if sum(score) < sum(best_score):
            best_score = score
            improved = True
    
    no_improvement = 0 if improved else no_improvement + 1




submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')
submission['assigned_day'] = schedule + 1
submission.to_csv('submission.csv', index=False)









#https://www.kaggle.com/golubev/c-stochastic-product-search-65ns
#https://www.kaggle.com/hengzheng/santa-s-seed-seeker




get_ipython().run_cell_magic('writefile', 'main.cpp', '#include <array>\n#include <cassert>\n#include <algorithm>\n#include <cmath>\n#include <fstream>\n#include <iostream>\n#include <vector>\n#include <thread>\n#include <atomic>\n#include <random>\n#include <string.h>\nusing namespace std;\n#include <chrono>\nusing namespace std::chrono;\n\nconstexpr array<uint8_t, 21> DISTRIBUTION{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5};\n// {2, 5} it\'s mean the first random family will brute force for choices 1-2 and the second random family will brute force for choices 1-5\n\nconstexpr int MAX_OCCUPANCY = 300;\nconstexpr int MIN_OCCUPANCY = 125;\nconstexpr int BEST_N = 10;\narray<uint8_t, 5000> n_people;\narray<array<uint8_t, 10>, 5000> choices;\narray<array<uint16_t, 10>, 5000> PCOSTM;\narray<array<double, 176>, 176> ACOSTM;\n\nstatic std::atomic<bool> flag(false);\nstatic array<uint8_t, 5000> global_assigned_days = {};\n\nauto START_TIME = high_resolution_clock::now();\nint END_TIME = 515; //475\n\nint N_JOBS = 4;\n\nstruct Index {\n    Index(array<uint8_t, 5000> assigned_days_) : assigned_days(assigned_days_)  {\n        setup();\n    }\n    array<uint8_t, 5000> assigned_days;\n    array<uint16_t, 100> daily_occupancy_{};\n    int preference_cost_ = 0;\n    void setup() {\n        preference_cost_ = 0;\n        daily_occupancy_.fill(0);\n        for (int j = 0; j < assigned_days.size(); ++j) {\n            daily_occupancy_[choices[j][assigned_days[j]]] += n_people[j];\n            preference_cost_ += PCOSTM[j][assigned_days[j]];\n        }\n    }\n    double calc(const array<uint16_t, 5000>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {\n        double accounting_penalty = 0.0;\n        auto daily_occupancy = daily_occupancy_;\n        int preference_cost = preference_cost_;\n        for (int i = 0; i < DISTRIBUTION.size(); ++i) {\n            int j = indices[i];\n            daily_occupancy[choices[j][assigned_days[j]]] -= n_people[j];\n            daily_occupancy[choices[j][       change[i]]] += n_people[j];\n\n            preference_cost += PCOSTM[j][change[i]] - PCOSTM[j][assigned_days[j]];\n        }\n\n        for (auto occupancy : daily_occupancy)\n            if (occupancy < MIN_OCCUPANCY)\n                return 1e12 * (MIN_OCCUPANCY - occupancy);\n            else if (occupancy > MAX_OCCUPANCY)\n                return 1e12 * (occupancy - MAX_OCCUPANCY);\n\n        for (int day = 0; day < 99; ++day)\n            accounting_penalty += ACOSTM[daily_occupancy[day] - 125][daily_occupancy[day + 1] - 125];\n\n        accounting_penalty += ACOSTM[daily_occupancy[99] - 125][daily_occupancy[99] - 125];\n        return preference_cost + accounting_penalty;\n    }\n    void reindex(const array<uint16_t, DISTRIBUTION.size()>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {\n        for (int i = 0; i < DISTRIBUTION.size(); ++i) {\n            assigned_days[indices[i]] = change[i];\n        }\n        setup();\n    }\n};\n\n\nvoid init_data() {\n    ifstream in("/kaggle/input/santa-workshop-tour-2019/family_data.csv");\n\n    assert(in && "family_data.csv");\n    string header;\n    int n, x;\n    char comma;\n    getline(in, header);\n    for (int j = 0; j < choices.size(); ++j) {\n        in >> x >> comma;\n        for (int i = 0; i < 10; ++i) {\n            in >> x >> comma;\n            choices[j][i] = x - 1;\n        }\n        in >> n;\n        n_people[j] = n;\n    }\n    array<int, 10> pc{0, 50, 50, 100, 200, 200, 300, 300, 400, 500};\n    array<int, 10> pn{0,  0,  9,   9,   9,  18,  18,  36,  36, 235};\n    for (int j = 0; j < PCOSTM.size(); ++j)\n        for (int i = 0; i < 10; ++i)\n            PCOSTM[j][i] = pc[i] + pn[i] * n_people[j];\n\n    for (int i = 0; i < 176; ++i)\n        for (int j = 0; j < 176; ++j)\n            ACOSTM[i][j] = i * pow(i + 125, 0.5 + abs(i - j) / 50.0) / 400.0;\n}\narray<uint8_t, 5000> read_submission(string filename) {\n    ifstream in(filename);\n    assert(in && "submission.csv");\n    array<uint8_t, 5000> assigned_day{};\n    string header;\n    int id, x;\n    char comma;\n    getline(in, header);\n    for (int j = 0; j < choices.size(); ++j) {\n        in >> id >> comma >> x;\n        assigned_day[j] = x - 1;\n        auto it = find(begin(choices[j]), end(choices[j]), assigned_day[j]);\n        if (it != end(choices[j]))\n            assigned_day[j] = distance(begin(choices[j]), it);\n    }\n    return assigned_day;\n}\n\n\ndouble calc(const array<uint8_t, 5000>& assigned_days, bool print = false) {\n    int preference_cost = 0;\n    double accounting_penalty = 0.0;\n    array<uint16_t, 100> daily_occupancy{};\n    for (int j = 0; j < assigned_days.size(); ++j) {\n        preference_cost += PCOSTM[j][assigned_days[j]];\n        daily_occupancy[choices[j][assigned_days[j]]] += n_people[j];\n    }\n    for (auto occupancy : daily_occupancy)\n        if (occupancy < MIN_OCCUPANCY)\n            return 1e12 * (MIN_OCCUPANCY - occupancy);\n        else if (occupancy > MAX_OCCUPANCY)\n            return 1e12 * (occupancy - MAX_OCCUPANCY);\n\n    for (int day = 0; day < 99; ++day)\n        accounting_penalty += ACOSTM[daily_occupancy[day] - 125][daily_occupancy[day + 1] - 125];\n\n    accounting_penalty += ACOSTM[daily_occupancy[99] - 125][daily_occupancy[99] - 125];\n    if (print) {\n        cout << preference_cost << " " << accounting_penalty << " " << preference_cost + accounting_penalty << endl;\n    }\n    return preference_cost + accounting_penalty;\n}\n\nbool time_exit_fn(){\n    return duration_cast<minutes>(high_resolution_clock::now()-START_TIME).count() < END_TIME;\n}\n\nconst vector<array<uint8_t, DISTRIBUTION.size()>> changes = []() {\n    vector<array<uint8_t, DISTRIBUTION.size()>> arr;\n    array<uint8_t, DISTRIBUTION.size()> tmp{};\n    \n    for (int i = 0; true; ++i) {\n        arr.push_back(tmp);\n\n        tmp[0] += 1;\n        for (int j = 0; j < DISTRIBUTION.size(); ++j)\n            if (tmp[j] >= DISTRIBUTION[j]) {\n                if (j >= DISTRIBUTION.size() - 1)\n                    return arr;\n                tmp[j] = 0;\n                ++tmp[j + 1];\n            }\n    }\n    return arr;\n}();\n\n\nvoid stochastic_product_search(array<uint8_t, 5000> assigned_days, double best_local_score) { // 15\'360\'000it/s  65ns/it  0.065µs/it\n    Index index(assigned_days);\n    thread_local std::mt19937 gen(std::random_device{}());\n    uniform_int_distribution<> dis(0, 4999);\n    array<uint16_t, 5000> indices;\n    iota(begin(indices), end(indices), 0);\n    array<uint16_t, DISTRIBUTION.size()> best_indices{};\n    array<uint8_t, DISTRIBUTION.size()> best_change{};\n    for (;time_exit_fn();) {\n        bool found_better = false;\n\n        for (int k = 0; k < BEST_N; ++k) {\n            for (int i = 0; i < DISTRIBUTION.size(); ++i) //random swap\n                swap(indices[i], indices[dis(gen)]);\n\n            for (const auto& change : changes) {\n                auto score = index.calc(indices, change);\n                if (score < best_local_score) {\n                    found_better = true;\n                    best_local_score = score;\n                    best_change = change;\n                    copy_n(begin(indices), DISTRIBUTION.size(), begin(best_indices));\n                }\n            }\n        }\n\n        if (flag.load() == true) {\n            return;\n        }\n\n        if (found_better && flag.load() == false) { // reindex from N best if found better\n            flag = true;\n\n            index.reindex(best_indices, best_change);\n            global_assigned_days = index.assigned_days;\n            return;\n        }\n    }\n}\n\n\n\narray<uint16_t, 5000> sort_indexes(const array<uint8_t, 5000> v) {\n\n    // initialize original index locations\n    array<uint16_t, 5000> idx;\n    iota(idx.begin(), idx.end(), 0);\n\n    // sort indexes based on comparing values in v\n    sort(idx.begin(), idx.end(),\n    [v](size_t i1, size_t i2) {return v[i1] > v[i2];});\n\n    return idx;\n}\n\n\nvoid seed_finding(array<uint8_t, 5000> assigned_days, double best_local_score, int order) { // 15\'360\'000it/s  65ns/it  0.065µs/it\n    thread_local std::mt19937 gen(std::random_device{}());\n    uniform_real_distribution<> dis(0.0, 1.0);\n\n    auto original_score = best_local_score;\n\n    auto indices = sort_indexes(n_people); // sort by descending\n\n    if (order == 0) { // sort by ascending\n        reverse(begin(indices), end(indices));\n    }\n\n    for (;time_exit_fn();) {\n        auto local_assigned_days = assigned_days;\n\n        if (order == 1) { // sort by random\n            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();\n            shuffle (indices.begin(), indices.end(), std::default_random_engine(seed));\n        }\n\n        for (int t = 0; t < 700; t++) {\n            for (auto& i : indices) {\n                for (int j = 0; j < 10; j++) {\n                    auto di = local_assigned_days[i];\n                    local_assigned_days[i] = j;\n                    auto cur_score = calc(local_assigned_days, false);\n\n                    double KT = 1;\n                    if (t < 5) {\n                        KT = 1.5;\n                    }\n                    else if ( t < 10) {\n                        KT = 4.5;\n                    }\n                    else {\n                        if (cur_score > best_local_score + 100) {\n                            KT = 3;\n                        }\n                        else if (cur_score > best_local_score + 50) {\n                            KT = 2.75;\n                        }\n                        else if (cur_score > best_local_score + 20) {\n                            KT = 2.5;\n                        }\n                        else if (cur_score > best_local_score + 10) {\n                            KT = 2;\n                        }\n                        else if (cur_score > best_local_score) {\n                            KT = 1.5;\n                        }\n                        else {\n                            KT = 1;\n                        }\n                    }\n\n                    if (cur_score <= best_local_score) {\n                        best_local_score = cur_score;\n                    }\n                    else {\n                        auto prob = exp(-(cur_score - best_local_score) / KT);\n                        if (dis(gen) < prob) {\n                            best_local_score = cur_score;\n                        }\n                        else {\n                            local_assigned_days[i] = di;\n                        }\n                    }\n                }\n            }\n\n            if (flag.load() == true) {\n                return;\n            }\n\n            if (best_local_score < original_score && flag.load() == false) {\n                flag = true;\n\n                global_assigned_days = local_assigned_days;\n\n                return;\n\n            }\n        }\n\n        if (best_local_score <= original_score && flag.load() == false) {\n            flag = true;\n\n            global_assigned_days = local_assigned_days;\n\n            return;\n        }\n    }\n}\n\nvoid save_sub(const array<uint8_t, 5000>& assigned_day) {\n    ofstream out("submission.csv");\n    out << "family_id,assigned_day" << endl;\n    for (int i = 0; i < assigned_day.size(); ++i)\n        out << i << "," << choices[i][assigned_day[i]] + 1 << endl;\n}\n\n\nint main() {\n    init_data();\n    auto assigned_days = read_submission("submission.csv");\n\n    double best_score = calc(assigned_days, true);\n\n    for (;time_exit_fn();) {\n\n        std::thread threads[N_JOBS];\n        for (int i = 0; i < N_JOBS; i++) {\n            //threads[i] = std::thread(stochastic_product_search, assigned_days, best_score);\n\n            if (i < 2) {\n                threads[i] = std::thread(stochastic_product_search, assigned_days, best_score);\n            }\n            else {\n                int order = i % 3;\n                threads[i] = std::thread(seed_finding, assigned_days, best_score, order);\n            }\n\n            //int order = i % 3;\n            //threads[i] = std::thread(seed_finding, assigned_days, best_score, order);\n        }\n        for (int i = 0; i < N_JOBS; i++) {\n            threads[i].join();\n        }\n\n        // global_assigned_days return from threads\n        best_score = calc(global_assigned_days, true);\n        save_sub(global_assigned_days);\n\n        flag = false;\n        assigned_days = global_assigned_days;\n    }\n\n\n    return 0;\n}')




get_ipython().system('g++ -pthread -lpthread -O3 -std=c++17 -o main main.cpp')




get_ipython().system('./main')




#https://www.kaggle.com/golubev/mip-optimization-preference-cost




# import numpy as np
# import pandas as pd
# from collections import defaultdict
# NUMBER_DAYS = 100
# NUMBER_FAMILIES = 5000
# MAX_BEST_CHOICE = 5
# data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
# submission = pd.read_csv('submission.csv')
# assigned_days = submission['assigned_day'].values
# columns = data.columns[1:11]
# DESIRED = data[columns].values

# COST_PER_FAMILY        = [0,50,50,100,200,200,300,300,400,500]
# COST_PER_FAMILY_MEMBER = [0, 0, 9,  9,  9, 18, 18, 36, 36,235]
# N_PEOPLE = data['n_people'].astype(int).values

# def get_daily_occupancy(assigned_days):
#     daily_occupancy = np.zeros(100, np.int32)
#     for i, r in enumerate(assigned_days):
#         daily_occupancy[r-1] += N_PEOPLE[i]
#     return daily_occupancy

# def cost_function(prediction):
#     N_DAYS = 100
#     MAX_OCCUPANCY = 300
#     MIN_OCCUPANCY = 125
#     penalty = 0
#     days = list(range(N_DAYS,0,-1))
#     tmp = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
#     family_size_dict = tmp[['n_people']].to_dict()['n_people']

#     cols = [f'choice_{i}' for i in range(10)]
#     choice_dict = tmp[cols].to_dict()

#     # We'll use this to count the number of people scheduled each day
#     daily_occupancy = {k:0 for k in days}
    
#     # Looping over each family; d is the day for each family f
#     for f, d in enumerate(prediction):
#         # Using our lookup dictionaries to make simpler variable names
#         n = family_size_dict[f]
#         choice_0 = choice_dict['choice_0'][f]
#         choice_1 = choice_dict['choice_1'][f]
#         choice_2 = choice_dict['choice_2'][f]
#         choice_3 = choice_dict['choice_3'][f]
#         choice_4 = choice_dict['choice_4'][f]
#         choice_5 = choice_dict['choice_5'][f]
#         choice_6 = choice_dict['choice_6'][f]
#         choice_7 = choice_dict['choice_7'][f]
#         choice_8 = choice_dict['choice_8'][f]
#         choice_9 = choice_dict['choice_9'][f]

#         # add the family member count to the daily occupancy
#         daily_occupancy[d] += n

#         # Calculate the penalty for not getting top preference
#         if d == choice_0:
#             penalty += 0
#         elif d == choice_1:
#             penalty += 50
#         elif d == choice_2:
#             penalty += 50 + 9 * n
#         elif d == choice_3:
#             penalty += 100 + 9 * n
#         elif d == choice_4:
#             penalty += 200 + 9 * n
#         elif d == choice_5:
#             penalty += 200 + 18 * n
#         elif d == choice_6:
#             penalty += 300 + 18 * n
#         elif d == choice_7:
#             penalty += 300 + 36 * n
#         elif d == choice_8:
#             penalty += 400 + 36 * n
#         elif d == choice_9:
#             penalty += 500 + 36 * n + 199 * n
#         else:
#             penalty += 500 + 36 * n + 398 * n

#     # for each date, check total occupancy
#     #  (using soft constraints instead of hard constraints)
#     for _, v in daily_occupancy.items():
#         if v > MAX_OCCUPANCY or v < MIN_OCCUPANCY:
#             penalty += 100000000

#     # Calculate the accounting cost
#     # The first day (day 100) is treated special
#     # using the max function because the soft constraints might allow occupancy to dip below 125
#     accounting_cost = max(0, (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5))
#     # Loop over the rest of the days, keeping track of previous count
#     yesterday_count = daily_occupancy[days[0]]
#     for day in days[1:]:
#         today_count = daily_occupancy[day]
#         diff = abs(today_count - yesterday_count)
#         accounting_cost += max(0, (today_count-125.0) / 400.0 * today_count**(0.5 + diff / 50.0))
#         yesterday_count = today_count

#     return penalty, accounting_cost, penalty + accounting_cost




# %%time
# from ortools.graph import pywrapgraph
# for num_members in range(2, 9): # Families have minimum 2 and maximum 8 members
#     daily_occupancy = get_daily_occupancy(assigned_days)
#     fids = np.where(N_PEOPLE == num_members)[0]

#     PCOSTM = {}
#     for fid in range(NUMBER_FAMILIES):
#         if fid in fids:
#             for i in range(MAX_BEST_CHOICE):
#                 PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]
#         else:
#             daily_occupancy[assigned_days[fid]-1] -= N_PEOPLE[fid]

#     offset = fids.shape[0]
#     solver = pywrapgraph.SimpleMinCostFlow()
#     for day in range(NUMBER_DAYS):
#         solver.SetNodeSupply(offset+day, int(daily_occupancy[day]//num_members))

#     for i in range(offset):
#         fid = fids[i]
#         solver.SetNodeSupply(i, -1)
#         for j in range(MAX_BEST_CHOICE):
#             day = DESIRED[fid][j]-1
#             solver.AddArcWithCapacityAndUnitCost(int(offset+day), i, 1, int(PCOSTM[fid, day]))
#     solver.SolveMaxFlowWithMinCost()

#     for i in range(solver.NumArcs()):
#         if solver.Flow(i) > 0:
#             assigned_days[fids[solver.Head(i)]] = solver.Tail(i) - offset + 1
#     print(cost_function(assigned_days))




# submission['assigned_day'] = assigned_days
# submission.to_csv('submission.csv', index=False)




from ortools.linear_solver import pywraplp


NUMBER_DAYS = 100
NUMBER_FAMILIES = 5000
data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
submission = pd.read_csv('submission.csv')
assigned_days = submission['assigned_day'].values
columns = data.columns[1:11]
DESIRED = data[columns].values
COST_PER_FAMILY        = [0,50,50,100,200,200,300,300,400,500]
COST_PER_FAMILY_MEMBER = [0, 0, 9,  9,  9, 18, 18, 36, 36,235]
N_PEOPLE = data['n_people'].values

def get_daily_occupancy(assigned_days):
    daily_occupancy = np.zeros(100, int)
    for fid, assigned_day in enumerate(assigned_days):
        daily_occupancy[assigned_day-1] += N_PEOPLE[fid]
    return daily_occupancy
    
def cost_function(prediction):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    days = list(range(N_DAYS,0,-1))
    tmp = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
    family_size_dict = tmp[['n_people']].to_dict()['n_people']

    cols = [f'choice_{i}' for i in range(10)]
    choice_dict = tmp[cols].to_dict()

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}
    
    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):
        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        choice_0 = choice_dict['choice_0'][f]
        choice_1 = choice_dict['choice_1'][f]
        choice_2 = choice_dict['choice_2'][f]
        choice_3 = choice_dict['choice_3'][f]
        choice_4 = choice_dict['choice_4'][f]
        choice_5 = choice_dict['choice_5'][f]
        choice_6 = choice_dict['choice_6'][f]
        choice_7 = choice_dict['choice_7'][f]
        choice_8 = choice_dict['choice_8'][f]
        choice_9 = choice_dict['choice_9'][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if  (v < MIN_OCCUPANCY): #(v > MAX_OCCUPANCY) or
            penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_costs = [max(0, accounting_cost)]
    diffs = [0]
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_costs.append(max(0, (today_count-125.0) / 400.0 * today_count**(0.5 + diff / 50.0)))
        yesterday_count = today_count

    return penalty, sum(accounting_costs), penalty + sum(accounting_costs)




from ortools.linear_solver import pywraplp
MAX_BEST_CHOICE = 5
NUM_SWAP = 2500
NUM_SECONDS = 1800
NUM_THREADS = 4
for _ in range(10):
    solver = pywraplp.Solver('Optimization preference cost', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    daily_occupancy = get_daily_occupancy(assigned_days).astype(float)
    fids = np.random.choice(range(NUMBER_FAMILIES), NUM_SWAP, replace=False)
    PCOSTM, B = {}, {}
    for fid in range(NUMBER_FAMILIES):
        if fid in fids:
            for i in range(MAX_BEST_CHOICE):
                PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]
                B[     fid, DESIRED[fid][i]-1] = solver.BoolVar('')
        else:
            daily_occupancy[assigned_days[fid]-1] -= N_PEOPLE[fid]

    solver.set_time_limit(NUM_SECONDS*NUM_THREADS*1000)
    solver.SetNumThreads(NUM_THREADS)

    for day in range(NUMBER_DAYS):
        if daily_occupancy[day]:
            solver.Add(solver.Sum([N_PEOPLE[fid] * B[fid, day] for fid in range(NUMBER_FAMILIES) if (fid,day) in B]) == daily_occupancy[day])
        
    for fid in fids:
        solver.Add(solver.Sum(B[fid, day] for day in range(NUMBER_DAYS) if (fid, day) in B) == 1)

    solver.Minimize(solver.Sum(PCOSTM[fid, day] * B[fid, day] for fid, day in B))
    sol = solver.Solve()
    
    status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']
    if status[sol] in ['OPTIMAL', 'FEASIBLE']:
        tmp = assigned_days.copy()
        for fid, day in B:
            if B[fid, day].solution_value() > 0.5:
                tmp[fid] = day+1
        if cost_function(tmp)[2] < cost_function(assigned_days)[2]:
            assigned_days = tmp
            submission['assigned_day'] = assigned_days
            submission.to_csv('submission.csv', index=False)
        print('Result:', status[sol], cost_function(tmp))
    else:
        print('Result:', status[sol])


#!/usr/bin/env python
# coding: utf-8



from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

NUMBER_DAYS = 100
NUMBER_FAMILIES = 5000
data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
submission = pd.read_csv('/kaggle/input/best-submission-31-12a/submission.csv')
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

def days_plot(assigned_days):
    daily_occupancy = get_daily_occupancy(assigned_days)
    best_choices = get_daily_occupancy(DESIRED[:,0])
    plt.rcParams['figure.figsize'] = [20, 5]
    plt.xticks(np.arange(1, 101, step=1), rotation=90)
    plt.axhline(y=125, color='gray', linestyle=':')
    plt.axhline(y=300, color='gray', linestyle=':')
    mondays125     = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 == 1 and daily_occupancy[day] == 125])
    other_mondays  = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 == 1 and daily_occupancy[day] != 125])
    weekends       = np.array([(day+1, daily_occupancy[day]) for day in range(100) if day % 7 in [2,3,4] or day == 0])
    not_weekends   = np.array([(day+1, daily_occupancy[day]) for day in range(1, 100) if day % 7 in [0,5,6]])
    plt.bar(*weekends.transpose()      , color = 'y', label = 'Weekends')
    plt.bar(*not_weekends.transpose()  , color = 'b', label = 'Thu-Wed-Tue')
    plt.bar(*other_mondays.transpose() , color = 'm', label = 'Mondays > 125')
    plt.bar(*mondays125.transpose()    , color = 'g', label = 'Mondays = 125')
    plt.plot(range(1,101), best_choices, color = 'k', label = 'Best choices')
    plt.ylim(0, 500)
    plt.xlim(0, 101)
    plt.xlabel('Days before Christmas', fontsize=14)
    plt.ylabel('Occupancy', fontsize=14)
    plt.legend()
    plt.show()
    
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
days_plot(assigned_days)
print("Score: ", cost_function(assigned_days))




print("Day allocations:")
print(np.arange(1, 100)[np.abs(np.diff(get_daily_occupancy(assigned_days)) * -1) > 25])




days_for_fix = np.array([37])
daily_occupancy = get_daily_occupancy(assigned_days)
fids = np.where(np.isin(assigned_days, days_for_fix))[0] # Ids of family for move
MAX_BEST_CHOICE = 3

solver = pywraplp.Solver('Setup occupation of days', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
PCOSTM, B = {}, {}
for fid in fids:
    for i in range(MAX_BEST_CHOICE):
        B[fid, DESIRED[fid][i]-1] = solver.BoolVar(f'b{fid, i}')
        PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]

lower_bounds = np.zeros(100)
upper_bounds = 300. - daily_occupancy
upper_bounds[np.arange(100)%7 == 1] = 0 # don't move to Mondays

# Daily occupation for special Mondays only 125
lower_bounds[days_for_fix-1] = 125
upper_bounds[days_for_fix-1] = 125

for j in range(NUMBER_DAYS):
    I = solver.IntVar(lower_bounds[j], upper_bounds[j], f'I{j}')
    solver.Add(solver.Sum([N_PEOPLE[i] * B[i, j] for i in range(NUMBER_FAMILIES) if (i,j) in B]) == I)
    
for i in fids:
    solver.Add(solver.Sum(B[i, j] for j in range(NUMBER_DAYS) if (i,j) in B) == 1)

solver.Minimize(solver.Sum(PCOSTM[i, j] * B[i, j] for i, j in B))
sol = solver.Solve()

status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']
if status[sol] == 'OPTIMAL':
    for i, j in B:
        if B[i, j].solution_value() > 0.5:
            assigned_days[i] = j+1
            
print('Solution: ', status[sol])
print("Score: ", cost_function(assigned_days))
days_plot(assigned_days)
submission['assigned_day'] = assigned_days
submission.to_csv('submission.csv', index=False)




def pytorch_optimim(MAX_CHOICE = 6):
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from pathlib import Path
    import tqdm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from numba import njit
    root_path = Path(r'../input/santa-workshop-tour-2019')
    best_submit_path = Path(r'../input/mip-optimization-preference-cost/')
    
    fpath = root_path / 'family_data.csv'
    data = pd.read_csv(fpath, index_col='family_id')
    data_choices = data.values

    fpath = root_path / 'sample_submission.csv'
    submission = pd.read_csv(fpath, index_col='family_id')
    
    dummies = []
    for i in range(MAX_CHOICE):
        tmp = pd.get_dummies(data[f'choice_{i}']).values*(data['n_people'].values.reshape(-1,1))
        dummies.append((
            np.concatenate([tmp, tmp[:, -1].reshape(-1,1)], axis=1)
                       ).reshape(5000, 101, 1))
    dummies = np.concatenate(dummies, axis=2)
    dummies = np.swapaxes(dummies, 1, 2)

    penalties = {n: [0, 50, 50 + 9 * n, 100 + 9 * n, 200 + 9 * n, 200 + 18 * n, 300 + 18 * n, 300 + 36 * n, 400 + 36 * n, 500 + 36 * n + 199 * n] for n in np.unique(data['n_people'])}

    mat = []
    for i in range(5000):
        n = data.iloc[i]['n_people']
        mat.append(penalties[n][:MAX_CHOICE])
    mat = np.array(mat)

    def create_init(initial_sub):

        fam_choices = data
        a = pd.merge(initial_sub, fam_choices, on='family_id')

        initial_choices = []
        for i in range(MAX_CHOICE):
            initial_choices.append(((a[f'choice_{i}'] == a['assigned_day'])).values.reshape(-1,1))
        initial_choices = np.concatenate(initial_choices, axis=1)
        initial_choices = torch.tensor(
           initial_choices*10
            , dtype=torch.float32)#.cuda()
        return initial_choices

    initial_sub = pd.read_csv('./submission.csv')
    initial_choices = create_init(initial_sub)

    family_sizes = data.n_people.values.astype(np.int8)
    cost_dict = {0:  [  0,  0],
                 1:  [ 50,  0],
                 2:  [ 50,  9],
                 3:  [100,  9],
                 4:  [200,  9],
                 5:  [200, 18],
                 6:  [300, 18],
                 7:  [300, 36],
                 8:  [400, 36],
                 9:  [500, 36 + 199],
                 10: [500, 36 + 398],
                }

    def cost(choice, members, cost_dict):
        x = cost_dict[choice]
        return x[0] + members * x[1]
    all_costs = {k: pd.Series([cost(k, x, cost_dict) for x in range(2,9)], index=range(2,9)) for k in cost_dict.keys()}
    df_all_costs = pd.DataFrame(all_costs)

    family_cost_matrix = np.zeros((100,len(family_sizes))) # Cost for each family for each day.

    for i, el in enumerate(family_sizes):
        family_cost_matrix[:, i] += all_costs[10][el] # populate each day with the max cost
        for j, choice in enumerate(data.drop("n_people",axis=1).values[i,:]):
            family_cost_matrix[choice-1, i] = all_costs[j][el]

    def accounting(today, previous):
        return ((today - 125) / 400 ) * today ** (.5 + (abs(today - previous) / 50))

    acc_costs = np.zeros([176,176])

    for i, x in enumerate(range(125,300+1)):
        for j, y in enumerate(range(125,300+1)):
            acc_costs[i,j] = accounting(x,y)

    @njit(fastmath=True)
    def cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix):
        N_DAYS = 100
        MAX_OCCUPANCY = 300
        MIN_OCCUPANCY = 125
        penalty = 0
        accounting_cost = 0
        max_occ = False

        daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int16)
        for i, (pred, n) in enumerate(zip(prediction, family_size)):
            daily_occupancy[pred - 1] += n
            penalty += family_cost_matrix[pred - 1, i]

        daily_occupancy[-1] = daily_occupancy[-2]
        for day in range(N_DAYS):
            n_next = daily_occupancy[day + 1]
            n = daily_occupancy[day]
            max_occ += MIN_OCCUPANCY > n
            max_occ += MAX_OCCUPANCY < n
            accounting_cost += accounting_cost_matrix[n-MIN_OCCUPANCY, n_next-MIN_OCCUPANCY]
        if max_occ: 
            return 1e11
        return penalty+accounting_cost

    print(cost_function(initial_sub['assigned_day'].values, family_sizes, family_cost_matrix, acc_costs))
    
    class Model(nn.Module):
        def __init__(self, mat, dummies):
            super().__init__()
            self.mat = torch.from_numpy(mat).type(torch.int16)#.cuda()
            self.dummies = torch.from_numpy(dummies).type(torch.float32)#.cuda()
            self.weight = torch.nn.Parameter(data=torch.Tensor(5000, MAX_CHOICE).type(torch.float32)#.cuda()
                                             , requires_grad=True)
            self.weight.data.copy_(initial_choices)

        def forward(self):
            prob = F.softmax(self.weight,dim=1)

            x = (prob * self.mat).sum()

            daily_occupancy = torch.zeros(101, dtype=torch.float32)#.cuda()
            for i in range(MAX_CHOICE):
                daily_occupancy += (prob[:, i]@self.dummies[:, i, :])

            diff = torch.abs(daily_occupancy[:-1] - daily_occupancy[1:])
            daily_occupancy = daily_occupancy[:-1]
            y = (
                torch.relu(daily_occupancy-125.0) / 400.0 * daily_occupancy**(0.5 + diff / 50.0)
            ).sum() 

            v = ((torch.relu(125-daily_occupancy))**2+(torch.relu(daily_occupancy-300))**2).sum()

            entropy_loss = -1.0 * (prob * F.log_softmax(self.weight, dim=1)).sum()
            return  x, y, v*10000, entropy_loss
        
    model = Model(mat, dummies)
    best_score = 10e10
    best_pos = None
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
    
    for epoch in tqdm.tqdm_notebook(range(100_001)):
        optimizer.zero_grad()
        x, y, v, ent = model()
        loss = x + y + v + 0*ent
        loss.backward()
        optimizer.step()

        pos = model.weight.argmax(1).cpu().numpy()
        pred = []
        for i in range(5000):
            pred.append(data_choices[i, pos[i]])
        pred = np.array(pred)
        score = cost_function(pred, family_sizes, family_cost_matrix, acc_costs)
        if (score < best_score):
            best_score = score
            best_pos = pred
            print(best_score)
            submission['assigned_day'] = best_pos
            submission.to_csv(f'submission.csv')
        if epoch % 1000 == 0:
                x = np.round(x.item(),1)
                y = np.round(y.item(),1)
                print(f'{epoch}\t{x}\t{y}    \t{np.round(score, 2)}')
                
    prev_best_score = best_score
    coef = 1
    count_failures = 0
    
    for _ in range(10_000):
        initial_sub = pd.read_csv('submission.csv')
        initial_choices = create_init(initial_sub)

        model = Model(mat, dummies)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

        for epoch in tqdm.tqdm_notebook(range(1_001)):
            optimizer.zero_grad()
            x, y, v, ent = model()
            loss = x + coef*y + v + 0*ent
            loss.backward()
            optimizer.step()

            pos = model.weight.argmax(1).cpu().numpy()
            pred = []
            for i in range(5000):
                pred.append(data_choices[i, pos[i]])
            pred = np.array(pred)
            score = cost_function(pred, family_sizes, family_cost_matrix, acc_costs)
            if (score < best_score):
                best_score = score
                best_pos = pred
                print(best_score)
                submission['assigned_day'] = best_pos
                submission.to_csv(f'submission.csv')
            if (epoch % 1000 == 0) and epoch != 0:
                    x = np.round(x.item(),1)
                    y = np.round(y.item(),1)
                    print(f'{epoch}\t{x}\t{y}    \t{np.round(score, 2)}')
        if best_score == prev_best_score:
            count_failures += 1
            if count_failures > 30:
                break
            coef = coef*1.01
    #         break
        else:
            prev_best_score = best_score
            count_failures = 0
            coef = 1
            
    prev_best_score = best_score
    coef = 1
    count_failures = 0
    
    for _ in range(10_000):

        initial_sub = pd.read_csv('submission.csv')
        initial_choices = create_init(initial_sub)

        model = Model(mat, dummies)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

        for epoch in tqdm.tqdm_notebook(range(1_001)):
            optimizer.zero_grad()
            x, y, v, ent = model()
            loss = coef*x + y + v + 10*ent
            loss.backward()
            optimizer.step()

            pos = model.weight.argmax(1).cpu().numpy()
            pred = []
            for i in range(5000):
                pred.append(data_choices[i, pos[i]])
            pred = np.array(pred)
            score = cost_function(pred, family_sizes, family_cost_matrix, acc_costs)
            if (score < best_score):
                best_score = score
                best_pos = pred
                print(best_score)
                submission['assigned_day'] = best_pos
                submission.to_csv(f'submission.csv')
            if (epoch % 1000 == 0) and epoch != 0:
                    x = np.round(x.item(),1)
                    y = np.round(y.item(),1)
                    print(f'{epoch}\t{x}\t{y}    \t{np.round(score, 2)}')
        if best_score == prev_best_score:
            count_failures += 1
            if count_failures > 20:
                break
            coef = coef*1.05
    #         break
        else:
            prev_best_score = best_score
            count_failures = 0
            coef = 1
            
    submission['assigned_day'] = best_pos
    submission.to_csv(f'submission.csv')




def optimization_preference(submission, MAX_BEST_CHOICE=7, NUM_SWAP=3000, NUM_SECONDS=2000,NUM_THREADS=4):
    from ortools.linear_solver import pywraplp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    NUMBER_DAYS = 100
    NUMBER_FAMILIES = 5000
    data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
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
    for _ in range(20):
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




def mincostflow(submission, MAX_BEST_CHOICE=4):
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from ortools.graph import pywrapgraph

    NUMBER_DAYS = 100
    NUMBER_FAMILIES = 5000

    data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
    assigned_days = submission['assigned_day'].values
    columns = data.columns[1:11]
    DESIRED = data[columns].values

    COST_PER_FAMILY        = [0,50,50,100,200,200,300,300,400,500]
    COST_PER_FAMILY_MEMBER = [0, 0, 9,  9,  9, 18, 18, 36, 36,235]
    N_PEOPLE = data['n_people'].astype(int).values

    def get_daily_occupancy(assigned_days):
        daily_occupancy = np.zeros(100, np.int32)
        for i, r in enumerate(assigned_days):
            daily_occupancy[r-1] += N_PEOPLE[i]
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
            if v > MAX_OCCUPANCY or v < MIN_OCCUPANCY:
                penalty += 100000000

        # Calculate the accounting cost
        # The first day (day 100) is treated special
        # using the max function because the soft constraints might allow occupancy to dip below 125
        accounting_cost = max(0, (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5))
        # Loop over the rest of the days, keeping track of previous count
        yesterday_count = daily_occupancy[days[0]]
        for day in days[1:]:
            today_count = daily_occupancy[day]
            diff = abs(today_count - yesterday_count)
            accounting_cost += max(0, (today_count-125.0) / 400.0 * today_count**(0.5 + diff / 50.0))
            yesterday_count = today_count

        return penalty, accounting_cost, penalty + accounting_cost

    _,_,reference = cost_function(submission['assigned_day'])
    
    for num_members in range(2, 9): # Families have minimum 2 and maximum 8 members
        daily_occupancy = get_daily_occupancy(assigned_days)
        fids = np.where(N_PEOPLE == num_members)[0]

        PCOSTM = {}
        for fid in range(NUMBER_FAMILIES):
            if fid in fids:
                for i in range(MAX_BEST_CHOICE):
                    PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]
            else:
                daily_occupancy[assigned_days[fid]-1] -= N_PEOPLE[fid]

        offset = fids.shape[0]
        solver = pywrapgraph.SimpleMinCostFlow()
        for day in range(NUMBER_DAYS):
            solver.SetNodeSupply(offset+day, int(daily_occupancy[day]//num_members))

        for i in range(offset):
            fid = fids[i]
            solver.SetNodeSupply(i, -1)
            for j in range(MAX_BEST_CHOICE):
                day = DESIRED[fid][j]-1
                solver.AddArcWithCapacityAndUnitCost(int(offset+day), i, 1, int(PCOSTM[fid, day]))
        solver.SolveMaxFlowWithMinCost()

        for i in range(solver.NumArcs()):
            if solver.Flow(i) > 0:
                assigned_days[fids[solver.Head(i)]] = solver.Tail(i) - offset + 1
        print(cost_function(assigned_days))
    
    _,_,new_reference = cost_function(submission['assigned_day'])
    
    if new_reference < reference:
        submission.to_csv("submission.csv", index=False)




get_ipython().run_cell_magic('writefile', 'main.cpp', '#include <array>\n#include <cassert>\n#include <algorithm>\n#include <cmath>\n#include <fstream>\n#include <iostream>\n#include <vector>\n#include <thread>\n#include <random>\nusing namespace std;\n#include <chrono>\nusing namespace std::chrono;\n\nconstexpr array<uint8_t, 14> DISTRIBUTION{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5}; // You can setup how many families you need for swaps and what best choice use for each family\n// {3, 5} it\'s mean the first random family will brute force for choices 1-2 and the second random family will brute force for choices 1-5\n\nconstexpr int MAX_OCCUPANCY = 300;\nconstexpr int MIN_OCCUPANCY = 125;\nconstexpr int BEST_N = 1000;\narray<uint8_t, 5000> n_people;\narray<array<uint8_t, 10>, 5000> choices;\narray<array<uint16_t, 10>, 5000> PCOSTM;\narray<array<double, 176>, 176> ACOSTM;\n\nvoid init_data() {\n    ifstream in("../input/santa-workshop-tour-2019/family_data.csv");\n    \n    assert(in && "family_data.csv");\n    string header;\n    int n,x;\n    char comma;\n    getline(in, header);\n    for (int j = 0; j < choices.size(); ++j) {\n        in >> x >> comma;\n        for (int i = 0; i < 10; ++i) {\n            in >> x >> comma;\n            choices[j][i] = x-1;\n        }\n        in >> n;\n        n_people[j] = n;\n    }\n    array<int, 10> pc{0, 50, 50, 100, 200, 200, 300, 300, 400, 500};\n    array<int, 10> pn{0,  0,  9,   9,   9,  18,  18,  36,  36, 235};\n    for (int j = 0; j < PCOSTM.size(); ++j)\n        for (int i = 0; i < 10; ++i)\n            PCOSTM[j][i] = pc[i] + pn[i] * n_people[j];\n    \n    for (int i = 0; i < 176; ++i)\n        for (int j = 0; j < 176; ++j)\n            ACOSTM[i][j] = i * pow(i+125, 0.5 + abs(i-j) / 50.0) / 400.0;\n}\narray<uint8_t, 5000> read_submission(string filename) {\n    ifstream in(filename);\n    assert(in && "submission.csv");\n    array<uint8_t, 5000> assigned_day{};\n    string header;\n    int id, x;\n    char comma;\n    getline(in, header);\n    for (int j = 0; j < choices.size(); ++j) {\n        in >> id >> comma >> x;\n        assigned_day[j] = x-1;\n        auto it = find(begin(choices[j]), end(choices[j]), assigned_day[j]);\n        if (it != end(choices[j]))\n            assigned_day[j] = distance(begin(choices[j]), it);\n    }\n    return assigned_day;\n}\nstruct Index {\n    Index(array<uint8_t, 5000> assigned_days_) : assigned_days(assigned_days_)  {\n        setup();\n    }\n    array<uint8_t, 5000> assigned_days;\n    array<uint16_t, 100> daily_occupancy_{};\n    int preference_cost_ = 0;\n    void setup() {\n        preference_cost_ = 0;\n        daily_occupancy_.fill(0);\n        for (int j = 0; j < assigned_days.size(); ++j) {\n            daily_occupancy_[choices[j][assigned_days[j]]] += n_people[j];\n            preference_cost_ += PCOSTM[j][assigned_days[j]];\n        }\n    }\n    double calc(const array<uint16_t, 5000>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {\n        double accounting_penalty = 0.0;\n        auto daily_occupancy = daily_occupancy_;\n        int preference_cost = preference_cost_;\n        for (int i = 0; i < DISTRIBUTION.size(); ++i) {\n            int j = indices[i];\n            daily_occupancy[choices[j][assigned_days[j]]] -= n_people[j];\n            daily_occupancy[choices[j][       change[i]]] += n_people[j];\n            \n            preference_cost += PCOSTM[j][change[i]] - PCOSTM[j][assigned_days[j]];\n        }\n\n        for (auto occupancy : daily_occupancy)\n            if (occupancy < MIN_OCCUPANCY)\n                return 1e12*(MIN_OCCUPANCY-occupancy);\n            else if (occupancy > MAX_OCCUPANCY)\n                return 1e12*(occupancy - MAX_OCCUPANCY);\n\n        for (int day = 0; day < 99; ++day)\n            accounting_penalty += ACOSTM[daily_occupancy[day]-125][daily_occupancy[day+1]-125];\n\n        accounting_penalty += ACOSTM[daily_occupancy[99]-125][daily_occupancy[99]-125];\n        return preference_cost + accounting_penalty;\n    }\n    void reindex(const array<uint16_t, DISTRIBUTION.size()>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {\n        for (int i = 0; i < DISTRIBUTION.size(); ++i) {\n            assigned_days[indices[i]] = change[i];\n        }\n        setup();\n    }\n};\n\ndouble calc(const array<uint8_t, 5000>& assigned_days, bool print=false) {\n    int preference_cost = 0;\n    double accounting_penalty = 0.0;\n    array<uint16_t, 100> daily_occupancy{};\n    for (int j = 0; j < assigned_days.size(); ++j) {\n        preference_cost += PCOSTM[j][assigned_days[j]];\n        daily_occupancy[choices[j][assigned_days[j]]] += n_people[j];\n    }\n    for (auto occupancy : daily_occupancy)\n        if (occupancy < MIN_OCCUPANCY)\n            return 1e12*(MIN_OCCUPANCY-occupancy);\n        else if (occupancy > MAX_OCCUPANCY)\n            return 1e12*(occupancy - MAX_OCCUPANCY);\n\n    for (int day = 0; day < 99; ++day)\n        accounting_penalty += ACOSTM[daily_occupancy[day]-125][daily_occupancy[day+1]-125];\n\n    accounting_penalty += ACOSTM[daily_occupancy[99]-125][daily_occupancy[99]-125];\n    if (print) {\n        cout << preference_cost << " " << accounting_penalty << " " << preference_cost+accounting_penalty << endl;\n    }\n    return preference_cost + accounting_penalty;\n}\n\nvoid save_sub(const array<uint8_t, 5000>& assigned_day) {\n    ofstream out("submission.csv");\n    out << "family_id,assigned_day" << endl;\n    for (int i = 0; i < assigned_day.size(); ++i)\n        out << i << "," << choices[i][assigned_day[i]]+1 << endl;\n}\n        \nconst vector<array<uint8_t, DISTRIBUTION.size()>> changes = []() {\n    vector<array<uint8_t, DISTRIBUTION.size()>> arr;\n    array<uint8_t, DISTRIBUTION.size()> tmp{};\n    for (int i = 0; true; ++i) {\n        arr.push_back(tmp);\n        tmp[0] += 1;\n        for (int j = 0; j < DISTRIBUTION.size(); ++j)\n            if (tmp[j] >= DISTRIBUTION[j]) {\n                if (j >= DISTRIBUTION.size()-1)\n                    return arr;\n                tmp[j] = 0;\n                ++tmp[j+1];\n            }\n    }\n    return arr;\n}();\n\ntemplate<class ExitFunction>\nvoid stochastic_product_search(Index index, ExitFunction fn) { // 15\'360\'000it/s  65ns/it  0.065µs/it\n    double best_local_score = calc(index.assigned_days);\n    thread_local std::mt19937 gen(std::random_device{}());\n    uniform_int_distribution<> dis(0, 4999);\n    array<uint16_t, 5000> indices;\n    iota(begin(indices), end(indices), 0);\n    array<uint16_t, DISTRIBUTION.size()> best_indices{};\n    array<uint8_t, DISTRIBUTION.size()> best_change{};\n    for (; fn();) {\n        bool found_better = false;\n        for (int k = 0; k < BEST_N; ++k) {\n            for (int i = 0; i < DISTRIBUTION.size(); ++i) //random swap\n                swap(indices[i], indices[dis(gen)]);\n            for (const auto& change : changes) {\n                auto score = index.calc(indices, change);\n                if (score < best_local_score) {\n                    found_better = true;\n                    best_local_score = score;\n                    best_change = change;\n                    copy_n(begin(indices), DISTRIBUTION.size(), begin(best_indices));\n                }\n            }\n        }\n        if (found_better) { // reindex from N best if found better\n            index.reindex(best_indices, best_change);\n//            save_sub(index.assigned_days);\n            calc(index.assigned_days, true);\n            \n        }\n    }\n    save_sub(index.assigned_days);\n}\n\nint main() {\n    init_data();\n    auto assigned_day = read_submission("./submission.csv");\n\n    Index index(assigned_day);\n    calc(index.assigned_days, true);\n//    auto forever = []() { return true; };\n//    auto count_exit = [start = 0]() mutable { return (++start <= 1000); };\n    auto time_exit = [start = high_resolution_clock::now()]() {\n        return duration_cast<minutes>(high_resolution_clock::now()-start).count() < 15; //5h55\n    };\n    \n    stochastic_product_search(index, time_exit);\n    return 0;\n}')




get_ipython().system('g++ -pthread -lpthread -O3 -std=c++17 -o main main.cpp')




trials = 1

for r in range(trials):
    pytorch_optimim(MAX_CHOICE=6)
    get_ipython().system('./main')
    submission = pd.read_csv("./submission.csv")
    optimization_preference(submission, MAX_BEST_CHOICE=6, NUM_SWAP=3000, NUM_SECONDS=2000, NUM_THREADS=4)
    submission = pd.read_csv("./submission.csv")
    mincostflow(submission, MAX_BEST_CHOICE=6)
    get_ipython().system('./main')




save = False

if save:
    from IPython.display import HTML

    def create_download_link(title = "Download CSV file", filename = "data.csv"):  
        html = '<a href={filename}>{title}</a>'
        html = html.format(title=title,filename=filename)
        return HTML(html)

    # create a link to download the dataframe which was saved with .to_csv method
    create_download_link(filename='submission.csv')


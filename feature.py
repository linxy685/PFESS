# %%
import pandas as pd
import numpy as np
import re
def Standardize(unit):
    for m in range(len(unit)-1,-1,-1):
        if unit[m] == '':
            unit.remove('')
    return unit

# Import data
print('Importing Data')

# Property data
raw_df = pd.read_excel('train-test.xlsx', sheet_name='property')


#Split sample to metal_x and metal_y
metal_x = []
metal_y = []
sample = raw_df['sample'].values.tolist()

for i in range(len(sample)):
    element=re.split('(?=[A-Z])', sample[i])
    Standardize(element)
    metal_x.append(element[0]) 
    metal_y.append(element[1])

raw_df['metal_x']=metal_x
raw_df['metal_y']=metal_y

#Import primary element features
ele_df = pd.read_excel('train-test.xlsx', sheet_name='element', index_col=0)
nx = []
ny = []
Sx = []
Sy = []
dx = []
dy = []
IEx = []
IEy = []    
Xx = []
Xy = []
Rx = []
Ry = []

for i in range(len(sample)):
    nx.append(ele_df['n'][raw_df['metal_x'][i]]) 
    ny.append(ele_df['n'][raw_df['metal_y'][i]])
    Sx.append(ele_df['S'][raw_df['metal_x'][i]]) 
    Sy.append(ele_df['S'][raw_df['metal_y'][i]])
    dx.append(ele_df['d'][raw_df['metal_x'][i]]) 
    dy.append(ele_df['d'][raw_df['metal_y'][i]])
    IEx.append(ele_df['IE'][raw_df['metal_x'][i]]) 
    IEy.append(ele_df['IE'][raw_df['metal_y'][i]])
    Xx.append(ele_df['X'][raw_df['metal_x'][i]]) 
    Xy.append(ele_df['X'][raw_df['metal_y'][i]])
    Rx.append(ele_df['R'][raw_df['metal_x'][i]]) 
    Ry.append(ele_df['R'][raw_df['metal_y'][i]])

raw_df['nx']=nx
raw_df['ny']=ny
raw_df['Sx']=Sx
raw_df['Sy']=Sy
raw_df['dx']=dx
raw_df['dy']=dy
raw_df['IEx']=IEx
raw_df['IEy']=IEy
raw_df['Xx']=Xx
raw_df['Xy']=Xy
raw_df['Rx']=Rx
raw_df['Ry']=Ry

print(raw_df)




# %%
import itertools

def optc(D_1_in, h_1_in, u_1_in, D_2_in, h_2_in, u_2_in, opt_flag):

    sz_1 = D_1_in.shape[1]
    sz_2 = D_2_in.shape[1]

    n_c2 = np.array(list(itertools.product(range(sz_1), range(sz_2))))
    print('n_c2=', n_c2)
    sz2 = np.size(n_c2, axis=0)   
    
    if opt_flag == '/' or opt_flag == '/u':
        D_out = np.zeros((D_1_in.shape[0], sz2 * 2))
        h_out = [None] * (sz2 * 2)
        u_out = np.zeros(sz2 * 2)
    else:
        D_out = np.zeros((D_1_in.shape[0], sz2))
        h_out = [None] * sz2
        u_out = np.zeros(sz2)

    
    for i in range(sz2):
        if opt_flag == '-':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = D_1_in[:, n_c2[i, 0]] - D_2_in[:, n_c2[i, 1]]
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '(' + h_1_in[n_c2[i, 0]] + '-' + h_2_in[n_c2[i, 1]] + ')'
        elif opt_flag == '+':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = D_1_in[:, n_c2[i, 0]] + D_2_in[:, n_c2[i, 1]]
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '(' + h_1_in[n_c2[i, 0]] + '+' + h_2_in[n_c2[i, 1]] + ')'
        elif opt_flag == '/':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = D_1_in[:, n_c2[i, 0]] / D_2_in[:, n_c2[i, 1]]
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '(' + h_1_in[n_c2[i, 0]] + '/' + h_2_in[n_c2[i, 1]] + ')'
                j = i + sz2
                D_out[:, j] = D_2_in[:, n_c2[i, 1]] / D_1_in[:, n_c2[i, 0]]
                u_out[j] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[j] = '(' + h_2_in[n_c2[i, 1]] + '/' + h_1_in[n_c2[i, 0]] + ')'
        elif opt_flag == '/u':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = D_1_in[:, n_c2[i, 0]] / D_2_in[:, n_c2[i, 1]]
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '(' + h_1_in[n_c2[i, 0]] + '/' + h_2_in[n_c2[i, 1]] + ')'
                j = i + sz2
                D_out[:, j] = D_2_in[:, n_c2[i, 1]] / D_1_in[:, n_c2[i, 0]]
                u_out[j] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[j] = '(' + h_2_in[n_c2[i, 1]] + '/' + h_1_in[n_c2[i, 0]] + ')'
        elif opt_flag == '*':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = D_1_in[:, n_c2[i, 0]] * D_2_in[:, n_c2[i, 1]]
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '(' + h_1_in[n_c2[i, 0]] + '*' + h_2_in[n_c2[i, 1]] + ')'
        elif opt_flag == '/abs':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = np.abs(D_1_in[:, n_c2[i, 0]] / D_2_in[:, n_c2[i, 1]])
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '|' + h_1_in[n_c2[i, 0]] + '/' + h_2_in[n_c2[i, 1]] + '|'
        elif opt_flag == '/absu':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = np.abs(D_1_in[:, n_c2[i, 0]] / D_2_in[:, n_c2[i, 1]])
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '|' + h_1_in[n_c2[i, 0]] + '/' + h_2_in[n_c2[i, 1]] + '|'
        elif opt_flag == '*abs':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:, i] = np.abs(D_1_in[:, n_c2[i, 0]] * D_2_in[:, n_c2[i, 1]])
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '|' + h_1_in[n_c2[i, 0]] + '*' + h_2_in[n_c2[i, 1]] + '|'
        elif opt_flag == '-abs':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:,i] = np.abs(D_1_in[:,n_c2[i,0]] - D_2_in[:,n_c2[i,1]])
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '|' + h_1_in[n_c2[i,0]] + '-' + h_2_in[n_c2[i,1]] + '|'
        elif opt_flag == '+abs':
            if u_1_in[n_c2[i, 0]] != u_2_in[n_c2[i, 1]]:
                D_out[:,i] = np.abs(D_1_in[:,n_c2[i,0]] + D_2_in[:,n_c2[i,1]])
                u_out[i] = np.abs(max(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]) ** 2 - min(u_2_in[n_c2[i, 1]],u_1_in[n_c2[i, 0]]))
                h_out[i] = '|' + h_1_in[n_c2[i,0]] + '+' + h_2_in[n_c2[i,1]] + '|'

        else:
            print('unknown operation')


    unique = np.where(np.all(D_out != 0, axis=0))[0]
    D_out = D_out[:,unique]
    u_out = u_out[unique]
    h_out = [h_out[i] for i in unique]
    
    return D_out, h_out, u_out

# %%
import itertools

def optc2(D_in, h_in, u_in, opt_flag):
    sz = np.shape(D_in)
    index = np.arange(0, sz[1])

    n_c2 = np.array(list(itertools.combinations(index, 2)))

    sz2 = np.size(n_c2, axis=0)  
    
    if opt_flag == '/' or opt_flag == '/u':
        D_out = np.zeros((sz[0], sz2 * 2))
        h_out = [None] * (sz2 * 2)
        u_out = np.zeros(sz2 * 2)
    else:
        D_out = np.zeros((sz[0], sz2))
        h_out = [None] * sz2
        u_out = np.zeros(sz2)

    
    for i in range(sz2):
        if opt_flag == '-':
            if u_in[n_c2[i, 0]] == u_in[n_c2[i, 1]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] - D_in[:, n_c2[i, 1]]
                u_out[i] = u_in[n_c2[i, 0]]
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '-' + h_in[n_c2[i, 1]] + ')'
        elif opt_flag == '+':
            if u_in[n_c2[i, 0]] == u_in[n_c2[i, 1]] and u_in[n_c2[i, 0]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] + D_in[:, n_c2[i, 1]]
                u_out[i] = u_in[n_c2[i, 0]]
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '+' + h_in[n_c2[i, 1]] + ')'
        elif opt_flag == '/':
            D_out[:, i] = D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]]
            u_out[i] = u_in[n_c2[i, 0]] 
            h_out[i] = '(' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + ')'
            j = i + sz2
            D_out[:, j] = D_in[:, n_c2[i, 1]] / D_in[:, n_c2[i, 0]]
            u_out[j] = u_in[n_c2[i, 0]] 
            h_out[j] = '(' + h_in[n_c2[i, 1]] + '/' + h_in[n_c2[i, 0]] + ')'
        elif opt_flag == '/u':
            if u_in[n_c2[i, 0]] == u_in[n_c2[i, 1]]  and u_in[n_c2[i, 0]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]]
                u_out[i] = u_in[n_c2[i, 0]] 
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + ')'
                j = i + sz2
                D_out[:, j] = D_in[:, n_c2[i, 1]] / D_in[:, n_c2[i, 0]]
                u_out[j] = u_in[n_c2[i, 0]] 
                h_out[j] = '(' + h_in[n_c2[i, 1]] + '/' + h_in[n_c2[i, 0]] + ')'
        elif opt_flag == '*':
            D_out[:, i] = D_in[:, n_c2[i, 0]] * D_in[:, n_c2[i, 1]]
            u_out[i] = u_in[n_c2[i, 0]] 
            h_out[i] = '(' + h_in[n_c2[i, 0]] + '*' + h_in[n_c2[i, 1]] + ')'
        elif opt_flag == '*u':
            if u_in[n_c2[i, 0]] == u_in[n_c2[i, 1]]  and u_in[n_c2[i, 0]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] * D_in[:, n_c2[i, 1]]
                u_out[i] = u_in[n_c2[i, 0]] 
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '*' + h_in[n_c2[i, 1]] + ')'
        elif opt_flag == '/abs':
            D_out[:, i] = np.abs(D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]])
            u_out[i] = u_in[n_c2[i, 0]] 
            h_out[i] = '|' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + '|'
        elif opt_flag == '/absu':
            if u_in[n_c2[i, 0]] == u_in[n_c2[i, 1]]:
                D_out[:, i] = np.abs(D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]])
                u_out[i] = u_in[n_c2[i, 0]] 
                h_out[i] = '|' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + '|'
        elif opt_flag == '*abs':
            D_out[:, i] = np.abs(D_in[:, n_c2[i, 0]] * D_in[:, n_c2[i, 1]])
            u_out[i] = u_in[n_c2[i, 0]] 
            h_out[i] = '|' + h_in[n_c2[i, 0]] + '*' + h_in[n_c2[i, 1]] + '|'
        elif opt_flag == '-abs':
            if u_in[n_c2[i,0]] == u_in[n_c2[i,1]]  and u_in[n_c2[i, 0]]:
                D_out[:,i] = np.abs(D_in[:,n_c2[i,0]] - D_in[:,n_c2[i,1]])
                u_out[i] = u_in[n_c2[i,0]]
                h_out[i] = '|' + h_in[n_c2[i,0]] + '-' + h_in[n_c2[i,1]] + '|'
        elif opt_flag == '+abs':
            if u_in[n_c2[i,0]] == u_in[n_c2[i,1]]:
                D_out[:,i] = np.abs(D_in[:,n_c2[i,0]] + D_in[:,n_c2[i,1]])
                u_out[i] = u_in[n_c2[i,0]]
                h_out[i] = '|' + h_in[n_c2[i,0]] + '+' + h_in[n_c2[i,1]] + '|'

        else:
            print('unknown operation')


    unique = np.where(np.all(D_out != 0, axis=0))[0]
    D_out = D_out[:,unique]
    u_out = u_out[unique]
    h_out = [h_out[i] for i in unique]
    
    return D_out, h_out, u_out

# %%
def optc3(D_in, h_in, u_in, opt_flag):
    sz = np.shape(D_in)
    index = np.arange(0, sz[1])

    n_c2 = np.array(list(itertools.combinations(index, 2)))
    sz2 = np.size(n_c2, axis=0)  
    
    if opt_flag == '/' or opt_flag == '/u':
        D_out = np.zeros((sz[0], sz2 * 2))
        h_out = [None] * (sz2 * 2)
        u_out = np.zeros(sz2 * 2)
    else:
        D_out = np.zeros((sz[0], sz2))
        h_out = [None] * sz2
        u_out = np.zeros(sz2)
    

    for i in range(sz2):
        if opt_flag == '-':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] - D_in[:, n_c2[i, 1]]
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]]) 
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '-' + h_in[n_c2[i, 1]] + ')'
        elif opt_flag == '+':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] + D_in[:, n_c2[i, 1]]
                u_out[i] = np.abs(u_in[n_c2[i, 1]] ** 2 - u_in[n_c2[i, 0]])
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '+' + h_in[n_c2[i, 1]] + ')'
        elif opt_flag == '/':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]]
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + ')'
                j = i +sz2
                D_out[:, j] = D_in[:, n_c2[i, 1]] / D_in[:, n_c2[i, 0]]
                u_out[j] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[j] = '(' + h_in[n_c2[i, 1]] + '/' + h_in[n_c2[i, 0]] + ')'
        elif opt_flag == '/u':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]]
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + ')'
                j = i +sz2
                D_out[:, j] = D_in[:, n_c2[i, 1]] / D_in[:, n_c2[i, 0]]
                u_out[j] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[j] = '(' + h_in[n_c2[i, 1]] + '/' + h_in[n_c2[i, 0]] + ')'
        elif opt_flag == '*':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = D_in[:, n_c2[i, 0]] * D_in[:, n_c2[i, 1]]
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[i] = '(' + h_in[n_c2[i, 0]] + '*' + h_in[n_c2[i, 1]] + ')'
        elif opt_flag == '/abs':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = np.abs(D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]])
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[i] = '|' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + '|'
        elif opt_flag == '/absu':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = np.abs(D_in[:, n_c2[i, 0]] / D_in[:, n_c2[i, 1]])
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]]) 
                h_out[i] = '|' + h_in[n_c2[i, 0]] + '/' + h_in[n_c2[i, 1]] + '|'
        elif opt_flag == '*abs':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:, i] = np.abs(D_in[:, n_c2[i, 0]] * D_in[:, n_c2[i, 1]])
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[i] = '|' + h_in[n_c2[i, 0]] + '*' + h_in[n_c2[i, 1]] + '|'
        elif opt_flag == '-abs':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:,i] = np.abs(D_in[:,n_c2[i,0]] - D_in[:,n_c2[i,1]])
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[i] = '|' + h_in[n_c2[i,0]] + '-' + h_in[n_c2[i,1]] + '|'
        elif opt_flag == '+abs':
            if u_in[n_c2[i, 0]] != u_in[n_c2[i, 1]]:
                D_out[:,i] = np.abs(D_in[:,n_c2[i,0]] + D_in[:,n_c2[i,1]])
                u_out[i] = min(u_in[n_c2[i, 1]],u_in[n_c2[i, 0]])
                h_out[i] = '|' + h_in[n_c2[i,0]] + '+' + h_in[n_c2[i,1]] + '|'

        else:
            print('unknown operation')


    unique = np.where(np.all(D_out != 0, axis=0))[0]
    D_out = D_out[:,unique]
    u_out = u_out[unique]
    h_out = [h_out[i] for i in unique]
    
    return D_out, h_out, u_out

# %%
def optc1(D_in, h_in, u_in, opt_flag):

    sz = np.size(D_in, axis=1)
    h_out = [None] * sz
    u_out = np.zeros(sz)

    for i in range(sz):
        if opt_flag == '^0.5':
            D_out = abs(D_in) ** 0.5
            h_out[i] = '(' + h_in[i] + ')' + opt_flag
            u_out = u_in 
        elif opt_flag == '^0.2':
            D_out = D_in ** 0.2
            h_out[i] = '(' + h_in[i] + ')' + opt_flag
            u_out = u_in 
        elif opt_flag == '^2':
            D_out = D_in ** 2
            h_out[i] = '(' + h_in[i] + ')' + opt_flag
            u_out = u_in 
        elif opt_flag == '^3':
            D_out = D_in ** 3
            h_out[i] = '(' + h_in[i] + ')' + opt_flag
            u_out = u_in
        elif opt_flag == 'log':
            D_out = np.log(abs(D_in))
            h_out[i] = '(' + h_in[i] + ')' + opt_flag
            u_out = u_in
        elif opt_flag == 'exp':
            D_out = np.exp(D_in)
            h_out[i] = '(' + h_in[i] + ')' + opt_flag
            u_out = u_in
        elif opt_flag == 'abs':
            D_out = abs(D_in)
            h_out[i] = '(' + h_in[i] + ')' + opt_flag
            u_out = u_in
        else:
            print('unknown operation')
    
    unique = np.where(np.all(D_out != 0, axis=0))[0]
    D_out = D_out[:,unique]
    u_out = u_out[unique]
    h_out = [h_out[i] for i in unique]
    
    return D_out, h_out, u_out

# %%
def genfeature(D_1_in, h_1_in, u_1_in, D_2_in, h_2_in, u_2_in, List, add_flag=None):
    if  add_flag == None:
        add_flag = 1

    DP = []
    hP = []
    uP = []
    for i in range(len(List)):
        DPTemp, hPTemp, uPTemp = optc(D_1_in, h_1_in, u_1_in, D_2_in, h_2_in, u_2_in, List[i])
        DP += DPTemp.T.tolist()
        hP += hPTemp
        uP += uPTemp.tolist()

    if (add_flag == 1):
        D_out = np.transpose(D_1_in.T.tolist() + D_2_in.T.tolist() + DP)
        h_out = h_1_in + h_2_in + hP
        u_out = np.transpose(u_1_in.tolist() + u_2_in.tolist() + uP)
    else:
        D_out = np.transpose(DP)
        h_out = hP
        u_out = np.transpose(uP)


    u_out = u_out.astype(int)
    
    return D_out, h_out, u_out

# %%
def genfeature2(D_in, h_in, u_in, List, add_flag=None):
    if  add_flag == None:
        add_flag = 1


    DP = []
    hP = []
    uP = []
    for i in range(len(List)):
        DPTemp, hPTemp, uPTemp = optc2(D_in, h_in, u_in, List[i])
        DP += DPTemp.T.tolist()
        hP += hPTemp
        uP += uPTemp.tolist()

    if (add_flag == 1):
        D_out = np.transpose(D_in.T.tolist() + DP)
        h_out = h_in + hP
        u_out = np.transpose(u_in.tolist() + uP)
    else:
        D_out = np.transpose(DP)
        h_out = hP
        u_out = np.transpose(uP)


    u_out = u_out.astype(int)
    
    return D_out, h_out, u_out

# %%
def genfeature1(D_in, h_in, u_in, List, add_flag=None):
    if  add_flag == None:
        add_flag = 1


    DP = []
    hP = []
    uP = []
    for i in range(len(List)):
        DPTemp, hPTemp, uPTemp = optc1(D_in, h_in, u_in, List[i])
        DP += DPTemp.T.tolist()
        hP += hPTemp
        uP += uPTemp.tolist()

    if (add_flag == 1):
        D_out = np.transpose(D_in.T.tolist() + DP)
        h_out = h_in + hP
        u_out = np.transpose(u_in.tolist() + uP)
    else:
        D_out = np.transpose(DP)
        h_out = hP
        u_out = np.transpose(uP)


    u_out = u_out.astype(int)
    
    return D_out, h_out, u_out

# %%
def genfeature3(D_in, h_in, u_in, List, add_flag=None):
    if  add_flag == None:
        add_flag = 1


    DP = []
    hP = []
    uP = []
    for i in range(len(List)):
        DPTemp, hPTemp, uPTemp = optc3(D_in, h_in, u_in, List[i])
        DP += DPTemp.T.tolist()
        hP += hPTemp
        uP += uPTemp.tolist()

    if (add_flag == 1):
        D_out = np.transpose(D_in.T.tolist() + DP)
        h_out = h_in + hP
        u_out = np.transpose(u_in.tolist() + uP)
    else:
        D_out = np.transpose(DP)
        h_out = hP
        u_out = np.transpose(uP)


    u_out = u_out.astype(int)

    return D_out, h_out, u_out

D_p = raw_df.values
loc=[4,5,6,7,14,15]
D_p = D_p[:,loc]

h_p = raw_df.columns.values
h_p= h_p[loc].tolist()


u_p= np.array([1,1,2,2,6,6])


List = ['^0.5', '^2']
D_fp1, h_fp1, u_fp1 = genfeature1(D_p, h_p, u_p, List, 1)

List = ['+', '-abs', '*u', '/u']
# List = ['+', '-abs']
D_fp2, h_fp2, u_fp2 = genfeature2(D_fp1, h_fp1, u_fp1, List, 1)
print(len(h_fp2))

List = ['*', '/']
D_fp3, h_fp3, u_fp3 = genfeature3(D_fp1, h_fp1, u_fp1, List, 1)

List = ['*', '/']
D_fp4, h_fp4, u_fp4 = genfeature(D_fp2, h_fp2, u_fp2, D_fp3, h_fp3, u_fp3, List, 1)
print(len(h_fp4))
print(len(u_fp4))
print(D_fp4.shape)





print(D_fp4[21][100])
print(h_fp4[100])



D_s = D_fp4
h_s = h_fp4
h_s=np.array(h_s)


c_col = np.where(np.any(np.isinf(D_s), axis=0) | np.any(np.isnan(D_s), axis=0))[0]
D_s = np.delete(D_s, c_col, axis=1)
h_s = np.delete(h_s, c_col, axis=0)

print('Running BIG DATA generation step!!')


C, ia, ic = np.unique(D_s.T, axis=0, return_index=True, return_inverse=True)
ia.sort()
D_s = D_s[:, ia]

# final headers 
h_s = h_s[ia]





# %%
Sample = raw_df['sample'].values.reshape(-1,1)

Property = raw_df['property'].values


h_s = h_s.tolist()


# %%
D_Total_s = (D_s - np.mean(D_s, axis=0)) / np.std(D_s, axis=0)
print(np.isnan(D_s).any())
print(np.isnan(D_Total_s).any())

# %%
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# %%
def RF_SIS(X, y,feature_number,coeff):

    
    f = open(f'RF_SIS_param_{feature_number}.txt', 'w')

    



    X_coeff = X[:,coeff]


    if len(coeff) > feature_number:
        rf_regressor = RandomForestRegressor(n_estimators=2000, random_state=42)
        rf_regressor.fit(X_coeff, y)
        feature_importances =  rf_regressor.feature_importances_
        sorted_feature_indices = np.argsort(feature_importances)[::-1]
        selected_feature_indices = sorted_feature_indices[:feature_number]
        selected_feature_indices  = selected_feature_indices.astype(np.int32)
        coeff=np.array(coeff)
        coeff=coeff[selected_feature_indices]





    coeff = list(coeff.astype(int))




    f.write('-------End of RF Step------------------\n')
    f.write(f'Defined Number of feature spaces: {feature_number}\n')
    
    f.write('list of Coeff. \n')
    f.write('index\tCoeff. \n')

    for j in range(len(coeff)):
        f.write(f'{coeff[j]}\t{h_s[coeff[j]]}\n')

    f.close()        
    
    return coeff




P_c_temp = Property
P_c = P_c_temp - np.mean(P_c_temp)


lm_max = np.max(np.abs(np.transpose(D_Total_s) @ P_c)) / len(P_c)
print('lambda_max=',lm_max)

all_number=D_Total_s.shape[1]

feature_number=int(all_number/5*4)
coeff = np.arange(all_number)  
coeff = list(coeff.astype(int))

while feature_number>1000:
    coeff=RF_SIS(D_Total_s,P_c,feature_number,coeff)
    feature_number=int(feature_number/2*1)




# %%
from itertools import combinations
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

test = coeff
index = np.arange(len(test))


h_s=np.array(h_s)
lda = np.logspace(0, -5, 5) * lm_max

for k in range(1, 2):  # Set range to k=1:5 to generate 4D and 5D
    des_num = k
    n_c2 = np.array(list(combinations(test, des_num)), dtype=np.int32)    
    sz = len(n_c2)
    rmse = np.zeros(sz)
    coefficient= np.zeros(sz)
    intercept= np.zeros(sz)

    

    for i in range(sz):
        d_test = D_s[:, n_c2[i]]   
        lasso1_model = LassoCV(alphas=list(lda), cv=10,max_iter=100000)
        lasso1_model.fit(d_test, Property)
        if 0 not in lasso1_model.coef_:
            #print(lasso1_model.coef_)
            rmse[i] = np.sqrt(mean_squared_error(y_true=Property, y_pred=lasso1_model.predict(d_test)))




    
    n_des=50
    print(rmse)
    I = np.argsort(rmse).tolist()
    print(I)
    for i in reversed(range(len(I))):
        if rmse[I[i]] == 0:
            del I[i]
    I = I[:50]
    print(rmse[I])

    f_best = open('best_descriptor_{}D.txt'.format(des_num), 'w')

    best_des = n_c2[I]
    for i in range(len(best_des)):
        lasso2_fit_model = LassoCV(alphas=list(lda), cv=10,max_iter=100000)
        lasso2_fit_model.fit(D_s[:, best_des[i]], Property)


        y_pred = lasso2_fit_model.predict(D_s[:, best_des[i]])

        MAE = np.mean(np.abs(Property - y_pred))
        MaxAE = np.max(np.abs(Property - y_pred))

        r2 = lasso2_fit_model.score(D_s[:, best_des[i]], Property)

        

        f_best.write('----------------\n')
        f_best.write('No of descriptors\n')
        f_best.write('----------------\n')
        f_best.write(f'Top {i}\n')
        f_best.write(f'lowest RMSE: {rmse[I[i]]}\n')
        f_best.write(f'MAE: {MAE}\n')
        f_best.write(f'R2: {r2}\n')
        f_best.write(f'Descriptors\n {h_s[list(best_des[i])]}\n')
        f_best.write(f'Descriptor Index\n {list(best_des[i])}\n')
        f_best.write(f'Coefficients\n {lasso2_fit_model.coef_}\n')
        f_best.write(f'Intercept\n {lasso2_fit_model.intercept_}\n')

            
        if i == 0:
            y_pred_best = y_pred

    f_best.close()


    figure1 = plt.figure()


    y_lim_max = round(np.max([y_pred_best, Property])) + 0.5
    y_lim_min = round(np.min([y_pred_best, Property])) - 0.5


    plt.plot([y_lim_min, y_lim_max], [y_lim_min, y_lim_max], '-', y_pred_best, Property, 'o')


    plt.ylim([y_lim_min, y_lim_max])
    plt.xlim([y_lim_min, y_lim_max])


    plt.xlabel('Predicted Property (eV)')
    plt.ylabel('Property (eV)')


    t = '{}D, RMSE = {}'.format(des_num, rmse[I[0]])
    plt.title(t)


    fname = '{}D.jpg'.format(des_num)
    plt.savefig(fname)


    wdata = np.column_stack((Property.T, y_pred_best.T))
    fname1 = '{}D.txt'.format(des_num)
    np.savetxt(fname1, wdata, delimiter='\t', fmt='%.6f')

# %%


#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --quiet cupy-cuda100==5.4.0')
get_ipython().system('pip install --quiet chainer-chemistry==0.5.0')
get_ipython().system('pip install --quiet chaineripy')
get_ipython().system('conda install -y --quiet -c rdkit rdkit')


# In[2]:


# Check correctly installed, and modules can be imported.
import chainer
import chainer_chemistry
import chaineripy
import cupy
import rdkit

print('chainer version: ', chainer.__version__)
print('cupy version: ', cupy.__version__)
print('chainer-chemistry version: ', chainer_chemistry.__version__)
print('rdkit version: ', rdkit.__version__)


# In[3]:


from contextlib import contextmanager
import gc
import numpy as np # linear algebra
import numpy
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from time import time, perf_counter

import seaborn as sns
import matplotlib.pyplot as plt

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

import rdkit
from rdkit import Chem


# In[4]:


# util functions...

@contextmanager
def timer(name):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print('[{}] done in {:.3f} s'.format(name, t1-t0))


# In[5]:


"""
Copied from
https://github.com/jensengroup/xyz2mol/blob/master/xyz2mol.py

Modified `chiral_stereo_check` method for this task's purpose.
"""
##
# Written by Jan H. Jensen based on this paper Yeonjoon Kim and Woo Youn Kim
# "Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity
# to Three-Dimensional Geometry" Bull. Korean Chem. Soc. 2015, Vol. 36, 1769-1777 DOI: 10.1002/bkcs.10334
#
from rdkit import Chem
from rdkit.Chem import AllChem
import itertools
from rdkit.Chem import rdmolops
from collections import defaultdict
import copy
import networkx as nx  # uncomment if you don't want to use "quick"/install networkx

global __ATOM_LIST__
__ATOM_LIST__ = [x.strip() for x in ['h ', 'he',                                      'li', 'be', 'b ', 'c ', 'n ', 'o ', 'f ', 'ne',                                      'na', 'mg', 'al', 'si', 'p ', 's ', 'cl', 'ar',                                      'k ', 'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',                                      'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',                                      'rb', 'sr', 'y ', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',                                      'cd', 'in', 'sn', 'sb', 'te', 'i ', 'xe',                                      'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',                                      'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w ', 're', 'os', 'ir', 'pt',                                      'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',                                      'fr', 'ra', 'ac', 'th', 'pa', 'u ', 'np', 'pu']]


def get_atom(atom):
    global __ATOM_LIST__
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def getUA(maxValence_list, valence_list):
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if maxValence - valence > 0:
            UA.append(i)
            DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, quick):
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = getUA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, quick)[0]

    return BO


def valences_not_too_large(BO, valences):
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def BO_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atomicNumList, charged_fragments):
    Q = 0  # total charge
    q_list = []
    if charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atomicNumList):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)

    if (BO - AC).sum() == sum(DU) and charge == Q and len(q_list) <= abs(charge):
        return True
    else:
        return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def clean_charges(mol):
    # this hack should not be needed any more but is kept just in case
    #

    rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
                  '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
                  '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
                  '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
                  '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
                  '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol, fragment)

    return mol


def BO2mol(mol, BO_matrix, atomicNumList, atomic_valence_electrons, mol_charge, charged_fragments):
    # based on code written by Paolo Toscani

    l = len(BO_matrix)
    l2 = len(atomicNumList)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and atomicNumList '
                           '{1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)
    mol = rwMol.GetMol()

    if charged_fragments:
        mol = set_atomic_charges(mol, atomicNumList, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge)
    else:
        mol = set_atomic_radicals(mol, atomicNumList, atomic_valence_electrons, BO_valences)

    return mol


def set_atomic_charges(mol, atomicNumList, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge):
    q = 0
    for i, atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if (abs(charge) > 0):
            a.SetFormalCharge(int(charge))

    # shouldn't be needed anymore bit is kept just in case
    # mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol, atomicNumList, atomic_valence_electrons, BO_valences):
    # The number of radical electrons = absolute atomic charge
    for i, atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if (abs(charge) > 0):
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1:]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, quick):
    bonds = get_bonds(UA, AC)
    if len(bonds) == 0:
        return [()]

    if quick:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]
        #           if quick and max_atoms_in_combo == 2*int(len(UA)/2):
        #               return UA_pairs
        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def AC2BO(AC, atomicNumList, charge, charged_fragments, quick):
    # TODO
    atomic_valence = defaultdict(list)
    atomic_valence[1] = [1]
    atomic_valence[6] = [4]
    atomic_valence[7] = [4, 3]
    atomic_valence[8] = [2, 1]
    atomic_valence[9] = [1]
    atomic_valence[14] = [4]
    atomic_valence[15] = [5, 4, 3]
    atomic_valence[16] = [6, 4, 2]
    atomic_valence[17] = [1]
    atomic_valence[32] = [4]
    atomic_valence[35] = [1]
    atomic_valence[53] = [1]

    atomic_valence_electrons = {}
    atomic_valence_electrons[1] = 1
    atomic_valence_electrons[6] = 4
    atomic_valence_electrons[7] = 5
    atomic_valence_electrons[8] = 6
    atomic_valence_electrons[9] = 7
    atomic_valence_electrons[14] = 4
    atomic_valence_electrons[15] = 5
    atomic_valence_electrons[16] = 6
    atomic_valence_electrons[17] = 7
    atomic_valence_electrons[32] = 4
    atomic_valence_electrons[35] = 7
    atomic_valence_electrons[53] = 7

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    for atomicNum in atomicNumList:
        valences_list_of_lists.append(atomic_valence[atomicNum])

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = list(itertools.product(*valences_list_of_lists))

    best_BO = AC.copy()

    # implemenation of algorithm shown in Figure 2
    # UA: unsaturated atoms
    # DU: degree of unsaturation (u matrix in Figure)
    # best_BO: Bcurr in Figure
    #

    for valences in valences_list:
        AC_valence = list(AC.sum(axis=1))
        UA, DU_from_AC = getUA(valences, AC_valence)

        if len(UA) == 0 and BO_is_OK(AC, AC, charge, DU_from_AC, atomic_valence_electrons, atomicNumList,
                                     charged_fragments):
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, quick)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, quick)
            if BO_is_OK(BO, AC, charge, DU_from_AC, atomic_valence_electrons, atomicNumList, charged_fragments):
                return BO, atomic_valence_electrons

            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO, valences):
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick):
    # convert AC matrix to bond order (BO) matrix
    BO, atomic_valence_electrons = AC2BO(AC, atomicNumList, charge, charged_fragments, quick)

    # add BO connectivity and charge info to mol object
    mol = BO2mol(mol, BO, atomicNumList, atomic_valence_electrons, charge, charged_fragments)

    return mol


def get_proto_mol(atomicNumList):
    mol = Chem.MolFromSmarts("[#" + str(atomicNumList[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atomicNumList)):
        a = Chem.Atom(atomicNumList[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def get_atomicNumList(atomic_symbols):
    atomicNumList = []
    for symbol in atomic_symbols:
        atomicNumList.append(get_atom(symbol))
    return atomicNumList


def read_xyz_file(filename):
    atomic_symbols = []
    xyz_coordinates = []

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
                else:
                    charge = 0
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atomicNumList = get_atomicNumList(atomic_symbols)

    return atomicNumList, charge, xyz_coordinates


def xyz2AC(atomicNumList, xyz):
    import numpy as np
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    dMat = Chem.Get3DDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms, num_atoms)).astype(int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * 1.30
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * 1.30
            if dMat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC, mol


def chiral_stereo_check(mol):
    # Chem.SanitizeMol(mol)
    num_error = Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
    if num_error != 0:
        print('error id', num_error)

    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return mol


def xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick):
    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC, mol = xyz2AC(atomicNumList, xyz_coordinates)

    # Convert AC to bond order matrix and add connectivity and charge info to mol object
    new_mol = AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick)

    # Check for stereocenters and chiral centers
    new_mol = chiral_stereo_check(new_mol)

    return new_mol


# In[6]:


def mol_from_xyz(filepath, to_canonical=False):
    charged_fragments = True  # alternatively radicals are made

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = True

    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)
    # print('atomicNumList', atomicNumList, 'charge', charge)
    mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)

    # Canonical hack
    if to_canonical:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(smiles)
    return mol


# In[7]:


# This script is referred from http://rdkit.blogspot.jp/2015/02/new-drawing-code.html
# and http://cheminformist.itmol.com/TEST/wp-content/uploads/2015/07/rdkit_moldraw2d_2.html
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
def moltosvg(mol,molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def render_svg(svg):
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return SVG(svg.replace('svg:',''))


# In[8]:


# Input data files are available in the "../input/" directory.
print(os.listdir("../input"))


# In[9]:


# Load each data...

input_dir = Path('../input/champs-scalar-coupling')
train_df = pd.read_csv(input_dir/'train.csv')
test_df = pd.read_csv(input_dir/'test.csv')
structures_df = pd.read_csv(input_dir/'structures.csv')
sample_submission_df = pd.read_csv(input_dir/'sample_submission.csv')

dipole_moments_df = pd.read_csv(input_dir/'dipole_moments.csv')
magnetic_shielding_tensors_df = pd.read_csv(input_dir/'magnetic_shielding_tensors.csv')
mulliken_charges_df = pd.read_csv(input_dir/'mulliken_charges.csv')
potential_energy_df = pd.read_csv(input_dir/'potential_energy.csv')
scalar_coupling_contributions_df = pd.read_csv(input_dir/'scalar_coupling_contributions.csv')


# In[10]:


train_df.head()


# In[11]:


test_df.head()


# In[12]:


structures_df.head()


# In[13]:


sample_submission_df.head()


# In[14]:


dipole_moments_df.head()


# In[15]:


potential_energy_df.head()


# In[16]:


potential_energy_df['potential_energy'].hist()


# In[17]:


magnetic_shielding_tensors_df.head()


# In[18]:


mulliken_charges_df.head()


# In[19]:


mulliken_charges_df['mulliken_charge'].hist()


# In[20]:


scalar_coupling_contributions_df.head()


# In[21]:


# Atom pair wise info
print('--- Atom pair wise info ---')
print(f'train_df {train_df.shape}')
print(f'test_df {test_df.shape}')
print(f'scalar_coupling_contributions_df {scalar_coupling_contributions_df.shape}')  # only contain train info
print(f'sample_submission_df {sample_submission_df.shape}')  # for test prediction

# Molecule wise info
print('--- Molecule wise info ---')
print(f'potential_energy_df {potential_energy_df.shape}')  # only contain train info
print(f'dipole_moments_df {dipole_moments_df.shape}')  # only contain train info

# Atom wise info
print('--- Atom wise info ---')
print(f'structures_df {structures_df.shape}')  # contains both train/test info
print(f'magnetic_shielding_tensors_df {magnetic_shielding_tensors_df.shape}')  # only contain train info
print(f'mulliken_charges_df {mulliken_charges_df.shape}')  # only contain train info


# In[22]:


del dipole_moments_df
del magnetic_shielding_tensors_df
del mulliken_charges_df
del potential_energy_df
del scalar_coupling_contributions_df

gc.collect()


# In[23]:


train_mol_names = train_df['molecule_name'].unique()
test_mol_names = test_df['molecule_name'].unique()

print(f'Number of molecules: train {len(train_mol_names)}, test {len(test_mol_names)}, train+test {len(train_mol_names) + len(test_mol_names)}')
print(f'Number of total atoms: train+test {len(structures_df)}')


# In[24]:


print('name of molecues in train', train_mol_names)
print('name of molecues in test', test_mol_names)


# In[25]:


structures_df['atom'].unique()


# In[26]:


mol_name_counts = structures_df['molecule_name'].value_counts()
mol_name_counts.head()


# In[27]:


mol_name_counts.hist()


# In[28]:


heavy_atoms_in_mol = structures_df[structures_df['atom'] != 'H']['molecule_name'].value_counts()


# In[29]:


heavy_atoms_in_mol.hist()


# In[30]:


# Histogram of heavy atoms
heavy_atoms_in_mol.value_counts()


# In[31]:


heavy_atoms_in_mol.sort_index()


# In[32]:


plt.figure()
plt.scatter(np.arange(len(heavy_atoms_in_mol)), heavy_atoms_in_mol.sort_index().values)
plt.xlabel('molecule_name')
plt.ylabel('Number of heavy atoms')
plt.show()
plt.close()


# In[33]:


for mol_id in ['000001', '001000', '010000', '133885']:
    mol_name = f'dsgdb9nsd_{mol_id}'
    filepath = input_dir/f'structures/{mol_name}.xyz'
    mol = mol_from_xyz(filepath, to_canonical=True)
    print('mol_name:', mol_name)
    display(mol)


# In[34]:


coupling_tyes = train_df['type'].unique()
print('coupling_tyes', coupling_tyes)


# In[35]:


plt.figure()
sns.distplot(train_df['scalar_coupling_constant'])
plt.show()
plt.close()


# In[36]:


grid = sns.FacetGrid(train_df[['type', 'scalar_coupling_constant']], col='type', hue='type', col_wrap=4)
grid.map(sns.distplot, 'scalar_coupling_constant')


# In[37]:


# Calculate distance feature...
# --- dist hist ---

# from https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark

def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures_df, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train_df = map_atom_info(train_df, 0)
train_df = map_atom_info(train_df, 1)

test_df = map_atom_info(test_df, 0)
test_df = map_atom_info(test_df, 1)

train_p_0 = train_df[['x_0', 'y_0', 'z_0']].values
train_p_1 = train_df[['x_1', 'y_1', 'z_1']].values
test_p_0 = test_df[['x_0', 'y_0', 'z_0']].values
test_p_1 = test_df[['x_1', 'y_1', 'z_1']].values

train_df['dist_speedup'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test_df['dist_speedup'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)


# In[38]:


# dist_speedup and atom positions are added to both train_df and test_df
train_df.head()


# In[39]:


grid = sns.FacetGrid(train_df[['type', 'dist_speedup']], col='type', hue='type', col_wrap=4)
grid.map(sns.distplot, 'dist_speedup')


# In[40]:


len(train_df)


# In[41]:


train_df_sampled = train_df.sample(10000)


# In[42]:


sns.scatterplot(x="dist_speedup", y="scalar_coupling_constant", hue='type', data=train_df_sampled)


# In[43]:


train_test_df = pd.concat([train_df, test_df], axis=0, sort=True)
train_test_df['coupling_type_id'] = train_test_df['type'].map(
    {'1JHC': 0, '1JHN': 1, '2JHH': 2, '2JHN': 3, '2JHC': 4, '3JHH': 5, '3JHC': 6, '3JHN': 7})


# In[44]:


import chainer

from chainer_chemistry.dataset.preprocessors.common import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.weavenet_preprocessor import construct_atom_type_vec, construct_formal_charge_vec,     construct_partial_charge_vec, construct_atom_ring_vec, construct_hybridization_vec, construct_hydrogen_bonding,     construct_aromaticity_vec, construct_num_hydrogens_vec
from chainer_chemistry.dataset.preprocessors.weavenet_preprocessor import construct_pair_feature
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler

import os
import numpy
from tqdm import tqdm_notebook as tqdm


# In[45]:


num_max_atoms = 29
atom_list = ['H', 'C', 'N', 'O', 'F']
include_unknown_atom = False

# Each feature can be calculated using functions provided by chainer-chemistry using RDKit `mol` instance
construct_atomic_number_array(mol)
atom_type_vec = construct_atom_type_vec(
    mol, num_max_atoms, atom_list=atom_list,
    include_unknown_atom=include_unknown_atom)
formal_charge_vec = construct_formal_charge_vec(mol, num_max_atoms)
partial_charge_vec = construct_partial_charge_vec(mol, num_max_atoms)
atom_ring_vec = construct_atom_ring_vec(mol, num_max_atoms)
hybridization_vec = construct_hybridization_vec(mol, num_max_atoms)
hydrogen_bonding = construct_hydrogen_bonding(mol, num_max_atoms)
aromaticity_vec = construct_aromaticity_vec(mol, num_max_atoms)


# In[46]:


# Each feature is in shape (atom_id, feature_dim), where atom_id is padded to "num_max_atoms" size when atom is not exist.
atom_type_vec.shape, formal_charge_vec.shape, partial_charge_vec.shape, atom_ring_vec.shape, hybridization_vec.shape, hydrogen_bonding.shape, aromaticity_vec.shape


# In[47]:


WEAVE_DEFAULT_NUM_MAX_ATOMS=20
def construct_atom_feature(mol, add_Hs,
                           num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS,
                           atom_list=None, include_unknown_atom=False):
    """construct atom feature

    Args:
        mol (Mol): mol instance
        add_Hs (bool): if the `mol` instance was added Hs, set True.
        num_max_atoms (int): number of max atoms
        atom_list (list): list of atoms to extract feature. If None, default
            `ATOM` is used as `atom_list`
        include_unknown_atom (bool): If False, when the `mol` includes atom
            which is not in `atom_list`, it will raise
            `MolFeatureExtractionError`.
            If True, even the atom is not in `atom_list`, `atom_type` is set
            as "unknown" atom.

    Returns (numpy.ndarray): 2 dimensional array. First axis size is
        `num_max_atoms`, representing each atom index.
        Second axis for feature.

    """
    atom_type_vec = construct_atom_type_vec(
        mol, num_max_atoms, atom_list=atom_list,
        include_unknown_atom=include_unknown_atom)
    # TODO(nakago): Chilarity
    formal_charge_vec = construct_formal_charge_vec(
        mol, num_max_atoms=num_max_atoms)
    partial_charge_vec = construct_partial_charge_vec(
        mol, num_max_atoms=num_max_atoms)
    atom_ring_vec = construct_atom_ring_vec(
        mol, num_max_atoms=num_max_atoms)
    hybridization_vec = construct_hybridization_vec(
        mol, num_max_atoms=num_max_atoms)
    hydrogen_bonding = construct_hydrogen_bonding(
        mol, num_max_atoms=num_max_atoms)
    aromaticity_vec = construct_aromaticity_vec(
        mol, num_max_atoms=num_max_atoms)
    if add_Hs:
        num_hydrogens_vec = construct_num_hydrogens_vec(
            mol, num_max_atoms=num_max_atoms)
        feature = numpy.hstack((atom_type_vec, formal_charge_vec,
                                partial_charge_vec, atom_ring_vec,
                                hybridization_vec, hydrogen_bonding,
                                aromaticity_vec, num_hydrogens_vec))
    else:
        feature = numpy.hstack((atom_type_vec, formal_charge_vec,
                                partial_charge_vec, atom_ring_vec,
                                hybridization_vec, hydrogen_bonding,
                                aromaticity_vec))
    return feature


# In[48]:


pair_feature = construct_pair_feature(mol, num_max_atoms=num_max_atoms)


# In[49]:


# Each feature is in shape (atom_id*atom_id, feature_dim).
pair_feature.shape


# In[50]:



def to_ndarray(feature_list):
    feat_array = np.empty(len(feature_list), dtype=np.ndarray)
    feat_array[:] = feature_list[:]
    return feat_array

def create_features(mol_names):
    atom_array_list = []
    adj_array_list = []
    to_array_list = []
    from_array_list = []
    label_array_list = []
    coupling_type_id_array_list = []

    for mol_name in tqdm(mol_names):
        mol = mol_from_xyz(input_dir/f'structures/{mol_name}.xyz')

        atom_array = construct_atom_feature(mol, add_Hs=True,
                                            num_max_atoms = num_max_atoms, atom_list=atom_list,
                                            include_unknown_atom=False)
        pair_feature = construct_pair_feature(
            mol, num_max_atoms=num_max_atoms)                                              
#         atom_ids = construct_atomic_number_array(mol)
#         df_mol = structure_df.loc[structure_df['molecule_name'] == mol_name, :]
#         atom_id_array = df_mol['atom_id'].values.astype(np.int32)
#         assert np.alltrue(atom_ids == atom_id_array), print(atom_ids, atom_id_array)

#         xyz_array = df_mol[['x', 'y', 'z']].values
#         dist_array = np.linalg.norm(xyz_array[:, None, :] - xyz_array[None, :, :], axis=2).astype(np.float32)

        df_mol2 = train_test_df.loc[train_test_df['molecule_name'] == mol_name, :]
        from_array = df_mol2['atom_index_0'].values.astype(np.int32)
        to_array = df_mol2['atom_index_1'].values.astype(np.int32)
        label_array = df_mol2['scalar_coupling_constant'].values.astype(np.float32)
        #label_array = df_mol2['scalar_coupling_constant_scaled'].values.astype(np.float32)
        coupling_type_id_array = df_mol2['coupling_type_id'].values.astype(np.int32)

        atom_array_list.append(atom_array)
        adj_array_list.append(pair_feature)
        from_array_list.append(from_array)
        to_array_list.append(to_array)
        coupling_type_id_array_list.append(coupling_type_id_array)
        label_array_list.append(label_array)
    return tuple(map(to_ndarray, [atom_array_list, adj_array_list, from_array_list, to_array_list, coupling_type_id_array_list, label_array_list]))


# In[51]:


# For debugging

k = create_features(train_mol_names[:2])


# In[52]:





# In[52]:


def load_dataset(filepath):
    if not os.path.exists(filepath):
        return None
    load_data = numpy.load(filepath, allow_pickle=True)
    result = []
    i = 0
    while True:
        key = 'arr_{}'.format(i)
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return NumpyTupleDataset(*result)

with timer('train'):
    train = load_dataset('../input/champs-weave-dataset/train_data_v2.npz')
with timer('test'):
    test = load_dataset('../input/champs-weave-dataset/test_data_v2.npz')


# In[53]:


atom_array, dist_array, from_array, to_array, coupling_type_id_array, label_array = train[0]


# In[54]:


atom_array.shape, dist_array.shape


# In[55]:


from_array, to_array, label_array, coupling_type_id_array


# In[56]:


from chainer.dataset.convert import _concat_arrays, to_device
from itertools import chain

def concat_with_padding(batch, device, i, padding):
    return to_device(device, _concat_arrays([example[i] for example in batch], padding))


def flatten_array(list_of_list, dtype=np.float32):
    flatten_list = list(chain.from_iterable(list_of_list))
    return np.array(flatten_list, dtype=dtype)

class EdgeConverter:

    def __init__(self, return_label=True):
        self.return_label = return_label

    def __call__(self, batch, device=None):
        # atom_array, dist_array, from_array, to_array, coupling_type_id_array, label_array, 
        atom = concat_with_padding(batch, device, 0, padding=None)
        adj = concat_with_padding(batch, device, 1, padding=None)
        from_array = to_device(device, flatten_array([example[2] for example in batch], dtype=np.int32))
        to_array = to_device(device,flatten_array([example[3] for example in batch], dtype=np.int32))
        coupling_type = to_device(device, flatten_array([example[4] for example in batch], dtype=np.int32))
        label = to_device(device, flatten_array([example[5] for example in batch], dtype=np.float32))
        batch_indices = to_device(device, flatten_array([np.ones(len(example[2]), dtype=np.int32) * i for i, example in enumerate(batch)], dtype=np.int32))

        if self.return_label:
            return atom, adj, from_array, to_array, coupling_type, batch_indices, label[:, None]
        else:
            return atom, adj, from_array, to_array, coupling_type, batch_indices


# In[57]:


converter = EdgeConverter()
#print('0', train[0])
#print('1', train[1])
batch = [train[0], train[1]]

input_data = converter(batch, device=-1)
atom, adj, from_array, to_array, coupling_type, batch_indices, label = input_data
print('input_data:', atom.shape, adj.shape, from_array.shape, to_array.shape, coupling_type.shape, batch_indices.shape, label.shape)


# In[58]:


class CouplingTypeScaler(chainer.Link):

    def __init__(self):
        super(CouplingTypeScaler, self).__init__()
        self.mean = None
        self.register_persistent('mean')
        self.std = None
        self.register_persistent('std')

        
    def set_param(self, mean, std):
        self.mean = mean
        self.std = std
        
    def inverse_transform(self, x, coupling_indices):
        assert x.ndim == 2, x.shape
        assert x.shape[1] == 1, x.shape
        assert x.shape[0] == len(coupling_indices)
        x = x * self.std[coupling_indices][:, None] + self.mean[coupling_indices][:, None]
        return x


# In[59]:


# scale each scalar couplint constant
scaler = StandardScaler()

mean_list = []
std_list = []

types =  ['1JHC', '1JHN',  '2JHH', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']
for target_type in types:
    print('scaling type', target_type)
    label = train_df.loc[train_df['type'] == target_type, 'scalar_coupling_constant'].values
    train_df.loc[train_df['type'] == target_type, 'scalar_coupling_constant_scaled'] = scaler.fit_transform(label)
    print('mean', scaler.mean[0], 'std', scaler.std[0])
    mean_list.append(scaler.mean[0])
    std_list.append(scaler.std[0])

coupling_type_scaler = CouplingTypeScaler()
coupling_type_scaler.set_param(
    np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32))


# In[60]:


import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.config import WEAVE_DEFAULT_NUM_MAX_ATOMS
from chainer_chemistry.links.readout.general_readout import GeneralReadout
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID

from chainer_chemistry.models.weavenet import LinearLayer, WeaveModule

class WeaveNet(chainer.Chain):
    """WeaveNet implementation

    Args:
        weave_channels (list): list of int, output dimension for each weave
            module
        hidden_dim (int): hidden dim
        n_atom (int): number of atom of input array
        n_sub_layer (int): number of layer for each `AtomToPair`, `PairToAtom`
            layer
        n_atom_types (int): number of atom id
        readout_mode (str): 'sum' or 'max' or 'summax'
    """

    def __init__(self, weave_channels=None, hidden_dim=16,
                 n_atom=WEAVE_DEFAULT_NUM_MAX_ATOMS,
                 n_sub_layer=1, n_atom_types=MAX_ATOMIC_NUM,
                 readout_mode='sum'):
        weave_channels = weave_channels or WEAVENET_DEFAULT_WEAVE_CHANNELS
        weave_module = [
            WeaveModule(n_atom, c, n_sub_layer, readout_mode=readout_mode)
            for c in weave_channels
        ]

        super(WeaveNet, self).__init__()
        with self.init_scope():
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
            self.weave_module = chainer.ChainList(*weave_module)
            # self.readout = GeneralReadout(mode=readout_mode)
        self.readout_mode = readout_mode

    def __call__(self, atom_x, pair_x, train=True):
        if atom_x.dtype == self.xp.int32:
            # atom_array: (minibatch, atom)
            atom_x = self.embed(atom_x)

        for i in range(len(self.weave_module)):
            atom_x, pair_x = self.weave_module[i].forward(atom_x, pair_x)
#             if i == len(self.weave_module) - 1:
#                 # last layer, only `atom_x` is needed.
#                 atom_x = self.weave_module[i].forward(atom_x, pair_x,
#                                                       atom_only=True)
#             else:
#                 # not last layer, both `atom_x` and `pair_x` are needed
#                 atom_x, pair_x = self.weave_module[i].forward(atom_x, pair_x)
                
        # x = self.readout(atom_x, axis=1)
        # return x
        return pair_x


# In[61]:


import chainer
import chainer.links as L
import chainer.functions as F

from chainer_chemistry.models.mlp import MLP


def calc_pair_feature(h1, h2):
    # Any permutation invariant is okay
    # calc_pair_feature(h1, h2) == calc_pair_feature(h2, h1)
    return F.concat([h1 + h2, h1 * h2], axis=1)


class EdgePredictor(chainer.Chain):

    def __init__(self, graph_conv, coupling_type_scaler=None, num_coupling_type=8, edge_embed_dim=16):
        super(EdgePredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            self.coupling_type_scaler = coupling_type_scaler
            self.embed = L.EmbedID(num_coupling_type, edge_embed_dim)
            self.mlp = MLP(out_dim=1, hidden_dim=16, n_layers=2)

    def forward(self, atom, adj, from_array, to_array, coupling_array, batch_indices):
        batchsize, num_node, ch = atom.shape
        # (batch, node*node, ch)
        h = self.graph_conv(atom, adj)
        assert h.shape[:2] == (batchsize, num_node*num_node)
        assert h.ndim == 3

        indices = num_node * from_array + to_array
        h = h[batch_indices, indices]
        h_coupling = self.embed(coupling_array)

        # (edge, ch)
        h = F.concat([h, h_coupling], axis=1)
        h = self.mlp(h)
        # (edge, 1)
        h = self.coupling_type_scaler.inverse_transform(h, coupling_array)
        return h


# In[62]:


import copy
from logging import getLogger

import numpy

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer.training.extensions import Evaluator
from chainer.training.extensions import Evaluator


def _get_1d_numpy_array(v):
    """Convert array or Variable to 1d numpy array

    Args:
        v (numpy.ndarray or cupy.ndarray or chainer.Variable): array to be
            converted to 1d numpy array

    Returns (numpy.ndarray): Raveled 1d numpy array

    """
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v).ravel()



class LogMAEEvaluator(Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, 
                 name=None, logger=None):
        super(LogMAEEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.name = name
        self.logger = logger or getLogger()

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = []
        t_total = []
        coupling_total = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train',
                                                                  False):
                y = eval_func(*in_arrays[:-1])
            t = in_arrays[-1]
            coupling = in_arrays[-2]
            y_data = _get_1d_numpy_array(y)
            t_data = _get_1d_numpy_array(t)
            coupling_data = _get_1d_numpy_array(coupling)
            y_total.append(y_data)
            t_total.append(t_data)
            coupling_total.append(coupling_data)

        y_total = numpy.concatenate(y_total).ravel()
        t_total = numpy.concatenate(t_total).ravel()
        coupling_total = numpy.concatenate(coupling_total).ravel()
        # metrics_value = self.metrics_fun(y_total, t_total)

        metrics_dict = {}
        num_coupling = 8
        num_valid_coupling = 0
        log_mae_mean = 0
        for i in range(num_coupling):  # 8 is num_coupling type
            this_coupling = coupling_total == i
            if np.sum(this_coupling) == 0:
                continue
            
            log_mae = np.log(np.mean(np.abs(y_total[this_coupling] - t_total[this_coupling])))
            log_mae_mean += log_mae
            num_valid_coupling += 1
            metrics_dict[f'logmae_{i}'] = log_mae
            #print('log_mae', i, log_mae, np.sum(this_coupling == i))
        log_mae_mean /= num_valid_coupling
        metrics_dict['logmae_mean'] = log_mae_mean
        
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(metrics_dict, self._targets['main'])
        return observation


# In[63]:


import numpy
import os

from chainer.datasets import split_dataset_random
from chainer import functions as F
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions as E

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.dataset.splitters.random_splitter import RandomSplitter


# In[64]:


num_max_atoms = 29

graph_conv = WeaveNet(
    weave_channels=[32, 32], hidden_dim=32,
    n_atom=num_max_atoms,
    n_sub_layer=2, n_atom_types=MAX_ATOMIC_NUM)
predictor = EdgePredictor(graph_conv, coupling_type_scaler)


# In[65]:



batchsize = 128
#device = -1  # Use CPU
device = 0    # Use GPU, much faster!
epoch = 3
out = 'results'
seed = 0


#splitter = RandomSplitter()
#train_data, valid_data = splitter.train_valid_split(train, return_index=False, frac_train=0.9, frac_val=0.1)
train_data, valid_data = split_dataset_random(train, int(len(train) * 0.9), seed)
train_iter = iterators.SerialIterator(train_data, batchsize)
valid_iter = iterators.SerialIterator(valid_data, batchsize, repeat=False,
                                      shuffle=False)

regressor = Regressor(predictor, lossfun=F.mean_squared_error,
                      #metrics_fun=metrics_fun,
                      device=device)
# Set up the optimizer.
optimizer = optimizers.Adam()
optimizer.setup(regressor)
converter.return_label = True

# Set up the updater.
updater = training.StandardUpdater(train_iter, optimizer, device=device,
                                   converter=converter)

# Set up the trainer.
trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
# trainer.extend(E.Evaluator(valid_iter, regressor, device=device,
#                            converter=converter))
trainer.extend(LogMAEEvaluator(valid_iter, regressor, device=device,
                               converter=converter, name='val', eval_func=predictor))
trainer.extend(E.snapshot(), trigger=(epoch, 'epoch'))
trainer.extend(E.LogReport(trigger=(1, 'epoch')))  # (100, 'iteration') can be used for frequent logging.
try:
    # https://github.com/grafi-tt/chaineripy
    # Override PrintReport & ProgressBar to use chaineripy on jupyter notebook
    from chaineripy.extensions import PrintReport, ProgressBar
    trainer.extend(PrintReport([
    'epoch', 'main/loss', 'validation/main/loss',
    'val/main/logmae_mean', 'val/main/logmae_0', 'val/main/logmae_1', 'val/main/logmae_2', 'val/main/logmae_3', 'val/main/logmae_4',
    'val/main/logmae_5', 'val/main/logmae_6', 'val/main/logmae_7',
    'elapsed_time']))
    trainer.extend(ProgressBar())
except ImportError:
    trainer.extend(E.PrintReport([
    'epoch', 'main/loss', 'validation/main/loss',
    'val/main/logmae_mean', 'val/main/logmae_0', 'val/main/logmae_1', 'val/main/logmae_2', 'val/main/logmae_3', 'val/main/logmae_4',
    'val/main/logmae_5', 'val/main/logmae_6', 'val/main/logmae_7',
    'elapsed_time']))
    trainer.extend(E.ProgressBar())


trainer.run()


# In[66]:


# Prediction for test label

converter.return_label = False
test_pred = regressor.predict(test, batchsize=128, converter=converter)


# In[67]:


test_pred.shape, test_pred.dtype


# In[68]:


# Check test prediction distribution

sns.distplot(test_pred)


# In[69]:


sample_submission_df.head()


# In[70]:


np.alltrue(test_df.id.values == sample_submission_df.id.values)


# In[71]:


sample_submission_df.shape


# In[72]:


# Assign test prediction to submit format.
sample_submission_df['scalar_coupling_constant'] = test_pred


# In[73]:


sample_submission_df.head()


# In[74]:


# Save to csv file and we are ready to submit!

sample_submission_df.to_csv('submission.csv', index=False)
print('saved to submission.csv')


# In[75]:





# In[75]:





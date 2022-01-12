#!/usr/bin/env python
from itertools import combinations
import pandas as pd
import networkx as nx
import os, pickle
import sympy as sp

# This script takes the conversion factors found in the spreadsheet
# ConversionFactors.xlsx and generates the additional factors that
# these ones implies.
# As a final result a dictionary is produced and saved in a pickle.

def generate_unit_matrix():
    if os.path.exists('./conversion_facts.pkl'):
        print("conversion_facts.pkl already exists, delete if you want to generate it again.")
        return None
    if not os.path.exists('./ConversionFactors.xlsx'):
        print("ConversionFactors.xlsx not found, exiting...")
        return None
    print("Reading values from spreadsheet...")
    conversions = pd.read_excel('./ConversionFactors.xlsx',None)
    conversion_facts = {}
    for conversion_type in conversions:
        these_factors = conversions[conversion_type]
        these_dict = dict(zip(list(map(tuple,these_factors[['Source','Destination']].values.tolist())),
              list(map(lambda x: x[0],these_factors[['Factor']].values.tolist()))))
        conversion_facts.update(these_dict)
    cfacts = conversion_facts
    unitgraph = nx.DiGraph()
    allunits = set()
    print("Creating the weighted graph...")
    for k,v in cfacts.items():
        unitgraph.add_node(k[0])
        unitgraph.add_node(k[1])
        v = sp.S(v)
        unitgraph.add_weighted_edges_from([(k[0],k[1],v)])
        unitgraph.add_weighted_edges_from([(k[1],k[0],1/v)])
        allunits.add(k[0])
        allunits.add(k[1])
    allunits = list(allunits)
    allpairs = list(combinations(allunits,2))
    print("Filling in the holes...")
    for pair in allpairs:
        try:
            path = nx.shortest_path(unitgraph,*pair)
        except:
            path = []
        if len(path) == 0:
            continue
        fromto = (path[0],path[-1])
        backto = (path[-1], path[0])
        total_factor = sp.S(1)
        for node_idx in range(len(path)-1):
            nodeA = path[node_idx]
            nodeB = path[node_idx+1]
            total_factor *= unitgraph[nodeA][nodeB]['weight']
        if fromto not in cfacts.keys():
            cfacts[fromto] = float(total_factor)
        if backto not in cfacts.keys():
            cfacts[backto] = float(1/total_factor)
    for k,v in cfacts.items():
        unitgraph.add_node(k[0])
        unitgraph.add_node(k[1])
        v = sp.S(v)
        unitgraph.add_weighted_edges_from([(k[0],k[1],v)])
        unitgraph.add_weighted_edges_from([(k[1],k[0],1/v)])
    print("Saving to pickle...")
    pickle.dump(cfacts,open('./conversion_facts.pkl','wb'))
    print("conversion_facts.pkl succesfully created!")
    return cfacts

if __name__ == '__main__':
    generate_unit_matrix()

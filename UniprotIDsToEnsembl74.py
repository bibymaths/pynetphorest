#!/usr/bin/env python

"""
To be used to convert UniprotIDs to Ensembl74 ENSPs for use with NetworKIN.py

usage Mouse python UniprotIDsToEnsembl74.py -o 10090 ResultsFile.txt

usage Human python UniprotIDsToEnsembl74.py -o 9606 ResultsFile.txt

"""

import argparse
import os
import csv
import sys

parser = argparse.ArgumentParser(description ='Generate NetworKIN Phosphosites Input File')

parser.add_argument('-o', "--organism", type=str,required = True, help="Declare whether data is Mouse (10090) or Human (9609)")
parser.add_argument('-r', "--results", type =str, required = True, help="Provide relative path to results file")
parser.add_argument('-f', "--fasta", type=str, required=True, help="Provide relative path to results file")

args = parser.parse_args()

sys.stderr.write("Path: %s\n"%os.path.dirname(os.path.abspath(__file__)))
sys.stderr.write("results: %s\n"%args.results)

if args.organism == "10090" or args.organism == "9606":
    sys.stderr.write("Organism: %s\n"%args.organism)
else:
    raise Exception("Organism not recognized.\nMust be 9606 for human data or 10090 for mouse data.")

uniprotDict = {}
geneNameDict = {}

with open(os.path.dirname(os.path.abspath(__file__)) +"/Data/20181002_Biomart_IdentifierConversion_Human_Mouse.csv") as infile:
    reader = csv.DictReader(infile)
    for line in reader:
        if args.organism == "9606":
            if not line["Uniprot_Human"] in (None, ""):
                uniprotDict[line["Uniprot_Human"]] = line["Ensembl74_Human"]
            if not reader["HGNC"] in (None, ""):
                geneNameDict[line["HGNC"]] = line["Ensembl74_Human"]
        else:
            if not line["Uniprot_Mouse"] in (None, ""):
                uniprotDict[line["Uniprot_Mouse"]] = line["Ensembl74_Human"]
            if not line["MGI"] in (None, ""):
                geneNameDict[line["MGI"]] = line["Ensembl74_Human"]

sys.stderr.write("Length uniprotDict: %s\n"%len(uniprotDict))
sys.stderr.write("Length geneNameDict: %s\n"%len(geneNameDict))

with open(os.path.dirname(os.path.abspath(__file__)) + "/PhosphoSites.txt", 'w+') as outfile:
    writer = csv.writer(outfile , delimiter = "\t")
    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + args.results) as infile:
        reader = csv.DictReader(infile)
        for line in reader:
            if line["Majority Protein ID"].split('-')[0] in uniprotDict:
                writer.writerow([uniprotDict[line["Majority Protein ID"].split('-')[0]]]+[line["Position within protein"]]+[line["Amino Acid"]])
            else:
                if line["Gene name"] in geneNameDict:
                    writer.writerow([geneNameDict[line["Gene name"]]]+[line["Position within protein"]]+[line["Amino Acid"]])

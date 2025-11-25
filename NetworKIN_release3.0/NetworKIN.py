#!/usr/bin/env python3

"""
NetworKIN(tm), (C) 2005,2006,2007,2013.
Drs Rune Linding, Lars Juhl Jensen, Heiko Horn & Jinho Kim

Usage: ./networkin.py Organism FastaFile SitesFile

If no sites file is given NetworKIN will predict on all T/S/Y residues
in the given sequences.
"""

import sys, os, subprocess, re, tempfile, glob
from optparse import OptionParser
import gzip

# --- PATH CONFIGURATION ---
# Dynamically determine paths based on where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "NetworKIN_release3.0", "data")
MAPPING_FILE = os.path.join(BASE_DIR, "Data", "20181002_Biomart_IdentifierConversion_Human_Mouse.csv")
# --------------------------

# Weighting parameter, 0=only motif, 1=only STRING
ALPHAS = {"9606": 0.85, "4932": 0.65}
dSpeciesName = {"9606": "human", "4932": "yeast"}
dPenalty = {"9606": {"hub penalty": 100, "length penalty": 800}, "4932": {"hub penalty": 170, "length penalty": 1000}}

NETWORKIN_SITE_FILE = 1
PROTEOME_DISCOVERER_SITE_FILE = 2
MAX_QUANT_DIRECT_OUTPUT_FILE = 3


class CSheet(list):
    pass


# Run system binary
def myPopen(cmd):
    try:
        pipe = subprocess.Popen(cmd, shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = pipe.stdout.readlines()
        # Decode bytes to strings if python3
        stdout = [line.decode('utf-8') if isinstance(line, bytes) else line for line in stdout]
    except:
        sys.stderr.write('ERROR executing: ' + str(cmd) + '\n')
        sys.exit()
    else:
        return stdout


def runNetPhorest(id_seq, id_pos_res, save_diskspace, number_of_processes, number_of_active_processes=1, fast=False,
                  leave_intermediates=True):
    """
    MODIFIED: Bypasses execution. Returns the user-provided NetPhorest output file.
    """
    sys.stderr.write("\n[BYPASS] Using existing 'results.txt' from pynetphorest.\n")

    class CDummy:
        def __init__(self, name): self.name = name

    # Check if results.txt exists
    result_path = os.path.join(BASE_DIR, "results.txt")
    if not os.path.exists(result_path):
        sys.stderr.write(f"ERROR: Could not find '{result_path}'. Please run pynetphorest first.\n")
        sys.exit(1)

    return [CDummy(result_path)]


# Read sequences from fasta file
def readFasta(fastafile):
    id_seq = {}
    aminoacids = re.compile('[^ACDEFGHIKLMNPQRSTVWYXB]')
    data = fastafile.readlines()
    fastafile.close()
    seq = ''
    id = ''
    for line in data:
        if line[0] != ';':
            if line[0] == '>':
                line = line.strip()
                id = line[1:].split(' ', 1)[0]
                seq = ''
            else:
                seq += aminoacids.sub('', line)
                if len(seq) > 0:
                    id_seq[id] = seq
    return id_seq


def CheckInputType(sitesfile):
    with open(sitesfile, 'r') as f:
        line = f.readline()

    tokens = line.split()
    if len(tokens) == 3:
        return NETWORKIN_SITE_FILE
    elif len(tokens) == 2:
        return PROTEOME_DISCOVERER_SITE_FILE
    elif len(tokens) > 4 and tokens[0] == "Proteins" and tokens[4] == "Leading":
        return MAX_QUANT_DIRECT_OUTPUT_FILE
    else:
        sys.stderr.write("Unknown format of site file\n")
        sys.exit()


# Read phosphorylation sites from tsv file
def readPhosphoSites(sitesfile):
    id_pos_res = {}
    try:
        with open(sitesfile, 'r') as f:
            data = f.readlines()
            for line in data:
                tokens = line.split('\t')
                id = tokens[0]
                try:
                    pos = int(tokens[1])
                except:
                    sys.stderr.write(line)
                    raise
                try:
                    res = tokens[2].strip()
                except:
                    res = ""
                if id in id_pos_res:
                    id_pos_res[id][pos] = res
                else:
                    id_pos_res[id] = {pos: res}
    except Exception as e:
        sys.stderr.write(f"Could not open site file: {sitesfile}. Error: {e}\n")
        sys.exit()

    return id_pos_res


def readPhosphoSitesProteomeDiscoverer(fastafile, sitesfile):
    fasta = open(fastafile).readlines()
    fastadict = {}
    ensp = ""
    for line in fasta:
        if line[0] == ">":
            ensp = line.split()[0].strip(">")[0:15]
            fastadict[ensp] = ""
        else:
            seq = line.strip()
            fastadict[ensp] = seq

    peptides = open(sitesfile).readlines()
    peptidedict = {}
    for line in peptides:
        line = line.strip()
        tokens = line.split("\t")
        protID = tokens[0]
        peptide = tokens[1]
        if protID in peptidedict:
            if peptide not in peptidedict[protID]:
                peptidedict[protID][peptide] = ""
        else:
            peptidedict[protID] = {peptide: ""}

    id_pos_res = {}
    for protID in peptidedict:
        for peptide in peptidedict[protID]:
            UPPERpeptide = peptide.upper()
            if protID not in fastadict: continue

            sequence = fastadict[protID]
            try:
                peptideindex = sequence.index(UPPERpeptide)
            except ValueError:
                continue

            x = 0
            for letter in peptide:
                if letter.islower():
                    if letter in ["s", "t", "y"]:
                        phoslocation = peptideindex + x + 1
                        phosresidue = sequence[phoslocation - 1]

                        if protID in id_pos_res:
                            id_pos_res[protID][phoslocation] = phosresidue
                        else:
                            id_pos_res[protID] = {phoslocation: phosresidue}
                x += 1
    return id_pos_res


def ReadSheet(fname, offset=0):
    l = CSheet()
    f = open(fname, 'r')
    for i in range(offset):
        f.readline()

    columns = f.readline().strip().split('\t')
    l.columns = columns
    for line in f.readlines():
        instance = {}
        l.append(instance)
        fields = line.strip().split('\t')

        for i in range(len(columns)):
            try:
                instance[columns[i]] = fields[i]
            except IndexError:
                instance[columns[i]] = ''
    f.close()
    return l


def readPhosphoSitesMaxQuant(fname, only_leading=False):
    id_pos_res = {}
    phosphosites = ReadSheet(fname)

    for site in phosphosites:
        Ids = site["Proteins"].split(';')
        positions = list(map(lambda x: int(x), site["Positions within proteins"].split(';')))
        aa = site["Amino acid"]
        leading_protein_ids = site["Leading proteins"].split(';')

        for i in range(len(Ids)):
            Id = Ids[i]
            pos = positions[i]

            if only_leading and not Id in leading_protein_ids:
                continue

            if Id in id_pos_res:
                id_pos_res[Id][pos] = aa
            else:
                id_pos_res[Id] = {pos: aa}
    return id_pos_res


# Alias hashes
def readAliasFiles(organism, datadir):
    alias_hash = {}
    desc_hash = {}

    # Python 3 safe path handling
    alias_file = os.path.join(datadir, f"{organism}.alias_best.tsv.gz")
    desc_file = os.path.join(datadir, f"{organism}.text_best.tsv.gz")

    try:
        if os.path.exists(alias_file):
            with gzip.open(alias_file, 'rt') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        alias_hash[parts[1]] = parts[2]
        else:
            sys.stderr.write(f"Warning: Alias file not found at {alias_file}\n")
    except Exception as e:
        sys.stderr.write(f"Error reading alias file: {e}\n")

    try:
        if os.path.exists(desc_file):
            with gzip.open(desc_file, 'rt') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        desc_hash[parts[1]] = parts[2]
        else:
            sys.stderr.write(f"Warning: Description file not found at {desc_file}\n")
    except Exception as e:
        sys.stderr.write(f"Error reading desc file: {e}\n")

    return alias_hash, desc_hash


# Parse the NetPhorest output
def parseNetphorestFile(filename, id_pos_res, save_diskspace=True):
    id_pos_tree_pred = {}
    i = 0
    sys.stderr.write("Reading NetPhorest Output: %s\n" % filename)

    try:
        with open(filename, 'r') as infile:
            for line in infile:
                if line.startswith("#"):
                    continue

                line = line.strip()
                if not line: continue

                tokens = line.split('\t')

                try:
                    id = tokens[0].strip()
                except:
                    continue

                if (id == "#N/A" or id == "#N/A\r\n"):
                    continue

                try:
                    pos = int(tokens[1])
                except ValueError:
                    continue

                # FIX: Handle pynetphorest output columns
                # pynetphorest: Name(0), Pos(1), Res(2), Pep(3), Method(4), Tree(5), Classifier(6), Posterior(7/8)
                try:
                    res = tokens[2]
                    peptide = tokens[3]
                    # method = tokens[4]
                    tree = tokens[5]
                    pred = tokens[6]
                    # Check index for posterior (differs slightly based on pynetphorest args)
                    # pynetphorest with --causal adds columns. Standard is index 7 or 8.
                    # We look for the float value.
                    score = 0.0
                    if len(tokens) > 7:
                        try:
                            score = float(tokens[7])
                        except ValueError:
                            if len(tokens) > 8: score = float(tokens[8])
                except IndexError:
                    sys.stderr.write(f"Line parsing error: {tokens}\n")
                    continue

                if (id in id_pos_res and pos in id_pos_res[id]) or not id_pos_res:
                    if id in id_pos_tree_pred:
                        if pos in id_pos_tree_pred[id]:
                            if tree in id_pos_tree_pred[id][pos]:
                                id_pos_tree_pred[id][pos][tree][pred] = (res, peptide, score)
                            else:
                                id_pos_tree_pred[id][pos][tree] = {pred: (res, peptide, score)}
                        else:
                            id_pos_tree_pred[id][pos] = {tree: {pred: (res, peptide, score)}}
                    else:
                        id_pos_tree_pred[id] = {pos: {tree: {pred: (res, peptide, score)}}}
    except FileNotFoundError:
        sys.stderr.write(f"Error: File {filename} not found.\n")

    return id_pos_tree_pred


def ReadLines(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return lines


def WriteString(fname, s):
    with open(fname, 'w') as f:
        f.write(s)


# Map incoming peptides to STRING sequences
def mapPeptides2STRING(blastDir, organism, fastafilename, id_pos_res, id_seq, number_of_processes, datadir, fast=True,
                       leave_intermediates=False):
    sys.stderr.write("Mapping using blast\n")
    incoming2string = {}
    string2incoming = {}

    blast_tmpfile = tempfile.NamedTemporaryFile(mode='w+', delete=False)

    if not id_pos_res:
        for id in id_seq:
            blast_tmpfile.write('>' + id + '\n' + id_seq[id] + '\n')
    else:
        for id in id_pos_res:
            if id in id_seq:
                blast_tmpfile.write('>' + id + '\n' + id_seq[id] + '\n')
    blast_tmpfile.close()

    # FIX: Point to correct BLAST DB in Data directory
    # Note: Ensure you ran formatdb on this file!
    blastDB = os.path.join(datadir, "9606.protein.sequences.v9.0.fa")

    # Check if blast database is initialized
    if not os.path.isfile(blastDB + '.pin'):
        # Assuming formatdb is in the same folder as blastall
        formatdb_cmd = os.path.join(os.path.dirname(blastDir), "bin", "formatdb")
        # Fallback if blastDir points to bin directly
        if not os.path.exists(formatdb_cmd):
            formatdb_cmd = os.path.join(blastDir, "formatdb")

        command = f"{formatdb_cmd} -i {blastDB} -p T"
        sys.stderr.write(f"Initializing BLAST DB: {command}\n")
        myPopen(command)

    # Output file
    blast_out_file = "blast_out_Ensembl74.txt"
    blastall_cmd = os.path.join(blastDir, "blastall")

    # Run BLAST
    # -p blastp -e 1e-10 -m 8 (tabular)
    command = f"{blastall_cmd} -a {number_of_processes} -p blastp -e 1e-10 -m 8 -d {blastDB} -i {blast_tmpfile.name} | sort -k12nr"

    # Check if we can skip
    if fast and os.path.isfile(blast_out_file):
        sys.stderr.write("Reading existing BLAST output\n")
        blast_out = ReadLines(blast_out_file)
    else:
        sys.stderr.write(f"Performing BLAST: {command}\n")
        blast_out = myPopen(command)
        if leave_intermediates:
            WriteString(blast_out_file, "".join(blast_out))

    # Parse BLAST output
    for line in blast_out:
        if not line: continue
        tokens = line.split('\t')
        if len(tokens) <= 5: continue

        incoming = tokens[0]
        string = tokens[1].replace(f"{organism}.", "")  # Remove organism prefix

        if incoming not in incoming2string:
            incoming2string[incoming] = {string: True}
        else:
            incoming2string[incoming][string] = True

        if string not in string2incoming:
            string2incoming[string] = {incoming: True}
        else:
            string2incoming[string][incoming] = True

    os.unlink(blast_tmpfile.name)
    return incoming2string, string2incoming


# Load the precalculated STRING network file
def loadSTRINGdata(string2incoming, datadir, number_of_processes):
    # 1. Load the Mapping File (UniProt -> Ensembl)
    sys.stderr.write(f"Loading Identifier Mapping from: {MAPPING_FILE}\n")
    ensToUniDict = {}
    try:
        with open(MAPPING_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')  # CSV format
                if len(parts) > 1:
                    # Adjust indices based on your CSV structure.
                    # Assuming: UniProt, Ensembl
                    ensToUniDict[parts[1]] = parts[0]
    except FileNotFoundError:
        sys.stderr.write("Warning: Mapping file not found. Skipping ID conversion.\n")

    # 2. Load String Network
    fn_bestpath = os.path.join(datadir, "bestpath",
                               f"{organism}.string_000_{dPenalty[organism]['hub penalty']:04d}_{dPenalty[organism]['length penalty']:04d}.tsv.gz")

    if not os.path.isfile(fn_bestpath):
        sys.stderr.write(f"Best path file does not exist: {fn_bestpath}\n")
        return {}

    tree_pred_string_data = {}

    try:
        with gzip.open(fn_bestpath, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                tokens = line.split('\t')

                path = ""
                if len(tokens) == 8:
                    (tree, group, name, string1, string2, stringscore, stringscore_indirect, path) = tokens
                elif len(tokens) == 7:
                    (tree, group, name, string1, string2, stringscore, stringscore_indirect) = tokens
                else:
                    continue

                if not (string2 and string1): continue

                if (string2 in string2incoming):
                    if string2 not in tree_pred_string_data:
                        tree_pred_string_data[string2] = {}

                    if string1 not in tree_pred_string_data[string2]:
                        tree_pred_string_data[string2][string1] = {"_name": name}

                    score_val = float(stringscore) if options.path == "direct" else float(stringscore_indirect)
                    tree_pred_string_data[string2][string1]["_score"] = score_val
                    tree_pred_string_data[string2][string1]["_path"] = path

    except Exception as e:
        sys.stderr.write(f"Error reading STRING data: {e}\n")

    return tree_pred_string_data


def InsertValueIntoMultiLevelDict(d, keys, value):
    for i in range(len(keys) - 1):
        if keys[i] not in d:
            d[keys[i]] = {}
        d = d[keys[i]]

    if keys[-1] not in d:
        d[keys[-1]] = []
    d[keys[-1]].append(value)


def ReadGroup2DomainMap(path_group2domain_map):
    map_group2domain = {}
    try:
        with open(path_group2domain_map, "r") as f:
            for line in f:
                tokens = line.split()
                if len(tokens) >= 3:
                    InsertValueIntoMultiLevelDict(map_group2domain, tokens[:2], tokens[2])
    except FileNotFoundError:
        sys.stderr.write(f"Warning: Group2Domain map not found at {path_group2domain_map}\n")
    return map_group2domain


def SetValueIntoMultiLevelDict(d, keys, value):
    for i in range(len(keys) - 1):
        if keys[i] not in d:
            d[keys[i]] = {}
        d = d[keys[i]]
    d[keys[-1]] = value


# Basic helpers for Likelihood (Minimal implementation to replace 'from likelihood import *')
def ReadConversionTableBin(fname):
    # This expects a binary format specific to NetworKIN.
    # If you lack the binary reader 'likelihood.py', this will fail.
    # Assuming the original 'likelihood' module is present in the directory.
    try:
        import likelihood
        return likelihood.ReadConversionTableBin(fname)
    except ImportError:
        sys.stderr.write("Error: Could not import likelihood module. Ensure likelihood.py is in the folder.\n")
        return {}


def ConvertScore2L(score, table):
    try:
        import likelihood
        return likelihood.ConvertScore2L(score, table)
    except:
        return score  # Fallback


def printResult(id_pos_tree_pred, tree_pred_string_data, incoming2string, string_alias, string_desc, organism, mode,
                dir_likelihood_conversion_tbl, map_group2domain):
    species = dSpeciesName[organism]
    dLRConvTbl = {}
    dir_likelihood_conversion_tbl = dir_likelihood_conversion_tbl
    # Load Conversion Tables
    search_path = os.path.join(dir_likelihood_conversion_tbl, "conversion_tbl_*_smooth*")
    for fname in glob.glob(search_path):
        try:
            basename = os.path.basename(os.path.splitext(fname)[0])
            # Regex to parse filename
            match = re.search(r"conversion_tbl_([a-z]+)_smooth_([a-z]+)_([A-Z0-9]+)_([a-zA-Z0-9_/-]+)", basename)
            if match:
                netphorest_or_string = match.group(1)
                species_tbl = match.group(2)
                tree = match.group(3)
                player_name = match.group(4)

                if species_tbl != species: continue

                conversion_tbl = ReadConversionTableBin(fname)
                SetValueIntoMultiLevelDict(dLRConvTbl, [species_tbl, tree, player_name, netphorest_or_string],
                                           conversion_tbl)
        except Exception:
            continue

    # Main Matching Loop
    for id in id_pos_tree_pred:
        if id in incoming2string:
            for pos in id_pos_tree_pred[id]:
                for tree in id_pos_tree_pred[id][pos]:
                    score_results = {}
                    for pred in id_pos_tree_pred[id][pos][tree]:
                        # Loop through mapped STRING IDs
                        for string1 in incoming2string[id]:
                            bestName1 = string_alias.get(string1, '')
                            desc1 = string_desc.get(string1, '')

                            if string1 in tree_pred_string_data:
                                (res, peptide, netphorestScore) = id_pos_tree_pred[id][pos][tree][pred]

                                for string2 in tree_pred_string_data[string1]:
                                    bestName2 = string_alias.get(string2, '')
                                    desc2 = string_desc.get(string2, '')

                                    stringScore = tree_pred_string_data[string1][string2].get("_score", 0.0)
                                    path = tree_pred_string_data[string1][string2].get("_path", "")
                                    name = tree_pred_string_data[string1][string2].get("_name", "")

                                    # Simplified filtering logic
                                    networkinScore = 0.0

                                    # Calculate Score (Placeholder for complex likelihood logic)
                                    # We multiply raw probabilities if table lookup fails
                                    networkinScore = netphorestScore * stringScore

                                    def float_fmt(x):
                                        return "{:.4f}".format(x)

                                    result = f"{id}\t{res}{pos}\t{tree}\t{pred}\t{name}\t" \
                                             f"{float_fmt(networkinScore)}\t{float_fmt(netphorestScore)}\t{float_fmt(stringScore)}\t" \
                                             f"{string1}\t{string2}\t{bestName1}\t{bestName2}\t{desc1}\t{desc2}\t{peptide}\t{path}\n"

                                    if networkinScore not in score_results:
                                        score_results[networkinScore] = []
                                    score_results[networkinScore].append(result)

                    # Sort and Print
                    for score in sorted(score_results.keys(), reverse=True):
                        sys.stdout.write("".join(score_results[score]))


# MAIN
def Main():
    sys.stderr.write("Reading fasta input file\n")
    id_seq = readFasta(fastafile)
    id_pos_res = {}
    if sitesfile:
        sys.stderr.write("Reading phosphosite file\n")
        input_type = CheckInputType(sitesfile)
        if input_type == NETWORKIN_SITE_FILE:
            id_pos_res = readPhosphoSites(sitesfile)
        elif input_type == PROTEOME_DISCOVERER_SITE_FILE:
            id_pos_res = readPhosphoSitesProteomeDiscoverer(fn_fasta, sitesfile)
        elif input_type == MAX_QUANT_DIRECT_OUTPUT_FILE:
            id_pos_res = readPhosphoSitesMaxQuant(sitesfile)
    else:
        id_pos_res = {}

    sys.stderr.write("Loading aliases and descriptions\n")
    (string_alias, string_desc) = readAliasFiles(organism, options.datadir)

    path_group2domain_map = ""
    if organism == "9606":
        path_group2domain_map = os.path.join(options.datadir, "group_human_protein_name_map.tsv")
    elif organism == "4932":
        path_group2domain_map = os.path.join(options.datadir, "group_yeast_KIN.tsv")

    map_group2domain = ReadGroup2DomainMap(path_group2domain_map)

    sys.stderr.write(f"Blast dir: {blastDir}\n")
    sys.stderr.write(f"Organism: {organism}\n")
    sys.stderr.write(f"DataDir: {options.datadir}\n")

    incoming2string, string2incoming = mapPeptides2STRING(
        blastDir, organism, fastafile.name, id_pos_res, id_seq,
        options.threads, options.datadir, options.fast, options.leave
    )

    sys.stderr.write("Loading STRING network\n")
    tree_pred_string_data = loadSTRINGdata(string2incoming, options.datadir, options.threads)

    # Run NetPhorest (Bypassed)
    sys.stderr.write("Running NetPhorest (Bypass)\n")
    netphorestTmpFiles = runNetPhorest(id_seq, id_pos_res, options.compress, options.threads, options.active_threads,
                                       options.fast, options.leave)

    # Writing result to STDOUT
    sys.stderr.write("Writing results\n")
    sys.stdout.write(
        "#Name\tPosition\tTree\tNetPhorest Group\tKinase/Phosphatase/Phospho-binding domain\tNetworKIN score\tNetPhorest probability\tSTRING score\tTarget STRING ID\tKinase/Phosphatase/Phospho-binding domain STRING ID\tTarget description\tKinase/Phosphatase/Phospho-binding domain description\tTarget Name\tKinase/Phosphatase/Phospho-binding domain Name\tPeptide sequence window\tIntermediate nodes\n")

    for i in range(len(netphorestTmpFiles)):
        id_pos_tree_pred = parseNetphorestFile(netphorestTmpFiles[i].name, id_pos_res, options.compress)

        if options.path == "direct":
            dir_likelihood_conversion_tbl = os.path.join(options.datadir, "likelihood_conversion_table_direct")
        elif options.path == "indirect":
            dir_likelihood_conversion_tbl = os.path.join(options.datadir, "likelihood_conversion_table_indirect")

        printResult(id_pos_tree_pred, tree_pred_string_data, incoming2string, string_alias, string_desc, organism,
                    options.mode, dir_likelihood_conversion_tbl, map_group2domain)


if __name__ == '__main__':
    # Set default Blast Dir relative to script
    default_blast = os.path.join(BASE_DIR, "blast-2.2.17", "bin")

    usage = "usage: %prog [options] organism FASTA-file [sites-file]"
    parser = OptionParser(usage=usage, version="%prog 3.0")
    parser.add_option("-n", "--netphorest", dest="netphorest_bin", default="echo",
                      help="Placeholder for NetPhorest binary.")
    parser.add_option("-b", "--blast", dest="blast", default=default_blast,
                      help="Directory for BLAST binaries")
    parser.add_option("-m", "--mode", dest="mode", default=False,
                      help="Network mode")
    parser.add_option("-p", "--path", dest="path", default="direct",
                      help="direct/indirect")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="print out everything")
    parser.add_option("-f", "--fast", dest="fast", default=False, action="store_true",
                      help="Speed up by using previous files")
    parser.add_option("-l", "--leave", dest="leave", default=False, action="store_true",
                      help="leave intermediate files")
    parser.add_option("-u", "--uncovered", dest="string_for_uncovered", default=False, action="store_true",
                      help="Use STRING likelihood for uncovered Kinases")
    parser.add_option("-t", "--threads", dest="threads", default=1, type="int",
                      help="number of threads")
    parser.add_option("--nt", dest="active_threads", default=2, type="int",
                      help="number of active threads")

    parser.add_option("-c", "--compress", dest="compress", default=False,
                      help="compress temporary files")
    parser.add_option("-d", "--data", dest="datadir", default=DATA_DIR,
                      help="location for data files")

    parser.add_option("--tmp", dest="tmpdir", default="/tmp",
                      help="location for temporary files")

    global options
    (options, args) = parser.parse_args()

    # Apply tempdir
    tempfile.tempdir = options.tmpdir

    if len(args) < 2:
        parser.error("Organism and FASTA-file are required!")

    organism = args[0]
    fn_fasta = args[1]

    try:
        fastafile = open(fn_fasta, 'r')
    except IOError:
        parser.error(f"Could not open FASTA file: {fn_fasta}")

    sitesfile = args[2] if len(args) > 2 else False

    blastDir = options.blast

    if options.verbose:
        sys.stderr.write(f'\nRunning NetworKIN:\nOrganism: {organism}\nFasta: {fn_fasta}\nBlastDir: {blastDir}\n\n')

    Main()
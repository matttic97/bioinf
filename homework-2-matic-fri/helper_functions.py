import pickle
import numpy as np
from os import path
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from collections import defaultdict


def load(organism_id: str) -> SeqRecord:
    """Load the NCBI record, use cached files if possible."""
    if not path.exists(path.join("data", f"{organism_id}.pkl.gz")):
        with Entrez.efetch(db="nucleotide", rettype="gb", id=organism_id) as handle:
            record = SeqIO.read(handle, "gb")
            with open(path.join("data", f"{organism_id}.pkl.gz"), "wb") as f:
                pickle.dump(record, f)
    else:
        with open(path.join("data", f"{organism_id}.pkl.gz"), "rb") as f:
            record = pickle.load(f)

    return record


def find_cds_regions(record):
    cds_regions = []
    for feature in record.features:
        if feature.type == 'CDS':
            cds_regions.append((
                feature.qualifiers['product'],
                feature.qualifiers['translation'][0]
            ))
    return cds_regions
    

def find_genome_spike_protein(genome):
    for feature in genome.features:
        if feature.type == 'CDS':
            if 'gene' in feature.qualifiers and 'S' in feature.qualifiers['gene']:
                return feature.qualifiers['translation'][0]

            elif 'spike protein' in feature.qualifiers['product']:
                return feature.qualifiers['translation'][0]
    return None


def hamming_distance(string1, string2):
    distance = 0
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            distance += 1
    return distance


def reverse_hamming_distance(string1, string2):
    score = 0
    for i in range(len(string1)):
        if string1[i] == string2[i]:
            score += 1
    return score


def match_global_alig(covids_aminoacids, scoring_function):
    col_names = []
    distance_matrix = np.zeros([len(covids_aminoacids), len(covids_aminoacids)])   
    for index1, (name1, seq1) in enumerate(covids_aminoacids):
        col_names.append(name1)
        for index2 in range(index1 + 1, len(covids_aminoacids)):
            name2, seq2 = covids_aminoacids[index2]
            alig1, alig2, score = global_alignment(seq1, seq2, scoring_function, "")
            distance = hamming_distance(alig1, alig2)
            distance_matrix[index1][index2] = distance
    return col_names, distance_matrix


def global_alignment(seq1, seq2, scoring_function, indel_simbol="-"):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    """

    table = defaultdict(int)
    prev = {}

    #Initialize table
    seq1 = indel_simbol + seq1
    seq2 = indel_simbol + seq2

    table[0, 0] = 0
    for i in range(1, len(seq1)):
        table[i, 0] = table[i-1, 0] + scoring_function(seq1[i], indel_simbol)
    for j in range(1, len(seq2)):
        table[0, j] = table[0, j-1] + scoring_function(indel_simbol, seq2[j])

    #Calculate score
    for i in range(1, len(seq1)):
        for j in range(1, len(seq2)):
            table[i, j], prev[i, j] = max(
                (table[i-1, j-1] + scoring_function(seq1[i], seq2[j]), (i-1, j-1)),
                (table[i, j-1] + scoring_function(indel_simbol, seq2[j]), (i, j-1)),
                (table[i-1, j] + scoring_function(seq1[i], indel_simbol), (i-1, j))
            )

    finale_score = table[len(seq1)-1, len(seq2)-1]

    #Traceback
    alig1 = ""
    alig2 = ""
    i = len(seq1) - 1
    j = len(seq2) - 1
    while i != 0 and j != 0:
        if prev[i, j] == (i-1, j-1):
            alig1 = seq1[i] + alig1
            alig2 = seq2[j] + alig2
        elif prev[i, j] == (i-1, j):
            alig1 = seq1[i] + alig1
            alig2 = "-" + alig2
        elif prev[i, j] == (i, j-1):
            alig1 = "-" + alig1
            alig2 = seq2[j] + alig2
        i, j = prev[i, j]

    return alig1, alig2, finale_score


def match_local_alig(seq1, reference_genomes_cds_regions, scoring_function, indel_simbol="-"):
    scores = []
    for reference, value in reference_genomes_cds_regions.items():
        for (gene, seq2) in value:
            alig1, alig2, score = local_alignment(seq1, seq2, scoring_function, indel_simbol)
            scores.append((reverse_hamming_distance(alig1, alig2), len(alig1), len(seq1), len(seq2), reference, gene))
    max_score = max(scores)
    return [s for s in scores if s[0] == max_score[0]]


def local_alignment(seq1, seq2, scoring_function, indel_simbol="-"):
    """Local sequence alignment using the Smith-Waterman algorithm.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    """

    # Initialize table
    table = defaultdict(int)
    prev = {}
    seq1 = indel_simbol + seq1
    seq2 = indel_simbol + seq2

    # Calculate score
    max_score = (0, (0, 0))
    for i in range(1, len(seq1)):
        for j in range(1, len(seq2)):
            table[i, j], prev[i, j] = max(
                (table[i - 1, j - 1] + scoring_function(seq1[i], seq2[j]), (i - 1, j - 1)),
                (table[i, j - 1] + scoring_function(indel_simbol, seq2[j]), (i, j - 1)),
                (table[i - 1, j] + scoring_function(seq1[i], indel_simbol), (i - 1, j))
            )
            if table[i, j] < 0:
                table[i, j] = 0
            if table[i, j] > max_score[0]:
                max_score = (table[i, j], (i, j))

    # Traceback
    alig1 = ""
    alig2 = ""
    i = max_score[1][0]
    j = max_score[1][1]
    while table[i, j] != 0:
        if prev[i, j] == (i - 1, j - 1):
            alig1 = seq1[i] + alig1
            alig2 = seq2[j] + alig2
        elif prev[i, j] == (i - 1, j):
            alig1 = seq1[i] + alig1
            alig2 = "-" + alig2
        elif prev[i, j] == (i, j - 1):
            alig1 = "-" + alig1
            alig2 = seq2[j] + alig2
        i, j = prev[i, j]

    return alig1, alig2, max_score[0]

from collections import namedtuple
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2

GffEntry = namedtuple(
    "GffEntry",
    [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ],
)


GeneDict = Dict[str, GffEntry]


def read_gff(fname: str) -> Dict[str, GffEntry]:
    gene_dict = {}

    with open(fname) as f:
        for line in f:
            if line.startswith("#"):  # Comments start with '#' character
                continue

            parts = line.split("\t")
            parts = [p.strip() for p in parts]

            # Convert start and stop to ints
            start_idx = GffEntry._fields.index("start")
            parts[start_idx] = int(parts[start_idx]) - 1  # GFFs count from 1..
            stop_idx = GffEntry._fields.index("end")
            parts[stop_idx] = int(parts[stop_idx]) - 1  # GFFs count from 1..

            # Split the attributes
            attr_index = GffEntry._fields.index("attribute")
            attributes = {}
            for attr in parts[attr_index].split(";"):
                attr = attr.strip()
                k, v = attr.split("=")
                attributes[k] = v
            parts[attr_index] = attributes

            entry = GffEntry(*parts)

            gene_dict[entry.attribute["gene_name"]] = entry

    return gene_dict


def split_read(read: str) -> Tuple[str, str]:
    """Split a given read into its barcode and DNA sequence. The reads are
    already in DNA format, so no additional work will have to be done. This
    function needs only to take the read, and split it into the cell barcode,
    the primer, and the DNA sequence. The primer is not important, so we discard
    that.

    The first 12 bases correspond to the cell barcode.
    The next 24 bases corresond to the oligo-dT primer. (discard this)
    The reamining bases corresond to the actual DNA of interest.

    Parameters
    ----------
    read: str

    Returns
    -------
    str: cell_barcode
    str: mRNA sequence

    """
    return (read[0:12], read[36:])


def map_read_to_gene(read: str, ref_seq: str, genes: GeneDict) -> Tuple[str, float]:
    """Map a given read to a gene with a confidence score using Hamming distance.

    Parameters
    ----------
    read: str
        The DNA sequence to be aligned to the reference sequence. This should
        NOT include the cell barcode or the oligo-dT primer.
    ref_seq: str
        The reference sequence that the read should be aligned against.
    genes: GeneDict

    Returns
    -------
    gene: str
        The name of the gene (using the keys of the `genes` parameter, which the
        read maps to best. If the best alignment maps to a region that is not a
        gene, the function should return `None`.
    similarity: float
        The similarity of the aligned read. This is computed by taking the
        Hamming distance between the aligned read and the reference sequence.
        E.g. catac and cat-x will have similarity 3/5=0.6.


    """
    aligned = pairwise2.align.localxx(ref_seq, read)
    best_aligned = [v.score for v in pairwise2.align.localxx(ref_seq, read)]
    best_aligned = np.where(best_aligned == np.max(best_aligned))[0]
    
    best_score = ("", 0)
    for index in best_aligned:
        alignment = aligned[index]
        seq = alignment.seqB[alignment.start : alignment.end]
        score = alignment.score/len(seq)
        if score > best_score[1]:
            best_score = (seq[0], score, alignment.start, alignment.end)
            
    if best_score[0] in genes:
        gene = genes[best_score[0]]
        if gene.start == best_score[2] and gene.end == best_score[3]:
            return (gene.attribute['gene_name'], best_score[1])
    
    return (None, best_score[1])


def generate_count_matrix(
    reads: str,
    ref_seq: str,
    genes: GeneDict,
    similarity_threshold: float
) -> pd.DataFrame:
    """

    Parameters
    ----------
    reads: List[str]
        The list of all reads that will be aligned.
    ref_seq: str
        The reference sequence that the read should be aligned against.
    genes: GeneDict
    similarity_threshold: float

    Returns
    -------
    count_table: pd.DataFrame
        The count table should be an N x G matrix where N is the number of
        unique cell barcodes in the reads and G is the number of genes in
        `genes`. The dataframe columns should be to a list of strings
        corrsponding to genes and the dataframe index should be a list of
        strings corresponding to cell barcodes. Each cell in the matrix should
        indicate the number of times a read mapped to a gene in that particular
        cell.

    """
    scores = []
    for read in reads:
        split = split_read(read)
        maped_gene = map_read_to_gene(split[1], ref_seq, genes)
        if maped_gene[1] >= similarity_threshold:
            scores.append([split[0], maped_gene])

    scores = np.array(scores, dtype=object)

    N = np.unique(scores[:, 0])
    G = list(genes.keys())
    matrix = np.zeros((len(N), len(G)), dtype=int)

    for n in range(len(N)):
        for score in scores[scores[:, 0] == N[n], 1]:
            index = G.index(score[0])
            matrix[n, index] += 1
            
    return pd.DataFrame(
            index=N,
            columns=G,
            data=np.array(matrix),
        )


def filter_matrix(
    count_matrix: pd.DataFrame,
    min_counts_per_cell: int,
    min_counts_per_gene: int,
) -> pd.DataFrame:
    """Filter a matrix by cell counts and gene counts.
    The cell count is the total number of molecules sequenced for a particular
    cell. The gene count is the total number of molecules sequenced that
    correspond to a particular gene. Filtering statistics should be computed on
    the original matrix. E.g. if you filter out the genes first, the filtered
    gene molecules should still count towards the cell counts.

    Parameters
    ----------
    count_matrix: pd.DataFrame
    min_counts_per_cell: float
    min_counts_per_gene: float

    Returns
    -------
    filtered_count_matrix: pd.DataFrame

    """
    data = count_matrix.values
    columns_sum = data.sum(axis=0)
    rows_sum = data.sum(axis=1)
    
    col_del = np.where(columns_sum < min_counts_per_gene)[0]
    row_del = np.where(rows_sum < min_counts_per_cell)[0]
    data = np.delete(data, row_del, 0)
    data = np.delete(data, col_del, 1)
    
    index = np.delete(count_matrix.index, row_del, None)
    columns = np.delete(count_matrix.columns, col_del, None)
    
    return pd.DataFrame(
        index=index,
        columns=columns,
        data=np.array(data),
    )


            
            
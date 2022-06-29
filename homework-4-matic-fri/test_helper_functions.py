import unittest
from helper_functions import kmers, de_bruijn_graph, assemble_genome


class TestKmers(unittest.TestCase):
    def test_kmers_1(self):
        s = "TAATGCCATGGGATGTT"
        result = list(kmers(s, 3, 1))
        expected = [
            "TAA",
            "AAT",
            "ATG",
            "TGC",
            "GCC",
            "CCA",
            "CAT",
            "ATG",
            "TGG",
            "GGG",
            "GGA",
            "GAT",
            "ATG",
            "TGT",
            "GTT",
        ]
        self.assertEqual(expected, result)

    def test_kmers_2(self):
        s = "TAATGCCATGGGATGTT"
        result = list(kmers(s, 5, 1))
        expected = [
            "TAATG",
            "AATGC",
            "ATGCC",
            "TGCCA",
            "GCCAT",
            "CCATG",
            "CATGG",
            "ATGGG",
            "TGGGA",
            "GGGAT",
            "GGATG",
            "GATGT",
            "ATGTT",
        ]
        self.assertEqual(expected, result)


class TestDeBruijnGraph(unittest.TestCase):
    def test_graph_1(self):
        seq = "TAATGCCATGGGATGTT"

        result = de_bruijn_graph([seq], k=3)
        expected = {
            "AAT": ["ATG"],
            "ATG": ["TGC", "TGG", "TGT"],
            "CAT": ["ATG"],
            "CCA": ["CAT"],
            "GAT": ["ATG"],
            "GCC": ["CCA"],
            "GGA": ["GAT"],
            "GGG": ["GGA"],
            "TAA": ["AAT"],
            "TGC": ["GCC"],
            "TGG": ["GGG"],
            "TGT": ["GTT"],
        }

        self.assertEqual(expected, result)

    def test_graph_2(self):
        seq = "TAATGCCATGGGATGTT"

        result = de_bruijn_graph([seq, seq, seq], k=3)
        expected = {
            "AAT": ["ATG", "ATG", "ATG"],
            "ATG": ["TGC", "TGG", "TGT", "TGC", "TGG", "TGT", "TGC", "TGG", "TGT"],
            "CAT": ["ATG", "ATG", "ATG"],
            "CCA": ["CAT", "CAT", "CAT"],
            "GAT": ["ATG", "ATG", "ATG"],
            "GCC": ["CCA", "CCA", "CCA"],
            "GGA": ["GAT", "GAT", "GAT"],
            "GGG": ["GGA", "GGA", "GGA"],
            "TAA": ["AAT", "AAT", "AAT"],
            "TGC": ["GCC", "GCC", "GCC"],
            "TGG": ["GGG", "GGG", "GGG"],
            "TGT": ["GTT", "GTT", "GTT"],
        }

        self.assertEqual(expected, result)


class TestAssembleGenome(unittest.TestCase):
    def test_assembly_1(self):
        seq = "TAATGCCATGGGATGTT"
        k = 3

        assemblies = assemble_genome([seq], k=k)

        self.assertEqual(len(assemblies), 2)
        self.assertIn("TAATGCCATGGGATGTT", assemblies)
        self.assertIn("TAATGGGATGCCATGTT", assemblies)

    def test_assembly_2(self):
        seq = "TAATGCCATGGGATGTT"
        k = 5

        assemblies = assemble_genome([seq], k=k)

        self.assertEqual(len(assemblies), 1)
        self.assertIn("TAATGCCATGGGATGTT", assemblies)

    def test_assembly_3(self):
        seqs = ["TAATGCCATGGG", "GGGATGTT"]
        k = 3

        assemblies = assemble_genome(seqs, k=k)

        self.assertEqual(len(assemblies), 2)
        self.assertIn("TAATGCCATGGGATGTT", assemblies)
        self.assertIn("TAATGGGATGCCATGTT", assemblies)

    def test_assembly_4(self):
        seq = "she sells sea shells by the sea shore"
        k = 3

        assemblies = assemble_genome([seq], k=k)

        self.assertEqual(len(assemblies), 8)
        self.assertIn(seq, assemblies)

    def test_assembly_5(self):
        with open("data/rudolph.txt") as f:
            seq = f.read()

        k = 10

        assemblies = assemble_genome([seq], k=k)
        self.assertEqual(len(assemblies), 4)
        self.assertIn(seq, assemblies)


if __name__ == "__main__":
    unittest.main()

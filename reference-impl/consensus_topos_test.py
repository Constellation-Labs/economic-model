import functools
import unittest

from consensus_topos import *
from consensus_topos_utils import *


class TestConsensusToposEvolution(unittest.TestCase):

    def test_pullback_limits(self):
        con_top_a = ConsensusTopos(4)
        con_top_b = ConsensusTopos(3)
        with self.assertRaises(ValueError, msg="must have matching subspace topology"): con_top_a.pullback(con_top_b)

    def test_pullback(self):
        con_top_a = ConsensusTopos(4)
        closed_mapping_space = con_top_a.convolution()
        self.assertIsNotNone(closed_mapping_space,
                             msg="convolution should exist for outer product space of protocol complex with internal hom")

    def test_lattice_simplex_product(self):
        con_top_a = ConsensusTopos(4)
        l_0_triangulation = con_top_a.triangulate_vertices(con_top_a.U, 1)
        l_1_triangulation = con_top_a.triangulate_vertices(con_top_a.U, 2)
        self.assertTrue(all(len(i) == 2 for i in functools.reduce(operator.iconcat, l_1_triangulation, [])),
                        msg="should form edges between elements (triangles) of triangulated surfaces")
        self.assertTrue(all(len(i) == 2 for i in functools.reduce(operator.iconcat, l_0_triangulation, [])),
                        msg="should form edges out of points")

    def test_algebraic_topological_model(self):
        con_top_a = ConsensusTopos(4)
        l_4_space = con_top_a.action_complex()
        l_4_chain_complex = l_4_space.chain_complex()
        l_4_cochain_complex = l_4_space.chain_complex(cochain=True)
        # Protocol Complexes: Definition 5.1 https://repositum.tuwien.at/bitstream/20.500.12708/13126/2/Topology%20in%20Distributed%20Computing.pdf
        chain_contraction_d4, d4_simplicial_chain_complex = l_4_space.algebraic_topological_model()
        self.assertTrue(
            all(i == 0 for i in l_4_chain_complex.differential(1) * chain_contraction_d4.iota().in_degree(1)),
            msg="algebraic_topological_model should have vanishing homology <=> vanishing differential")
        self.assertTrue(
            all(i == 0 for i in
                l_4_cochain_complex.differential(1) * chain_contraction_d4.dual().iota().in_degree(1)),
            msg="algebraic_topological_model should have vanishing cohomology in the absence of sheafification")

    def test_graded_induced_homology(self):
        con_top_a = ConsensusTopos(4)
        closed_mapping_space = con_top_a.convolution()
        from sage.homology.homology_morphism import InducedHomologyMorphism
        self.assertIsInstance(closed_mapping_space, InducedHomologyMorphism)
        self.assertTrue(ascii_art(closed_mapping_space)._matrix[0] == 'Graded vector space morphism:')

    def test_protocol_configuration_complex(self):
        con_top_a = ConsensusTopos(4)
        from sage.homology.algebraic_topological_model import algebraic_topological_model_delta_complex
        random_normal_phi, M_random_normal = algebraic_topological_model_delta_complex(
            con_top_a.G.configuration_complex, GF(4))
        uniform_phi, M_uniform = algebraic_topological_model_delta_complex(con_top_a.U.configuration_complex, GF(4))
        homology_random_normal = M_random_normal.homology()
        cohomology_random_normal = M_random_normal.dual().homology()
        con_top_homology_uniform = M_uniform.homology()
        cohomology_uniform = M_uniform.dual().homology()
        self.assertTrue(homology_random_normal == con_top_homology_uniform,
                        msg="homologies should be equal")
        self.assertTrue(cohomology_random_normal == cohomology_uniform,
                        msg="cohomologies should be equal")

    def test_configuration_complex_coherence(self):
        con_top_a = ConsensusTopos(4)
        decoherence = con_top_a.coherence()
        ideal_partition_function = estimate_shannon_entropy(list(i[1] for i in con_top_a.U.configuration_simplex.set()))
        self.assertGreaterEqual(decoherence, ideal_partition_function,
                                msg="information gain of uniform partition always greater than random")

    def test_coherence_complex(self):
        con_top_a = ConsensusTopos(4)
        decoherence_complex = con_top_a.coherence_complex()
        ideal_complex = con_top_a.U.protocol_complex.cells()
        ideal_partition_function = {k: estimate_shannon_entropy(v) for k, v in ideal_complex.items()}
        ideal_info_gain = dict(
            [(k, ideal_partition_function.get(k, 0.0) + ideal_partition_function.get(k, 0.0)) for k in
             set(list(ideal_partition_function.keys()) + list(ideal_partition_function.keys()))])
        self.assertGreaterEqual(sum(decoherence_complex.keys()), sum(ideal_info_gain.keys()),
                                msg="coherence = U - G, U is uniform, G is disordered")

    def test_address_balance_transformation(self):
        con_top_a = ConsensusTopos(4)
        address_balance_space = con_top_a.address_balance_transformation()
        from sage.modules.vector_space_morphism import VectorSpaceMorphism
        self.assertIsInstance(address_balance_space, VectorSpaceMorphism)
        self.assertTrue('Vector space morphism' in ascii_art(address_balance_space)._matrix[0])

    def test_action_space(self):
        con_top_a = ConsensusTopos(4)
        action = con_top_a.action_complex()
        self.assertTrue(con_top_a.hom.codomain() == action,
                        msg="ConsensusTopos should form internal hom")


if __name__ == '__main__':
    unittest.main()

from consensus_topos_utils import *
from sage.homology.examples import RandomComplex
from sage.homology.simplicial_complex import lattice_paths


class ConsensusTopos():
    """
    The following constructs cellular action of the ConsensusTopos as well as homology/cohomology groups and their
    associated chain complexes as well as quadratic forms (9.1.1). It can further be extended to
    calculate stabilizers (see 10.2 https://pure.mpg.de/rest/items/item_3121687/component/file_3121688/content)

    This can further be extended symbollically using sage.Manifold and to construct markov transition matrices using a
    Hopf algebra (https://sci-hub.se/https://doi.org/10.1016/j.jcta.2015.02.001)

    note the free module construction turns set-theoretic products into tensor products Thus, it preserves algebraic
    objects (such as monoid objects https://ncatlab.org/nlab/show/free+module#as_a_monoidal_functor
    """

    def __init__(self, num_vertices):
        random_normal_complex, uniform_complex = protocol_configuration_simplices()
        self.base_category = HyperGraphCategory(ZZ)
        self.U = Protocol(self.base_category, num_vertices, uniform_complex)
        self.G = Protocol(self.base_category, num_vertices, random_normal_complex)
        self.hom = Hom(self.U.protocol_complex, self.action_complex())

    def market_matrix(self):
        """
        Compute Ellerman's market matrix out of price index.
        :return: (Sequence_generic, matrix) the lyapunov exponents and market matrix
        """
        partials_of_constraints = self.price_vector()
        mu = vector(RR, list(partials_of_constraints)) * vector(RR, list(partials_of_constraints))
        solutions_boundary = self.lagrangian(mu)
        del_f = list(solutions_boundary.values())
        g_vec = -matrix(RR, vector(RR, list(partials_of_constraints)))
        jacobian = g_vec.transpose() * g_vec
        # g_mat.eigenvalues() gives lyuponov exponents and when negative, the system is stable. Positive => chaotic
        lyapunov_spectrum = jacobian.eigenvalues()
        market_matrix = matrix([del_f[:-1]] + list(list(i) for i in jacobian))
        return (lyapunov_spectrum, market_matrix)

    def price_vector(self):
        """
        Calculate constraints from normalized cofactors of balance transformations
        :return: FreeModuleElement_generic_dense containing the price vector
        """
        constraint_matrix = self.address_balance_transformation().matrix()
        lagrange_hom = linear_transformation(matrix(RR, constraint_matrix.numpy()), side='right')
        normalized_cofactors = vector(RR, random_address_manifold())
        partials_of_constraints = lagrange_hom(normalized_cofactors)
        return partials_of_constraints

    def lagrangian(self, mu):
        """
        Calculate rate of transformation of marginal market data satisfying constraint mu
        :return: Dict of derived gradient
        """
        var('DAG, LTX, STAR, BTC')
        f = DAG + LTX + STAR + BTC
        f_DAG = diff(f, DAG)
        f_LTX = diff(f, LTX)
        f_STAR = diff(f, STAR)
        f_BTC = diff(f, BTC)
        gradf = [f_DAG, f_LTX, f_STAR, f_BTC]
        var('lamb')
        g = 1 / DAG + 1 / LTX + 1 / STAR + 1 / BTC
        # todo define gradg to incorporate chain data in protocol constraints M U p
        solns_boundary = solve(
            [f_DAG == lamb * mu, f_LTX == lamb * mu, f_STAR == lamb * mu, f_BTC == lamb * mu, g == 0],
            (DAG, LTX, STAR, BTC, lamb), solution_dict=True)

        return solns_boundary[0]

    def coherence(self):
        """
        Calculate disorder in schedule of configuration complex.
        :return: Relative entropy
        """
        ideal_partition_function = estimate_shannon_entropy(list(i[1] for i in self.U.configuration_simplex.set()))
        state_partition_function = estimate_shannon_entropy(list(i[1] for i in self.G.configuration_simplex.set()))
        return ideal_partition_function + state_partition_function

    def coherence_complex(self):
        """
        Calculate disorder between ideal and stateful simplicial complex. See https://arxiv.org/pdf/1603.07135.pdf

        note that the protocol complex is a rips complex as edges are formed from points that have a distance metric
        given by cryptographic hash.
        :return: Dict of relative entropy across ranks of complexes
        """
        state_complex = self.G.protocol_complex.cells()
        ideal_complex = self.U.protocol_complex.cells()
        coherence_complex = dict([(k, list(state_complex.get(k, [])) + list(ideal_complex.get(k, []))) for k in
                                  set(list(state_complex.keys()) + list(ideal_complex.keys()))])
        ideal_partition_function = {k: estimate_shannon_entropy(v) for k, v in ideal_complex.items()}
        state_partition_function = {k: estimate_shannon_entropy(v) for k, v in coherence_complex.items()}
        info_gain = dict([(k, ideal_partition_function.get(k, 0.0) + state_partition_function.get(k, 0.0)) for k in
                          set(list(ideal_partition_function.keys()) + list(state_partition_function.keys()))])
        return info_gain

    def address_balance_transformation(self):
        """
        Define address balance transformation as proper linear transformation, enforcing dimensionality for later
        calculations
        :return: VectorSpaceMorphism of rewards distribution
        """
        rewards_manifold = distro_rewards(random_balance_distribution())
        xdim, ydim = rewards_manifold.shape
        reward_transform = linear_transformation(RR ** xdim, RR ** ydim, matrix(RR, rewards_manifold))
        return reward_transform

    def triangulate_vertices(self, protocol, rank=1):
        vertex_hyper_edge = protocol.protocol_complex.cells()[rank]
        return lattice_paths(vertex_hyper_edge, vertex_hyper_edge)

    def pullback(self, other):
        """
        A morphism of simplicial complexes, which is the morphism from the space of the fiber product to the codomain.

        :param other: ConsensusTopos
        :return: SimplicialComplexMorphism of fiber product to codomain
        """
        f = vertex_map(self.hom)
        g = vertex_map(other.hom)
        pullback_bundle = self.hom(f).fiber_product(other.hom(g), rename_vertices=False)
        pullback_bundle.domain().set_immutable()
        pullback_bundle.codomain().set_immutable()
        return pullback_bundle

    def convolution(self):
        """
        a pullback of functions along this map makes an involution of the convolution algebra
        :return: Graded vector space morphism
        """
        fiber_pullback = self.pullback(self)
        return fiber_pullback.induced_homology_morphism()

    def action_complex(self, other_U=None):
        """
        rank 4 (3d + 1) state complex describing concurrent 3d processes (consensus)
        :param other_U:
        :return: SimplicialComplex U -> U x U
        """
        if other_U:
            product_space = self.U.protocol_complex.product(other_U.protocol_complex,
                                                            rename_vertices=False, is_mutable=False)
        else:
            product_space = self.U.protocol_complex.product(self.U.protocol_complex,
                                                            rename_vertices=False, is_mutable=False)
        product_space.set_immutable()
        return product_space

    def compose(self, other_topos):
        """
        Only consider finite ZZ topologies for now. This can be extended to U by handling non-commutativity via braiding
        logic
        :param other_topos:
        :return: the total complex of their protocols i.e. the chain complex of their chain complexes
        """
        other_protocol_chain_complex = other_topos.U.protocol_complex.chain_complex()
        total_complex = self.U.protocol_complex.chain_complex().tensor(other_protocol_chain_complex)
        return total_complex


class HyperGraphCategory(UniqueRepresentation, Field):
    def __init__(self, base):
        Field.__init__(self, base)

    def _repr_(self):
        return "HyperGraphCategory(%s)" % repr(self.base())

    def base_ring(self):
        return self.base().base_ring()

    def characteristic(self):
        return self.base().characteristic()


class Protocol(FieldElement):
    def __init__(self, parent, num_vertices, configuration_simplex):
        self.configuration_simplex = configuration_simplex
        self.configuration_complex = DeltaComplex({configuration_simplex: True})
        self.protocol_complex = RandomComplex(d=2, n=num_vertices)
        self.protocol_complex.set_immutable()
        FieldElement.__init__(self, parent)


def vertex_map(hom):
    """
    Dict mapping domain to codomain
    :param hom:
    :return: a dictionary with keys exactly the vertices of the domain and values vertices of the codomain
    """
    hom_f = dict(zip(hom.domain().vertices(), hom.codomain().vertices()))
    return hom_f


if __name__ == "__main__":
    con_top_a = ConsensusTopos(4)
    con_top_b = ConsensusTopos(4)
    con_top_a.address_balance_transformation()
    composed = con_top_a.compose(con_top_b)
    lyapunov_spectrum, market_matrix = con_top_a.market_matrix()
    print(ascii_art(composed))
    print(ascii_art(con_top_a.action_complex().chain_complex()))

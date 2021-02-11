import collections
from balance_transition import *


j_index_size = len(node_set)
base_ring = GF(j_index_size)
j_index_numeric = list(i for i in range(j_index_size))


def filter_valid_complex(chain_dict):
    dim = max(chain_dict)
    valid_chain_dict = chain_dict.copy()
    chain_tups = valid_chain_dict.copy()
    for n in chain_tups:
        chain_tups[n] = tuple(chain_tups[n])

    for d in range(dim, 1, -1):
        for s in chain_tups[d]:  # s is a d-simplex
            faces = chain_tups[d - 1]
            for j in range(d + 1):
                if not all(faces[s[j]][i] == faces[s[i]][j - 1] for i in range(j)):
                    valid_chain_dict[d].remove(s)
                    break

    return valid_chain_dict


def to_chain_dict(configuration_complex):
    new_data = {-1: ((),)}  # add the empty cell
    for dim in range(0, len(configuration_complex)):
        if isinstance(configuration_complex[dim], (list, tuple)):
            new_data[dim] = configuration_complex[dim]
    return new_data


def estimate_shannon_entropy(element_sequence):
    m = len(element_sequence)
    bases = collections.Counter([tmp_base for tmp_base in element_sequence])

    shannon_entropy_value = 0
    for base in bases:
        # number of residues
        n_i = bases[base]
        # n_i (# residues type i) / M (# residues in column)
        p_i = n_i / float(m)
        entropy_i = p_i * (math.log(p_i, 2))
        shannon_entropy_value += entropy_i

    return shannon_entropy_value * -1


def protocol_configuration_simplices():
    # Input Complex for decision problem, configurations are flattened into float values
    # see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.9562&rep=rep1&type=pdf
    random_normal_simplex_vertices = list(zip(list(i for i in range(j_index_size)), random_rewards_distribution(j_index_size)))
    uniforml_simplex_vertices = list((i, 1/j_index_size) for i in range(j_index_size))
    random_normal_simplex = Simplex(random_normal_simplex_vertices)
    uniform_simplex = Simplex(uniforml_simplex_vertices)
    return (random_normal_simplex, uniform_simplex)
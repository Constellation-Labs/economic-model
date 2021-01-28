from token_types import *

total_dag = 1000000000.0
reward_per_snapshot = 1000.0
address_space = ["a1", "a2", "a3", "a4", "b5", "b6", "b7", "c8", "c9", "c10", "d1", "d2", "d3", "d4", "d5",
                 "d6"]  ##todo make sure sorted always
node_set = ["a1", "a2", "a3", "b5", "b6", "b7", "c8", "c9"]
validator_map = {
    "DAG": ["a1", "a2", "a3", "a4", "b5", "b6", "b7", "c8", "c9"],
    "LTX": ["a1", "b5", "b6", "b7"],
    "STAR": ["a2", "c8", "c9", "c10"],
    "BTC": ["d1"]
}
channels = list(validator_map.keys())
external_price_index_dims = channels + ["USDT"]
channel_count = len(channels)
channel_marketshare = {
    "DAG": 0.6,
    "LTX": 0.1,
    "STAR": 0.2,
    "BTC": 0.1
}

# todo calculate percent_price_index_flux from trade/emit between addresses
market_flux_dag = [1.01, -0.01, 0.02]  # dag moved via btc pair drop
market_flux_btc = [0.01, 1.02, 0.01]  # dag and usdt move into btc
market_flux_usdt = [-0.02, -0.01, 0.97]  # usdt moved into btc and dag
percent_price_index_flux = np.array([market_flux_dag, market_flux_btc, market_flux_usdt])

# todo randomize fluctuations or connect to price stream. Price index is equivalent to price level, for finite velocity. Call the diff a "moment"
dag_pairs = [1.0, 0.00000024736, 0.007735]
btc_pairs = [4042690.81501, 1.0, 0.00003225806]
usdt_pairs = [129.282482224, 31000.0, 1.0]
starting_price_index = np.array([dag_pairs, btc_pairs, usdt_pairs])
starting_price_index_upper_tri = np.triu([dag_pairs, btc_pairs, usdt_pairs])


def distro_rewards(starting_balances, num_channels=channel_count, validator_map=validator_map,
                   address_space=address_space):
    '''
    This is just an untyped tensor of the address distribution function Q and rewards. The order here is important and
    should be enforced by a monoidal braiding. This is where we will want to implement ConTop as a Functorial
    construction.
    https://doc.sagemath.org/html/en/reference/categories/sage/categories/primer.html#functorial-constructions
    :param num_channels:
    :return:
    '''
    l_0_rewards = list(map(lambda x: x * reward_per_snapshot,
                           random_rewards_distribution(num_channels)))  # random_rewards_distribution(num_channels) #
    dag = dag_rewards(l_0_rewards[0], validator_map, address_space)  # off
    ltx = ltx_rewards(l_0_rewards[1], validator_map, address_space, starting_balances)  # 69.17494226569694
    star = star_rewards(l_0_rewards[2], validator_map, address_space)
    btc = btc_bridge_rewards(l_0_rewards[3], validator_map, address_space)
    return np.array([dag, ltx, star, btc])


def external_pair_flux_tensor(index_dims=external_price_index_dims):
    price_index_surface_shear = np.multiply(starting_price_index, percent_price_index_flux)
    price_index_change = price_index_surface_shear + (starting_price_index - np.identity(len(starting_price_index)))
    return price_index_change


def transitive_pair_flux(index_dims=external_price_index_dims, price_index_shift=external_pair_flux_tensor()):
    transitive_pairs = np.dot(price_index_shift[0][np.newaxis, :].T, np.array([0.97, 0.01, 0.01, 0.01])[np.newaxis, :])
    return transitive_pairs


def random_address_manifold():
    address_eigenvalues = random_rewards_distribution(len(address_space))
    weighted_address_vectors = list(map(lambda x: total_dag * x, address_eigenvalues))
    address_magnitudes = np.array(weighted_address_vectors)  # actual starting $DAG amounts
    return address_magnitudes


def plot_transition_figures(starting_balances, rewards_manifold,
                            price_index_point_change, price_index_point_flux,
                            price_index_change, transitive_dag_flux_pairs,
                            address_magnitudes, transitive_flux_pairs):
    plot_address_manifold_evolution = plot_evolution_3d(starting_balances, rewards_manifold,
                                                        title_a="starting_balances", title_b="rewards_manifold")
    plot_market_flux = plot_evolution_3d(price_index_point_change, price_index_point_flux, fig_name=200,
                                         title_a="external_markets", title_b="transaction_manifold")  # flux_moment
    plot_transitive_flux_moment = plot_evolution_3d(price_index_change, transitive_dag_flux_pairs, fig_name=300,
                                                    title_a="price_index_change", title_b="transitive_dag_flux_pairs")
    plot_global_transitive_flux = plot_evolution_3d(address_magnitudes, transitive_flux_pairs, fig_name=400,
                                                    title_a="address_magnitudes", title_b="transitive_flux_pairs")
    return plt.show()


if __name__ == "__main__":
    hyper_graph_liquidity_manifold = np.array([0.97, 0.01, 0.01, 0.01])[np.newaxis, :]
    address_vectors = np.array(
        list(random_rewards_distribution(channel_count) for _ in range(len(address_space)))).transpose()
    address_magnitudes = random_address_manifold()
    starting_balances = address_vectors * address_magnitudes
    rewards_manifold = distro_rewards(starting_balances)
    price_index_point_change = external_pair_flux_tensor()
    transitive_flux_pairs = transitive_pair_flux()  # forms btc and tether edges for hypergraph tokens
    final_state = starting_balances + rewards_manifold
    transitive_flux_index_shift = hyper_graph_liquidity_manifold.T * percent_price_index_flux[0][np.newaxis, :]
    transitive_flux_index_shift_scaled = transitive_flux_index_shift - transitive_flux_pairs.T

    plot_transition_figures(starting_balances, rewards_manifold,
                            price_index_point_change, percent_price_index_flux,
                            price_index_point_change, transitive_flux_pairs.T[1:],
                            # transitive flux on unlisted HG tokens
                            transitive_flux_index_shift_scaled, transitive_flux_pairs.T)  # global transitive flux

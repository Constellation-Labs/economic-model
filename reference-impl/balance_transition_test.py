import unittest

from balance_transition import *

error_bounds = 0.01


class TestEvolution(unittest.TestCase):

    def test_random_rewards_distribution(self):
        channel_space = random_rewards_distribution(num_channels=channel_count)
        self.assertLessEqual(sum(channel_space) - 1.0, error_bounds, "rewards manifold should be normalized")

    def test_rewards(self):
        address_vectors = np.array(
            list(random_rewards_distribution(channel_count) for _ in range(len(address_space)))).transpose()
        address_magnitudes = random_address_manifold()
        starting_balances = address_vectors * address_magnitudes
        self.assertLess(sum(distro_rewards(starting_balances).flat) - reward_per_snapshot, error_bounds,
                        "rewards manifold should be normalized")

    def test_starting_balances(self):
        address_vectors = np.array(
            list(random_rewards_distribution(channel_count) for _ in range(len(address_space)))).transpose()
        address_magnitudes = random_address_manifold()
        starting_balances = address_vectors * address_magnitudes  # [:, np.newaxis]
        self.assertLess(sum(starting_balances.flat) - total_dag, error_bounds, "Should be total_dag")

    def test_final_state(self):
        address_vectors = np.array(
            list(random_rewards_distribution(channel_count) for _ in range(len(address_space)))).transpose()
        address_magnitudes = random_address_manifold()
        starting_balances = address_vectors * address_magnitudes
        final_state = starting_balances + distro_rewards(starting_balances)
        self.assertLess(sum(final_state.flat) - (total_dag + reward_per_snapshot), error_bounds,
                        "final_state should be normalized")

    def test_proof_of_work(self):
        rewards = 1.0
        rewards_distro = proof_of_work(rewards, validator_map, address_space, "DAG")
        self.assertLess(sum(rewards_distro) - rewards, error_bounds,
                        "proof of work rewards should be normalized")

    def test_work_share(self):
        rewards = 1.0
        rewards_distro = shared_work(rewards, validator_map, address_space, "STAR")
        self.assertLess(sum(rewards_distro) - rewards, error_bounds,
                        "work share rewards should be normalized")

    def test_compound_work(self):
        rewards = 1.0
        rewards_distro = shared_work(rewards, validator_map, address_space, "LTX")
        self.assertLess(sum(rewards_distro) - rewards, error_bounds,
                        "compound work share rewards should be normalized")


if __name__ == '__main__':
    unittest.main()

from balance_transition_utils import *


def proof_of_work(rewards, validator_map, address_space, token_name):
    '''
    rewards model where rewards are distributed only for provable work done
    :return:
    '''
    node_pool = validator_map[token_name]
    non_validators = list(set(address_space) - set(node_pool))
    non_validator_rewards = list((x, 0.0) for x in non_validators)
    reward_partitions = list(zip(sorted(node_pool), random_rewards_distribution(len(node_pool))))
    all_rewards = list(rewards * i[1] for i in sorted(non_validator_rewards + reward_partitions))
    return all_rewards


def shared_work(rewards, validator_map, address_space, token_name):
    '''
    rewards model where rewards are shared amongst the addresses in the liquidity pool
    :return:
    '''
    node_pool = validator_map[token_name]
    dividends = rewards * 0.01
    node_reward_amount = rewards - dividends
    non_validators = list(set(address_space) - set(node_pool))
    non_validator_rewards = list((x, dividends/len(non_validators)) for x in non_validators)
    reward_partitions = list(zip(sorted(node_pool), random_rewards_distribution(len(node_pool))))
    node_rewards = list((i[0], node_reward_amount * i[1]) for i in sorted(reward_partitions))
    all_rewards = list(i[1] for i in sorted(non_validator_rewards + node_rewards))
    return all_rewards


def compound_work(rewards, validator_map, address_space, starting_balances, token_name):
    '''
    rewards model where rewards are shared amongst the addresses in the liquidity pool by a compounding factor. Note
    that in this case the compounding factor is stake or fraction of total ownership
    :return:
    '''
    channels = list(validator_map.keys())
    tensor_idx = sorted(channels).index(token_name)
    pool_stake_distribution = starting_balances[tensor_idx]
    total_pool_stake = sum(pool_stake_distribution)
    normalized_pool_stake_distribution = list(map(lambda x: x / total_pool_stake, pool_stake_distribution))
    node_rewards = list(rewards * normalized_pool_stake_distribution[i] for i in range(len(address_space)))
    return node_rewards


def dag_rewards(rewards, validator_map, address_space):
    all_rewards = proof_of_work(rewards, validator_map, address_space, "DAG")
    return all_rewards


def ltx_rewards(rewards, validator_map, address_space, starting_balances):
    all_rewards = compound_work(rewards, validator_map, address_space, starting_balances, "LTX")
    return all_rewards


def star_rewards(rewards, validator_map, address_space):
    all_rewards = proof_of_work(rewards, validator_map, address_space, "STAR")
    return all_rewards


def btc_bridge_rewards(rewards, validator_map, address_space):
    all_rewards = proof_of_work(rewards, validator_map, address_space, "BTC")
    return all_rewards
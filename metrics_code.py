#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File    :   metrics_code.py
@Version :   1.0
@Author  :   Forrest Stone
@Contact :   ysbrilliant@163.com
@Time    :   2023/02/04 11:06:27
@Desc    :   The implements of diversity metric in recommednation
'''

# here put the import lib
import numpy as np
from math import log
from copy import deepcopy


# calculate the ILAD of single user
def get_ILAD_per_user_at_k(predicted, item_representations, topk):
    """
    The intra-list average distance, the average dis-similarity of every pair items in the list, here we adopt the (1 - cos similarity) to calculate the distance between two items

    Args:
        predicted (list): The list of predicted items of single user, [i1,i2,...]
        item_representations (array): The representation of all items [number_item * dimension_item]
        topk (int): The length of the list we want to calculate

    Returns:
        float: The value of ILAD
    """
    local_depth = min(len(predicted), topk)
    list_items = item_representations[predicted][:local_depth]
    list_items = list_items / np.linalg.norm(
        list_items, axis=-1, keepdims=True)
    dis_matrix = np.dot(list_items, list_items.T)
    dis_matrix = 1 - dis_matrix
    dis_sum = np.sum(dis_matrix) / 2
    ILAD = np.divide(dis_sum, dis_matrix.shape[0] * (dis_matrix.shape[0] - 1))
    return ILAD


# calculate the ILAD of all users
def get_ILAD_at_k(predicted, item_representations, topk):
    ILAD = 0.0
    num_users = len(predicted)
    for user_id in range(num_users):
        ILAD += get_ILAD_per_user_at_k(predicted[user_id],
                                       item_representations, topk)
    return ILAD / num_users


# The item-level coverage, calculate signle user's coverage is useless since the recommednation list of every user is unchanged, always topk
def coverage_items_system_at_k(predicted, items, topk):
    """
    The system-level coverage, the set of recommended items for all users divided by all items

    Args:
        predicted (list): The list of predicted items of all users, [[i1,i2,...], [i1,i2,...]]
        items (list): The all items list
        topk (int): The length of the list we want to calculate

    Returns:
        float: The value of coverage
    """
    items_list = []
    num_users = len(predicted)
    for user_id in range(num_users):
        local_depth = min(len(predicted[user_id]), topk)
        item_list_per_user = predicted[user_id][:local_depth]
        items_list.append(item_list_per_user)
    item_set = set(sum(items_list, []))
    coverage = len(item_set) / len(items)
    return coverage


# The subtopic-level coverage, two aspects, this one is the user level, this is, calculate the coverage of a single user and average all users
def coverage_subtopics_user_at_k(predicted, topics, items_topics_dict, topk):
    """
    The user-level coverage subtopic, first, calculate every user's subtopic in this recommendation list, and then divide by the total topics, finally, sum all users' results, and then average

    Args:
        predicted (list): The list of predicted items of all users, [[i1,i2,...], [i1,i2,...]]
        topics (list): The topics list
        items_topics_dict (dict): The items and topics dict, {item1:[topic1, topic2, ...], item2:[topic1, topic2,...]}
        topk (int): The length of the list we want to calculate

    Returns:
        float: The value of user-level subtopic coverage
    """
    coverage_items_user = 0.0
    num_users = len(predicted)
    for user_id in range(num_users):
        topics_set_one_user = set()
        local_depth = min(len(predicted[user_id]), topk)
        item_list_per_user = predicted[user_id][:local_depth]
        for item_id in range(local_depth):
            topic_list_per_user = items_topics_dict[item_list_per_user[item_id]]
            topics_set_one_user = topics_set_one_user | set(
                topic_list_per_user)
        topics_per_user = len(set(topics_set_one_user)) / topic_num
        coverage_items_user += topics_per_user
    return coverage_items_user / num_users


# The subtopic-level coverage, two aspects, this one is the system level, this is, calculate the coverage of all users directly
def coverage_subtopics_system_at_k(predicted, topics, items_topics_dict, topk):
    """
    The system-level coverage subtopic, first, calculate every user's subtopic in this recommendation list, and then get all users topics, then divide by all topics

    Args:
        predicted (list): The list of predicted items of all users, [[i1,i2,...], [i1,i2,...]]
        topics (list): The topics list
        items_topics_dict (dict): The items and topics dict, {item1:[topic1, topic2, ...], item2:[topic1, topic2,...]}
        topk (int): The length of the list we want to calculate

    Returns:
        float: The value of system-level subtopic coverage
    """
    topics_list_all_user = []
    topics_list_one_user = []
    num_users = len(predicted)
    for user_id in range(num_users):
        local_depth = min(len(predicted[user_id]), topk)
        item_list_per_user = predicted[user_id][:local_depth]
        for item_id in range(local_depth):
            topic_list_per_user = items_topics_dict[item_list_per_user[item_id]]
            topics_list_one_user.append(topic_list_per_user)
        topics_per_user = list(set(sum(topics_list_one_user, [])))
        topics_list_all_user.append(topics_per_user)
    topics_all_user_set = set(sum(topics_list_all_user, []))
    coverage_items_system = len(topics_all_user_set) / len(topics)
    return coverage_items_system


def get_ideal_ranking(predicted, user_topic_list, items_topics_dict, alpha, topk):
    ideal_ranking = []
    topics_user = set(user_topic_list)
    topics_number_of_occurrences = dict(
        zip(topics_user, np.zeros(len(topics_user))))

    item_candidates = set(deepcopy(predicted))

    while len(item_candidates) > 0 and len(ideal_ranking) < topk:
        bestValue = float("-inf")
        whoIsBest = "noOne"
        topics_of_best = set()

        topics_intersections = {}
        for item in item_candidates:
            topics_intersections[item] = deepcopy(
                set(items_topics_dict[item]) & topics_user)

        for item in item_candidates:
            value = 0.0

            for topic in topics_intersections[item]:
                value += ((1 - alpha)**topics_number_of_occurrences[topic]) / log(
                    2 + len(ideal_ranking), 2)

            if value > bestValue:
                bestValue = deepcopy(value)
                whoIsBest = deepcopy(item)
                topics_of_best = deepcopy(topics_intersections[item])

        for topic in topics_of_best:
            topics_number_of_occurrences[topic] += 1
        ideal_ranking.append(deepcopy(whoIsBest))
        item_candidates.remove(whoIsBest)

    return ideal_ranking


def get_alpha_DCG_per_user_at_k(ranking, user_topic_list, items_topics_dict, alpha, topk):
    topics_user = set(user_topic_list)
    topics_number_of_occurrences = dict(
        zip(topics_user, np.zeros(len(topics_user))))

    local_depth = min(topk, len(ranking))
    local_dcg_values = np.zeros(local_depth)

    value = 0.0
    for i in range(0, local_depth):
        topics_intersection = (
            set(items_topics_dict[ranking[i]]) & topics_user)

        for topic in topics_intersection:
            value += ((1 - alpha) **
                      topics_number_of_occurrences[topic]) / log(2 + i, 2)
            topics_number_of_occurrences[topic] += 1
        local_dcg_values[i] = deepcopy(value)
    return local_dcg_values


def alpha_nDCG_per_user_at_k(predicted, user_topic_list, items_topics_dict, alpha, topk):
    """
    The alpha_nDCG for a singal user methods

    Args:
        predicted (list): The list of predicted items of single user, [i1,i2,...]
        user_topic_list (list): The topics of one user, [topic1, topic2,...]
        items_topics_dict (dict): The items and topics dict, {item1:[topic1, topic2, ...], item2:[topic1, topic2,...]}
        alpha (float): The redundancy penalty
        topk (int): The length of the list we want to calculate

    Reference:
        https://github.com/Pabfigueira/alpha-nDCG

    Returns:
        list: The alpha-nDCG of every position for a single user
    """
    local_depth = min(len(predicted), topk)
    ideal_ranking = get_ideal_ranking(
        predicted, user_topic_list, items_topics_dict, alpha, topk)
    dcg_target_ranking = get_alpha_DCG_per_user_at_k(
        predicted, user_topic_list, items_topics_dict, alpha, local_depth)
    dcg_ideal_ranking = get_alpha_DCG_per_user_at_k(
        ideal_ranking, user_topic_list, items_topics_dict, alpha, local_depth)

    ndcg_values = np.zeros(local_depth)

    for i in range(0, local_depth):
        if dcg_target_ranking[i] == 0.0:
            ndcg_values[i] = 0.0
        else:
            ndcg_values[i] = deepcopy(
                dcg_target_ranking[i] / dcg_ideal_ranking[i])

    return ndcg_values


def get_alpha_DCG_all_user_at_k(predicted, user_topic_dict, items_topics_dict, alpha, topk):
    num_users = len(predicted)
    local_dcg_values = {}

    for user_id in range(num_users):
        topics_user = set(user_topic_dict[user_id])
        topics_number_of_occurrences = dict(
            zip(topics_user, np.zeros(len(topics_user))))

        local_depth = min(topk, len(predicted[user_id]))
        local_dcg_values[user_id] = np.zeros(local_depth)

        value = 0.0
        for i in range(0, local_depth):
            topics_intersection = (
                set(items_topics_dict[predicted[user_id][i]])
                & topics_user)

            for topic in topics_intersection:
                value += (
                    (1 - alpha) **
                    topics_number_of_occurrences[topic]) / log(2 + i, 2)
                topics_number_of_occurrences[topic] += 1
            local_dcg_values[user_id][i] = deepcopy(value)
    return local_dcg_values


def get_alpha_nDCG_all_user_at_k(predicted, user_topic_dict, items_topics_dict, alpha, topk):
    """
    The alpha_nDCG for all users methods

    Args:
        predicted (list): The list of predicted items of all users, [[i1,i2,...], [i1,i2,...]]
        user_topic_dict (dict): The users and topics dict, {user1:[topic1, topic2, ...], user2:[topic1, topic2,...]}
        items_topics_dict (dict): The items and topics dict, {item1:[topic1, topic2, ...], item2:[topic1, topic2,...]}
        alpha (float): The redundancy penalty
        topk (int): The length of the list we want to calculate

    Reference:
        https://github.com/Pabfigueira/alpha-nDCG

    Returns:
        dict: The alpha-nDCG of every position for all users
    """
    dcg_values = deepcopy(get_alpha_DCG_all_user_at_k(
        predicted, user_topic_dict, items_topics_dict, alpha, topk))
    ndcg_values = {}

    num_users = len(predicted)
    for user_id in range(num_users):
        local_depth = min(topk, len(predicted[user_id]))
        idealRanking = get_ideal_ranking(
            predicted, user_topic_dict[user_id], items_topics_dict, alpha, topk)
        auxiliarDict = []
        auxiliarDict.append(deepcopy(idealRanking))
        dcg_ideal_ranking = get_alpha_DCG_all_user_at_k(
            auxiliarDict, user_topic_dict, items_topics_dict, alpha, topk)

        ndcg_values[user_id] = np.zeros(local_depth)

        for i in range(0, local_depth):
            if dcg_values[user_id][i] == 0.0:
                ndcg_values[user_id][i] = 0.0
            else:
                ndcg_values[user_id][i] = (dcg_values[user_id][i] /
                                           dcg_ideal_ranking[user_id][i])
    return ndcg_values

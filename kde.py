#!/usr/bin/env python
# coding:utf-8

"""
Kidney-inspired algorithm for optimization problems
@author: KaiVen
"""

import copy
import numpy as np


def initialize(dim, size, pop_range):
    """
    initialize the population
    :return: population
    """
    pop = np.zeros((size, dim))
    # 最后一列为function值
    for i in range(size):
        pop[i] = np.random.uniform(pop_range[0], pop_range[1], [1, dim])
    return pop


def DeJong(pop):
    pop_evaluate = sum(pop ** 2)
    return pop_evaluate


def Create_New_Pop(pop, best_pop):
    """
    generate the new population.
    :param pop: the old population
    :param best_pop: the best population of old
    :return:
    """
    pop_size, pop_dim = np.shape(pop)
    pop_new = np.zeros((pop_size, pop_dim))
    for i in range(pop_size):
        pop_new[i] = create_point(pop[i], best_point=best_pop)

    return pop_new


def create_point(old_point, best_point):
    # 创建新的point
    point_new = old_point + np.random.random() * (best_point - old_point)
    return point_new.reshape((256,))


def filtration(pop, alpha=0.8):
    """
    moving solutes and water from blood to the tubule.
    :return: fr
    """
    pop_size = np.shape(pop)[0]
    pop_evaluate = [DeJong(x) for x in pop]
    fr = alpha * sum(pop_evaluate) / pop_size
    fb = [x for x in pop if DeJong(x) < fr]
    w = [x for x in pop if DeJong(x) >= fr]
    return fr, fb, w


def reabsorption(point, fb, w):
    """
    transporting useful water and solutes from the tubule back into the bloodstream.
    :return: 再吸收：把W中满足条件的吸收到fb中。
    """
    w = np.delete(w, point, 1)
    fb = np.row_stack((fb, point))
    return fb, w


def secrection(point, fb, w):
    """
    transfering extra and harmful substances from the blood into the tubular.
    :return: 分泌：把fb中不满足条件的分泌到w中。
    """
    if len(w) == 0:
        fb = np.delete(fb, point, 0)
        w.append(point)
    else:
        fb = np.delete(fb, point, 0)
        w = np.row_stack((w, point))
    return fb, w


def evaluate(pop):
    """
    返回集合中最大值、最小值和索引
    :param pop: 集合
    :return: 最差值和索引
    """
    pop_size = np.shape(pop)[0]
    pop_evaluate = np.zeros(pop_size)
    for i in range(pop_size):
        pop_evaluate[i] = DeJong(pop[i])
    rank_evaluate = sorted(pop_evaluate)
    min_evaluate = rank_evaluate[0]
    for i in range(pop_size):
        if DeJong(pop[i]) == min_evaluate:
            min_pop = pop[i]
            break
    max_evaluate = rank_evaluate[-1]
    for i in range(pop_size):
        if DeJong(pop[i]) == max_evaluate:
            max_pop = pop[i]
            break
    return max_evaluate, max_pop, min_evaluate, min_pop


def run():
    dim = 256
    size = 100
    pop_range = [-5.12, 5.12]
    pop = initialize(dim, size, pop_range)
    fr, fb, w = filtration(pop, alpha=1)
    worst, worst_pop, best, best_pop = evaluate(fb)
    iterations = 100
    for i in range(iterations):
        # 如果新的解标记为W集合
        for j in range(np.shape(w)[0]):
            # 生成新的解
            si = create_point(w[j], best_pop)
            # 添加到W集合
            w = np.row_stack((w, si))
            w = np.delete(w, j, 0)
            # 生成新的解
            pop_new = create_point(si, best_pop)
            if DeJong(pop_new) < fr:
                # 满足条件再吸收
                reabsorption(pop_new, fb, w)
            else:
                # 不满足条件重新生成新的解
                np.delete(w, si, 1)
                w = np.row_stack((w, create_point(pop_new, best_pop)))
        # 如果新的解标记为FB集合
        for k in range(np.shape(fb)[0]):
            si = create_point(fb[k], best_pop)
            # 添加到FB集合
            fb = np.row_stack((fb, si))
            # 从fb集合里面分泌较差的解
            if DeJong(si) < worst:
                secrection(worst_pop, fb, w)
                fb = np.delete(fb, k, 0)
            else:
                secrection(si, fb, w)
                fb = np.delete(fb, k, 0)
        pop = np.vstack([fb, w])
        print(np.shape(pop))
        fr, fb, w = filtration(pop, alpha=1)
        worst, worst_pop, best, best_pop = evaluate(fb)
        print("The iterations is: %s, The best result is:%f" % (i + 1, best))
    print("The optimal solution is: %s" % best_pop)


run()

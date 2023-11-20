import csv
import operator

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import math

cities = []
sol = []

# TSP파일 열어서 cities로 저장
with open('TSP.csv', mode='r', newline='') as tsp:
    # read TSP city map
    reader = csv.reader(tsp)
    for row in reader:
        cities.append(row)
    #print(cities)
    # cities에 node들이 저장.


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(float(self.x) - float(city.x))
        yDis = abs(float(self.y) - float(city.y))
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
            #self.fitness = 55000 - float(self.routeDistance())
        return self.fitness



def createRoute(cityList):
    # 랜덤하게 cityList에서 순서를 섞어서 반환함.
    route = random.sample(cityList, len(cityList))
    return route


# 초기 population 생성
def initialPopulation(popSize, cityList):
    population = []
    # popSize 개수만큼 초기 gene을 생성함.
    for i in range(0, popSize):
        population.append(createRoute(cityList))

    return population


# population 랭킹 정렬
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    # fitness 결과 내림차순으로 정렬 후 반환
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


#################################################
# Test 룰렛휠
def selectOne(population, popRanked):

    # max = float(sum([chromosome.fitness for chromosome in popRanked]))
    max = 0
    for i in range(0, len(popRanked)):
        max += float(popRanked[i][1])

    point = random.uniform(0, max)

    current = 0
    # 룰렛휠로 뽑은 유전자의 번호를 population에서 찾아 해당 유전자 반환
    for i in range(0, len(popRanked)):
        # fitness가 작은 순서로 더하면서 어떤 유전자에 해당되는지
        # popRanked는 fitness가 큰 순서로 되어 있으니까.
        current += popRanked[len(popRanked) - i - 1][1]
        if current > point:
            idx = popRanked[len(popRanked) - i - 1][0]
            return population[idx]


def makeChildren(population, popRanked):
    children = []

    parent1 = selectOne(population, popRanked)
    parent2 = selectOne(population, popRanked)

    print('parent1 : ' + str(parent1))
    print('parent2 : ' + str(parent2))

    # 자식세대 생성 - 사이클, 일점, 다점 ....
    for i in range(0, len(population)):
        child = crossover(parent1, parent2)
        children.append(child)

    return children

# 일점 교차
def crossover(parent1, parent2):
    child = []
    idx = len(parent1) / 2

#    for i in range(0, idx):
#        child[i] = parent1[i]

#    for i in range

    child = random.sample(parent1, len(parent1))

    return child


#################################################

# 변이
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# 돌연변이 적용
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# 다음 세대 생성 루틴
def nextGeneration(currentGen, mutationRate):
    # elite size 필요 없고
    # rankRoutes도 필요 없고
    # 룰렛휠 사용해야 하고

    popRanked = rankRoutes(currentGen)
    # 현재 세대, rank로 다음 세대 만들기(Selection-룰렛휠, Crossover)
    children = makeChildren(currentGen, popRanked)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# 유전알고리즘 반복 초기 집단 생성, 초기 거리, 세대 반복
def geneticAlgorithm(population, popSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, mutationRate)
        #print(" distance: " + str(1 / rankRoutes(pop)[0][1]))

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# 그래프로 표현
def geneticAlgorithmPlot(population, popSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


if __name__ == '__main__':
    population_size = 50
    n_generations = 100
    cityList = []

    # Cities cityList로 변환
    for i, j in cities:
        x = i
        y = j
        cityList.append(City(x=x, y=y))

    # GA알고리즘으로 TSP문제 풀기
    # geneticAlgorithm(population=cityList, popSize=100, mutationRate=0.02, generations=100)

    # 그래프로 표현
    geneticAlgorithmPlot(population=cityList, popSize=100, mutationRate=0.01, generations=100)



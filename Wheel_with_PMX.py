"""
    1. 초기집단 랜덤 샘플링
    2. 선택 연산 - 룰렛휠 선택
    3. 교차 연산 - PMX(부분 매핑 교차, Partially Matched Crossover)
    4. 변이 - Swap 변이
"""

import csv
import operator
from copy import deepcopy
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

cities = []

# 교수님 주신 TSP파일 열어서 cities로 저장
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


# population 각각에 대한 fitness 관련 클래스
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
        return self.fitness


# City 1000개 랜덤 방문
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# 초기 population 생성
def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))

    return population


# 한 세대의 유전자들을 fitness로 정렬
def rankRoutes(population):
    fitnessResults = {}  # 키 - 밸류
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    # fitness 결과 내림차순으로 정렬 후 반환
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


# 선택 2 - 룰렛 휠 선택
def select_wheel(population, popRanked):

    max = 0
    for i in range(0, len(popRanked)):
        max += float(popRanked[i][1])

    point = random.uniform(0, max)

    current = 0
    # 룰렛휠로 뽑은 유전자의 번호를 population에서 찾아 해당 유전자 반환
    for i in range(0, len(popRanked)):
        current += popRanked[i][1]

        if current > point:
            idx = popRanked[i][0]
            return population[idx]


# 자식세대 생성
# Select, Crossover
def makeChildren(population, popRanked):
    children = []

    # 자식세대 생성
    for i in range(0, len(population)):
        # 부모세대에서 룰렛휠 기반으로 2개를 선택
        parent1 = select_wheel(population, popRanked)
        parent2 = select_wheel(population, popRanked)

        # 선택된 부모들로 자식세대 생성
        child = PMX(parent1, parent2)
        children.append(child)

    return children


# 교차 2 - PMX(Partially Matched Crossover)
# 부분 매핑 교차
def PMX(parent1, parent2):
    # 자식값을 저장할 리스트 생성
    childP1 = []
    childP2 = []
    childP3 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    # 랜덤한 2개의 좌표 지정
    for i in range(startGene, endGene):
        # 좌표 안의 값을 먼저 자식해의 값을 지정
        childP1.append(parent1[i])
    for i in range(0, startGene):
        # 2번쨰 부모의 앞부분의 값을 자식해에 임시로 저장
        childP2.append(parent2[i])
    for i in range(endGene, len(parent1)):
        # 2번째 부모의 뒷부분값을 자식해에 임시로 저장
        childP3.append(parent2[i])
    for i in range(0, len(childP2)):
        if childP2[i] in childP1:
            # 2번째 부모에서 가져온 값이 1번째 부모에서 가져온 값과 같은 경우
            while True:
                # 매핑을 통해서 같은 값이 나오지 않을때까지 찾아서 변경
                pos = parent1.index(childP2[i])
                childP2[i] = parent2[pos]
                if childP2[i] not in childP1:
                    break

    for i in range(0, len(childP3)):
        if childP3[i] in childP1:
            # 2번째 부모에서 가져온 값이 1번째 부모에서 가져온 값과 같은 경우
            while True:
                # 매핑을 통해서 같은 값이 나오지 않을때까지 찾아서 변경
                pos = parent1.index(childP3[i])
                childP3[i] = parent2[pos]
                if childP3[i] not in childP1:
                    break

    # 바뀐 값들 다 합쳐서 리턴
    child = childP2 + childP1 + childP3
    return child


# 변이
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# 변이 적용
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# 다음 세대 생성 루틴
def nextGeneration(currentGen, mutationRate):
    popRanked = rankRoutes(currentGen)
    children = makeChildren(currentGen, popRanked)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# 유전알고리즘 - 초기 집단 생성, 세대 진화 반복
def geneticAlgorithm(population, popSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# 유전 알고리즘 결과 그래프로 표현
def geneticAlgorithmPlot(population, popSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    # 결과 출력
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    bestDistance = 1 / rankRoutes(pop)[0][1]

    for i in range(0, generations):
        pop = nextGeneration(pop, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
        currDistance = 1 / rankRoutes(pop)[0][1]
        print(" generation: " + str(i))
        print("currDis = " + str(currDistance))

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    writeSolution(bestRoute, population)

    plt.plot(progress)
    plt.title('Roulette Wheel with PMX')
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return bestRoute


# 경로 csv 파일로 저장
def writeSolution(bestRoute, cityList):
    sol = []
    print("cities : " + str(cityList))
    print("bestRoute : " + str(bestRoute))

    for i in range(0, len(bestRoute)):
        idx = cityList.index(bestRoute[i])
        sol.append(idx)

    print("sol : " + str(sol))

    f = open("solution13.csv", "w")
    for i in range(len(sol)):
        f.write(str(sol[i]) + '\n')

    f.close()


if __name__ == '__main__':

    cityList = []

    # Cities cityList로 변환
    for i, j in cities:
        x = i
        y = j
        cityList.append(City(x=x, y=y))

    # GA알고리즘으로 TSP문제 풀기
    #geneticAlgorithm(population=cityList, popSize=100, mutationRate=0.02, generations=100)

    # GA 과정 및 결과 그래프로 표현
    geneticAlgorithmPlot(population=cityList, popSize=100, mutationRate=0.01, generations=100)


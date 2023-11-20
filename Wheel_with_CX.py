"""
    1. 초기집단 랜덤 샘플링
    2. 선택 연산 - 룰렛휠 선택
    3. 교차 연산 - 사이클 교차
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

def new_rankRoutes(population):
    fitnessResults = {}  # 키 - 밸류
    popLength = len(population)
    for i in range(0, len(population)):
        fitnessResults[i] = (popLength*(Fitness(population[i]).routeFitness()))**2

    # fitness 결과 내림차순으로 정렬 후 반환
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

# 한 세대의 유전자들을 fitness로 정렬
def rankRoutes(population):
    fitnessResults = {}  # 키 - 밸류
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    # fitness 결과 내림차순으로 정렬 후 반환
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


# 선택 2 - 룰렛 휠 선택
def select_wheel(population, popRanked):
    probRoute = []

    # 적합도 총 합
    sum = 0
    for j in range(0, len(popRanked)):
        sum += float(popRanked[j][1])

    # 튜플형식은 수정이안되서 list로 바꿈
    # list형태
    for k in range(0, len(popRanked)):
        probRoute.append(popRanked[k][1] / sum)

    tmp = probRoute[0]
    probRoute[0] = 0
    # 확률의 누적합리스트
    for z in range(0, len(probRoute)):

        if z == 0:
            probRoute[0] = tmp
        else:
            probRoute[z] = probRoute[z] + probRoute[z-1]

    # 어느부분인지 찾을거
    point = random.uniform(0, probRoute[len(probRoute) - 1])
    # 맞는 유전자 찾기
    for i in range(0, len(probRoute)):
        if point <= probRoute[i]:
            return population[i]

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
        child = CX(parent1, parent2)
        children.append(child)

    return children


# 교차 1 - Cycle crossover
def CX(parent1, parent2):
    cycles = [-1] * len(parent1)
    cyclestart = (i for i, v in enumerate(cycles) if v < 0)
    cycle_no = 1
    for pos in cyclestart:

        while cycles[pos] < 0:
            cycles[pos] = cycle_no
            pos = parent1.index(parent2[pos])

        cycle_no += 1

    child1 = [parent1[i] if n % 2 else parent2[i] for i, n in enumerate(cycles)]
    return child1


# 변이
# 여기서 변이는 세대 전체를 만들고 변이를 적용
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
    popRanked = new_rankRoutes(currentGen)
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
    plt.title('Roulette Wheel with Cycle crossover')
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
    geneticAlgorithmPlot(population=cityList, popSize=200, mutationRate=0.005, generations=200)

"""
    1. 초기집단 - K-mean 클러스터링 사용
    2. 선택 연산 - 토너먼트 선택
    3. 교차 연산 - CX 사이클 교차
    4. 변이 - 군집 내에서만 변이 일어나도록 함.
"""

import csv
import operator
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from operator import itemgetter
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
cities = []
sol = []


# 교수님 주신 TSP파일 열어서 cities로 저장
with open('TSP.csv', mode='r', newline='') as tsp:
    # read TSP city map
    reader = csv.reader(tsp)
    for row in reader:
        row[0] = float(row[0])
        row[1] = float(row[1])
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

    def Getx(self):
        return self.x

    def Gety(self):
        return self.y

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


# 초기 집단 생성 - K-mean 클러스터링 사용
def createNew():
    arr = np.empty((0, 2), float)

    # 배열을 추가합니다.
    # 추가하는 배열의 요소수는 초기화했던 길이와 동일해야합니다.
    # axis = 0은 행으로 추가한다는 뜻입니다.
    for i in range(1000):
        arr = np.append(arr, np.array([cities[i]]), axis=0)
    model = KMeans(n_clusters=4)
    model.fit(arr)
    labels = model.predict(arr)
    x = arr[:, 0]
    y = arr[:, 1]

    AF_list1 = []
    AF_list2 = []
    AF_list3 = []
    AF_list4 = []

    for i in range(len(arr)):
        if model.labels_[i] == 0:
            AF_list1.append(arr[i])
        if model.labels_[i] == 1:
            AF_list2.append(arr[i])
        if model.labels_[i] == 2:
            AF_list3.append(arr[i])
        if model.labels_[i] == 3:
            AF_list4.append(arr[i])

    Final_list = []
    Return_citylist = []
    Final_list = AF_list1 + AF_list2 + AF_list4 + AF_list3
    BF = [len(AF_list1),len(AF_list1)+len(AF_list2),len(AF_list1)+len(AF_list2)+len(AF_list4)]
    for i in range(len(Final_list)):
        Return_citylist.append(City(x=Final_list[i][0],y=Final_list[i][1]))
    return Return_citylist, BF


# 초기 population 생성
def initialPopulation(popSize):
    population = []
    cities_1,BF = createNew()
    # popSize 개수만큼 초기 gene을 생성함.0
    for i in range(0, popSize):
        First = random.sample(cities_1[0:BF[0]], BF[0])
        Second = random.sample(cities_1[BF[0]:BF[1]], BF[1] - BF[0])
        Third = random.sample(cities_1[BF[1]:BF[2]], BF[2] - BF[1])
        Fourth = random.sample(cities_1[BF[2]:len(cities_1)], len(cities_1) - BF[2])
        final = First+Second+Third+Fourth
        population.append(final)
    return population, BF


# 한 세대의 유전자들을 fitness 로 정렬
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    # fitness 결과 내림차순으로 정렬 후 반환
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


# 토너먼트 선택
def select_tmt(population, popRanked):

    # population의 10%를 k로 설정하여 후보자 리스트 만듦
    # 만약 100개면 10개를 뽑아서 토너먼트 진행
    k = int(len(popRanked) * 0.15)
    candidates=random.sample(popRanked, k)
    top_fit = 0
    top_idx = 0
    for i in range(0, len(candidates)):
        if top_fit < candidates[i][1]:
            top_fit = candidates[i][1]
            top_idx = candidates[i][0]
    # 토너먼트 결과 유전자 반환
    return population[top_idx]


# 자식세대 생성
# Select, Crossover
def makeChildren(population, popRanked):
    children = []

    # 자식세대 생성 - 선택, 교차
    for i in range(0, len(population)):
        parent1 = select_tmt(population, popRanked)
        parent2 = select_tmt(population, popRanked)
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


# 변이 - 군집 내에서만 swap 이 일어나도록 함.
def mutate(individual, mutationRate, BF):
    swapWith = 0
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            check = np.random.randint(4)
            if check == 0 and swapped < BF[0]:
                swapWith = int(random.random() * BF[0])
            if check == 1 and BF[0] <= swapped < BF[1]:
                swapWith = BF[0]+int(random.random() * (BF[1]-BF[0]))
            if check == 2 and BF[1] <= swapped < BF[2]:
                swapWith = BF[1]+int(random.random() * (BF[2]-BF[1]))
            if check == 3 and BF[2] <= swapped < len(individual):
                swapWith = BF[2]+int(random.random() * len(individual)-BF[2])

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# 변이 적용
def mutatePopulation(population, mutationRate, BF):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate, BF)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# 다음 세대 생성 루틴
def nextGeneration(currentGen, mutationRate, BF):
    popRanked = rankRoutes(currentGen)
    children = makeChildren(currentGen, popRanked)
    nextGeneration = mutatePopulation(children, mutationRate, BF)
    return nextGeneration


# 유전알고리즘 반복 초기 집단 생성, 초기 거리, 세대 반복
def geneticAlgorithm(population, popSize, mutationRate, generations):
    pop = initialPopulation(popSize)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# 유전 알고리즘 결과 그래프로 표현
def geneticAlgorithmPlot(population, popSize, mutationRate, generations):
    pop, BF = initialPopulation(popSize)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, mutationRate, BF)
        progress.append(1 / rankRoutes(pop)[0][1])
        currDistance = 1 / rankRoutes(pop)[0][1]
        print(" generation: " + str(i))
        print("currDis = " + str(currDistance))

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    writeSolution(bestRoute, population)

    plt.plot(progress)
    plt.title('K-mean, Tournament with CX')
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return bestRoute


# 경로 csv 파일로 저장
def writeSolution(bestRoute, cityList):
    sol = []
    print("cities : " + str(cityList))
    print("bestRoute : " + str(bestRoute))
    route_pos = []
    list_pos = []
    for i in range(0, len(bestRoute)):
        route_pos.append([bestRoute[i].Getx(), bestRoute[i].Gety()])
        list_pos.append([cityList[i].Getx(), cityList[i].Gety()])
    for i in range(0,len(route_pos)):
        idx = list_pos.index(route_pos[i])
        sol.append(idx)
    print("sol : " + str(sol))

    f = open("solution13.csv", "w")
    for i in range(len(sol)):
        f.write(str(sol[i]) + '\n')
    print(len(sol))
    f.close()


if __name__ == '__main__':

    cityList = []
    start = time.time()

    # Cities cityList로 변환

    for i, j in cities:
        x = i
        y = j
        cityList.append(City(x=x, y=y))

    # GA알고리즘으로 TSP문제 풀기
    # geneticAlgorithm(population=cityList, popSize=100, mutationRate=0.02, generations=100)
    # 그래프로 표현
    geneticAlgorithmPlot(population=cityList,popSize=300, mutationRate=0.001, generations=10000)
import random
import numpy as np
import pickle



class GeneticSearch:
	"""
	A class for selecting best features subset for ML feature selection.
	Each individual created is a boolean vector. You should interpret this vector as you with in your
	fitness object.
	"""
	gfitnessObj = None

	from enum import Enum
	class InitialPopulation(Enum):
		SINGLETONS = 0
		RANDOM = 1
		TARGET_SUBSET_PORTION = 2


	def __init__(self, individualSize, logFileName=None, numOfCategories=2, fitnessFunc=None, fitnessObj=None):
		"""

		:param individualSize:
		:param numOfCategories: The number of possible values in each entry of the individual's vector. Default is
		2 = binary.
		:param logFileName:
		:param fitnessFunc:
		:param fitnessObj:
		"""
		#	C'tor:
		#	set probability parameters:
		self.feature_probability = 0.3
		self.mutation_probability = 0.05
		self.crossover_probability = 0.4
		self.bit_crossover_probability = 0.3
		self.populationSize = 200
		self.populationBottleneck = 0.8

		#	set given data parameters:
		self.numOfCategories = numOfCategories
		self.individualSize = individualSize
		self.population = []
		self.populationFitnesses = []
		self.fitnessFunc = fitnessFunc
		self.fitnessObj = fitnessObj
		GeneticSearch.gfitnessObj = fitnessObj
		self.logFileName = logFileName

		#	initialize others:
		random.seed()
		self.bestAllTimes = []
		self.bestAllTimesScore = np.inf
		self.numOfGensSinceBest = 0


	def createIndividual(self, mask=None):
		"""
		:param mask: only for binary.
		:return:
		"""
		#	add new subgroup of features to the population:
		#	if mask is set - retain these features. mask is a boolean list of size 'featuresLen'
		#		indicating what features to retain
		individual = []
		if mask is None:
			mask = [True for f in range(self.individualSize)]

		#	try more random feature subsets:
		#individualFeatureProba = random.random()

		individual = np.random.randint(0, self.numOfCategories, self.individualSize)
		self.population.append(individual)
		return individual


	def constructPopulation(self, size, constructionType, mask=None, verbose=True):
		"""

		:param size: the size of the population in each generation
		:param constructionType: should be value of the Enum GeneticSearch.InitialPopulation.
		RANDOM means generate 'size' individuals, each with random number of features choosen randomly.
		SINGLETON means start a population with size equal to the number of features and each individual is a single
		feature.
		TARGET_SUBSET_PORTION means you give the portion of the whole set (number) you want, for example 0.6 for 60%,
		and the algorithm searches around that area.
		:param mask: if you want to retain some features or remove them, use the mask.
		:param verbose:
		:return:
		"""
		self.populationSize = size
		#	create a population of size 'size'. use mask to retain certain features of the data at first generation
		if constructionType == GeneticSearch.InitialPopulation.RANDOM:
			# from multiprocessing import Pool
			# with Pool() as pool:
			# 	self.population = pool.map(self.createIndividual, [mask for i in range(size)])

			self.population = [self.createIndividual() for i in range(size)]

			#	Add an individual that represents the center point:
			middle = (self.numOfCategories - 1) // 2
			self.population.append([middle for i in range(self.individualSize)])

		elif constructionType == GeneticSearch.InitialPopulation.SINGLETONS:
			#	Start from the bottom up:
			self.population = [[True if i == j else False for j in range(self.individualSize)] for i in range(self.individualSize)]
		self.computePopulationFitness()

		if verbose:
			self.log("Generation number 0 - Init stage")
			sizes = np.array([np.uint64(np.sum(ind)) for ind in self.population], dtype=np.uint64)
			self.log("Max subset size = " + str(sizes.max()))
			self.log("Min subset size = " + str(sizes.min()))
			self.log("Current Gen Fitness Avg: " + str(self.getCurrentGenAvgFitness()))
			self.getLastBestSubset(True)
			self.log("============================================================")
			self.getBestSubsetAllTimes(True)
			self.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		return self.population


	def fitness(self, individual, fitFunc=None):
		return self.fitnessObj.evaluateFitness(individual)


	def getCurrentGenAvgFitness(self):
		return np.average(self.populationFitnesses)


	def computePopulationFitness(self):
		self.populationFitnesses = np.array([self.fitness(self.population[i]) for i in range(len(self.population))])

		# from multiprocessing import Pool
		# import itertools
		# with Pool() as pool:
		# 	self.populationFitnesses = pool.map(self.fitness, self.population)
		# 	# self.populationFitnesses = pool.map(self.fitnessObj.evaluateFitness, self.population)
		self.updateBest()


	def startEvolution(self, maxNumOfImproveTries=15, maxNumOfGenerations=None, verbose=False):
		#	start evolution process. use default of 15 generations limit on the number of sequential tries to improve
		#	the all times best. if maximum number of iterations is given, use it also as a limit

		#	try setting a general mutation probability:
		self.mutation_probability = 1.0 / (self.individualSize)

		iCurrentGen = 1
		while self.numOfGensSinceBest < maxNumOfImproveTries and \
				(maxNumOfGenerations is None or maxNumOfGenerations >= iCurrentGen) and self.bestAllTimesScore > 0:
			self.evolveNextGen()
			if verbose:
				self.log("Generation number " + str(iCurrentGen))
				sizes = np.array([np.uint64(np.sum(ind)) for ind in self.population], dtype=np.uint64)
				self.log("Max subset size = " + str(sizes.max()))
				self.log("Min subset size = " + str(sizes.min()))
				self.log("Current Gen Fitness Avg: " + str(self.getCurrentGenAvgFitness()))
				self.getLastBestSubset(True)
				self.log("============================================================")
				self.getBestSubsetAllTimes(True)
				self.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
				iCurrentGen += 1


	def evolveNextGen(self):
		#	evolve the current generation to next generation
		newPopulation = []

		#	set probabilities:
		populationProbabilities = self.getProbabilitiesByFitness()

		#	grow new population:
		while len(newPopulation) < int(self.populationSize):
			parentsIdx = np.random.choice([i for i in range(len(self.population))], size=2, p=populationProbabilities)
			parents = [self.population[parentsIdx[i]] for i in range(len(parentsIdx))]
			children = []
			#	randomly do crossover:
			if random.random() < self.crossover_probability:
				#	do crossover between parents:
				children = self.crossoverEachBit(parents[0],parents[1], self.bit_crossover_probability)
			else:
				children = parents

			#	randomly do mutation:
			children[0] = self.mutateByChance(children[0])
			children[1] = self.mutateByChance(children[1])

			#	add to the new population: 
			newPopulation.append(children[0])
			newPopulation.append(children[1])
		self.population = newPopulation
		self.computePopulationFitness()


	def getProbabilitiesByFitness(self):
		max_h = max(self.populationFitnesses)
		min_h = min(self.populationFitnesses)
		certaintyLevel = 5
		if min_h == max_h:
			tranform_h = lambda h: 5
		else:
			tranform_h = lambda h: certaintyLevel*(max_h - h)/((max_h - min_h))
		populationProbabilities = np.array([tranform_h(self.populationFitnesses[i]) for i in range(len(self.populationFitnesses))])
		#sumOfFit = np.sum(populationProbabilities)
		#populationProbabilities = [populationProbabilities[i]/sumOfFit for i in range(len(populationProbabilities))]
		sumOfFit = np.sum(np.exp(populationProbabilities))
		softmax = lambda h: (np.exp(h))/sumOfFit
		populationProbabilities = [softmax(populationProbabilities[i]) for i in range(len(populationProbabilities))]
		return populationProbabilities

		
	def mutateByChance(self, individual):
		mutateVal = lambda val: (val + np.random.randint(0, self.numOfCategories-1) + 1) % self.numOfCategories
		mutateParam = np.random.rand(self.individualSize) < self.mutation_probability
		individual = [mutateVal(f_i) if m else f_i for f_i, m in zip(individual, mutateParam)]
		return individual

	def crossoverEachBit(self, parent1, parent2, crossBitPribability):
		#	do crossover between parents:
		child1 = list(parent1)
		child2 = list(parent2)

		crossOvers = np.random.rand(self.individualSize) < crossBitPribability

		newCld1 = [child2[i] if c else child1[i] for i, c in zip(range(self.individualSize), crossOvers)]
		newCld2 = [child1[i] if c else child2[i] for i, c in zip(range(self.individualSize), crossOvers)]

		#	ensure class is still in the subset:
		return [newCld1, newCld2]


	def crossover(self, parent1, parent2):
		#	do crossover between parents:
		crossPlace = random.randint(0, self.individualSize)
		child1 = [parent1[i] for i in range(crossPlace)] + [parent2[i] for i in range(crossPlace, self.individualSize)]
		child2 = [parent2[i] for i in range(crossPlace)] + [parent1[i] for i in range(crossPlace, self.individualSize)]
		return [child1, child2]


	def getLastBestSubset(self, verbose=False):
		#	return the best individual's subset in lastest generation (list of boolean values)
		best = self.getLastBestIndividual()
		best_list = self.population[best]
		if verbose:
			self.log("Best Current Gen Fitness Score: " + str(self.populationFitnesses[best]) + " Of Size: " + str(len(best_list)))
		return best_list


	def compareIndividuals(self, individual1Idx, individual2Idx):
		"""
		return which of the individuals is better. positive number => individual1 is better,
												   negative number => individual2 is better
		:param individual1Idx: index of an individual
		:param individual2Idx: index of an individual
		:return: positive number if individual1 is better than individual2,
		negative if individual2 is better and 0 if they're equal.
		"""
		individual1Fit = self.populationFitnesses[individual1Idx]
		individual2Fit = self.populationFitnesses[individual2Idx]
		return self.compareByFit(individual1Fit, individual2Fit)

	def compareByFitAndSize(self, individual1Fit, individual1Size, individual2Fit, individual2Size):
		"""

		:param individual1Fit:
		:param individual1Size:
		:param individual2Fit:
		:param individual2Size:
		:return:
		return positive (+) number if individual 1 has better fit or if they have equal fit and individual 1 has smaller
		set of features.
		return 0 if they have same fit & size
		return negative else
		"""
		if individual1Fit < individual2Fit or \
				(individual1Fit == individual2Fit and individual1Size < individual2Size):
			return 1
		elif individual1Fit == individual2Fit and individual1Size == individual2Size:
			return 0
		else:
			return -1

	def compareByFit(self, individual1Fit, individual2Fit):
		"""

		:param individual1Fit:
		:param individual1Size:
		:param individual2Fit:
		:param individual2Size:
		:return:
		return positive (+) number if individual 1 has better fit or if they have equal fit and individual 1 has smaller
		set of features.
		return 0 if they have same fit & size
		return negative else
		"""
		if individual1Fit < individual2Fit:
			return 1
		else:
			return -1


	def getLastBestIndividual(self):
		#	get the best individual of latest generation (actual list of booleans)
		best = 0
		for i in range(1, len(self.populationFitnesses)):
			if self.compareIndividuals(i, best) > 0:
				best = i
		return best

	def updateBest(self):
		#	updates the current best all times variable:
		curBest = self.getLastBestIndividual()
		curBestFit = self.populationFitnesses[curBest]
		if (self.bestAllTimesScore is None) or \
				(self.compareByFit(curBestFit, self.bestAllTimesScore) > 0):
			self.bestAllTimesScore = self.populationFitnesses[curBest]
			self.bestAllTimes = np.copy(self.population[curBest])
			self.numOfGensSinceBest = 0
		else:
			self.numOfGensSinceBest += 1


	def getBestSubsetAllTimes(self, verbose=False):
		best_list = self.bestAllTimes
		if verbose:
			self.log("Best All Times Gen Fitness Score: " + str(self.bestAllTimesScore) + " Of Size: " + str(len(best_list)))
		return best_list, self.bestAllTimesScore


	def log(self, msg, verbose=True):
		if verbose:
			print(msg)

		if self.logFileName is not None:
			with open(self.logFileName, "a") as file:
				file.write(str(msg) + '\n')

	def toPickle(self, filename):
		with open(filename, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


	def fromPickle(self, filename):
		with open(filename, 'rb') as input:
			self = pickle.load(input)




'''
	Abstract class for heuristic of a fittness.
	Pass an object that inherits from this class to calculate the fitness
'''
#HeuristicDataSets = namedtuple('HeuristicDataSets', ['trainData','trainClass','testData','testClass'])

class WrapperHeuristic:
	def __init__(self, dataset, classCol, baseModelList):
		#	C'tor
		#self.dataset = dataset
		self.baseModelList = baseModelList
		#self.classCol = classCol

		from sklearn.model_selection import train_test_split
		self.trainData, self.testData, self.trainClass, self.testClass = \
			train_test_split(dataset, classCol, train_size=0.65, test_size=0.35)
		#self.dataSetsTuple = HeuristicDataSets(trainData, trainClass, testData, testClass)



	def evaluateFitness(self, featuresMask):
		'''
		Calculates a score of how well the given subset of features is. Lower score = better, 0 is best.
		The score is actually the error of a learining model (or avg of multiple models) on that subset of features.
		:param featuresSubset:
		:return:
		'''
		featuresSubset = self.trainData.columns[featuresMask]
		if len(featuresSubset) < 1:
			return np.inf


		for baseModel in self.baseModelList:
			baseModel.fit(self.trainData.loc[:, featuresSubset], self.trainClass)
		scores = np.mean(
			[baseModel.score(self.testData.loc[:, featuresSubset], self.testClass) for baseModel in self.baseModelList])


		#	split data into test and train subgroups:
		# from sklearn.model_selection import cross_val_score
		# scores = np.mean([cross_val_score(baseModel, self.dataset.loc[:, featuresSubset], self.classCol, cv=5) for baseModel in self.baseModelList])
		return (1 - np.mean(scores)) * 100





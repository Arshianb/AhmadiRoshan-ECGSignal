import numpy as np
class ParticlSwarmOptimization:
    def __init__(self, pop_obj, W= 0.4, C1 = 1, C2 = 1.5, 
                 max_gens=[], min_gens=[], gap=10, IndexofGensToMainChromosome=[]):
        self.pop_obj = pop_obj
        self.particleDirection = [[0]*len(self.pop_obj.population[0])]*len(self.pop_obj.population)
        self.BestExperiencePosition = [[0]*len(self.pop_obj.population[0])]*len(self.pop_obj.population)
        self.BestExperienceFitness = [0]*len(self.pop_obj.population)
        self.BestParticlePosition = [0]*len(self.pop_obj.population[0])
        self.BestParticleFitness = 0
        self.IndexofGensToMainChromosome=IndexofGensToMainChromosome
        self.W = W
        self.C1 = C1
        self.C2 = C2
        self.max_gens = max_gens
        self.min_gens = min_gens
        self.generation=0
        self.gap=gap
    def CheckTheChromosome(self, newChromosome):
        Genindex=-1
        for i in self.IndexofGensToMainChromosome:
            Genindex+=1
            if newChromosome[Genindex]>self.max_gens[i]:
                newChromosome[Genindex]=self.max_gens[i]
            if newChromosome[Genindex]<self.min_gens[i]:
                newChromosome[Genindex]=self.min_gens[i]
        return newChromosome
    def MoveParticle(self, popFitness):
        if self.generation>self.gap:
            newPopulation=[]
            indexOfChromosome=-1
            for chromosome in self.pop_obj.population:
                indexOfChromosome+=1
                r1 = np.random.rand()
                r2 = np.random.rand()
                Velocity_P = np.multiply(self.W, self.particleDirection[indexOfChromosome]) + \
                    np.multiply(self.C1*r1, np.subtract(self.BestExperiencePosition[indexOfChromosome] ,chromosome)) + \
                    np.multiply(self.C2*r2, np.subtract(self.BestParticlePosition[indexOfChromosome] ,chromosome))
                new_Cromosome=chromosome+Velocity_P
                new_Cromosome=self.CheckTheChromosome(new_Cromosome)
                self.particleDirection[indexOfChromosome]=Velocity_P
                newPopulation.append(new_Cromosome)
            self.pop_obj.population=new_Cromosome
        else:
            # Fix Later
            pass
        if popFitness[indexOfChromosome]>self.BestParticleFitness:
            self.BestParticleFitness = popFitness[indexOfChromosome]
            self.BestParticlePosition= chromosome
        if popFitness[indexOfChromosome]>self.BestExperienceFitness[indexOfChromosome]:
            self.BestExperienceFitness[indexOfChromosome] = popFitness[indexOfChromosome]
            self.BestExperiencePosition[indexOfChromosome]= chromosome


        self.generation+=1
"lab2"

import logging
from pprint import pprint, pformat
from collections import namedtuple
import random
from copy import deepcopy

import numpy as np


def nim_sum(state: Nim) -> int:
    tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
    xor = tmp.sum(axis=0) % 2
    return int("".join(str(_) for _ in xor), base=2)


def analize(raw: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = dict()
    for ply in (Nimply(r, o) for r, c in enumerate(raw.rows) for o in range(1, c + 1)):
        tmp = deepcopy(raw)
        tmp.nimming(ply)
        cooked["possible_moves"][ply] = nim_sum(tmp)
    return cooked


def optimal(state: Nim) -> Nimply:
    analysis = analize(state)
    logging.debug(f"analysis:\n{pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns != 0]
    if not spicy_moves:
        spicy_moves = list(analysis["possible_moves"].keys())
    ply = random.choice(spicy_moves)
    return ply



Nimply = namedtuple("Nimply", "row, num_objects")

class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects

def strategy_random(state: Nim) -> Nimply:
    """A completely random move"""
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)

population[0]

def strategy1(state:Nim)->Nimply:
  rows=state.rows
  chosen=np.argmax(rows)
  ply=Nimply(chosen,rows[chosen])
  return ply

def optimal(state: Nim) -> Nimply:
    analysis = analize(state)
    logging.debug(f"analysis:\n{pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns != 0]
    if not spicy_moves:
        spicy_moves = list(analysis["possible_moves"].keys())
    ply = random.choice(spicy_moves)
    return ply

def strategy2(state: Nim) -> Nimply:
  analysis = analize(state)
  if embed_nim(state)==3:
    print("3 ok")
    logging.debug(f"analysis:\n{pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns == 0]
  else:
    spicy_moves = list(analysis["possible_moves"].keys())
    ply = random.choice(spicy_moves)
  return ply

def embed_nim(nim_instance: Nim):
    np0 =nim_sum(nim_instance)==0
    l=len([r for r in nim_instance.rows if r > 0])
    even = l% 2 == 0

    if np0 and even:
        e=0
    elif np0 and not even:
        e=1
    elif not np0 and even:
        e=2
    elif not np0 and not even:
        e=3
    return e,l

def initialization(strategies:list):
  alpha = np.ones(len(strategies))
  map = np.random.dirichlet(alpha, size=16)
  return map

def pick_strategy(state:Nim,map:np.array,strategies:list):

  embedding=nim_sum(state) #here
  probs=map[embedding,:]
  result=np.random.multinomial(1, probs)
  index=np.argmax(result)
  strategy_chosen=strategies[index]
  return strategy_chosen

"""## Oversimplified match"""

def tournament(n,map,strategies:list):
  #logging.getLogger().setLevel(logging.INFO)

  state = Nim(5)
  #logging.info(f"init : {nim}")
  player = 0
  while state:
    if player==0:
      strategy=pick_strategy(state,map,strategies)
    else:
      strategy=optimal
    ply = strategy(state)
      #logging.info(f"ply: player {player} plays {ply}")
    state.nimming(ply)
      #logging.info(f"status: {nim}")
    player = 1 - player
  return player

def fitness(individual,accuracy):
  score=0
  i=1
  while i<=accuracy:
    score+=tournament(4,individual,strategies)
    i+=1
  return 1-score/accuracy

def select_parent(pop,pressure):
  parent= random.choice(pop[:pressure])
  return parent

def mutate(individual,strategies,std):
  perturbations = np.random.normal(loc=0, scale=std, size=individual.shape)
  perturbations=np.abs(perturbations)
  mutated=individual+perturbations
  norms = np.sum(mutated, axis=1)
  for i in range(individual.shape[0]):
    mutated[i,:]/=norms[i]
  return mutated

def fusion(p1,p2):
  o=(p1+p2)/2
  return o

off=10
mp=.15
generations=100
accuracy=10
population_size=100
pressure=4

population=list()
strategies=[strategy1,optimal,strategy_random]

for i in range(population_size):
  map=initialization(strategies)
  population.append(map)

population.sort(key=lambda i:fitness(i,accuracy),reverse=True)
best_pop=fitness(population[0],accuracy)
limit_pop=fitness(population[limit],accuracy)
print(best_pop)

for generation in range(100):
  offspring=list()
  for counter in range(off):
    if random.random()<mp:
      p=select_parent(population,pressure)  #tenere conto dei migliori
      o=mutate(p,strategies,10)
    else:
      p1=select_parent(population,pressure) #tenere conto dei migliori
      p2=select_parent(population,pressure)
      o=fusion(p1,p2)
    offspring.append(o)
  offspring.sort(key=lambda i:fitness(i,accuracy),reverse=True)
  off_best=fitness(offspring[0],accuracy)
  if off_best>=limit_pop:
    print("good offspring")
    population.extend(offspring)
    population.sort(key=lambda i:fitness(i,accuracy),reverse=True)
    population=population[:population_size]
    best_pop=fitness(population[0],accuracy)
    limit_pop=fitness(population[limit],accuracy)
    print(best_pop)
  else:
    print("bad")

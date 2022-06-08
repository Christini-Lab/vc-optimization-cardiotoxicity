import pickle
from os import listdir


def get_high_fitness(ga_result):
    best_individual = ga_result.generations[0][0]

    for i, gen in enumerate(ga_result.generations):
        best_in_gen = ga_result.get_high_fitness_individual(i)
        if best_in_gen.fitness > best_individual.fitness:
            best_individual = best_in_gen

    return best_individual


files = listdir('./ga_results')
for f in files:
    if 'DS' in f:
        continue
    if 'csv' in f:
        continue

    if 'opt' in f:
        proto = pickle.load(open(f'./ga_results/{f}', 'rb'))
    else:
        try:
            ga_result = pickle.load(open(f'./ga_results/{f}', 'rb'))
            proto = get_high_fitness(ga_result)
            proto = proto.protocol
        except:
            import pdb
            pdb.set_trace()


    proto.to_csv(f'./ga_results/{f.split(".")[0]}.csv')




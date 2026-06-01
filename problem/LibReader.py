import os

def TSPLIBReader(filename):
    '''
        Acquire description of a TSP problem from a TSPLIB-formatted file
        Parameters:
        - filename: the name of the TSPLIB-formatted file. 
        Returns:
        - name: the name of the TSP problem.
        - dimension: the number of nodes in the TSP problem. (int)
        - locs: the coordinates of nodes in the TSP problem. 
    '''
    with open(filename, 'r') as f:
        dimension = 0
        started = False
        locs = []
        edge_weight_type = None

        for line in f:
            loc = []
            if started:
                if line.startswith("EOF"):
                    break
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)
            if line.startswith("NAME"):
                name = line.strip().split()[-1]
            if line.startswith("DIMENSION"):
                dimension = int(line.strip().split()[-1])
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.strip().split()[-1]
                if line.strip().split()[-1] not in ["EUC_2D", "CEIL_2D"]:
                    return None, None, None, None
            if line.startswith("NODE_COORD_SECTION"):
                started = True

    assert len(locs) == dimension
    return name, dimension, locs, edge_weight_type

def CVRPLIBReader(filename):
    '''
        Acquire description of a CVRP problem from a CVRPLIB-formatted file
        Parameters:
        - filename: the name of the CVRPLIB-formatted file. 
        Returns:
        - name: the name of the CVRP problem.
        - dimension: the number of nodes in the CVRP problem. (int)
        - locs: the coordinates of nodes in the CVRP problem. 
        - demand: A list of node demands. 
        - capacity: The capacity of the vehicle. (int)
    '''
    with open(filename, 'r') as f:
        dimension = 0
        started_node = False
        started_demand = False
        locs = []
        demand = []
        for line in f:
            loc = []
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    break
                demand.append(int(line.strip().split()[-1]))
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node = False
                    started_demand = True
            if started_node:
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)

            if line.startswith("NAME"):
                name = line.strip().split()[-1]
            if line.startswith("DIMENSION"):
                dimension = float(line.strip().split()[-1]) - 1 # depot is not counted
            if line.startswith("EDGE_WEIGHT_TYPE"):
                if line.strip().split()[-1] not in ["EUC_2D", "CEIL_2D"]:
                    return None, None, None, None, None, None, None
                edge_weight_type = line.strip().split()[-1]
            if line.startswith("CAPACITY"):
                capacity = float(line.strip().split()[-1])
            if line.startswith("NODE_COORD_SECTION"):
                started_node = True
    cost_file = filename.replace('.vrp', '.sol')
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as f:
            for line in f:
                if line.startswith("Cost"):
                    cost = float(line.split()[1])
    else:
        raise FileNotFoundError(f"Cost file {cost_file} not found." +
                                "Please make sure the cost file with the same name as the problem file and with .sol extension exists in the same directory.")
    assert len(locs) == dimension + 1  # +1 for depot
    assert len(demand) == dimension + 1  # +1 for depot
    return name, int(dimension), locs, demand, capacity, cost, edge_weight_type

tsplib_cost = {
   # TSPLIB, which includes 81 TSP instances with EUC_2D or CEIL_2D edge weight type. 
   # All optimal, http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
    "a280": 2579,
    "berlin52": 7542,
    "bier127": 118282,
    "brd14051": 469385,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "d15112": 1573084,
    "d18512": 645238,
    "dsj1000": 18660188, # (CEIL_2D)
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "fl3795": 28772,
    "fnl4461": 182566,
    "gil262": 2378,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "kroB200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "nrw1379": 56638,
    "p654": 34643,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pcb3038": 137694,
    "pla7397": 23260728, # (CEIL_2D)
    "pla33810": 66048945, # (CEIL_2D)
    "pla85900": 142382641, # (CEIL_2D)
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "rl5915": 565530,
    "rl5934": 556045,
    "rl11849": 923288,
    "st70": 675,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    "usa13509": 19982859,
    "vm1084": 239297,
    "vm1748": 336556, 
    
    # http://webhotel4.ruc.dk/~keld/research/LKH/DIMACS_results.html
    'E10k.0' : 71865826,
    'E10k.1' : 72031630,
    'E10k.2' : 71822483,
    'E31k.0' : 127282138,
    'E31k.1' : 127452384,
    'E100k.0': 225784127,
    'E100k.1': 225654639,
    'E316k.0': 401301206,
    'E1M.0'  : 713187688,
    'E3M.0'  : 1267295461,
    'E10M.0' : 2253040309,
}

TEST_CONFIG = {
        "tsp": {
            "model_path": "./pretrained/tsp_best.pt",
            "lower_neighbors_num": 20,
            "distributions": {
                "uniform": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/TSP",
                    "cases": {
                        1000:   {"episodes": 128, "batch_size": 128, "filename": "MCTS_tsp1000_test_concorde.txt"},
                        5000:   {"episodes":  16, "batch_size":  16, "filename": "test_tsp5000_lkh3_n16.txt"},
                        10000:  {"episodes":  16, "batch_size":  16, "filename": "MCTS_tsp10000_test_concorde.txt"},
                        50000:  {"episodes":  16, "batch_size":  16, "filename": "test_tsp50000_lkh3_n16.txt"},
                        100000: {"episodes":  16, "batch_size":  16, "filename": "test_tsp100000_lkh3_n16.txt"},
                    },
                },
                "clustered1": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/TSP",
                    "cases": {
                        5000: {"episodes": 20, "batch_size": 20, "filename": "INViT_test_tsp5000_nums20_clustered1.pt"},
                    },
                },
                "explosion": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/TSP",
                    "cases": {
                        5000: {"episodes": 20, "batch_size": 20, "filename": "INViT_test_tsp5000_nums20_explosion.pt"},
                    },
                },
                "implosion": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/TSP",
                    "cases": {
                        5000: {"episodes": 20, "batch_size": 20, "filename": "INViT_test_tsp5000_nums20_implosion.pt"},
                    },
                },
                "tsplib": {
                    "type": "benchmark",
                    "data_dir": "./dataset/benchmark/TSPLIB",
                    "problem_sizes": 1000,
                    "max_problem_size": 100000, # 100K
                },
                "dimacs_e": {
                    "type": "benchmark",
                    "data_dir": "./dataset/benchmark/DIMACS_E",
                    "problem_sizes": 10000,
                    "max_problem_size": 1000 *10000, # 10 million
                },
            },
        },
        "cvrp": {
            "model_path": "./pretrained/cvrp_best.pt",
            "lower_neighbors_num": 50,
            "distributions": {
                "uniform": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/CVRP",
                    "cases": {
                        1000:   {"episodes": 100, "batch_size": 100, "filename": "TAM_test_cvrp1000_capacity200_nums100_uniform.txt"},
                        5000:   {"episodes": 100, "batch_size": 100, "filename": "TAM_test_cvrp5000_capacity300_nums100_uniform.txt"},
                        10000:  {"episodes":  16, "batch_size":  16, "filename": "TAM_test_cvrp10000_capacity300_nums16_uniform.txt"},
                        50000:  {"episodes":  16, "batch_size":  16, "filename": "TAM_test_cvrp50000_capacity300_nums16_uniform.txt"},
                        100000: {"episodes":  16, "batch_size":  16, "filename": "TAM_test_cvrp100000_capacity300_nums16_uniform.txt"},
                    },
                },
                "clustered1": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/CVRP",
                    "cases": {
                        5000: {"episodes": 20, "batch_size": 20, "filename": "INViT_test_cvrp5000_capacity50_nums20_clustered1.pt"},
                    },
                },
                "explosion": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/CVRP",
                    "cases": {
                        5000: {"episodes": 20, "batch_size": 20, "filename": "INViT_test_cvrp5000_capacity50_nums20_explosion.pt"},
                    },
                },
                "implosion": {
                    "type": "synthetic",
                    "data_root": "./dataset/synthetic/CVRP",
                    "cases": {
                        5000: {"episodes": 20, "batch_size": 20, "filename": "INViT_test_cvrp5000_capacity50_nums20_implosion.pt"},
                    },
                },
                "cvrplib": {
                    "type": "benchmark",
                    "data_dir": "./dataset/benchmark/CVRPLIB",
                    "problem_sizes": 3000,
                    "max_problem_size": 100000, # 100K
                },
            },
        },
    }
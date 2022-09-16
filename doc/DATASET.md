# Datasets

## Formats
We consider the following formats of SAT problems:
1. CNF dataset with Bipartite settings: `cnf_dataset.npz`.
2. Circuit converted from CNF via `cube and conquer`: `ckt_dataset.npz`.
3. AIG converted from CNF via `cnf2aig`: `aig_dataset.npz`.
4. AIG converted from CNF via `cnf2aig` with simulated probability supervision: `aig_prob_dataset.npz`.
5. Optimized AIG via `abc`: `optaig_dataset.npz`.
6. Optimized AIG via `abc` with simulated probability supervision: `optaig_prob_dataset.npz`.


## Random SAT
1. Toy example: 100 SR3-10: `data/random_sr3_10_100`
2. SR3-10:
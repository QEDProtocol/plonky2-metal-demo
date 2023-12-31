# Plonky2 Metal Demo

This is a demonstration of using a Metal compute shader to accelerate plonky2/starky proving. QED no longer uses classic poseidon as our default hasher, but we decided release our early implementation since it still provides a **40-50% speed up** to proving time despite being extremely janky.
We hope that this can be the starting point for people that want to experiment with optimizing plonky2.

Notably, proofs generated by this demo are drop in compatible with vanilla plonky2.


Note: This uses a modified version of plonky2 which adds `MerkleHasher` to GenericConfig. 
If you don't want to use the fork used by this repo, the changes are relatively simple and can be found [here](https://github.com/QEDProtocol/plonky2-merkle-config/commit/4dfb8791505210408c6ed4c1e04d173ecc0e7639).


Running:
```bash
make main
```

Output Example:
```
num_gates: 16384, degree: 32768
proving: metal took 1362ms
num_gates: 16384, degree: 32768
proving: cpu took 2147ms
metal is 57.635829662261386% faster than cpu+rayon
```

(c) 2023 Zero Knowledge Labs Limited - MIT - ∀


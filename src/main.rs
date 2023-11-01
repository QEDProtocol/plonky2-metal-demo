use std::marker::PhantomData;

use plonky2_metal_demo::gpu::metal::poseidon_interleaved::MetalRuntime;
use plonky2::{
    field::{
        extension::quadratic::QuadraticExtension, goldilocks_field::GoldilocksField, types::{Field, Sample},
    },
    hash::{
        hash_types::{HashOut, HashOutTarget},
        merkle_tree::MerkleTree,
        poseidon::PoseidonHash,
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, CircuitData},
        config::{
            AlgebraicHasher, GenericConfig, GenericConfigMerkleHasher, PoseidonGoldilocksConfig,
        }, proof::ProofWithPublicInputs,
    }, iop::witness::{PartialWitness, WitnessWrite},
};
use serde::Serialize;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PoseidonGoldilocksGPUMerkleHasher(PhantomData<()>);

/// Configuration using GPU Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize)]
pub struct PoseidonGoldilocksGPUConfig;
impl GenericConfig<2> for PoseidonGoldilocksGPUConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = PoseidonHash;
    type InnerHasher = PoseidonHash;
    type MerkleHasher = PoseidonGoldilocksGPUMerkleHasher;
}

impl GenericConfigMerkleHasher<GoldilocksField, PoseidonHash>
    for PoseidonGoldilocksGPUMerkleHasher
{
    fn new_merkle_tree(
        leaves: Vec<Vec<GoldilocksField>>,
        cap_height: usize,
    ) -> MerkleTree<GoldilocksField, PoseidonHash> {
        let tree_height = (leaves.len() as f64).log2().ceil() as usize;

        if cap_height == tree_height || tree_height < 13 {
            // use cpu for small trees
            MerkleTree::new(leaves, cap_height)
        }else{
            MetalRuntime::new_merkle_tree(leaves, cap_height)
        }
    }
}


struct TestCircuit<const D: usize, C: GenericConfig<D>>{
    data: CircuitData<C::F, C, D>,
    seed_target: HashOutTarget,
}
impl<const D: usize, C: GenericConfig<D>> TestCircuit<D, C> {
    pub fn new<H: AlgebraicHasher<C::F>>(num_iters: usize) -> Self {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder: CircuitBuilder<C::F, D> = CircuitBuilder::<C::F, D>::new(config);
        let seed_target = builder.add_virtual_hash();
        let mut cur = seed_target;
        for i in 0..num_iters {
            let i_constant = builder.constant(C::F::from_canonical_usize(i));
    
            cur = builder.hash_n_to_hash_no_pad::<H>(vec![
                i_constant,
                i_constant,
                i_constant,
                i_constant,
                cur.elements[0],
                cur.elements[1],
                cur.elements[2],
                cur.elements[3],
            ]);
        }
        builder.register_public_inputs(&cur.elements);
    
        let num_gates = builder.num_gates();
        let data = builder.build::<C>();
        println!("num_gates: {}, degree: {}", num_gates, data.common.degree());
        TestCircuit {
            data,
            seed_target
        }
    }
    pub fn prove(&self, seed_hash: HashOut<C::F>)->ProofWithPublicInputs<C::F, C, D> {
        let mut pw = PartialWitness::new();
        pw.set_hash_target(self.seed_target, seed_hash);
        self.data.prove(pw).unwrap()
    }
}


fn main() {
    MetalRuntime::get().warm_up();

    let seed_hash = HashOut{elements: GoldilocksField::rand_array::<4>()};


    let gpu = TestCircuit::<2, PoseidonGoldilocksGPUConfig>::new::<PoseidonHash>(1<<14);

    let gpu_start_time = std::time::Instant::now();
    let gpu_proof = gpu.prove(seed_hash);
    let gpu_total = gpu_start_time.elapsed().as_millis();
    gpu.data.verify(gpu_proof).unwrap();
    println!("proving: metal took {}ms", gpu_total);


    let cpu = TestCircuit::<2, PoseidonGoldilocksConfig>::new::<PoseidonHash>(1<<14);
    
    let cpu_start_time = std::time::Instant::now();
    let cpu_proof = cpu.prove(seed_hash);
    let cpu_total = cpu_start_time.elapsed().as_millis();
    cpu.data.verify(cpu_proof).unwrap();
    println!("proving: cpu took {}ms", cpu_total);

    println!("metal is {}% faster than cpu+rayon", (((cpu_total as f64)/(gpu_total as f64)*100f64)-100f64))

}

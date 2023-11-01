use lazy_static::lazy_static;
use metal::{objc::rc::autoreleasepool, *};
use plonky2::{
    field::goldilocks_field::GoldilocksField,
    hash::{
        hash_types::HashOut,
        merkle_tree::{MerkleCap, MerkleTree},
        poseidon::PoseidonHash,
    },
    plonk::config::Hasher,
};

use std::{ptr, sync::Mutex};

fn get_node_hash_index_in_digests(
    num_layers: usize,
    tree_length: usize,
    level: usize,
    index: usize,
) -> usize {
    let leaf_index = index << level;
    let tree_index = index >> (num_layers - level);
    let pair_index = ((leaf_index) & ((1 << num_layers) - 1)) >> level;
    let parity = pair_index & 1;
    let pair_index = pair_index >> 1;

    let siblings_index = (pair_index << (level + 1)) + (1 << level) - 1;
    let d_index = 2 * siblings_index + parity;
    return d_index + (tree_index * tree_length);
}

unsafe fn from_buf_raw<T>(ptr: *const T, elts: usize) -> Vec<T> {
    let mut dst = Vec::with_capacity(elts);

    // SAFETY: Our precondition ensures the source is aligned and valid,
    // and `Vec::with_capacity` ensures that we have usable space to write them.
    ptr::copy(ptr, dst.as_mut_ptr(), elts);

    // SAFETY: We created it with this much capacity earlier,
    // and the previous `copy` has initialized these elements.
    dst.set_len(elts);
    dst
}

fn get_size_for_count(count: usize) -> MTLSize {
    let max_dim = 32768;
    if count <= max_dim {
        return MTLSize {
            width: count as u64,
            height: 1,
            depth: 1,
        };
    } else {
        return MTLSize {
            width: 32768,
            height: (count / 32768) as u64,
            depth: 1,
        };
    }
}
const SHADERLIB: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher.metallib"
));

pub struct MetalRuntime {
    pub device: Mutex<Device>,
    f_poseidon_hash_leaves: Function,
    f_poseidon_hash_tree_level: Function,
    command_queue: CommandQueue,
    init_time: u128,
}

unsafe impl Sync for MetalRuntime {}

lazy_static! {
    static ref RUNTIME: MetalRuntime = {
        let device = Device::system_default().unwrap();
        let lib = device.new_library_with_data(SHADERLIB).unwrap();
        let command_queue = device.new_command_queue();

        let f_poseidon_hash_leaves = lib.get_function("poseidon_hash_leaves", None).unwrap();
        let f_poseidon_hash_tree_level = lib.get_function("poseidon_hash_tree_level", None).unwrap();


        MetalRuntime {
            device: Mutex::new(Device::system_default().unwrap()),
            //lib: Mutex::new(lib),
            command_queue,
            f_poseidon_hash_leaves,//: Mutex::new(f_poseidon_hash_leaves),
            f_poseidon_hash_tree_level,//: Mutex::new(f_poseidon_hash_tree_level),
            //func_ref: HashMap::new(),
            init_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros(),
        }
    };
}

impl MetalRuntime {
    pub fn get() -> &'static MetalRuntime {
        &RUNTIME
    }
    pub fn warm_up(&self) -> u128 {
        self.init_time
    }

    pub fn get_poseidon_hash_tree_level_pipeline_state(&self) -> ComputePipelineState {
        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&self.f_poseidon_hash_tree_level));

        let pipeline_state = self
            .device
            .lock()
            .unwrap()
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();
        pipeline_state
    }
    pub fn get_poseidon_hash_leaves_pipeline_state(&self) -> ComputePipelineState {
        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&self.f_poseidon_hash_leaves));

        let pipeline_state = self
            .device
            .lock()
            .unwrap()
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();
        pipeline_state
    }
    pub fn alloc(&self, len: usize) -> Buffer {
        self.device
            .lock()
            .unwrap()
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn new_merkle_tree(
        leaves: Vec<Vec<GoldilocksField>>,
        cap_height: usize,
    ) -> MerkleTree<GoldilocksField, PoseidonHash> {
        let leaf_length = leaves[0].len();
        let leaf_count = leaves.len();
        let tree_height = (leaf_count as f64).log2().ceil() as usize;

        if cap_height == tree_height || tree_height < 13 {
            return MerkleTree::new(leaves, cap_height);
        }
        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;

        let total_digests = total_tree_hashes - (num_caps * 2 - 1);
        let num_layers = tree_height - cap_height;
        let tree_length = total_digests >> cap_height;


        let leaf_u64s = leaves.concat();
        let digests = autoreleasepool::<Vec<HashOut<GoldilocksField>>,_>(|| {
            let leaves_buffer = Self::get().device.lock().unwrap().new_buffer_with_data(
                unsafe { std::mem::transmute(leaf_u64s.as_ptr()) },
                (leaf_length * leaf_count * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            );
            let result = Self::get().hash_merkle_tree_buf_ho(
                leaves_buffer,
                tree_height,
                leaf_length,
                cap_height,
            );
            result
        });

        let mut caps: Vec<HashOut<GoldilocksField>> = Vec::with_capacity(num_caps);
        let cap_child_level = tree_height - cap_height - 1;
        for i in 0..num_caps {
            let left_index =
                get_node_hash_index_in_digests(num_layers, tree_length, cap_child_level, i * 2);
            caps.push(<PoseidonHash as Hasher<GoldilocksField>>::two_to_one(
                digests[left_index],
                digests[left_index + 1],
            ));
        }

        MerkleTree {
            leaves,
            digests,
            cap: MerkleCap(caps),
        }
    }

    pub fn hash_merkle_tree_buf_ho(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> Vec<HashOut<GoldilocksField>> {
        let leaf_count = 1usize << tree_height;

        assert!(
            cap_height < (tree_height as usize),
            "cap height must be less than tree height"
        );
        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;

        let total_digests = total_tree_hashes - (num_caps * 2 - 1);
        let num_layers = tree_height - cap_height;
        let tree_length = total_digests >> cap_height;

        let mut uniforms: [u32; 4] = [0, tree_length as u32, num_layers as u32, leaf_length as u32];

        let digests_buffer = self.alloc((total_digests * 4) * std::mem::size_of::<u64>());

        let pipeline_poseidon_hash_leaves = self.get_poseidon_hash_leaves_pipeline_state();
        let pipeline_poseidon_hash_tree_level = self.get_poseidon_hash_tree_level_pipeline_state();

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        encoder.set_compute_pipeline_state(&pipeline_poseidon_hash_leaves);

        encoder.set_buffer(0, Some(&leaves_buffer), 0);
        encoder.set_buffer(1, Some(&digests_buffer), 0);

        let uniforms_buffer = self.device.lock().unwrap().new_buffer_with_data(
            unsafe { std::mem::transmute(uniforms.as_ptr()) },
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeWriteCombined,
        );
        encoder.set_buffer(2, Some(&uniforms_buffer), 0);
        let num_threads = pipeline_poseidon_hash_leaves.thread_execution_width();

        let lg = (leaf_count as NSUInteger + num_threads) / num_threads;
        let thread_group_count = get_size_for_count(lg as usize);

        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

        encoder.end_encoding();

        // start tree hasher

        for i in 1..num_layers {
            let compute_pass_descriptor = ComputePassDescriptor::new();

            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_poseidon_hash_tree_level);

            uniforms[0] = i as u32;

            encoder.set_bytes(1, 16, uniforms.as_ptr() as *mut core::ffi::c_void);
            encoder.set_buffer(0, Some(&digests_buffer), 0);

            let num_threads = pipeline_poseidon_hash_tree_level.thread_execution_width();

            let lg = ((leaf_count >> i) as NSUInteger + num_threads) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = digests_buffer.contents() as *mut HashOut<GoldilocksField>;
        let res = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(ptr, total_digests) };
        res
    }
}

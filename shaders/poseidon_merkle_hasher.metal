#include <metal_stdlib>

#include "goldilocks.metal"

#include "poseidon_goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;


inline uint compute_digest_index(uint tree_length, uint num_layers, uint level_in, uint index_in) {
    ulong index = (ulong)index_in;
    ulong level = (ulong)level_in;

    ulong leaf_index = index<<(level);
    ulong tree_index = index>>(((ulong)num_layers)-level);
    ulong pair_index = ((leaf_index)&((1 << ((ulong)num_layers)) - 1))>>level;
    ulong parity = pair_index&1;
    pair_index >>=1;

    ulong siblings_index = (pair_index << (level + 1)) + (1 << level) - 1;
    ulong d_index = 2 * siblings_index + parity;
    return (uint)(d_index+((ulong)tree_index*(ulong)tree_length));

}

struct Uniforms {
    uint level;
    uint tree_length;
    uint num_layers;
    uint leaf_size;
};

kernel void poseidon_hash_leaves(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant Uniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
        
    Fp p2_state[12];
    
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;


    
    uint offset = (gid[1] * 32768 + gid[0]) * uniforms.leaf_size;
    uint num_full_rounds = uniforms.leaf_size / 8;

    for (uint i = 0; i < num_full_rounds; i++) {
        p2_state[0] = leaf_inputs[offset];
        p2_state[1] = leaf_inputs[offset + 1];
        p2_state[2] = leaf_inputs[offset + 2];
        p2_state[3] = leaf_inputs[offset + 3];

        p2_state[4] = leaf_inputs[offset + 4];
        p2_state[5] = leaf_inputs[offset + 5];
        p2_state[6] = leaf_inputs[offset + 6];
        p2_state[7] = leaf_inputs[offset + 7];
        poseidon_permute(p2_state);
        offset+=8;
    }

    uint remaining = uniforms.leaf_size - num_full_rounds * 8;
    if (remaining != 0) {
        for(uint i=0;i<remaining;i++){
            p2_state[i] = leaf_inputs[offset + i];
        }
        poseidon_permute(p2_state);
    }

    offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, 0, (gid[1] * 32768 + gid[0]) ) * 4;
    output[offset] = static_cast<ulong>(p2_state[0]);
    output[offset + 1] = static_cast<ulong>(p2_state[1]);
    output[offset + 2] = static_cast<ulong>(p2_state[2]);
    output[offset + 3] = static_cast<ulong>(p2_state[3]);
}

kernel void poseidon_hash_tree_level(
    constant Uniforms & uniforms[[buffer(1)]],
    device ulong * output[[buffer(0)]],
    uint2 gid[[thread_position_in_grid]]
) {
    Fp p2_state[12];
    uint offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level - 1, (gid[1] * 32768 + gid[0])  * 2) * 4;

    p2_state[0] = output[offset];
    p2_state[1] = output[offset + 1];
    p2_state[2] = output[offset + 2];
    p2_state[3] = output[offset + 3];

    p2_state[4] = output[offset + 4];
    p2_state[5] = output[offset + 5];
    p2_state[6] = output[offset + 6];
    p2_state[7] = output[offset + 7];
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;
    poseidon_permute(p2_state);
    
    offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level, (gid[1] * 32768 + gid[0]) ) * 4;
    
    output[offset] = static_cast<ulong>(p2_state[0]);
    output[offset + 1] = static_cast<ulong>(p2_state[1]);
    output[offset + 2] = static_cast<ulong>(p2_state[2]);
    output[offset + 3] = static_cast<ulong>(p2_state[3]);
}
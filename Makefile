main:
	xcrun -sdk macosx metal -c ./shaders/poseidon_merkle_hasher.metal -o ./shaders/poseidon_merkle_hasher.air
	xcrun -sdk macosx metallib ./shaders/poseidon_merkle_hasher.air -o ./shaders/poseidon_merkle_hasher.metallib
	rm ./shaders/poseidon_merkle_hasher.air
	cargo run --release
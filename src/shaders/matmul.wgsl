// Matrix multiplication: C = A × B
// Each workgroup computes a TILE_SIZE × TILE_SIZE block of C.
// Uses shared memory tiling for better cache behavior.

const TILE_SIZE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: vec4<u32>; // x=M, y=N, z=K

var<workgroup> tileA: array<f32, 256>; // TILE_SIZE * TILE_SIZE
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let M = uniforms.x;
  let N = uniforms.y;
  let K = uniforms.z;

  let row = gid.y;
  let col = gid.x;

  var sum: f32 = 0.0;
  let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A into shared memory
    let aCol = t * TILE_SIZE + lid.x;
    if (row < M && aCol < K) {
      tileA[lid.y * TILE_SIZE + lid.x] = a[row * K + aCol];
    } else {
      tileA[lid.y * TILE_SIZE + lid.x] = 0.0;
    }

    // Load tile of B into shared memory
    let bRow = t * TILE_SIZE + lid.y;
    if (bRow < K && col < N) {
      tileB[lid.y * TILE_SIZE + lid.x] = b[bRow * N + col];
    } else {
      tileB[lid.y * TILE_SIZE + lid.x] = 0.0;
    }

    workgroupBarrier();

    // Compute partial dot product for this tile
    for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
      sum = sum + tileA[lid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + lid.x];
    }

    workgroupBarrier();
  }

  // Write result
  if (row < M && col < N) {
    c[row * N + col] = sum;
  }
}

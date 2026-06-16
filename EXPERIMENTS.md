### Experiment 1

1. Create voxel map
2. Compute average coord of each voxel cluster (or voxel)
3. Compute positional embedding for each coord above
4. Add (or concat) into final embedding for each patch
5. ("Optional") Use NN/encoder to map

### Experiment 2

1. Create voxel map
2. Create patches the same size as one voxel
3. Compute positional embeddings by using the voxel as X, Y, Z coords
4. Grid search to find best voxel/patch size

### Experiment 3

1. Compute average depth per patch
2. Go pixel by pixel in current frame and assign it to a patch. If pixel depth is close to average depth of the patch its in, place it there. If it is above some threshold (maybe just do KNN here), then assign to different patch
3. Compute positional embeddings

### Experiment 4

Create nonuniform patches by clustering regions of the same depth/object (RGB), and respecitive positional embeddings. Is not compatible with ViT used here (Sigflip v1).

### Experiment 5

Positional embeddings of pixel distributions within a patch from centroid, via some combination of NN or sinusoidal

### Current built-in modes (`world_position_embedding_type`)

Composite string, 3 composable axes (llava_arch.py:384-517):

- **Coord reduction** (pick one): `avg` mean of patch | `sample1/5/9` N sampled pts | `minmax` min&max corner.
- **PE encoder** (pick one): `sin3d` fixed sinusoidal (concat pts into slots) | `mlp` learned 3→512→hidden.
- **Modifiers** (optional): `discrete` quantize coords to voxel grid first | `mrope` use discrete coords as multimodal RoPE.

PE is added to the patch feature. e.g. `sample9_sin3d`, `avg_mlp_discrete`.

### Benchmarks

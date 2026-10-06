# InternVideo2 retrieval for three datasets

Goal: extract features and evaluate MSR-VTT, VATEX, and ActivityNet Captions on the user's downloaded combined subsets, then push to KarenYuusha/visual-retrieval.

Architecture: retain the proven standalone model and corrected CLS-only tokenizer. Dataset adapters return gallery items and caption queries; extraction caches features; evaluation scores a combined gallery separately for each dataset. A runner handles all three with explicit paths.

Constraints: Python 3.11, explicit CUDA PyTorch wheels, no training, no silent missing captions or incomplete-gallery evaluation, preserve existing CLIP tools and data. MSR-VTT cache settings remain compatible. ActivityNet defaults to caption segments, matching the existing CLIP extraction; an explicit video mode joins captions as paragraphs.

Tasks:
1. Write failing adapter and temporal sampling tests, implement strict subset/annotation loaders and interval sampling, run tests.
2. Integrate adapters into extraction, manifests, evaluation and search. Test cached end-to-end extraction/evaluation for all datasets and corrupted/incomplete inputs.
3. Add all-dataset runner, installation/run documentation and ignore rules. Run the full suite and annotation preflight against repository datasets. Obtain final code review, resolve significant findings, push one commit to main with a head lease.

Review focus: timestamp boundaries, shared source files with different segments, train/validation overlap, query/gallery label alignment, old cache reuse, incomplete gallery rejection, dataset-specific output locations, CUDA dependencies.

Publication is explicitly authorized by the user. No full checkpoint inference is possible in this workspace without a GPU/user videos; report that limitation accurately.

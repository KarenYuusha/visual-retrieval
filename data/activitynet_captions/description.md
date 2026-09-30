# ActivityNet Captions Subset

Dataset:
ActivityNet Captions

Annotations:
Official ActivityNet Captions train.json and val_1.json.

Video source:
TornadoLabs/activitynet direct MP4 mirror.

Seed:
67

Train target:
500

Train successfully downloaded:
500

Validation target:
500

Validation successfully downloaded:
500

## IDs

Files retain the original ActivityNet Captions identifier.

Example:

v_QOlSCBRmfWY.mp4

## Captions

train.json and val_1.json contain:

- duration
- timestamps
- sentences

Each video contains multiple temporally localized captions.

## Evaluation

For a whole-video retrieval baseline, the segment sentences can
be concatenated to form one paragraph query for the video.

subset.json defines the exact videos used in this project.

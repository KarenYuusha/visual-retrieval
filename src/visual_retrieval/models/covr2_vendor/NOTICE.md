# CoVR-2 source notice

- `Qformer.py`: [lucas-ventura/CoVR](https://github.com/lucas-ventura/CoVR), revision
  `dd8a1a54e965d743314ce305eb1b8d64ed4e5860`, `src/model/blip2/Qformer.py`.
  Copyright Salesforce; BSD-3-Clause header retained. See [component license](LICENSE).
- `eva_vit.py`: Salesforce/LAVIS, revision `05cb670f47296522e24e7727e366b4a997d28582`,
  `lavis/models/eva_vit.py`. See [license](LICENSE); original notices retained.

Changes: removed the LAVIS download factory; preserved VisionTransformer and
position interpolation; CPU stochastic-depth linspace supports meta-device loading.
Vendored whitespace is normalized without behavior changes. The local inference
wrapper omits training, distributed evaluation and generative decoder dependencies;
it is not an official authors' package.
The upstream CoVR repository's MIT license is included as [COVR-LICENSE](COVR-LICENSE).

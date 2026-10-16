# CoVR-2 inference components

Qformer.py: lucas-ventura/CoVR, commit dd8a1a54e965d743314ce305eb1b8d64ed4e5860,
src/model/blip2/Qformer.py. Copyright Salesforce, BSD-3-Clause (header retained).

eva_vit.py: Salesforce/LAVIS lavis/models/eva_vit.py, commit
05cb670f47296522e24e7727e366b4a997d28582.
The LAVIS network-downloading factory was removed; the original VisionTransformer
implementation and position interpolation are preserved. The stochastic-depth
linspace is explicitly created on CPU to permit meta-device weight loading. Original copyright and
license notices are retained. See LICENSE in this directory.

Vendored source whitespace is normalized without changing behavior.

The local inference wrapper deliberately omits training, distributed evaluation,
and generative decoder dependencies. It is not an official authors' package.

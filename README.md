# FaceTrackerT

DlibをPyCall経由で実行し、顔の情報を取得する。

## MEMO

```julia
using Conda
Conda.add("dlib"; channel="conda-forge")
Conda.add("opencv"; channel="conda-forge")
Conda.add("numpy")
```
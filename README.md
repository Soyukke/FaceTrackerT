# FaceTrackerT

DlibをPyCall経由で実行し、顔の情報を取得する。

## MEMO

condaでinstallするとCUDNNパスを設定していても`DLIB_USE_CUDA=true`とならなかったので、pipでinstallしてbuildする。
Windowsの場合、CMake, Visual Studioをインストールする必要がある。
```julia
using PyCall
# Dlib install by pip
run(`$(PyCall.python) -m pip install dlib`)
```

```julia
using Conda
# Conda.add("dlib"; channel="conda-forge")
Conda.add("opencv"; channel="conda-forge")
Conda.add("numpy")
```

## カメラ画像の取得

- opencv, VideoIOでOBS Virtual Cameraの画像を取得するには、標準の仮想カメラ機能から取得できない。
https://github.com/obsproject/obs-studio/issues/3635

```julia
using Makie, VideoIO
# FIXME カメラ名で取得できない
f = VideoIO.opencamera()
img = read(f)
imshow(img)

scene = Makie.Scene(resolution = size(img))
makieimg = Makie.image!(scene, img; show_axis=false, scale_plot=true)

while !eof(f)
    # OffsetArrays -> Array
    img=read(f)
    makieimg[1] = img
    sleep(1/30)
end
```

## TODO

* [ ] ポイントからJulia Structに変換
* [ ] Julia ServerでREST対応する
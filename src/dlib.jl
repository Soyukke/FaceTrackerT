using Conda
using PyCall
using Plots
using Images
using Makie, VideoIO
using LinearAlgebra

using CoordinateTransformations, OffsetArrays, Rotations

export main0, drawcamera, img2pyimg, setmarkerfunc, track

"""
Matrix{Int32}
row: [x, y, w, h]
"""
function faceposition(cascade, grayimg)
    faces = cascade.detectMultiScale(grayimg, minSize=(100, 100))
end

"""
points
Tuple of (x, y) list
"""
function landmark(detector, predictor, grayimg)
    rects = detector(grayimg, 1)
    results = []
    for rect in rects
        ps = predictor(grayimg, rect).parts()
        push!(results, ps)
    end
    if 0 < length(results)
        return first(results)
    else
        return []
    end
end

"""
test capture landmarks.
"""
function main0()
    # fnimg = "dev/input01.jpg"
    dlib = pyimport("dlib")
    cv2 = pyimport("cv2")
    # Array{UInt8, 3}
    img = cv2.imread("dev\\input01.jpg")
    # 顔認識
    fn_cascade = "dev/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(fn_cascade)
    # Array{UInt8, 2}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 矩形リスト
    faces = faceposition(cascade, gray)
    detector = dlib.get_frontal_face_detector()
    fn_shape = "dev/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(fn_shape)
    mark = landmark(detector, predictor, gray)
    color = (21, 255, 12)

    for point2d in mark
        img = cv2.drawMarker(img, (point2d.x, point2d.y), color)
    end
    # BGR -> RGB
    # drawMarkerの戻り値はUMatというPyObjectで、`.get()`で値を取得できる。`
    img_rgb = mapslices(x -> RGB((reverse(Float64.(x) ./ 255))...), img.get(), dims=[3])[:, :, 1]
    # imshow(img_rgb)
    return img_rgb, img.get()
end

function setmarker(cv2, detector, predictor, jlimg)
    img = img2pyimg(jlimg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mark = landmark(detector, predictor, gray)
    color = (21, 255, 12)
    if 0 < length(mark)
    # Set marker
        for point2d in mark
            img = cv2.drawMarker(img, (point2d.x, point2d.y), color)
        end
        img_rgb = mapslices(x -> RGB((reverse(Float64.(x) ./ 255))...), img.get(), dims=[3])[:, :, 1]
        return img_rgb
    else
        return jlimg
    end
end

function setmarkerfunc()
    dlib = pyimport("dlib")
    cv2 = pyimport("cv2")
    fn_shape = "dev/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(fn_shape)
    f(jlimg) = setmarker(cv2, detector, predictor, jlimg)
    return f
end

"""
drawcamera

Stream video camera.
"""
function drawcamera(;transformer=x -> x, isshow=true)
    f = VideoIO.opencamera()
    img = read(f)
    # affinmap = recenter(RotMatrix(pi / 2), Images.center(img))
    # img = parent(warp(read(f), affinmap))
    # to small image
    img = imresize(img, round.(Int, size(img) ./ 2))
    if isshow
        scene = Makie.Scene(resolution=size(img))
        makieimg = Makie.image!(scene, img; show_axis=false, scale_plot=true)
        display(scene)
    end

    while !eof(f)
        @show "hoge"
        # use transformer function
        img = read(f)
        # img = parent(warp(read(f), affinmap))
        # to small image
        img = imresize(img, round.(Int, size(img) ./ 2))
        img = transformer(img)
        if isshow
            makieimg[1] = img
        end
        sleep(1 / 120)
    end
end


"""
drawcamera

Stream video camera.
"""
function track(;niter=20)
    dlib = pyimport("dlib")
    cv2 = pyimport("cv2")
    fn_shape = "dev/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(fn_shape)

    f = VideoIO.opencamera()
    iter = 0
    while !eof(f) && iter < niter
        iter += 1
        # use transformer function
        img = read(f)
        # img = parent(warp(read(f), affinmap))
        # to small image
        img = imresize(img, round.(Int, size(img) ./ 2.5))
        pyimg = img2pyimg(img)
        graypyimg = cv2.cvtColor(pyimg, cv2.COLOR_BGR2GRAY)
        mark = landmark(detector, predictor, graypyimg)
        if 0 < length(mark)
            # Set marker
            # TODO ここでcallback実行する
            p₀ = point2vec(mark[40])
            p₁ = point2vec(mark[38])
            p₂ = point2vec(mark[42])
            v₁ = p₁ - p₀
            v₂ = p₂ - p₀
            marklist = [x for x in mark]
            # right eye
            ps_right = point2vec.(marklist[37:42])
            # left eye
            ps_left = point2vec.(marklist[43:48])
            mouse = point2vec(marklist[67])[2] - point2vec(marklist[63])[2] 
            vs = point2vec.(marklist)
            eye_right = EAR(vs[40], vs[39], vs[38], vs[37], vs[42], vs[41])
            eye_left = EAR(vs[43:48]...)
            # @show areaofpoints(ps_right), areaofpoints(ps_left), mouse
            @show eye_right, eye_left
            @show 180 * acos(abs(dot(v₁, v₂) / (norm(v₁) * norm(v₂))))
            # @show mark[39].y - mark[41].y, mark[44].y - mark[48].y
        else
            @show "nasi"
        end
    end
end

"""
点で囲まれた領域の面積
"""
function areaofpoints(points)
    area = 0.0
    for i in 1:length(points)
        j = i + 1
        if length(points) < j
            j = 1
        end
        area += points[i][1] * points[j][2] - points[j][1] * points[i][2]
    end
    return abs(area) / 2
end

function point2vec(p)
    return [p.x, p.y, zero(typeof(p.x))]
end

"""
参考
https://qiita.com/mogamin/items/a65e2eaa4b27aa0a1c23

0.2以下で目が閉じている可能性が高い
解像度が低すぎるとうまく働かない
"""
function EAR(p₁, p₂, p₃, p₄, p₅, p₆)
    return (norm(p₂ - p₆) + norm(p₃ - p₅)) / (2 * norm(p₁ - p₄))
end

"""
Transform for read by Python.
julia: Matrix{RGB}, Float64 0 ~ 1
python: (width, height, BGR), UInt8 0 ~ 255
"""
function img2pyimg(img)
    pyimg = zeros(UInt8, size(img)..., 3)
    img = img .* 255
    for i in 1:size(img, 1), j in 1:size(img, 2)
        pyimg[i, j, :] .= round(UInt8, img[i, j].b), round(UInt8, img[i, j].g), round(UInt8, img[i, j].r)
    end
    return pyimg
end
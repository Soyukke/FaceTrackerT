using Conda
using PyCall
using Plots
using Images
using Makie, VideoIO

using CoordinateTransformations, OffsetArrays, Rotations

export main0, drawcamera, img2pyimg, setmarkerfunc

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
    return results
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
    marks = landmark(detector, predictor, gray)
    color = (21, 255, 12)

    for point2d in marks[1]
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
    marks = landmark(detector, predictor, gray)
    color = (21, 255, 12)
    if 0 < length(marks)
    # Set marker
        for point2d in marks[1]
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
function drawcamera(;transformer=x -> x)
    f = VideoIO.opencamera()
    img = read(f)
    # affinmap = recenter(RotMatrix(pi / 2), Images.center(img))
    # img = parent(warp(read(f), affinmap))
    # to small image
    img = imresize(img, round.(Int, size(img) ./ 2))
    scene = Makie.Scene(resolution=size(img))
    makieimg = Makie.image!(scene, img; show_axis=false, scale_plot=true)
    display(scene)

    while !eof(f)
        # use transformer function
        img = read(f)
        # img = parent(warp(read(f), affinmap))
        # to small image
        img = imresize(img, round.(Int, size(img) ./ 2))
        img = transformer(img)
        makieimg[1] = img
        sleep(1 / 120)
    end
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
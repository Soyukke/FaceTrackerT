using Conda
using PyCall

export main0

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
    results = landmark(detector, predictor, gray)
end
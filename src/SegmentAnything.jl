module SegmentAnything

using CondaPkg
using ColorTypes
using GeometryBasics
using PythonCall
using ColorTypes: N0f8, RGB

const cv = PythonCall.pynew()
const sam = PythonCall.pynew()
const np = PythonCall.pynew()

export MaskPredictor, MaskGenerator, ImageMask

function __init__()
    PythonCall.pycopy!(cv, pyimport("cv2"))
    PythonCall.pycopy!(sam, pyimport("segment_anything"))
    PythonCall.pycopy!(np, pyimport("numpy"))
end

include("mask.jl")

end

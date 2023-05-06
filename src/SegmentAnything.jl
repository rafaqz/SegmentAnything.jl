module SegmentAnything

using CondaPkg
using ColorTypes
using GeometryBasics
using PythonCall
using ColorTypes: N0f8, RGB

const cv = PythonCall.pynew()
const sam = PythonCall.pynew()
const np = PythonCall.pynew()
const torch = PythonCall.pynew()
const gc = PythonCall.pynew()

export ImageMask, SamPredictor, SamAutomaticMaskGenerator, predict, generate

const DEFAULT_MODEL_CHECKPOINT = realpath(joinpath(@__DIR__, "../deps/sam_vit_h_4b8939.pth"))
const CURRENT_MODEL_CHECKPOINT = Base.RefValue{String}(DEFAULT_MODEL_PATH)
const CURRENT_MODEL = Base.RefValue{Any}(nothing)

function __init__()
    PythonCall.pycopy!(cv, pyimport("cv2"))
    PythonCall.pycopy!(sam, pyimport("segment_anything"))
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(torch, pyimport("torch"))
    PythonCall.pycopy!(gc, pyimport("gc"))
end

include("mask.jl")
include("polygons.jl")

end

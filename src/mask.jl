using ColorTypes: N0f8, RGB

const DEFAULT_MODEL = realpath(joinpath(@__DIR__, "../deps/sam_vit_h_4b8939.pth"))

"""
    MaskPredictor

    MaskPredictor(; model_path=DEFAULT_MODEL, device="cuda")

A wrapper for "SamPredictor"

- `model_path`: The path to an alternate model file.
- `device`: can be "cuda" or "cpu".
"""
struct MaskPredictor
    predictor::Any#PythonCall.Py
end
function MaskPredictor(; model_path=DEFAULT_MODEL, device="cuda")
    model = sam.build_sam(model_path)
    model.to(device=device)
    predictor = sam.SamPredictor(model)
    return MaskPredictor(predictor)
end

function set_image(predictor::MaskPredictor, image::Matrix{<:Colorant})
    img = convert(Matrix{RGB{N0f8}}, image)
    # bring into correct form:
    A = permutedims(reinterpret(reshape, UInt8, img), (2, 3, 1))
    predictor.predictor.set_image(np.asarray(A))
end

"""
    MaskPredictor

    MaskPredictor(; model_path=DEFAULT_MODEL, device="cuda")

A wrapper for "SamAutomaticMaskGenerator".

- `model_path`: The path to an alternate model file.
- `device`: can be "cuda" or "cpu".
"""
struct MaskGenerator
    generator::Any#PythonCall.Py
end
function MaskGenerator(; model_path=DEFAULT_MODEL, device="cuda")
    model = sam.build_sam(model_path)
    model.to(device=device)
    generator = sam.SamAutomaticMaskGenerator(model)
    return MaskGenerator(generator)
end

function generate(generator::MaskGenerator, image::Matrix{<:Colorant})
    img = convert(Matrix{RGB{N0f8}}, image)
    # bring into correct form:
    generator.generator.generate(permutedims(reinterpret(reshape, UInt8, img), (2, 3, 1)))
end

"""
    ImageMask

    ImageMask(predictor::MaskPredictor, image::AbstractMatrix{<:Colorant}; kw...)

# Arguments

- `predictor`: a MaskPredictor object.
- `image`: the image to make masks of, as a AbstractMatrix{<:Colorant}.

# Keywords

-`multimask`: Return multiple masks, `true` by default.
-`points`: a `Vector` of `Tuple` or GeometryBasics.jl `Point`.
-`labels`: a `Vector{Int}` or `Vector{Bool}`
-`box`: A GeometryBasics `Rect` or a `[x1, y2, x2, y1]` bounding box.
"""
mutable struct ImageMask
    predictor::MaskPredictor
    original::AbstractMatrix{<:Colorant}
    masks::Array{Bool,3}
    scores::Vector{Float32}
    logits::Array{Float32,3}
    input::Union{Nothing,NamedTuple}
end
function ImageMask(predictor::MaskPredictor, image::AbstractMatrix{<:Colorant}; kw...)
    set_image(predictor, image)
    mimg = ImageMask(
        predictor, 
        image, 
        zeros(Bool, 3, size(image)...), 
        Float32[], 
        zeros(Float32, 0, 0, 0), 
        nothing,
    )
    if !isempty(kw)
        get_mask!(mimg; kw...)
    end
    return mimg
end

function get_mask!(image::ImageMask; 
    multimask=true, points=nothing, labels=nothing, box=nothing
)
    if all(isnothing, (points, labels, box))
        error("use `automatic_masks(predictor, image)")
    else
        kw = (; multimask, points, labels, box)
        masks, scores, logits = get_mask(image.predictor; kw...)
        image.masks = pyconvert(Array, masks)
        image.scores = pyconvert(Array, scores)
        image.logits = pyconvert(Array, logits)
        image.input = kw
        return masks, scores, logits
    end
end
function get_mask(predictor::MaskPredictor;
    multimask=true, points=nothing, labels=nothing, box=nothing
)
    if all(isnothing, (points, labels, box))
        error("use `MaskGenerator` if you have not points or boxes")
    else
        masks, scores, logits = predictor.predictor.predict(
            point_coords=convert_points(points),
            point_labels=convert_labels(labels),
            box=convert_box(box),
            multimask_output=multimask,
        )
    end
end

convert_points(points::Nothing) = points
convert_points(point::Point{2}) = convert_points([point])
function convert_points(
    points::Vector{<:Union{Vec{2,T},Point{2,T},NTuple{2,T}}}
) where T<:Number
    return np.asarray(permutedims(reinterpret(reshape, T, points), (2, 1)))
end

convert_labels(labels::Nothing) = labels
convert_labels(labels::Vector{<:Number}) = np.asarray(labels)

convert_box(box::Nothing) = box
function convert_box(box::Rect2)
    ((xmin, ymin), (xmax, ymax)) = extrema(box)
    return np.asarray([xmin ymin xmax ymax])
end
function convert_box(box::AbstractArray)
    if size(box) != (1, 4)
        error("Needs to be a matrix of 4 values, e.g. `[xmin ymin xmax ymax]`")
    end
    return np.asarray(collect(box))
end

"""
    SamPredictor

    SamPredictor(; checkpoint=DEFAULT_MODEL, device="cuda")

A wrapper for "SamPredictor"

- `model_type`: The sam model type. "vit_h" by default.
- `checkpoint`: The path to an alternate model file.
- `device`: can be "cuda" or "cpu".
"""
struct SamPredictor
    predictor::PythonCall.Py
end
function SamPredictor(; kw...)
    sam = sam.sam_model_registry["<model_type>"](checkpoint="")
    predictor = sam.SamPredictor(_load_model(; kw...))
    return SamPredictor(predictor)
end

"""

    predict(predictor::SamPredictor[, image::Matrix{<:Colorant}])

Wrapper function to call `predictor.predict(args...)`.

It may have an `image` argument that will update the predictor
image with `set_image!` prior to running predictions. 

# Keywords

- `multimask`: Return multiple masks, `true` by default.
- `points`: a `Vector` of `Tuple` or GeometryBasics.jl `Point`.
- `labels`: a `Vector{Int}` or `Vector{Bool}`
- `box`: A GeometryBasics `Rect` or a `[x1, y2, x2, y1]` bounding box.
"""
function predict(predictor::SamPredictor, image::Matrix{<:Colorant}=nothing;
     point_labels=nothing,
     point_coords=nothing,
     box=nothing,
     multimask=true
)
    all(isnothing, (points, labels, box)) && error("use SamAutomaticMaskGenerator if you have no points, labels or bbox")

    isnothing(image) || set_image!(predictor, image)
    masks, scores, logits = predictor.predictor.predict(
        point_coords=convert_points(points),
        point_labels=convert_labels(labels),
        box=convert_box(box),
        multimask_output=multimask,
    )
end

function set_image!(predictor::SamPredictor, image::Matrix{<:Colorant})
    A = _python_image(image)
    predictor.predictor.set_image(A)
    return A
end

"""
    SamAutomaticMaskGenerator

    SamAutomaticMaskGeneratorredictor(; model_path=DEFAULT_MODEL, device="cuda")

A wrapper for "SamAutomaticMaskGenerator".

- `model_path`: The path to an alternate model file.
- `device`: can be "cuda" or "cpu".
"""
struct SamAutomaticMaskGenerator
    generator::PythonCall.Py
end
function SamAutomaticMaskGenerator(; kw...)
    model = _load_model(; kw...)
    generator = sam.SamAutomaticMaskGenerator(model)
    return MaskGenerator(generator)
end

function generate(generator::SamAutomaticMaskGenerator, image::Matrix{<:Colorant})
    return generator.generator.generate(_python_image(A))
end

"""
    ImageMask

    ImageMask(predictor::SamPredictor, image::AbstractMatrix{<:Colorant}; kw...)

ImageMask is a wrapper around `SamPredictor` and `predict` funcion that, makes it a
little nicer to work with, and converts the results to julia objects. 

# Arguments

- `predictor`: a SamPredictor object.
- `image`: the image to make masks of, as a AbstractMatrix{<:Colorant}.

# Keywords (as for `predict`)

-`multimask`: Return multiple masks, `true` by default.
-`points`: a `Vector` of `Tuple` or GeometryBasics.jl `Point`.
-`labels`: a `Vector{Int}` or `Vector{Bool}`
-`box`: A GeometryBasics `Rect` or a `[x1, y2, x2, y1]` bounding box.
"""
mutable struct ImageMask
    original::AbstractMatrix{<:Colorant}
    masks::Array{Bool,3}
    scores::Vector{Float32}
    logits::Array{Float32,3}
    input::Union{Nothing,NamedTuple}
end
function ImageMask(predictor::SamPredictor, image::AbstractMatrix{<:Colorant}; kw...)
    set_image!(predictor, image)
    mimg = ImageMask(
        image, 
        zeros(Bool, 3, size(image)...),
        Float32[], 
        zeros(Float32, 0, 0, 0), 
        nothing,
    )
    if !isempty(kw)
        get_mask!(mimg; predictor, kw...)
    end
    return mimg
end

function get_mask!(image::ImageMask; kw...)
    masks, scores, logits = predict(image.predictor; kw...)
    image.masks = pyconvert(Array, masks)
    image.scores = pyconvert(Array, scores)
    image.logits = pyconvert(Array, logits)
    image.input = values(kw)
    return masks, scores, logits
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
function _convert_box(box::Rect2)
    ((xmin, ymin), (xmax, ymax)) = extrema(box)
    return np.asarray([xmin ymin xmax ymax])
end
function convert_box(box::AbstractArray)
    if size(box) != (1, 4)
        error("Needs to be a matrix of 4 values, e.g. `[xmin ymin xmax ymax]`")
    end
    return np.asarray(collect(box))
end

function unsafe_empty_cache() 
    gc.collect()
    torch.cuda.empty_cache()
end

function _load_model(; 
    checkpoint=DEFAULT_MODEL_CHECKPOINT, 
    model_type="vit_h"
    device="cuda",
)
    # It seems easy to exaust GPU memory by loading the model multiple times,
    # As for some reason pytorch does not seem to garbage collect them.
    # So we cache models in a global Ref, and only load them again
    # if a new model_path is passed in. Its really best to only use one
    # model per session.
    model = sam.sam_model_registry[model_type](checkpoint = checkpoint)
    model.to(device=device)
    # model = if isnothing(model_path)
    #     if isnothing(CURRENT_MODEL[])
    #         # Probably the first call, load the default model
    #         # CURRENT_MODEL[] = _load_model(DEFAULT_MODEL_PATH, "cuda")
    #     end
        # We already have a predictor loaded, use it
        # return CURRENT_MODEL[]
    # else
    #     # We want a different model to the one loaded
    #     if model_path != CURRENT_MODEL_PATH[]
    #         model = sam.sam_model_registry[model_type](checkpoint = checkpoint)
    #         model = sam.build_sam(model_path)
    #         return model
    #     end
    # end
end

function _python_image(image)
    img = convert(Matrix{RGB{N0f8}}, image)
    permuted = permutedims(reinterpret(reshape, UInt8, img), (2, 3, 1))
    return np.asarray(permuted)
end

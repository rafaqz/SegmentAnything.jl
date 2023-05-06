"""
    MaskPredictor

    MaskPredictor(; model_path=DEFAULT_MODEL, device="cuda")

A wrapper for "SamPredictor"

- `model_path`: The path to an alternate model file.
- `device`: can be "cuda" or "cpu".
"""
struct MaskPredictor
    predictor::PythonCall.Py
end
function MaskPredictor(; kw...)
    # Its easy to exaust GPU memory by loading the model multiple times,
    # As for some reason pytorch does not seem to garbage collect them.
    # So we cache models in a global Ref, and only load them again
    # if a new model_path is passed in. Its really best to only use one
    # model per session.
    predictor = sam.SamPredictor(_load_model(; kw...))
    return MaskPredictor(predictor)
end

function set_image(predictor::MaskPredictor, image::Matrix{<:Colorant})
    A = python_image(image)
    predictor.predictor.set_image(A)
end

function python_image(image)
    img = convert(Matrix{RGB{N0f8}}, image)
    permuted = permutedims(reinterpret(reshape, UInt8, img), (2, 3, 1))
    return np.asarray(permuted)
end

"""
    MaskGenerator

    MaskPGeneratorredictor(; model_path=DEFAULT_MODEL, device="cuda")

A wrapper for "SamAutomaticMaskGenerator".

- `model_path`: The path to an alternate model file.
- `device`: can be "cuda" or "cpu".
"""
struct MaskGenerator
    generator::PythonCall.Py
end
function MaskGenerator(; kw...)
    model = _load_model(; kw...)
    generator = sam.SamAutomaticMaskGenerator(model)
    return MaskGenerator(generator)
end

function generate(generator::MaskGenerator, image::Matrix{<:Colorant})
    A = python_image(image)
    return generator.generator.generate(A)
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
    original::AbstractMatrix{<:Colorant}
    masks::Array{Bool,3}
    scores::Vector{Float32}
    logits::Array{Float32,3}
    input::Union{Nothing,NamedTuple}
end
function ImageMask(predictor::MaskPredictor, image::AbstractMatrix{<:Colorant}; kw...)
    set_image(predictor, image)
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

function get_mask!(image::ImageMask; 
    predictor, multimask=true, points=nothing, labels=nothing, box=nothing
)
    if all(isnothing, (points, labels, box))
        error("use `automatic_masks(predictor, image)")
    else
        kw = (; multimask, points, labels, box)
        masks, scores, logits = get_mask(predictor; kw...)
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

function unsafe_empty_cache() 
    gc.collect()
    torch.cuda.empty_cache()
end

function _load_model(; model_path=DEFAULT_MODEL, device="cuda")
    predictor = if isnothing(model_path)
        if isnothing(CURRENT_MODEL[])
            # Probably the first call, load the default model
            CURRENT_MODEL[] = _load_model(DEFAULT_MODEL_PATH, "cuda")
        end
        # We already have a predictor loaded, use it
        return CURRENT_MODEL[]
    else
        # We want a different model to the one loaded
        if model_path != CURRENT_MODEL_PATH[]
            model = sam.build_sam(model_path)
            model.to(device=device)
            return model
        end
    end
end

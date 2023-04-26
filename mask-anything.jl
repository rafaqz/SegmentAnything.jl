using ColorTypes
using ColorTypes: N0f8, RGB

struct MaskPredictor
    predictor::PyObject
end

py"""
from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
"""

const cv = pyimport("cv2")
const np = pyimport("numpy")

const DEFAULT_MODEL = joinpath(@__DIR__, "sam_vit_h_4b8939.pth")

function MaskPredictor(; model_path=DEFAULT_MODEL, device="cuda")
    model = py"build_sam"(model_path)
    model.to(device=device)
    predictor = py"SamPredictor"(model)
    return MaskPredictor(predictor)
end

function set_image(predictor::MaskPredictor, image::Matrix{<: Colorant})
    img = convert(Matrix{RGB{N0f8}}, image)
    # bring into correct form:
    predictor.predictor.set_image(permutedims(reinterpret(reshape, UInt8, img), (2, 3, 1)))
end

convert_points(points::Nothing) = points
convert_points(point::Point{2}) = convert_points([point])
function convert_points(points::Vector{<: Union{Vec{2, T}, Point{2, T}, NTuple{2, T}}}) where T <: Number
    return permutedims(reinterpret(reshape, T, points), (2, 1))
end

struct MaskGenerator
    generator::PyObject
end

function MaskGenerator(; model_path=DEFAULT_MODEL, device="cuda")
    model = py"build_sam"(model_path)
    model.to(device=device)
    generator = py"SamAutomaticMaskGenerator"(model)
    return MaskGenerator(generator)
end

function generate(generator::MaskGenerator, image::Matrix{<:Colorant})
    img = convert(Matrix{RGB{N0f8}}, image)
    # bring into correct form:
    generator.generator.generate(permutedims(reinterpret(reshape, UInt8, img), (2, 3, 1)))
end

convert_labels(labels::Nothing) = labels
convert_labels(labels::Vector{<: Number}) = labels

convert_box(labels::Nothing) = labels

function convert_box(box::Rect2)
    ((xmin, ymin), (xmax, ymax)) = extrema(box)
    return [xmin ymin xmax ymax]
end

function convert_box(box::AbstractArray)
    if size(box) != (1, 4)
        error("Needs to be a matrix of 4 values, e.g. `[xmin ymin xmax ymax]`")
    end
    return collect(box)
end

mutable struct ImageMask
    predictor::MaskPredictor
    original::AbstractMatrix{<: Colorant}
    masks::Array{Bool, 3}
    scores::Vector{Float32}
    logits::Array{Float32,3}
    input::Union{Nothing, NamedTuple}
end

function ImageMask(predictor::MaskPredictor, image::AbstractMatrix{<: Colorant}; kw...)
    set_image(predictor, image)
    mimg = ImageMask(predictor, image, zeros(Bool, 3, size(image)...), Float32[], zeros(Float32, 0, 0, 0), nothing)
    if !isempty(kw)
        get_mask!(mimg; kw...)
    end
    return mimg
end

function get_mask!(image::ImageMask; multimask=true, points=nothing, labels=nothing, box=nothing)
    if all(isnothing, (points, labels, box))
        error("use `automatic_masks(predictor, image)")
    else
        nt = (; multimask, points, labels, box)
        masks, scores, logits = get_mask(image.predictor; nt...)
        image.masks = masks
        image.scores = scores
        image.logits = logits
        image.input = nt
        return masks, scores, logits
    end
end

function get_mask(predictor::MaskPredictor; multimask=true, points=nothing, labels=nothing, box=nothing)
    if all(isnothing, (points, labels, box))
        error("use `automatic_masks(predictor, image)")
    else
        masks, scores, logits = predictor.predictor.predict(
            point_coords=convert_points(points),
            point_labels=convert_labels(labels),
            box=convert_box(box),
            multimask_output=multimask,
        )
    end
end


function polygons(image::ImageMask)
    if image.input.multimask
        masks = image.masks
        return map(1:size(masks, 1)) do i
            contour = find_contours(masks[i, :, :])[1]
            return Polygon(map(contour) do xy
                x, y = Tuple(xy)
                return Point2f(y, size(masks, 2) - x)
            end)
        end
    end
end

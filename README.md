# SegmentAnything

A thin wrapper for installing and using 
[segment-anything](https://github.com/facebookresearch/segment-anything) 
in Julia.

It provides `SamPredictor` and a `predict` function and
`SamAutomaticMaskGenerator` and `generate`, as in segment-anything.

`predict` and `generate` return the raw outputs of their namesake python methods.

`ImageMask` provides a more julia wrapper for `SamPredictor` and `predict`.

## Example

```julia
using SegmentAnything, GLMakie, FileIO

# Load the predictor model
predictor = SamPredictor()

# Download an image
using Downloads
url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Beagle_and_sleeping_black_and_white_kitty-01.jpg"
Downloads.download(url, "pic.jpg")
img = load("pic.jpg")

# Plot it
p = Makie.image(rotr90(img); transparency=true)
p.axis.autolimitaspect = 1
hidedecorations!(p.axis)

# Mask the kitten, and plot
mask1 = ImageMask(predictor, img; 
  point_coords=[(400, 800)],
  point_labels=[true],
)
m1 = rotr90(map(mask1.masks[2, :, :]) do x
     x ? RGBAf(1, 0, 0, 1) : RGBAf(0, 0, 0, 0)
end)
image!(p.axis, m1; transparency=true)

# Now mask the beagle, and plot
mask2 = ImageMask(predictor, img; 
  point_coords=[(400, 600)],
  point_labels=[true],
)
m2 = rotr90(map(mask2.masks[2, :, :]) do x
     x ? RGBAf(0, 0.0, 1.0, 1) : RGBAf(0, 0, 0, 0)
end)
image!(p.axis, m2; transparency=true)

save("beagle_and_kitten.png", p.figure)
```

![beagle_and_kitten](https://user-images.githubusercontent.com/2534009/234685142-9483bd40-1af0-4912-bb25-6024ed0e06fa.png)

We can also use the automatic mask generator:

```julia
using SegmentAnything, GLMakie, FileIO, PythonCall
p = Makie.image(rotr90(img); transparency=true)

# Load the predictor model
generator = SamAutomaticMaskGenerator()
out = generate(generator, img)

# Create an array of zeros to write into
segments = rotr90(PythonCall.pyconvert(Array, out[1]["segmentation"])) .* 0

# Loop over the returned objects, which is an array of python dictionaries.
for (n, x) in enumerate(out)
    A = rotr90(PythonCall.pyconvert(Array, x["segmentation"]))
    for i in eachindex(A)
        if A[i] 
           segments[i] = n
        end
    end
end
Makie.image!(p.axis, segments; transparency=true, colormap=(:tableau_20, 2.0))

save("auto_segmentation.png", p.figure)
```

![auto_segmentation](https://user-images.githubusercontent.com/2534009/236788199-974810dc-8e44-496c-8095-bd38d4078f07.png)

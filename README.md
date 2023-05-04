# SegmentAnything

A thin wrapper for installing and using 
[segment-anything](https://github.com/facebookresearch/segment-anything) 
in Julia.

## Status:

The single object `MaskPredictor` works well and reliably, as in the example below. 

The multi-object mask `MaskGenerator` works with the `generate` funcion, but currently
returns an unwrapped python object.Calling `generate(maskgenerator)` multiple times seems to 
quickly fill GPU memory without freeing it. It's not clear if the problem is on the Python 
side or the Julia side.

`SegmentAnything.unsafe_empty_cache()` attempts to free memory, but doesn't
generally seem to work. PRs solving this would be appreciated!

## Example

```julia
using SegmentAnything, GLMakie, FileIO

# Load the predictor model
predictor = MaskPredictor()

# Download an image
using Downloads
url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Beagle_and_sleeping_black_and_white_kitty-01.jpg"
Downloads.download(url, "pic.jpg")
image = load("pic.jpg")

# Plot it
p = Makie.image(rotr90(image); transparency=true)
p.axis.autolimitaspect = 1
hidedecorations!(p.axis)

# Mask the kitten, and plot
mask1 = ImageMask(predictor, image; 
  points=[(400, 800)],
  labels=[true],
)
m = rotr90(map(mask1.masks[2, :, :]) do x
     x ? RGBAf(1, 0, 0, 1) : RGBAf(0, 0, 0, 0)
end)
image!(p.axis, m; transparency=true)

# Now mask the beagle, and plot
mask2 = ImageMask(predictor, image; 
  points=[(400, 600)],
  labels=[true],
)
m = rotr90(map(mask2.masks[2, :, :]) do x
     x ? RGBAf(0, 0.0, 1.0, 1) : RGBAf(0, 0, 0, 0)
end)
image!(p.axis, m; transparency=true)

save("beagle_and_kitten.png", p.figure)
```

![beagle_and_kitten](https://user-images.githubusercontent.com/2534009/234685142-9483bd40-1af0-4912-bb25-6024ed0e06fa.png)

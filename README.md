# SegmentAnything

A thin wrapper for installing and using 
[segment-anything](https://github.com/facebookresearch/segment-anything) 
in Julia.

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

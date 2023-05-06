using SegmentAnything

@testset "run the readme example, without Makie" begin
    # Load the predictor model
    predictor = MaskPredictor()

    # Download an image
    using Downloads
    url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Beagle_and_sleeping_black_and_white_kitty-01.jpg"
    Downloads.download(url, "pic.jpg")
    image = load("pic.jpg")

    # Mask the kitten
    mask1 = ImageMask(predictor, image; 
      points=[(400, 800)],
      labels=[true],
    )
    m = rotr90(map(mask1.masks[2, :, :]) do x
         x ? RGBAf(1, 0, 0, 1) : RGBAf(0, 0, 0, 0)
    end)

    # Now mask the beagle, and plot
    mask2 = ImageMask(predictor, image; 
      points=[(400, 600)],
      labels=[true],
    )
    m = rotr90(map(mask2.masks[2, :, :]) do x
         x ? RGBAf(0, 0.0, 1.0, 1) : RGBAf(0, 0, 0, 0)
    end)
end

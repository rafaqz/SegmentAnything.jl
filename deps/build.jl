using Downloads

filename = "sam_vit_h_4b8939.pth"
url = "https://dl.fbaipublicfiles.com/segment_anything/$filename"
path = joinpath(@__DIR__, filename)

println("Downloading segment-anything dataset to $path")
Downloads.download(url, path)

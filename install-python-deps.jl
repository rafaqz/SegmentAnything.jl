# TODO, how to install them with CondaPkg? I run into one problem after the other and then gave up
using Downloads
# Download model
Downloads.download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", joinpath(@__DIR__, "sam_vit_h_4b8939.pth"))

run(`$(PyCall.python) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`)
run(`$(PyCall.python) -m pip install git+https://github.com/facebookresearch/segment-anything.git`)
run(`$(PyCall.python) -m pip install opencv-python`)

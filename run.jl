using PyCall, FileIO
using GLMakie
using ImageTransformations
using ImageShow
using Images
using Tyler, GLMakie
using TileProviders
using MapTiles
using ImageShow

# First time, include the install process!
# include("install-python-deps.jl")
include("mask-anything.jl")
include("mouse-selection.jl")

predictor = MaskPredictor()

function gui(img)
    fig = Figure(resolution=reverse(size(img)))
    ax = Axis(fig[2, 1]; aspect=DataAspect())
    img_rot = rotr90(img)
    ax, pl = image!(ax, img_rot; axis=(; aspect=DataAspect()))
end

function axis_colorbuffer(ax)
    scene = ax.scene
    s = Makie.getscreen(scene)
    c = Makie.colorbuffer(s)
    ((xmin, ymin), (xmax, ymax)) = extrema(scene.px_area[])

    w, h = size(c)
    return (c)[(w-ymax):(w-ymin), xmin:xmax]
end

function gui!(fig, ax)
    fig[0, 1] = buttongrid = GridLayout(tellwidth=false)
    masking_enabled = buttongrid[1, 1] = Button(fig, label="Enable")
    buttons = buttongrid[1, 2:4] = [Button(fig, label="Mask $i") for i in 1:3]
    mask_button = buttongrid[1, 5] = Button(fig, label="mask! (enter)", buttoncolor=:lightblue)
    clear_button = buttongrid[1, 6] = Button(fig, label="clear", buttoncolor=:red)
    pixelsp = campixel(ax.scene)
    on(ax.scene.px_area; update=true) do area
        translate!(pixelsp, minimum(area)..., 1)
    end
    interactions = copy(Makie.interactions(ax))
    enabled = Observable(false)
    masker = nothing
    img = axis_colorbuffer(ax)
    on(masking_enabled.clicks) do n
        if isodd(n)
            img = axis_colorbuffer(ax)
            new_size = reverse(size(img))
            if new_size != size(mask_obs[1][])
                foreach(i-> mask_obs[i][] = fill(RGBAf(0,0,0,0), new_size), 1:3)
            end
            masker = ImageMask(predictor, img)
            enabled[] = true
            masking_enabled.label = "Disable"
            foreach(x -> Makie.deregister_interaction!(ax, x), keys(interactions))
        else
            enabled[] = false
            masking_enabled.label = "Enable"
            foreach(((k, (_, i)),)-> Makie.register_interaction!(ax, k, i), interactions)
        end
    end

    colors = RGBAf.(RGBf.(Makie.wong_colors()), 0.5)
    mask_obs = map(1:3) do i
        visible = map((e, x)-> e && iseven(x), enabled, buttons[i].clicks)
        mask = Observable(fill(RGBAf(0,0,0,0), reverse(size(img))))
        image!(pixelsp, mask, colormap=colors[i], transparency=true, visible=visible)
        return mask
    end

    ms = MouseSelector(ax; enabled=enabled)
    function clear!()
        empty!(ms.points[])
        empty!(ms.point_modifier[])
        ms.drag_rect[] = Rect(NaN, NaN, NaN, NaN)
        foreach(obs-> (fill!(obs[], RGBAf(0, 0, 0, 0)); notify(obs)), mask_obs)
        notify(ms.point_modifier)
        notify(ms.points)
    end

    function to_pixel(p)
        (x, y) = Makie.project(ax.scene.camera, :data, :pixel, p)[Vec(1, 2)]
        return Point2f(x, size(img, 1) - y)
    end

    function create_mask()
        attributes = Dict{Symbol, Any}()
        rect = ms.drag_rect[]
        if all(x-> !isnan(x), (rect.origin..., rect.widths...))
            mini, maxi = to_pixel.(extrema(rect))
            attributes[:box] = [mini[1] mini[2] maxi[1] maxi[2]]
        end
        points = ms.points[]
        labels = ms.point_modifier[]
        if !isempty(points)
            attributes[:points] = map(to_pixel, points)
            attributes[:labels] = map(x-> x == 1 ? 1 : 0, labels)
        end
        masks, scores, _ = get_mask!(masker; attributes...)
        for i in 1:3
            x = rotr90(view(masks, i, :, :))
            color = colors[i]
            map!(mask_obs[i][], x) do c
                return c ? color : RGBAf(0, 0, 0, 0)
            end
            notify(mask_obs[i])
            buttons[i].label[] = "Mask $i ($(round(scores[i]; digits=2)))"
        end
    end
    on(ax.scene.events.keyboardbutton) do b
        enabled[] || return Consume(false)
        if ispressed(ax.scene.events, Exclusively(Keyboard.enter))
            create_mask()
        end
    end
    on(mask_button.clicks) do n
        enabled[] || return Consume(false)
        create_mask()
    end
    on(clear_button.clicks) do n
        enabled[] || return Consume(false)
        clear!()
    end
    hidedecorations!(ax)
    return fig
end


# select a map provider
provider = TileProviders.Esri(:WorldImagery)
lat = 34.2013;
lon = -118.1714;
# convert to point in web_mercator
pts = Point2f(MapTiles.project((lon, lat), MapTiles.wgs84, MapTiles.web_mercator))
# set how much area to map in degrees
delta = 0.01;
# create rectangle for display extents in web_mercator
frame = Rect2f(lon - delta / 2, lat - delta / 2, delta, delta)
# show map

# enable: enable masking mode... May take a second to set the image for the predictor
# left_alt + left mouse drag: rectangle selection
# left mouse click: inside selection
# right mouse click: don't include in selection
# enter: create mask
begin
    m = Tyler.Map(frame; provider, figure=Figure(resolution=(1000, 1200)));
    hidedecorations!(m.axis)
    wait(m)
    gui!(m.figure, m.axis)
end

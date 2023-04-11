

function under_mouse_segmentation!(ax, img)
    on(ax.scene.events.mousebutton) do button
        if ispressed(fig.scene, Mouse.left)
            mp = ((round.(Int, mouseposition(ax.scene))))
            mp = (mp[1], size(simi, 2) - mp[2])
            masks, scores, logits = predictor.predict(
                point_coords=np.array([[mp...]]),
                point_labels=np.array([1]),
                multimask_output=true,
            )
            for i in 1:3
                mask = masks_obs[i]
                color = colors[i]
                mask[] = rotr90(map(masks[i, :, :]) do x
                    x ? color : RGBAf(0, 0, 0, 0)
                end)
            end
        end
    end
end

function interactive_segmentation(img)
    fig, ax, pl = GLMakie.image(simi; axis=(; aspect=DataAspect()), figure=(; resolution=size(simi)))
    [Makie.deregister_interaction!(ax, x) for x in keys(Makie.interactions(ax))]
    masks_obs = map(1:3) do i
        mask = Observable(zeros(RGBAf, size(simi)))
        image!(ax, mask; transparency=true)
        mask
    end
    on(fig.scene.events.mousebutton) do button
        if ispressed(fig.scene, Mouse.left)
            mp = ((round.(Int, mouseposition(ax.scene))))
            mp = (mp[1], size(simi, 2) - mp[2])
            masks, scores, logits = predictor.predict(
                point_coords=np.array([[mp...]]),
                point_labels=np.array([1]),
                multimask_output=true,
            )
            for i in 1:3
                mask = masks_obs[i]
                color = colors[i]
                mask[] = rotr90(map(masks[i, :, :]) do x
                    x ? color : RGBAf(0, 0, 0, 0)
                end)
            end
        end
    end

    display(fig)
end

pix2point(img, point) = Point2f(point[1], size(img, 2) - point[2])

function plot_img(img::ImageMask; colors = RGBAf.(RGBf.(Makie.wong_colors()), 0.5))
    image = rotr90(img.original)
    fig, ax, pl = GLMakie.image(image; axis=(; aspect=DataAspect()), figure=(; resolution=size(image)))

    for i in 1:3
        color = colors[i]
        mask = rotr90(map(img.masks[i, :, :]) do x
            x ? color : RGBAf(0, 0, 0, 0)
        end)
        image!(ax, mask)
    end
    if !isnothing(img.input) && haskey(img.input, :points)
        points = pix2point.((image,), img.input.points)
        colors = map(x -> x == 0 ? :red : :green, img.input.labels)
        scatter!(ax, points; markersize=20, marker=:cross, color=colors, strokewidth=1, strokecolor=:white)
    end
    return fig, ax
end

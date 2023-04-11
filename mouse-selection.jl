using GLMakie


struct MouseSelector
    drag_rect::Observable{Rect2{Float64}}
    points::Observable{Vector{Point2f}}
    point_modifier::Observable{Vector{Int}}
    modifier_keys::Vector{Any}
end

function MouseSelector(ax;
        modifiers=[Mouse.left, Mouse.right],
        delete=Mouse.left & Keyboard.left_shift, modifier_colors=[:blue, :red],
        marker=:star6,
        markersize=25,
        enabled=Observable(true))

    ms = MouseSelector(
        Observable(Rect2(NaN, NaN, NaN, NaN)),
        Observable(Point2f[]),
        Observable(Int[]),
        modifiers
    )

    visible = Observable(false)
    rect = Makie.RectangleZoom(ax; modifier=Keyboard.left_alt) do rect
        enabled[] || return false
        ms.drag_rect[] = Rect2{Float64}(rect)
        visible[] = true
    end

    rectplot = poly!(ax, ms.drag_rect, color=(:black, 0.2), strokecolor=:blue, strokewidth=2, visible=visible)

    Makie.register_interaction!(ax, :rect_select, rect)

    events = ax.scene.events
    colors = map(ax.scene, ms.point_modifier) do i
        return modifier_colors[i]
    end

    scatter_plot = scatter!(ax, ms.points, color=colors, marker=marker, markersize=markersize, strokewidth=1, strokecolor=:white)

    on(ax.scene, events.mousebutton) do mb
        enabled[] || return Consume(false)
        if ispressed(events, Exclusively(delete))
            plot, idx = pick(ax.scene)
            if plot === scatter_plot
                deleteat!(ms.points[], idx)
                deleteat!(ms.point_modifier[], idx)
                notify(ms.points)
                notify(ms.point_modifier)
                return Consume(true)
            elseif plot in Makie.flatten_plots(rectplot)
                visible[] = false
                ms.drag_rect[] = Rect2(NaN, NaN, NaN, NaN)
            end
        end
        return Consume(false)
    end

    onany(ax.scene, events.mousebutton, events.keyboardbutton) do mb, kb
        enabled[] || return Consume(false)
        for (i, mod) in enumerate(modifiers)
            if ispressed(events, Exclusively(mod))
                # Todo on first click without window focus, mouse position hasn't been triggered yet
                # if events.mouseposition[] == (0.0, 0.0)
                # get it directly from the window in that case
                #     screen = Makie.getscreen(ax.scene)
                #     glfwGetCursorPos(screen)
                # end
                mp = mouseposition(ax.scene)
                push!(ms.points[], mp)
                push!(ms.point_modifier[], i)
                notify(ms.points); notify(ms.point_modifier)
                return Consume(true)
            end
        end
        return Consume(false)
    end

    return ms
end

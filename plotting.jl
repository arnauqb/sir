using PyPlot
using Printf

function plot_curves(curves, labels)
    n = length(curves)
    f, ax = subplots(1, n, figsize = (15, 5))
    for (curve, label) in zip(curves, labels)
        ax[1].plot(curve[1], label = label)
        ax[2].plot(curve[2], label = label)
        if n == 3
            ax[3].plot(curve[3], label = label)
        end
    end
    ax[1].set_title("S")
    ax[2].set_title("I")
    if n == 3
        ax[3].set_title("R")
    end
    for i in 1:n
        ax[i].legend()
    end
    return f
end

function plot_jacobians(jacobians, labels; figsize=(12,10), qs = ["S", "I", "R"], xs = ["β", "γ"])
    f, ax = subplots(size(jacobians[1])[1], size(jacobians[1])[2], figsize = figsize)
    for (jacobian, label) in zip(jacobians, labels)
        for i in 1:size(jacobian)[1]
            for j in 1:size(jacobian)[2]
                ax[i, j].plot(jacobian[i, j], label = label)
                title = @sprintf("d%s / d%s", qs[i], xs[j])
                ax[i,j].set_title(title)
            end
        end
    end
    ax[1, 1].legend()
    return f
end

using StochasticAD, Distributions, PyPlot, Statistics, ForwardDiff, GumbelSoftmax, Random

struct GS end
struct SAD end

function sample_bernoulli(::SAD, probs)
	return map(rand ∘ Bernoulli, probs) 
end

function sample_bernoulli(::GS, probs)
	probs_cat = [probs 1.0 .- probs]
	return sample_gumbel_softmax(probs_cat, 0.1)[:, 1]
end

function step(diff_mode, infected, p, N)
	susceptible = 1.0 .- infected
	trans = sum(infected) .* susceptible / N
	probs = @. 1.0 - exp(-trans * p)
	samples = sample_bernoulli(diff_mode, probs)
	infected = infected + samples
	return infected
end


function sample_rounds(diff_mode, n_rounds; N = 1000, p = 0.1, initial_fraction_infected = 0.1)
	n_infected = round(Int, initial_fraction_infected * N)
	infected = vcat(ones(n_infected), zeros(N - n_infected))
	ret = typeof(p)[sum(infected)/N]
	for i in 1:n_rounds
		infected = step(diff_mode, infected, p, N)
		push!(ret, sum(infected) / N)
	end
	return ret
end

function do_n_steps(diff_mode, infected, p, N, n_steps)
    ret = []
    for i in 1:n_steps
        infected = step(diff_mode, infected, p, N)
        push!(ret, infected)
    end
    return ret, sum(stack(ret), dims=1)[:] / N
end

function sample_c(diff_mode, n_rounds, gradient_horizon; N = 1000, p = 0.1, initial_fraction_infected = 0.1)
	n_infected = round(Int, initial_fraction_infected * N)
	infected = vcat(ones(n_infected), zeros(N - n_infected))
    # until gradient horizon just simulate normally and get derivative
    infected_hist, I = do_n_steps(diff_mode, infected, p, N, gradient_horizon)
    I_deriv = derivative_estimate(x -> do_n_steps(diff_mode, infected, x, N, gradient_horizon)[2], p)
    # from gradient horizon then we need to start from infected at time t - gradient_horizon
    for i = gradient_horizon+1:n_rounds
        infected = infected_hist[end - gradient_horizon + 1]
        inf_t, I_t = do_n_steps(diff_mode, infected, p, N, gradient_horizon)
        infected_hist = vcat(infected_hist, inf_t[end,:])
        I = vcat(I, I_t[end])
        I_deriv = vcat(I_deriv, derivative_estimate(x -> do_n_steps(diff_mode, infected, x, N, gradient_horizon)[2][end], p))
    end
    return I, I_deriv
end

I, I_deriv = sample_c(SAD(), 100, 1, p = 0.1)

f, ax= subplots()
ax.plot(I)
display(f)

f, ax = subplots()
ax.plot(I_deriv)
display(f)

##

beta = 0.1
nt = 100
ts = sample_rounds(SAD(), nt, p = beta)
ts2 = sample_rounds(GS(), nt, p = beta)
f, ax = subplots()
ax.plot(ts)
ax.plot(ts2)
ax.plot(I)
display(f)
##

## gradients

grads = mean([derivative_estimate(x -> sample_rounds(SAD(), nt, p = x), beta, backend=SmoothedFIsBackend()) for i in 1:1])
grads2 = ForwardDiff.derivative(x -> sample_rounds(SAD(), nt, p = x), beta)
grads3 = ForwardDiff.derivative(x -> sample_rounds(GS(), nt, p = x), beta)
#grads4 = mean([sample_c(SAD(), nt, 50, p = beta)[2] for i in 1:100]);

##
##
f, ax = subplots()
ax.plot(grads, label = "non-smooth")
ax.plot(grads2, label = "smooth")
ax.plot(grads3, label = "gs")
#ax.plot(grads4, label = "custom");
ax.legend()
display(f)

##
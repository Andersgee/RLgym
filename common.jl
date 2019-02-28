function adam!(θ, ∇, m, v, αᵪ, β₁, β₂, ϵ)
    #ADAM optimizer. see (Kingma & Ba, 2017) https://arxiv.org/pdf/1412.6980v9
    @. m = β₁*m + (1-β₁)*∇
    @. v = β₂*v + (1-β₂)*(∇^2)
    @. θ -= m/(sqrt(v)+ϵ) * αᵪ
    @. ∇ = 0.0
    nothing
end

function adjust!(opt, model, s)
    α = 0.001
    β₁ = 0.9
    β₂ = 0.999
    ϵ = 1e-8
    αᵪ = α*sqrt(1-β₂^s)/(1-β₁^s)
    for p in keys(model), i=1:length(model[p])
        adam!(model[p][i].data, model[p][i].grad, opt[p][1][i], opt[p][2][i], αᵪ, β₁, β₂, ϵ)
    end
    nothing
end

xent(t, z) = -sum(t .* logsoftmax(z), dims=1)

xavier(a,b) = sqrt(0.5*(a+b))

function haar(d)
    #initial random orthogonal see (Saxe et al., 2014) https://arxiv.org/abs/1312.6120
    Q,R = qr(randn(d,d))
    W=Q*Diagonal(sign.(diag(R)));
    return W
end

function discount(x, γ)
    #a more descriptive name would be decayed sum of future values
    T = length(x)
    discounted_sumx = zeros(1,T)
    running_add = 0
    for t = T:-1:1
        running_add = γ*running_add + x[t]
        discounted_sumx[t] = running_add
    end
    return discounted_sumx
end

function index2onehot(x, d=4)
    T = length(x)
    onehot = zeros(d, T)
    for t = 1:T
        onehot[x[t],t]=1.0
    end
    return onehot
end
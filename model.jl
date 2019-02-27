function policy(x, m)
    z1 = leakyrelu.(m["W"][1]*x .+ m["b"][1])
    z2 = leakyrelu.(m["W"][2]*z1 .+ m["b"][2])
    z3 = m["W"][3]*z2 .+ m["b"][3]
    p = softmax(z3.data)

    action = wsample(p[:])
    #action = argmax(p[:])
    #if rand() < 0.1
    #    action = rand(1:4)
    #end
    return z3, p, action
end

xavier(a,b) = sqrt(0.5*(a+b))

#https://github.com/gabrielgarza/openai-gym-policy-gradient/blob/master/policy_gradient_layers.py
function initmodel()
    L = [8,10,10,4]

    W1=randn(L[2], L[1])./xavier(L[1],L[2])
    b1=ones(L[2],1).*0.0

    W2=randn(L[3], L[2])./xavier(L[2],L[3])
    b2=ones(L[3],1).*0.0

    W3=randn(L[4], L[3])./xavier(L[3],L[4])
    b3=ones(L[4],1).*0.0

    model=Dict("W"=>[param(W1),param(W2),param(W3)], "b"=>[param(b1),param(b2),param(b3)])
    opt = Dict("W"=>[[zero(W1),zero(W2),zero(W3)] for _=1:2], "b"=>[[zero(b1),zero(b2),zero(b3)] for _=1:2])
    
    return model, opt
end

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

function haar(d)
    #initial random orthogonal (Saxe et al., 2014) https://arxiv.org/abs/1312.6120
    Q,R = qr(randn(d,d))
    W=Q*Diagonal(sign.(diag(R)));
    return W
end


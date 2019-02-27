function actor(x, m)
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

function critic(x, m)
    z1 = leakyrelu.(m["W"][1]*x .+ m["b"][1])
    z2 = leakyrelu.(m["W"][2]*z1 .+ m["b"][2])
    z3 = m["W"][3]*z2 .+ m["b"][3]
    return z3
end

#https://github.com/gabrielgarza/openai-gym-policy-gradient/blob/master/policy_gradient_layers.py
function actorparams()
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

function criticparams()
    L = [8,10,10,1]

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



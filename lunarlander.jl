using Flux
using Distributions: wsample, std, mean
using LinearAlgebra
using Gym
import JLD
include("model.jl")
include("common.jl")

function train(λ, Nepisodes)
    env = GymEnv("LunarLander-v2")
    #model=JLD.load("model.jld"); opt=JLD.load("opt.jld")
    model, opt = actorparams()

    maxt=1000
    backdecay = zeros(1,maxt)
    rewards = zeros(1,maxt)
    decayed_sumreward = zeros(1,maxt)
    xs = zeros(8,maxt)
    qs = zeros(4,maxt)
    actions_onehot = zeros(4,maxt)

    sumreward=zeros(Nepisodes)

    for episode=1:Nepisodes
        x = reset!(env)
        fill!(backdecay, 0.0)
        fill!(xs, 0.0)
        fill!(actions_onehot, 0.0)
        fill!(decayed_sumreward, 0.0)
        
        t=0
        terminal=false
        while !terminal
            t+=1
            #interact
            q, p, action = actor(x, model)
            x_next, reward, terminal, information = step!(env, action-1)

            #store
            backdecay .*= λ; backdecay[t] = 1.0
            xs[:,t] = x
            actions_onehot[action,t] = 1.0
            decayed_sumreward += reward.*backdecay
            sumreward[episode] += reward

            #move
            x = x_next
        end

        normalized_decayed_sumreward = normalize_r(decayed_sumreward[1:t])

        qs, ps, _ = actor(xs[:,1:t], model)
        neglogprob = -sum(actions_onehot[:,1:t] .* logsoftmax(qs), dims=1) #treat picked actions as target of softmax output
        #loss = mean(neglogprob .* decayed_sumreward[1:t]') #multiply by some "score" or "advantage" function, in this case decayed sumreward
        loss = mean(neglogprob .* normalized_decayed_sumreward')
        Flux.Tracker.back!(loss)
        adjust!(opt, model, episode)

        if episode%200==0
            println("episode: ",episode,"/",Nepisodes," (saved checkpoint)")
            println("since last mean(sumreward): ",mean(sumreward[episode-199:episode]))

            JLD.save("model.jld", model)
            #JLD.save("opt.jld", opt)
        end
    end
    nothing
end

function main()
    @show λ = 0.99 #reward decay aka discount
    Nepisodes=50000
    train(λ, Nepisodes)
end

main()
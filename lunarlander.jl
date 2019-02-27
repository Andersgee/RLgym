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
    amodel, aopt = actorparams()
    cmodel, copt = criticparams()

    maxt=1000
    backdecay = zeros(1,maxt)
    rewards = zeros(1,maxt)
    decayed_sumreward = zeros(1,maxt)
    xs = zeros(8,maxt)
    qs = zeros(4,maxt)
    actions_onehot = zeros(4,maxt)

    sumreward=zeros(Nepisodes)

    entropy_β=0.01 #determines how important the loss for uncertainty of actor is

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
            logitp = actor(x, amodel)
            action = wsample(softmax(logitp.data)[:])
            #action = argmax(softmax(logitp.data)[:])
            x_next, reward, terminal, information = step!(env, action-1)

            #store "replay"
            xs[:,t] = x
            actions_onehot[action,t] = 1.0

            backdecay .*= λ; backdecay[t] = 1.0
            decayed_sumreward += reward.*backdecay #assign this reward to all visited states but with decay backwards

            sumreward[episode] += reward #for debugging

            #move
            x = x_next
        end

        #now do supervised learning on the stored data
        x = xs[:,1:t] #visited states (input)
        Rs = decayed_sumreward[:,1:t] #target of critic is actual sum of rewards (decayed) recieved after that state was visited
        target = actions_onehot[:,1:t] #target of actor is just the picked actions, but each loss will be weighted by some score/advantage value which might be negative

        #handle critic
        qs = critic(x, cmodel)
        criticloss = mean((qs .- Rs).^2) #MSE for linear output

        #handle actor
        logitps = actor(x, amodel)
        actorloss = -sum(target .* logsoftmax(logitps), dims=1) #crossentropy for softmax output
        actorloss_weighted = mean(actorloss .* (Rs .- qs.data)) #but weighted (this particular weighting is called "advantage" but there are other ways of choosing this score)
        #note for later: can use negativeAdvantage=qs-Rs here (which is used in critic MSE) if we use logprob instead of neglogprob..
        #less clear imho but more "elegant" since the two minus signs cancel

        #an extra "entropy" (crossentropy for actor using its output probabilities as target instead of picked actions)
        #this loss is low if actor is certain. so encourage exploration by subtracting this loss..
        actor_entropyloss = entropy_β * mean(-sum(softmax(logitps) .* logsoftmax(logitps), dims=1))

        loss = 0.5*criticloss + actorloss_weighted - actor_entropyloss
        Flux.Tracker.back!(loss)
        adjust!(aopt, amodel, episode)
        adjust!(copt, cmodel, episode)

        if episode%200==0
            println("episode: ",episode,"/",Nepisodes," (saved checkpoint)")
            println("since last mean(sumreward): ",mean(sumreward[episode-199:episode]))

            JLD.save("amodel.jld", amodel)
            JLD.save("cmodel.jld", cmodel)
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
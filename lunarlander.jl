using Flux
using Distributions: wsample, std, mean
using LinearAlgebra
using Gym
import JLD
include("model.jl")
include("common.jl")

function train(γ)
    env = GymEnv("LunarLander-v2")
    amodel, aopt = actorparams()
    cmodel, copt = criticparams()

    sumreward_ema = 0
    episode = 0
    while true
        episode += 1

        #generate a "replay" using the current actor
        xs=[]
        actions=[]
        rewards=[]
        x = reset!(env)
        while true
            logitp = actor(x, amodel)
            action = wsample(softmax(logitp.data)[:])
            x_next, reward, done, information = step!(env, action-1)
            push!(xs, x)
            push!(actions, action)
            push!(rewards, reward)
            x = x_next
            if done; break; end
        end
        sumreward_ema = 0.99*sumreward_ema + 0.01*sum(rewards)

        #now do supervised learning on the replay
        x = hcat(xs...)
        t_critic = discount(hcat(rewards...), γ)
        t_actor = index2onehot(hcat(actions...), 4)

        qs = critic(x, cmodel)
        criticloss = 0.5*mean((qs.-t_critic).^2) #MSE loss

        logitps = actor(x, amodel)
        actorloss = mean(xent(t_actor, logitps) .* (t_critic .- qs.data)) #crossentropy loss, but multiplied by an advantage score (in this case what we got, minus what critic thought we would get)

        actor_actionentropy = mean(xent(softmax(logitps), logitps)) #an extra entropy value for actor output probabilities, subtract this to artificially decrease loss when actor is unsure (encourages exploration)
        loss = criticloss + actorloss - 0.01*actor_actionentropy

        #backprop and adjust
        Flux.Tracker.back!(loss)
        adjust!(aopt, amodel, episode)
        adjust!(copt, cmodel, episode)

        #info
        if episode%200==0
            println("episode: ",episode," sumreward_ema: ",round(sumreward_ema, digits=2), " (saving models)")
            JLD.save("amodel.jld", amodel)
            JLD.save("cmodel.jld", cmodel)
            if sumreward_ema>200
                println("Condition met. consider it solved")
                break
            end
        end
    end
end

function main()
    γ = 0.99
    train(γ)
end

main()
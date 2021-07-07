
using Distributions

struct RewardParameters
    probs::Vector{Float64}
    rewards::Vector{Float64}
    rates::Vector{Float64}
end


mutable struct RewardState
    parameters::RewardParameters
    rewardQ::Vector{Int}
end

RewardState(parameters::RewardParameters)=RewardState(parameters,zeros(Int64,length(parameters.rewards)))

function makeRewardParameters(n::Int,alpha::Float64,beta::Float64,reward::Float64,rate::Float64)
    probs=rand(Beta(alpha,beta), n)
    rewards=reward*ones(Float64,n)
    rates=rate*ones(Float64,n)
    RewardParameters(probs,rewards,rates)
end

function updateReward(state::RewardState)
    reward=0.0::Float64
    for (i,r) in enumerate(state.rewardQ)
        givenOut=0::Int64
        for c in 1:r
            if rand()<state.parameters.rates[i]
                reward+=state.parameters.rewards[i]
                givenOut+=1
            end
        end
        state.rewardQ[i]-=givenOut
        
    end
    reward
end

function updateState(state::RewardState,choice::Int)
    if rand()<state.parameters.probs[choice]
            state.rewardQ[choice]+=1
    end
end

        
#state=RewardState(makeRewardParameters(4,0.5,0.5,1.0,0.1))

#for t in 1:100
#    updateState(state,rand([1,2,3,4]))
#    println(updateReward(state)," ",state.rewardQ)
#end


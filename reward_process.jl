
using Distributions

struct RewardParameters
    probs::Vector{Float64}
    rewards::Vector{Float64}
    qLength::Int
end


mutable struct RewardState
    parameters::RewardParameters
    rewardQ::Vector{Float64}
    qC::Int
end

RewardState(parameters::RewardParameters)=RewardState(parameters,zeros(Int64,qLength),1)

function makeRewardParameters(n::Int,alpha::Float64,beta::Float64,reward::Float64,qL::Int)
    probs=rand(Beta(alpha,beta), n)
    rewards=reward*ones(Float64,n)
    RewardParameters(probs,rewards,qL)
end

function update(state::RewardState,choice::Int)
    reward=state.rewardQ[state.qC]
    if rand()<state.parameters.probs[choice]
        location=rand(1:state.parameters.qLength)
        state.rewardQ[(state.qC+location)%state.parameters.qLength+1]+=state.parameters.rewards[choice]
    end
    state.qC=state.qC+1
    if state.qC>state.parameters.qLength
        state.qC=1
    end
    reward
end


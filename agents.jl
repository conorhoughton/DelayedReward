using Distributions

mutable struct DeltaRule
    learningRate::Float64
    predicted::Vector{Float64}
end

DeltaRule(learningRate::Float64,n::Int64)=DeltaRule(learningRate,zeros(Float64,n))

mutable struct RandomAgent
    n::Int64
end

mutable struct TDLearning
    learningRate::Float64
    discount::Float64
    predicted::Vector{Float64}
end
    
TDLearning(learningRate::Float64,discount::Float64,n::Int64)=TDLearning(learningRate,discount,zeros(Float64,n))

mutable struct PastLearning
    learningRate::Float64
    discount::Float64
    predicted::Vector{Float64}
    past::Vector{Float64}
end

PastLearning(learningRate::Float64,discount::Float64,n::Int64)=PastLearning(learningRate,discount,zeros(Float64,n),zeros(Float64,n))

mutable struct MemoryLearning
    learningRate::Float64
    memoryL::Int
    memory::Vector{Int}
    mC::Int
    predicted::Vector{Float64}
end

MemoryLearning(learningRate::Float64,memoryL::Int64,n::Int64)=MemoryLearning(learningRate,memoryL,zeros(Int,memoryL),1,zeros(Float64,n))

function randomizeAgent(agent::DeltaRule,r0::Float64,r1::Float64)
    n=length(agent.predicted)
    agent.predicted=rand(Uniform(r0,r1),n)
end



function update(agent::MemoryLearning,reward::Float64,choice::Int64)
    choices=zeros(Float64,length(agent.predicted))
    total=0.0::Float64
    for m in agent.memory
        if m>0
            choices[m]+=1.0
            total+=1.0
        end
    end
    agent.memory[agent.mC]=choice
    agent.mC+=1
    if agent.mC>length(agent.memory)
        agent.mC=1
    end
    if total != 0
        for i in 1:length(agent.predicted)
            agent.predicted[i]+=choices[i]/total*agent.learningRate*(reward-agent.predicted[i])
        end
    end
end


function updateWTA(agent::MemoryLearning,reward::Float64,choice::Int64)
    choices=zeros(Float64,length(agent.predicted))

    for m in agent.memory
        if m>0
            choices[m]+=1.0
        end
    end
    agent.memory[agent.mC]=choice
    agent.mC+=1
    if agent.mC>length(agent.memory)
        agent.mC=1
    end
    
    maxC=maximum(choices)
    winner = rand([i for (i,c) in enumerate(choices) if c==maxC])

    agent.predicted[winner]+=agent.learningRate*(reward-agent.predicted[winner])
    
end


function update(agent::PastLearning,reward::Float64,choice::Int64)
    agent.past*=agent.discount
    agent.past[choice]+=1.0
    totalDiscount=sum(agent.past)
    for i in 1:length(agent.predicted)
        agent.predicted[i]+=agent.past[i]/totalDiscount*agent.learningRate*(reward-agent.predicted[i])
    end

end

function update(agent::TDLearning,reward::Float64,choice::Int64)
    agent.predicted[choice]+=agent.learningRate*(reward-(1-agent.discount)*agent.predicted[choice])
end


function update(agent::DeltaRule,reward::Float64,choice::Int64)
    agent.predicted[choice]+=agent.learningRate*(reward-agent.predicted[choice])
end

    
struct EpsilonLapse
    lapseRate::Float64
end

    
struct EpsilonLapseDecay
    lapseRate::Float64
    decayRate::Float64
end

struct EpsilonLapseRatDecay
    lapseRate::Float64
    decayRate::Float64
end
 

function decision(agent,decisionModel::EpsilonLapse)
    if rand()<1.0-decisionModel.lapseRate
        maxR=maximum(agent.predicted)
        return rand([i for (i,r) in enumerate(agent.predicted) if r==maxR])
    end
    return rand(1:length(agent.predicted))
end


function decision(agent,decisionModel::EpsilonLapseDecay,trialC::Int64)
    if rand()<1.0-decisionModel.lapseRate*decisionModel.decayRate^(trialC-1)
        maxR=maximum(agent.predicted)
        return rand([i for (i,r) in enumerate(agent.predicted) if r==maxR])
    end
    return rand(1:length(agent.predicted))
end


function decision(agent,decisionModel::EpsilonLapseRatDecay,trialC::Int64)
    if rand()<1.0-decisionModel.lapseRate*1/(1+decisionModel.decayRate*(trialC-1))
        maxR=maximum(agent.predicted)
        return rand([i for (i,r) in enumerate(agent.predicted) if r==maxR])
    end
    return rand(1:length(agent.predicted))
end



function decision(agent::RandomAgent)
    return rand(1:agent.n)
end

function decision(reward::RewardParameters)
    max=maximum(reward.probs)
    return rand([i for (i,x) in enumerate(reward.probs) if x==max])
end

#agent=DeltaRule(0.5,4)
#decisionModel=EpsilonLapse(0.0)

#randomizeAgent(agent,-1.0,1.0)

#println(decision(agent,decisionModel))

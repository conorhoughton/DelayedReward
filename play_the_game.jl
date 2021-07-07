include("reward_process.jl")
include("agents.jl")


dayN=1000
trialN=100
choiceN=4
reward=1.0
qLength=10

rewardRandom=0.0
rewardDelta=0.0
rewardOptimal=0.0
rewardPast=0.0
rewardMemory=0.0
lapseRate=0.1
epsilonDecay=0.75
discount=0.9

deltaLR=0.1

deltaAgent=DeltaRule(deltaLR,choiceN)
randomAgent=RandomAgent(choiceN)
tDAgent=TDLearning(deltaLR,discount,choiceN)
pastAgent=PastLearning(deltaLR,discount,choiceN)
memoryAgent=MemoryLearning(deltaLR,qLength,choiceN)
decisionModel=EpsilonLapseRatDecay(lapseRate,epsilonDecay)

for _ in 1:dayN
    parameters=makeRewardParameters(choiceN,0.5,0.5,reward,qLength)

    deltaState=RewardState(parameters)
    randomState=RewardState(parameters)
    pastState=RewardState(parameters)
    optimalState=RewardState(parameters)
    memoryState=RewardState(parameters)
    
    for trialC in 1:trialN
        global rewardDelta,rewardOptimal,rewardRandom,rewardPast,rewardMemory
        
        choice=decision(deltaAgent,decisionModel,trialC)
        reward=update(deltaState,choice)
        update(deltaAgent,reward,choice)
        rewardDelta+=reward

        choice=decision(pastAgent,decisionModel,trialC)
        reward=update(pastState,choice)
        update(pastAgent,reward,choice)
        rewardPast+=reward

        choice=decision(memoryAgent,decisionModel,trialC)
        reward=update(memoryState,choice)
        update(memoryAgent,reward,choice)
        rewardMemory+=reward
        
        choice=decision(randomAgent)
        reward=update(randomState,choice)
        rewardRandom+=reward

        choice=decision(parameters)
        reward=update(optimalState,choice)
        rewardOptimal+=reward
        
    end
end

println((rewardDelta-rewardRandom)/(rewardOptimal-rewardRandom))
println((rewardPast-rewardRandom)/(rewardOptimal-rewardRandom))
println((rewardMemory-rewardRandom)/(rewardOptimal-rewardRandom))
        

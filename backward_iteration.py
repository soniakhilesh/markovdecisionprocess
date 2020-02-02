'''Input=List of states, dictionary for actions corresponding to each state,
    transition probabilities=dictionary with keys of form (t,state,new state, action), 
    Rewards=dictionary with keys of form (time,state,action),
    horizon length=number of periods'''
    
'Output=Optimal Policy, value vector'
import numpy as np
import copy

def backward_iteration(states,action_sets,transition_prob,rewards,horizon):
    #define dictionary to store policy
    policy={};
    #initialise value vector as a zeros vector
    value_vector=dict(zip(states,[0]*len(states)))
    #iterate over each period
    for t in range(horizon,0,-1):
        temp_value_vector=copy.deepcopy(value_vector);
        for s in states:
            #storing value corresponding to each action for current state
            temp_value={};
            for a in action_sets[s]:
                expected_reward=sum(transition_prob[(t,s,s_temp,a)]*value_vector[s_temp]
                    for s_temp in states if (t,s,s_temp,a) in transition_prob.keys())
                temp_value[a]=rewards[(t,s,a)]+expected_reward;
            #store the value of value vector in temp for current period
            temp_value_vector[s]=max(temp_value.values())
            #store which action to take
            policy[(t,s)]=max(temp_value,key=temp_value.get)
        #updating value vector
        value_vector=copy.deepcopy(temp_value_vector);
    return policy, value_vector

#Sample data for mdp process
def example_mdp():
    #define states
    states=[1,2];
    #define horizon
    horizon=6;
    #define actions sets
    action_sets=dict();
    action_sets[1]=['emp2300','emp3000'];
    action_sets[2]=['ad300','ad600'];
    #transition_prob
    transition_prob=dict();
    for t in range(1,horizon+1):
        transition_prob[(t,1,1,'emp2300')]=3/5;
        transition_prob[(t,1,2,'emp2300')]=2/5;
        transition_prob[(t,1,1,'emp3000')]=4/5;
        transition_prob[(t,1,2,'emp3000')]=1/5;
        transition_prob[(t,2,1,'ad300')]=7/10;
        transition_prob[(t,2,2,'ad300')]=3/10;
        transition_prob[(t,2,1,'ad600')]=9/10;
        transition_prob[(t,2,2,'ad600')]=1/10;
    #define rewards
    rewards=dict();
    for t in range(1,horizon+1):
        rewards[(t,1,'emp2300')]=-2300;
        rewards[(t,1,'emp3000')]=-3000;
        rewards[(t,2,'ad300')]=-4300;
        rewards[(t,2,'ad600')]=-4600;
    return states,action_sets,transition_prob,rewards,horizon

if __name__=='__main__':
    states,action_sets,transition_prob,rewards,horizon=example_mdp();
    policy, value_vector=backward_iteration(states,action_sets,transition_prob,rewards,horizon)
    print('Policy:',policy,'\n Value vector"',value_vector)
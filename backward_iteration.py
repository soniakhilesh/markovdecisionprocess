'''Input=List of states, dictionary for actions corresponding to each state,
    transition probabilities=dictionary with keys of form (t,state,new state, action), 
    Rewards=dictionary with keys of form (time,state,action),
    horizon length=number of periods'''
    
'Output=Optimal Policy, value vector'
import numpy as np
import copy
from graph_structure import Graph
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
            if temp_value_vector[(s)]>-1000:
                print('t=',t,'s=',s,'Arc:',policy[(t,s)],'Reward=',temp_value_vector[(s)])
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



def layered_network():
    states=[('1234'),('123'),('124'),('234'),('134'),('12'),('13'),('14'),('23'),('24'),('34'),('1'),('2'),('3'),('4')]
    empty_dict={}
    #constrcut a graph using graph class
    layered_network=Graph(empty_dict)
    for s in states:
        layered_network.add_vertex(s)
    for ori_node in states:
        dest_nodes=[ori_node.translate({ord(k): None}) for k in ori_node]
        for temp_dest_node in dest_nodes:
            layered_network.add_edge(ori_node,temp_dest_node)

    #define horizon
    horizon=4;

    #define action sets
    action_sets=dict();
    for ori_node in states:
        action_sets[ori_node]=layered_network.arcs_connected(ori_node)

    #define transition prob
    transition_prob=dict();
    #initialise with 0
    for t in range(1,horizon+1):
        for (i,j) in layered_network.edges():
                transition_prob[(t,i,j,(i,j))]=0;
    #figure out which are 1s
    for t in range(1,horizon+1):
        for (i,j) in layered_network.edges():
            if t==1 and len(i)==4:
                transition_prob[(t,i,j,(i,j))]=1;
            if t==2 and len(i)==3:
                transition_prob[(t,i,j,(i,j))]=1;
            if t==3 and len(i)==2:
                transition_prob[(t,i,j,(i,j))]=1;
            if t==4 and len(i)==1:
                transition_prob[(t,i,j,(i,j))]=1;

    #define rewards
    rewards=dict();
    p={'1':1,'2':2,'3':3,'4':4}
    for i in range(1,horizon+1):
        for s in states:
            for a in action_sets[s]:
                jobs_done=[j for j in list('1234') if j not in list(s)];
                for j in list(s):
                    if j not in list(a[1]):
                        job_done_current_action=j;
                        jobs_done.append(j)
                time_elapsed=0;
                for j in jobs_done:
                    time_elapsed+=p[j]
                c={'1':max(0,time_elapsed-2),'2':max(0,time_elapsed-7),'3':max(0,time_elapsed-5),'4':max(0,time_elapsed-6)}
                #assign high penalty first and then lower it for necessary cases
                rewards[(i,s,a)]=-99999999999;
                if i==1 and len(s)==4:
                    rewards[(i,s,a)]=-c[job_done_current_action]*time_elapsed;
                if i==2 and len(s)==3:
                    rewards[(i,s,a)]=-c[job_done_current_action]*time_elapsed;
                if i==3 and len(s)==2:
                    rewards[(i,s,a)]=-c[job_done_current_action]*time_elapsed;
                if i==4 and len(s)==1:
                    rewards[(i,s,a)]=-c[job_done_current_action]*time_elapsed;
    return states,action_sets,transition_prob,rewards,horizon

if __name__=='__main__':
    states,action_sets,transition_prob,rewards,horizon=layered_network();
    policy, value_vector=backward_iteration(states,action_sets,transition_prob,rewards,horizon)
# =============================================================================
#     print('Policy:',policy)
# =============================================================================
# =============================================================================
#     print('Policy:',policy,'\n Value vector"',value_vector)
# =============================================================================

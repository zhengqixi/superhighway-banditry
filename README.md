# superhighway-banditry
Fall 2020 Foundations of Machine Learning Final Project

In resource allocation, one common problem that often arises is the delivery of resources
from point A to point B. This can be seen in the delivery of physical goods,
as products move through various ports, as well as in the routing of internet packets.
Typically, there are multiple paths from point A to B. Routers have multiple ways to route
packets, and delivery companies have a variety of transportation methods through different
ports. We can easily represent this using a graph. Because the number of paths from point A
to B is limited, and can be easily enumerated via a search algorithm such as BFS, we will simplify
the graph away to a set of independent random variables. Each variable represents a path and its
associated "reward." There are a variety of metrics to determine the "reward" of a given path; this can range
from timieless in the case of routing packets to monetary cost in the case of physical transportation.


For this project, we will abstract these underlying variables away into a single metric per node.
We have now formuated this problem as a typical bandit problem: Presented with 
various arms with differing payoffs and given a certain
number of packets (goods) to deliver, which set of routes do we take to maximize our reward?
However, this is too standard of a problem; we will add a complication to this setup to model failure.
In reality, transportation can be downed for various reasons, and routers experience failures.
We also need to take into account the fact that a single bad node in a path blocks the entire path, 
and a given node can be shared among multiple paths.
To model this, we use a set of variables sigma_i...j, where j<=k, k being the number of total paths.
These variables emit {-c,0} with a fixed probability k_i. Tunable parameter that belongs to the environment,
although this parameter is made available to any agent for use in determining appropriate paths.
Multiple paths will be tied to a single sigma_j. The total reward for any given path tied to a sigma_j
will be the sum of the reward from the path itself with the result of sigma_j.
In reality, this penalty c can be thought of as a timeout that needs 
to be waited on until whether a packet can be considered
lost, or a reputation penalty to a delivery company in the case of physical goods.
This mapping will be made available, and can be used by any algorithm to appropriately
select paths.


While bandit problems are typically done in an online setting, recent works have considered
the problem in a batched setting. To give an analogy, an online approach would be akin to a TCP based
connection, where each packet is acknowledged before the next one is sent. For each packet i, we know
the results of the previous packet (in this case, latency as well as any necessary amount of retries due to packet loss).
A fully batched setting would be a UDP connection, where packets are sent without regard to whether the previous has succeeded. 
Our approach would be somewhere in between, where packets are sent in batches of size S, and we wait for the result of the 
entire batch before sending the next batch, using the results of the previous batch to inform our selection of routes
for the next batch.
We will be working off of the algorithm described in this paper:
https://papers.nips.cc/paper/2019/file/20f07591c6fcb220ffe637cda29bb3f6-Paper.pdf
We will first apply the algorithm to the setting described above. We will then attempt to come up with
a better approach, taking the topology of the sigma nodes into account to achieve a better result.

# Environment File 
To ensure consistent and reproducable starting environments, we store the starting state
in a JSON file.

The two necessary properties in this file are 'arms' and 'nodes'.
'arms' define the arms to be pulled, and 'nodes' define the common
nodes. Both of these are lists of tuples.

For 'arms', a tuple is of the pattern [arm_id, mean, std]

For 'nodes', a tuple is of the pattern [node_id, failure_probability]. Failure probability is a number between 0 and 1.

mapping is a list of tuples of the form [arm_id, node_id], linking
an arm to a node.

'failure_penalty' is the penalty applied to a arm where a mapped node has failed. This is optional. The default is 0. 

'seed' is an optional value for seeding the distribution.

Both node_id and arm_id are arbitrary unique strings. 
A node_id and arm_id may have the same string id,
but any two given node_id must be unique. Likewise for arm_id 


See [reference.json](reference.json) for examples
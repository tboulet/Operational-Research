# Operation Research

OR methods for Linear Programming and Integer Programming, with a focus on the VM Placement problem

# VM Placement Problem

This problem consist of placing a set of VMs on a set of servers, with the goal of minimizing the number of servers used. This is a NP-hard problem, and can be formulated as a Linear Programming problem, in particular, in it's original form, as a Binary Integer Programming problem.

### Problem Formulation

- We note $x_{ij}$ the binary decision variable that is equal to 1 if the VM $j$ is placed on the server $i$, and 0 otherwise.
- There are $n$ servers and $m$ VMs
- The goal is to minimize the number of servers used, so we introduce the variable $y_i$ which is equal to 1 if the server $i$ is used, and 0 otherwise. The goal is to minimize the following objective function:
  $$\sum_{i=1}^{n} y_i$$
- For each ressource $k$ (CPU, RAM, Disk, Network), we note $c_{i,k}$ the capacity of the server $i$ for the ressource $k$, and $a_{j,k}$ the consumption of the VM $j$ for the ressource $k$.
- Any server cannot be overloaded, so we have the following constraint for each server $i$ and each ressource $k$:
    $$\sum_{j=1}^{m} a_{j,k} \times x_{i,j} \leq c_{i,k} \times y_i$$
- Any VM must be placed on exactly (or more than?) one server, so we have the following constraint for each VM $j$:
  $$\sum_{i=1}^{n} x_{i,j} = 1$$
  We could potentially relax this constraint to allow the VM to be placed on several servers.

# Problem Generation

The VMP instances are generated artificially. The VM data is generated following random distributions (uniform, constant...) that can be defined in the `configs/problem/<problem tag>.yaml` files.

The server data is generated from the VM data, which allows the problem to be feasible and provides an almost-lower bound on the optimal solution. In details, it does the following :

- generate the m VMs
- group those VMs in n-k groups, with k >= n//2
- generate n-k servers (the optimal servers) whose capacities are the sum of the requirements of those matrix, plus a small (stochastic) bonus
- generate k other servers, each of them being 'altered copies' from one of the optimal servers, with capacities reduced by a (stochastic) malus.

This unsure that :
- the problem is always solvable (by using the initial configuration of the optimal servers for example)
- the optimal solution's value is at most n-k (the feasible optimal server is an upper bound on the optimal solution)
- if the other server malus is sufficiently high, they become not interesting to pick which ensures that the optimal solution is the feasible optimal server
- this ensures that the optimal solution value is n-k, if the bonus is sufficiently small and the malus is sufficiently high.

The configuration for the problem generation can be found in the `configs/problem/vmp.yaml` file.

# Solving the problem

For solving this problem, you can use one of the implemented algorithms using this command :

```bash
python run.py algo=<algo tag> problem=vmp
```

Where `<algo tag>` is the tag of the algorithm you want to use, and `vmp` is the tag of the problem you want to solve.

For example, for using Pyomo, you can use the following command :

```bash
python run.py algo=pyo problem=vmp
```

This will print the data generation as well as the solution found by the algorithm.

Algorithms currently implemented are :
- `pyomo` : formalize the problem under the Pyomo framework and solve it using a MILP solver.
- `greedy` : a greedy algorithm that places the VMs on the servers in a greedy way, by placing the VM on the server with the most available resources.
- `random` : a random algorithm that places the VMs on the servers in a random way until it has found a solution.
- `lp_around` : solve the LP relaxation of the problem, and search randomly around it the valid integer solution.


# Algorithms

### Pyomo

This is not an algorithm made by myself, but simply an implementation of the problem under the Pyomo framework, which is a Python library for optimization problems. This very efficient commercial library gives lower bound on the solution of any algorithms I will certainly implement, and is much faster.

### Greedy

This algorithm places the VMs on the servers in a greedy way, by placing the VM on the server with the most available resources. This is a very simple algorithm, and is not guaranteed to give the optimal solution.

### Random

This algorithm places the VMs on the servers in a random way until it has found a solution. This is a very simple algorithm. It is technically guaranteed to give the optimal solution, but the probability of finding it is very low, so the time to find the solution is very high.

### LP Relaxation + Random Around

This algorithm solve the relaxation of the problem (using the scipy.optimize.linprog function) and then try many solutions around the relaxation. It does that by randomly rounding to the inferior or superior integer each integer variable that was not integer in the relaxation solution, so mathematically, it is searching in the ball $B(x_{LP}^*,1)$ under the norm $L_{\infty}$.

<p align="center">
  <img src="assets/lp_around.png" alt="Title" width="60%"/>
</p>

It does that until it has found a solution that is integer. It then keep searching to eventually improve the solution's objective value.

Note that because there are this algorithm is very slow, because there are $2^{m}$ solutions to test, and that the search is not informed.

It is not guaranteed to give the optimal solution, and even to give a valid solution,, because the problem has no guarantee to have a valid solution in $B(x_{LP}^*,1)$, as you can see in this example :

<p align="center">
  <img src="assets/no_integer_solution_in_B1.png" alt="Title" width="60%"/>
</p>

Improvements :
- It can do slightly better by sampling the solutions without replacement. 
- If no solutions are found in the ball, it can increase the radius of the ball and try again.


# VM Placement Problem variants

### 1) Affinity rules between some of the VMs

This can happen for example if some VM work better if deployed on the same physical machine, or if some VMs should not be deployed on the same physical machine for security reason for example.

In the same-server case, it is equivalent to additionate the two VMs capacities, i.e. to merge the two VMs into one.

In the different-server case, we have the following constraint : for any pair (j,j') of uncompatible VMs, and any server $i$:
$$x_{i,j} + x_{i,j'} \leq 1$$
This means that if the VM $j$ is placed on the server $i$, then the VM $j'$ cannot be placed on the same server.

You can use the following command to solve the problem with this variant (the tag is `vmp_incomp`):

```bash
python run.py algo=<algo tag> problem=vmp_incomp
```

### 2) Case where all servers are partly occupied vs totally empty and all with the same characteristics

The case where all servers have not the same capacities is the default case in this implementation.

The case where all servers are identical and are initially empty can be obtained by associating all server capacities to the max capacities. You can obtain this by setting the `problem` tag to `vmp_empty` in the command line :

```bash
python run.py algo=<algo tag> problem=vmp_empty
```


#### 3) VMs could be splitted over several servers

This is a case where it is possible to split the deployment of any VM on $k$ servers, with $k$ a given integer. This is not equivalent to the linear relaxation case, where $k$ would be equal to the number of servers. In this case the VM can be split but not infinitely.

In this case, we can relax the integer constraint and set the bounds at (0,1). We now need to force the number of non-null $x_{i,j}$ to be inferior to $k$.

For this, we introduce the variable $z_{i,j}$ which is equal to 1 if the VM $j$ is partially placed on the server $i$, and 0 otherwise : $z_{i,j} = 1_{x_{i,j} \neq 0}$. 

We can obtain this with the constraints $z_{i,j} \geq x_{i,j}$ and $z_{i,j} * \epsilon \leq x_{i,j}$, with $\epsilon$ a small positive number.

We then have the following constraint for each VM $j$:
$$\sum_{i=1}^{n} z_{i,j} \leq k$$

If we want each assignment of a VM to the server to be at least equal to a given fraction $\alpha$ of the VM capacity, we simply have to set $\epsilon$ to $\alpha$.

#### 4) Consider VMs families, each family is given a criticity level between 1 to 3

In this case, it is imperative to assign the VMs of high criticity, and less important to assign the VMs of low criticity. This can be done as a soft or hard way.

#### 5) Online VMP

In this case, the VMs are arriving one by one, and the goal is to place them as they arrive. This is a case of online optimization, and the goal is to minimize the number of migrations.
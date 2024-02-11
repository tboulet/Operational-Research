# Operation Research

OR methods for Linear Programming and Integer Programming, with a focus on the VM Placement problem

## VM Placement Problem

This problem consist of placing a set of VMs on a set of servers, with the goal of minimizing the number of servers used. This is a NP-hard problem, and can be formulated as a Linear Programming problem.

## Problem Formulation

- We note $x_{ij}$ the decision variable, which is equal to 1 if the VM $j$ is placed on the server $i$, and 0 otherwise.
- There are $n$ servers and $m$ VMs
- The goal is to minimize the number of servers used, so we introduce the variable $y_i$ which is equal to 1 if the server $i$ is used, and 0 otherwise. The goal is to minimize the following objective function:
  $$\sum_{i=1}^{n} y_i$$
- For each ressource $k$ (CPU, RAM, Disk, Network), we note $c_{i,k}$ the capacity of the server $i$ for the ressource $k$, and $a_{j,k}$ the consumption of the VM $j$ for the ressource $k$.
- Any server cannot be overloaded, so we have the following constraint for each server $i$ and each ressource $k$:
    $$\sum_{j=1}^{m} a_{j,k} \times x_{i,j} \leq c_{i,k} \times y_i$$
- Any VM must be placed on exactly (or more than?) one server, so we have the following constraint for each VM $j$:
  $$\sum_{i=1}^{n} x_{i,j} = 1$$
  We could potentially relax this constraint to allow the VM to be placed on several servers.

## VM Placement Problem variants

#### 1) Affinity rules between some of the VMs

This can happen for example if some VM work better if deployed on the same physical machine, or if some VMs should not be deployed on the same physical machine for security reason for example.

In the same-server case, it is equivalent to additionate the two VMs capacities.

In the different-server case, we have the following constraint : for any *(j,j') pair of uncompatible VMs, and any server $i$:
$$x_{i,j} + x_{i,j'} \leq 1$$
This means that if the VM $j$ is placed on the server $i$, then the VM $j'$ cannot be placed on the same server.

#### 2) Case where all servers are partly occupied vs totally empty and all with the same characteristics

TODO

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
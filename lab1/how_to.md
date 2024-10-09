## Distributed Data Parallel
- Run in notebook `original.py` file

## Running distributed tasks
### Locally

- Run `torchrun --nnodes 1 --nproc-per-node 4 ddp.py` in terminal (files are on lab machine)

## Students
- To make it work run
```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```
- Run `lk_booted_pcs` and take machines u can connect to and connect to them to make connections passwordless
- Run `mpirun --hostfile hosts -np 4 -x MASTER_ADDR=pink00 -x MASTER_PORT=1234 ./run_training.sh` where master addess is one of hostnames in hosts file and port is any free port on this computer
- Here I've got the errors but i've done everything from tutorial right :/
- Write down processing times of 3 different configurations for number of hosts and workers `(red01 max_slots = n)`

### Excercise 1
`torchrun --nnodes 1 --nproc-per-node 4 ddp.py`
Learning took 43.70656609535217s
`torchrun --nnodes 1 --nproc-per-node 4 ddp.py`
Learnning took 48.722736120224s
`torchrun --nnodes 1 --nproc-per-node 2 ddp.py`
Learning took 30.902313232421875s

I cant run more nodes becouse on students it does not work even if I did every step but the time is lower
when there are less workers becouse maybe locally creating workers or communication is expensive. This is the same for every worker becouse they wait for final results of all workers.

## On Entropy using Slurm
- Copy files from `files_to_send_on_entropy` to entropy
- Change `--qos=your_qos` to output of `entropy_account_info` output qos
- Run `sbatch slurm-ddp.sh` (and retry when it fails on downloading dataset)

### Excercise 2
Schedule a job such that every task will invoke `/bin/hostname` where:

a) There is exactly one task on each of 3 nodes.

b) There are 9 tasks and 3 nodes.

c) Run 3 tasks on arnold and 3 on bruce

d) Ask `sbatch` or `salloc` for 3 nodes and 6 tasks. Invoke srun 2 times: one with 2 nodes and 2 tasks per node, second without any parameters.

Have you noticed something unusual in the results? Do you understand why they look this way?

### Excercise 3

Run the training for several (at least 4) different allocations in the entropy. Write down the processing times of those runs. Can you explain those numbers? How do they compare to the runs in the labs?

- This can be done either by `salloc` or `sbatch`.

## Distributed communication in pytorch
### Excercise 4
These 2 can be run using torchrun on students machine

- Example 1
```
Rank 0: Sending 42 to rank 1
Rank 1: Received 42.0 from rank 0
```
- Example 2
```
Rank 0: Sending 0
Rank 1: Sending 1
Rank 3: Sending 3
Rank 2: Sending 2
Rank 0: Sum of all messages is 6
```
a) On students on average 2.9s real time
```
sj429144@students:~/Pulpit$ torchrun --nnodes 2 --nproc-per-node 1 example2.py
-bash: torchrun: nie znaleziono polecenia
sj429144@students:~/Pulpit$ torchrun --nnodes 1 --nproc-per-node 2 example2.py
```

b, c) implemented

d) using time command (time execution)
real time lib all_reduce average 3 (example 5)
my all_reduce average 3 (example 4)
lib reduce average 2.9 (example 2)
my reduce average 2.9 (example 3)

In parenthasis is numeration of script on students.
My implementation is full as 0 process sends 0 but it can be exacly the same doing same trick as I've
done in all reduce (send 0 message to all except 0 before ;) 



red14, green15, cyan15
# Running on students and entropy

The following lines needs to be written to run on students or entropy
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch
pip install torchvision
```

## Distributed Data Parallel
- Run in notebook `original.py` file

## Running distributed tasks
### Locally

- Run `torchrun --nnodes 1 --nproc-per-node 4 ddp.py` in terminal
## Students
- To make it work run
```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```
- Run `lk_booted_pcs` and take machines u can connect to and connect to them to make connections passwordless
- Run `mpirun --hostfile hosts -np 4 -x MASTER_ADDR=pink00 -x MASTER_PORT=1234 ./run_training.sh` where master addess is one of hostnames in hosts file and port is any free port on this computer

### Excercise 1

Run the training for several (at least 3) different configurations of the number of hosts and workers. Write down the processing times of those runs. Can you explain those numbers?

| Number of hosts | Number of workers | time |
| --------------- | ----------------- | ---- |
|        3        |        4          |  46s |
|        3        |        3          |  42s |
|        2        |        4          |  47s |
|        2        |        1          |  37s |

The time is lower when there are less workers becouse maybe locally creating workers or communication is expensive. This is the same for every worker becouse they wait for final results of all workers.

## On Entropy using Slurm
- Copy files from `files_to_send_on_entropy` to entropy
- Change `--qos=your_qos` to output of `entropy_account_info` output qos
- Run `sbatch slurm-ddp.sh` (and retry when it fails on downloading dataset)

$${\color{red}\textrm{slurm-ddp.sh does not work}}$$

### Excercise 2
Scripts doing the task are on entropy and in this repo in `files_to_send_on_entropy`. Just launch them with sbatch.

**Results of a)**
arnold
asusgpu1
asusgpu2
**Results of b)**
arnold
arnold
arnold
asusgpu1
asusgpu2
asusgpu1
asusgpu2
asusgpu1
asusgpu2
**Results of c on arnold)**
arnold
arnold
arnold
**Results of c on bruce)**
bruce
bruce
bruce
$${\color{red}\textrm{This on bruce does not work :(}}$$

### Excercise 3

Run the training for several (at least 4) different allocations in the entropy. Write down the processing times of those runs. Can you explain those numbers? How do they compare to the runs in the labs?

- This can be done either by `salloc` or `sbatch`.

$${\color{red}\textrm{This is still not done}}$$

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

a) Measure the time it takes to send the message between two ranks when they are:

- on the same node
- on a different nodes
You can choose to do it in the lab or entropy. Repeat measurment many times to decrease variability.

On students on average 2.9s real time
```
sj429144@students:~/Pulpit$ torchrun --nnodes 2 --nproc-per-node 1 example2.py
sj429144@students:~/Pulpit$ torchrun --nnodes 1 --nproc-per-node 2 example2.py
```

b) Implement reduce using only point-to-point communication.

My implementation is full as 0 process sends 0 but it can be exacly the same doing same trick as I've
done in all reduce (send 0 message to all except 0 before ;) 

```python
import os
import time
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
RANK = int(os.environ['RANK']) # int(os.environ['SLURM_PROCID'])
dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)


if RANK == 0:
  print(f"Rank {RANK}: Sending {RANK}") # is 0 so ignore it xD
  s = torch.tensor(0)
  message = torch.tensor(RANK)
  for d in range(1, WORLD_SIZE):
    dist.recv(message, d)
    s += message
  print(f"Sum of all messages is {s.item()}")
else:
    print(f"Rank {RANK}: Sending {RANK}")
    dist.send(torch.tensor(RANK), 0)
```
c) Implement all_reduce using only point-to-point communication.
```python
import os
import time
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
RANK = int(os.environ['RANK']) # int(os.environ['SLURM_PROCID'])
dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)

if RANK == 0:
  print(f"Rank {RANK}: Sending {RANK}") # is 0 so ignore it xD
  s = torch.tensor(0)
  message = torch.tensor(RANK)
  for d in range(1, WORLD_SIZE):
    dist.recv(message, d)
    s += message
  print(f"Rank {RANK}: Sum of all messages is {s.item()}")

  for d in range(1, WORLD_SIZE):
    dist.send(s, d)
else:
    print(f"Rank {RANK}: Sending {RANK}")
    dist.send(torch.tensor(RANK), 0)
    s = torch.tensor(0)
    dist.recv(s, 0)
    print(f"Rank {RANK}: Sum of all messages is {s.item()}")
```

d) Measure the time of your implementations against the library ones (you can choose to do it in the lab o entropy).

Using time command (time execution)

```bash
real time lib all_reduce average 3 (example 5)
my all_reduce average 3 (example 4)
lib reduce average 2.9 (example 2)
my reduce average 2.9 (example 3)
```

In parenthasis is numeration of script on students.

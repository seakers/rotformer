from numba import cuda, objmode
import numpy
from timeit import default_timer as timer
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"

@cuda.jit
def add(a, b, c):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    c[idx] = a[idx] * b[idx]
    

def test_on_add():
    n = 262144
    dumb_var = cuda.to_device(numpy.zeros(1))
    t1 = timer()
    add[1,1](dumb_var, dumb_var, dumb_var)
    t10 = timer() - t1
    print(t10)

    for x in range(5):
        a = numpy.random.rand(524288)
        b = numpy.random.rand(524288)
        c = numpy.random.rand(524288)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dc = cuda.to_device(c)
        t1 = timer()
        add[1024,768](da, db, dc)
        t10 = timer() - t1
        t2 = timer()
        add[768,1024](da, db, dc)
        t20 = timer() - t2
        print(t10, t20)

        plt.scatter(x,t10,color="red", label="first")
        plt.scatter(x,t20,color="blue", label = "second")

    for x in range(5):
        start_cpu = timer()
        c_cpu = [a[i]*b[i] for i in range(len(c))]
        cpu = timer() - start_cpu
        plt.scatter(x,cpu,color="green")

    if numpy.allclose(c_cpu, c):
        print("Correct results!")
        print(len(c_cpu))
        print(c.shape)

    plt.legend()
    plt.show()

@cuda.jit
def matmul(W,
           x,
           output):
    
    seq_idx = cuda.blockIdx.x
    head_idx = cuda.blockIdx.y
    dq_idx = cuda.threadIdx.x
    summation = 0
    
    for i in range(len(x[0])):
        summation += + x[seq_idx][i] * W[head_idx][i][dq_idx]
    output[head_idx][seq_idx][dq_idx] = summation

def test_matmul(n_heads = 16, sequence_len = 20, d_emb = 512, d_q = 128, check_send=False):
    W_q = torch.rand((n_heads ,d_emb, d_q)).cuda()
    q = torch.rand((sequence_len, d_emb)).cuda()
    out = torch.zeros((n_heads ,sequence_len, d_q)).cuda()
    out_gpu = None
    q_F = None
    W_q_F = None

    if check_send:
        gpu_send_start = timer()
        W_q = cuda.to_device(W_q)
        q = cuda.to_device(q)
        out = cuda.to_device(out)
        gpu_send_time = timer() - gpu_send_start

    blocks_per_grid = (sequence_len, n_heads)
    threads_per_block = (d_q,)
    print("blocks =",sum(blocks_per_grid),", threads =",sum(threads_per_block))
    matmul[blocks_per_grid, threads_per_block](W_q, q, out)

    gpu_compute_start = timer()
    matmul[blocks_per_grid, threads_per_block](W_q, q, out)
    gpu_compute_time = timer() - gpu_compute_start

    if check_send:
        out_gpu = out.copy_to_host()
        q_F = q.copy_to_host(q_F)
        W_q_F = W_q.copy_to_host()
    else:
        out_gpu = out.cpu()
        q_F = q.cpu()
        W_q_F = W_q.cpu()

    cpu_start = timer()
    out_cpu = [q_F@W_q_F[i] for i in range(W_q.shape[0])]
    cpu_time = timer() - cpu_start


    if check_send: print("gpu send time:", gpu_send_time, gpu_send_time+gpu_compute_time)
    print("gpu compute time:", gpu_compute_time)
    print("cpu time:", cpu_time)

    if numpy.allclose(out_gpu, out_cpu):
        print("Correct results!")

def test_torch(n_heads = 16, sequence_len = 20, d_emb = 512, d_q = 128):
    W_q = torch.rand((n_heads ,d_emb, d_q))
    q = torch.rand((sequence_len, d_emb))

    cpu_compute_start = timer()
    out_cpu = torch.div(W_q, 2)
    cpu_compute_time = timer() - cpu_compute_start

    dW_q = W_q.to(device="cuda")
    dq = q.to(device="cuda")
    scale = torch.Tensor([2]).to(device="cuda")

    torch.div(dW_q, scale)
    gpu_compute_start = timer()
    out_gpu = torch.div(dW_q, scale)
    gpu_compute_time = timer() - gpu_compute_start

    out_gpu = out_gpu.cpu()

    if numpy.allclose(out_gpu, out_cpu):
        print("Correct results!")
        print(out_gpu.shape)
        print(out_cpu.shape)


    print("gpu compute time:", gpu_compute_time)
    print("cpu time:", cpu_compute_time)

x = torch.rand((3,4))
print(x)
print(torch.softmax(x, dim=0))
print(torch.softmax(x, dim=1))
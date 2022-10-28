---
title: "Scaling Deep Learning Applications"
center: true
theme: black
transition: slide
margin: 0.04
revealOptions:
   transition: 'slide'
css: 
- ./css/custom.css
---

<grid drop="5 20" drag="90 25" style="font-family:'Inter',sans-serif;background-color:#303030;border-radius:8px!important;padding:auto;align:center;">
#### Scaling Deep Learning Applications <!-- .element style="font-family:'Inter';color:#F8F8F8;" align="stretch" -->
    
<i class="fab fa-github"></i> [`CompPerfWorkshop/05_scaling-DL`](https://github.com/argonne-lcf/CompPerfWorkshop/tree/main/05_scaling-DL)
</grid>

<grid drop="0 55" drag="100 30" style="line-height:0.6em;" align="top">
Sam Foreman <!-- .element style="font-family:'Inter';font-size:1.2em;font-weight:500;line-height:0.6;color:#E0E0E0!important;vertical-align:bottom!important;" -->

May, 2022 <!-- .element style="font-family:'Nanum Pen Script'; font-size:1.6em;color:#616161;vertical-align:top;font-weight:400;" -->
</grid>

<grid drag="100 10" drop="bottom" flow="row" align="left" style="font-size:1.5em;">

::: block <!-- .element style="margin-left:2%;margin-bottom:2%;" -->

[<i class="fab fa-github"></i>](https://github.com/argonne-lcf/CompPerfWorkshop)
[<i class="fas fa-home"></i>](https://samforeman.me)
[<i class="fab fa-twitter"></i>](https://twitter.com/saforem2)

:::

</grid>

<grid drag="30 30" drop="70 70" align="bottomright">
<img src="https://raw.githubusercontent.com/saforem2/anl-job-talk/main/docs/assets/anl.svg" width="100%" align="center" ></img>
</grid>

---

::: block <!-- .element class="note" style="max-width:75%;padding:1%;" -->

# Distributed Training

:::

---
## Why train on multiple GPUs?
- Large batches may not fit in GPU memory
- Splitting the data across workers has the effect of increasing the batch size
- Smooth loss landscape
- Improved gradient estimators
- Less iterations for same number of epochs
    - May need to train for more epochs if another change is not made
    - e.g. boosting the learning rate
- See [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)See Large Batch Training of Convolutional Networks

---

# Recent Progess

::: block <!-- .element style="font-size:0.6em;" -->
   
|              Year               |                  Author                  |            Batch Size            |                  Processor                  |           DL Library            |                Time                |                         Accuracy |
|:-------------------------------:|:----------------------------------------:|:--------------------------------:|:-------------------------------------------:|:-------------------------------:|:----------------------------------:| --------------------------------:|
| <span id="blue">**2016**</span> |   <span id="blue">He et al. [1]</span>   |    <span id="blue">256</span>    |    <span id="blue">Tesla P100 x8</span>     |  <span id="blue">Caffe</span>   |              <span id="blue">29 Hrs</span>               |                            <span id="blue">75.3%</span> |
|                                 |             Goyal et al. [2]             |               8192               |                 Tesla P100                  |             Caffe 2             |               1 hour               |                            76.3% |
|                                 |             Smith et al. [3]             |         8192 ->  16,384          |                full TPU pod                 |           TensorFlow            |              30 mins               |                            76.1% |
|                                 |             Akiba et al. [4]             |              32,768              |              Tesla P100 x1024               |             Chainer             |              15 mins               |                            74.9% |
|                                 |              Jia et al. [5]              |              65,536              |              Tesla P40  x2048               |           TensorFLow            |              6.6 mins              |                            75.8% |
|                                 |             Ying et al. [6]              |              65,536              |                TPU v3 x1024                 |           TensorFlow            |              1.8 mins              |                            75.2% |
|                                 |            Mikami et al. [7]             |              55,296              |              Tesla V100 x3456               |               NNL               |              2.0 mins              |                           75.29% |
| <span id="red">**2019**</span>  | <span id="red">**Yamazaki et al**</span> | <span id="red">**81,920**</span> | <span id="red">**Tesla V100 x 2048**</span> | <span id="red">**MXNet**</span> | <span id="red">**1.2 mins**</span> | <span id="red">**75.08%**</span> |

:::

---

# Distributed Training
- For additional information / general guidance, refer to [Writing Distributed Applications with PyTorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html), or see [`[1]`](https://eng.uber.com/horovod/), [`[2]`](https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy), [`[3]`](https://www.arxiv.org/abs/1802.05799)

<grid drop="bottom" drag="100 40" class="footnote" style="color:#757575;font-size:0.8em;">
1. Sergeev, A., Del Balso, M. (2017) [Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow](https://eng.uber.com/horovod/)
2. Sergeev, A. (2017) [Horovod - Distributed TensorFlow Made Easy](https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy).
3. Sergeev, A., Del Balso, M. (2018) Horovod: fast and easy distributed deep learning in TensorFlow [arXiv:1802.05799](https://www.arxiv.org/abs/1802.05799).
</grid>

---

# Development Trajectory 

::: block <!-- .element style="font-size:0.7em;" align="center" -->

1. ❌ <span style="color:#757575;">Use single device training if the data and model can fit on one GPU, and training speed is not a concern  (not relevant for this section)</span>

2. ❌ <span style="color:#757575;">Use single-machine multi-GPU <a href="https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html"> `DataParallel` </a>  to make use of multiple GPUs on a single machine to speed up training with minimal code changes (not recommended, <tt>DDP</tt> preferred)</span>

3. ✅ Use single-machine multi-GPU <a href="https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"> <tt>DistributedDataParallel</tt>  </a>  if you would like to further speed up training and are willing to write a little more code to set it up

4. ✅ Use multi-machine <a href="https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"><tt>DistributedDataParallel</tt></a> and the <a href="https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md"> launching script</a> if the application needs to scale across machine boundaries

5. ❓ <span style="color:#757575;">Use <a href="https://pytorch.org/docs/stable/distributed.elastic.html"> <tt>torch.distributed.elastic</tt></a> to launch distributed training if errors (e.g., out-of-memory) are expected or if resources can join and leave dynamically during training  (not covered in this tutorial)</span>

:::

---


::: block <!-- .element class="note" style="max-width:75%;padding:1%;" -->

# Model vs Data Parallelism

:::

---

## Model Parallelism
- Disjoint subsets of a neural network are assigned to different devices

    - All communication associated with subsets are distributed
    
- Communication happens between devices whenever there is dataflow between two subsets

- Typically **more complicated** to implement than data parallel training

- Suitable when the model is too large to fit onto a single device (CPU / GPU)

---

## Model Parallelism
- Suitable when the model is too large to fit onto a single device (CPU / GPU)

    - Partitioning the model into different subsets is not an easy task
    
    - Might potentially introduce load imbalancing issues limiting the scaling efficiency
    
- 🤗[huggingface/transformers](https://github.com/huggingface/transformers) is a useful reference, and they have an excellent series of posts in their documentation on [Model Parallelism](https://huggingface.co/docs/transformers/parallelism)

---

![](assets/model-parallel.svg) <!-- .element align="stretch" -->

---

## Data Parallelism
- Typically easier / simpler to implement

- Existing frameworks ([Horovod](https://horovod.readthedocs.io/en/stable/index.html), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [DDP](https://pytorch.org/docs/stable/notes/ddp.html), etc.)

    - Typically relatively simple to get up and running (minor modifications to existing code)
    
    
- Our recent presentation on data-parallel training is available on [youtube](https://youtu.be/930yrXjNkgM) 

---

## Data Parallelism 

- All of the workers own a replica of the model

- Global batch of data is **split into multiple mini-batches** and processed by **different workers**

- **Each worker computes the loss + gradients** w.r.t its local data

- Loss + Grads are averaged across workers before updating parameters

    - Relatively simple to implement, `MPI_Allreduce` is the only communication operation required
    
    - [Concepts — Horovod documentation](https://horovod.readthedocs.io/en/stable/concepts_include.html)
    
---

## Data Parallelism


- In the data-parallel approach, all workers own a replica of the model. The global batch of data is split into multiple minibatches, and processed by different workers.

---


![](assets/data-parallel.svg) <!-- .element align="stretch" -->

---

## Data Parallelism

- Each worker computes the corresponding loss and gradients with respect to the local data it possesses. Before updating the parameters at each epoch, the loss and gradients are averaged among all the workers through a collective `allreduce` operation.
    - `MPI` defines the function `MPI_Allreduce` to reduce values from all ranks and broadcast the result of the reduction such that all processes have a copy of it at the end of the operation.
    - There are multiple different ways to implement the allreduce, and it may vary from problem to problem[^4], [^5]

---
    
    
![](assets/avgGrads.svg)
    



---
<style>
:root {
   --r-heading-font: 'Inter', sans-serif;
  font-size: 34px;
}
.horizontal_dotted_line{
  border-bottom: 2px dotted gray;
} 
.footer {
  font-size: 60%;
  vertical-align:bottom;
  color:#bdbdbd;
  font-weight:400;
  margin-left:-5px;
  margin-bottom:1%;
}
.note {
  padding:auto;
  margin: auto;
  text-align:center!important;
  border-radius: 8px!important;
  background-color: rgba(53, 53, 53, 0.5);
}
.reveal ul ul,
.reveal ul ol,
.reveal ol ol,
.reveal ol ul {
  margin-bottom: 10px;
}
.callout {
  background-color: #35353550
  color: #eeeeee;
}
.callout-content {
  overflow-x: auto;
  color: #eeeeee;
  background-color:#353535;
  padding: 5px 15px;
}
</style>
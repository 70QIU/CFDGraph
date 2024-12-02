<h1 align="center">
    <br>
    CFDGraph
</h1>
<p align="center">
    A Privacy-Preserving Graph Processing System for Large-Scale Collaborative Fraud Detection
</p>


### Dependencies

- A C++ compiler compliant with C++-11 standard. 
- openmp


### Building CFDGraph
Building CFDGraph using make. Go to the root directory of CFDGraph and execute the following command, you will get a target program **CFDGraph**.

```bash
make
```
We provide an example of running the PageRank algorithm on Bitcoin-Alpha graph using CFDGraph, just execute the following command.

```bash
sh run.sh
```
Modify the file baseline.sh and run.sh to change the dataset you want to execute on. 


### Dataset
All datasets we used:

| Graph   | Vertex | Edge | 
| ------- | ------ | ---- |
| Bitcoin-Alpha(BA) | 3,783 | 24,186 |
| Bitcoin-OTC(BO) | 5,881 | 35,592 |
|Twitter (TW)| 41,652,230| 1,468,365,182|
|Kron (KR) | 33,554,432| 1,174,405,120|
|uk-2005 (UK) | 39,454,746| 936,364,282|
|it-2004 (IT) | 41,290,682| 1,150,725,436|
|Friendster (FR) | 65,608,366| 1,806,067,135|

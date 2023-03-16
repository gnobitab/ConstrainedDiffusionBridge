# Diffusion Bridges on Constrained Domains
This is the official implementation of ICLR2023 Spotlight paper 
## [Learning Diffusion Bridges on Constrained Domains](https://openreview.net/forum?id=WH1yCa0TbB) 
by *Xingchao Liu, Lemeng Wu, Mao Ye, Qiang Liu* from UT Austin

<img src=./github_misc/fig1.png width=300 />


# Introduction
We present a simple and unified framework to learn diffusion models on constrained and structured domains. It can be easily adopted to various
types of domains, including product spaces of any type (be it bounded/unbounded, continuous/discrete, categorical/ordinal, or their mix).

In our model, the diffusion process is driven by a drift force that is a sum of two terms: one singular force designed by Doob’s h-transform that ensures all outcomes of the process to belong to the desirable domain, and one non-singular neural force field that is
trained to make sure the outcome follows the data distribution statistically.

# Colab Notebook

An interactive Colab notebook on a toy example is provided [here](https://colab.research.google.com/drive/1-Pf9IfQ85qBObglON_wER_ekj7puDugP?usp=sharing)

# Mixed-type Tabular Data

See instructions in ```./MixedTypeTabularData```



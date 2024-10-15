---
marp: true
paginate: true
math: mathjax
---

<style>
    section {
        font-family: "Fira sans", Arial;
    }
    /* h1, h2 {
        font-family: "Fira Mono", monospace;
    } */
</style>


<!-- _class: invert -->
# Multiresolution Representation of Images <br/> using Neural Networks

#### Understanding Deep Learning 2024 - ICMC, USP

## Hallison Paz

### October 15th, 2024

![bg](img/)

<!-- _footer: Presentation recorded and [available on Youtube](https://www.youtube.com/live/WmaWecH0ThU?si=cWvOgvNIpkO9_0zl) -->

<!-- _paginate: false -->
---

# Hallison Paz

- AI Graphics Researcher
- PhD Candidate 
@IMPA, @Visgraf

- Alma Mater: 
Instituto Militar de Engenharia

- Co-founder
@Programação Dinâmica


![bg left](img/hallpaz_lion.png)

---
<!-- _class: invert -->

![bg ](img/pgdinamica.png)

---
# Codec Avatars

- Research Scientist Intern at Reality Labs Research (Meta). Pittsbugh, PA, USA.

![bg right:50%](img/codec-avatar.gif)

<!-- _footer: Lex Fridman [podcast using avatars](https://youtu.be/MVYrJJNdrEg?si=yG_Hhx1JsBrBXHNa) -->

----
###### What do you mean by?
# Multiresolution representation of images

<video width="1200" height="800" controls>
  <source src="img/multiresolution-masp.mp4" type="video/mp4">
</video>

<!-- _class: inver -->

----
# Representational Networks

> You put water into a cup, <br/>it becomes the cup
<br/>– **Bruce Lee**

![bg right:40% height:80%](img/bruce_lee.jpg)

<!-- _footer: Apurba Kanti Roy, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons -->

<!-- _footer: check more details [at this video](https://www.youtube.com/live/voVBM6BYs8k?si=lgOZT6BRh9eL8fz3). -->
---

# Representational Networks

- Continuous function
* Compact
* New methods/operations

![bg right](img/cosine_approximation.gif)

<!-- _footer: Image: training of a ReLu MLP to fit a cosine wave -->

---


![bg 80%](img/image_as_function.png)

<!-- _footer: Source: https://youtu.be/Wo0QVVM5jXE?si=9uxYNqsLNk0jAr8X -->

---
# Implicit Models

![bg right:75% 90%](img/circlewithdistances.png)
![bg 90%](img/bunny2d-sdf.png)

---

# Deep SDF

![bg right:70% 90%](img/deep_sdf_bunny.ppm.png)

----
# Neural Radiance Fields (NeRFs)

<video width="1200" height="800" controls>
  <source src="img/nerf-example.mp4" type="video/mp4">
</video>

----

# Let's fit an MLP to it!


![bg right height:400px](img/masp.gif)

----

![bg](img/paper-spectral-bias.png)

----

![](img/paper-fourier-features.png)

<!-- _footer: Check [paper website](https://bmild.github.io/fourfeat/) -->

----

# Looks better! 



![bg right height:400](img/fourier_masp.gif)

----

![](img/paper-siren.png)

<!-- _footer: Check [paper website](https://www.vincentsitzmann.com/siren/) -->

----

# Now, we are talking!

![bg right height:400](img/siren_masp.gif)

----

<!-- _class: invert -->

# Demo

### [Training a representational network for images](https://github.com/hallpaz/nov23google/blob/main/code/representational_networks.ipynb)
<a href="https://colab.research.google.com/github/hallpaz/nov23google/blob/main/code/representational_networks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<table>
  <tr>
    <td> <img src="img/masp.gif"  alt="1" ></td>
    <td><img src="img/fourier_masp.gif" alt="2"></td>
    <td><img src="img/siren_masp.gif" alt="3"></td>
   </tr> 
</table>

---

# What about the multiresolution?

<!-- _class: invert -->
<!-- _paginate: false -->

----

![bg fit](img/mrnet-paper.png)

----

# Isolating frequencies

<!-- _class: invert -->
<!-- _paginate: false -->

----

# Shallow Network - low frequencies

![](img/thesis-low-freqs.png)

----

# Shallow Network - high frequencies

![](img/thesis-high-freqs.png)


----

# 1 Hidden Layer - low frequencies

![bg right:60% fit](img/thesis-hidden-layer-captures-low.png)

<!-- _paginate: false -->


----

# 1 Hidden Layer - high frequencies

![bg right:60% fit](img/thesis-hidden-layer-captures-high.png)


<!-- _paginate: false -->

---

# What can we do with this information?

<!-- _class: invert -->
<!-- _paginate: false -->

---

# Multiscale Decomposition

- Let $\mathscr{f}:\mathcal{D}\to \mathcal{C}$ be a *ground-truth signal*

- We decompose it into a sum of $N$ stages: 

$$\mathscr{f}=\mathscr{g}_0 + \dots + \mathscr{g}_{N-1}$$
, 
<!-- where $\gt{g}_0$ captures the coarsest approximation of the signal and $\gt{g}_i$, for $i>0$, progressively introduces higher-frequency components.  -->
- $\mathscr{g}_0$ representes the coarse features

---

# Multiscale decomposition

The *level of detail* at stage $i$ is defined as:

$$
\mathscr{f}_i = \mathscr{g}_0 + \cdots + \mathscr{g}_i \quad \text{or} \quad \mathscr{f}_i = \mathscr{f} - \sum_{j=i+1}^{N-1} \mathscr{g}_j.
$$

Each stage $\mathscr{g}_i$ is computed as:
$$
\mathscr{g}_i = \mathscr{f}_{i+1} - K * \mathscr{f}_{i+1}, \quad \text{where } \mathscr{f}_{N-1} = \mathscr{f}.
$$

---

Exemplo - imagem filtrada e subtraida

![](img/details.png)

<!-- _backgroundColor: #000000-->

<!-- _class: invert -->

---

## Multiresolution [Sinusoidal] Neural Networks (MR-Net)

<br/>

$$f:\mathcal{D} \times [0,N] \to \mathcal{C}$$
<br/>

$$f(x,t) = c_0(t) g_0(x) + \cdots + c_{N-1}(t) g_{N-1}(x),
$$

![bg right fit](img/mr-net-stages-v2.png)


---

# Shallow Network (S-Net)

![h:400](img/snet.jpg)

---

# Laplacian Network (L-Net)

![h:400](img/lnet.jpg)

---

# Modulated Network (M-Net)

![h:400](img/mnet.jpg)

---

# Multiresolution image representation

![h:500](img/mr-cameraman.png)


---

# Texture magnification / minification

![h:500](img/mr-tapete.png)

---

# Continuous Scale

<!-- ![](img/WhatsApp%20Video%202023-08-30%20at%2008.36.38.mp4) -->

<video width="1102" height="600" controls>
  <source src="img/continuous_scale.mov" type="video/mp4">
</video>

---

# Anti-aliasing

![h:500px](img/mr-antialiasing.png)


---

<!-- _class: invert -->

# Demo

### [Training a MR-Net for images](https://github.com/hallpaz/nov23google/blob/main/code/representational_networks.ipynb)
<a href="https://colab.research.google.com/github/hallpaz/nov23google/blob/main/code/representational_networks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- <table>
  <tr>
    <td> <img src="img/masp.gif"  alt="1" ></td>
    <td><img src="img/fourier_masp.gif" alt="2"></td>
    <td><img src="img/siren_masp.gif" alt="3"></td>
   </tr> 
</table> -->

---

# There's more to it...


<!-- _footer: To know about follow-up work on material textures, please check [this Youtube video](https://www.youtube.com/live/voVBM6BYs8k?si=Zxj4gtWJe893uzuV&t=4225) or [this Siggraph Poster](). --> 

---
# Thank you!

#### Reach out:

- hallpaz@impa.br
- [@pgdinamica](https://youtube.com/@pgdinamica) on Youtube
- @hallpaz at social media

<!-- _class: invert -->

----

# References:


- NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. [Website](https://www.matthewtancik.com/nerf).
- Occupancy Networks - Learning 3D Reconstruction in Function Space. [Website](https://github.com/autonomousvision/occupancy_networks).
- DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. [Github](https://github.com/facebookresearch/DeepSDF).
- Implicit Neural Representations with Periodic Activation Functions. [Website](https://www.vincentsitzmann.com/siren/).
- AI Graphics Theory and Practice. [IMPA Course](https://lvelho.impa.br/i3d23/).
- The making of MR-Net and a vision for multiresolution media representation. [Visgraf Seminar](https://www.youtube.com/live/voVBM6BYs8k?si=lgOZT6BRh9eL8fz3).

---

# References

- On the spectral bias of neural networks
N Rahaman*, A Baratin*, D Arpit, F Draxler, M Lin, F Hamprecht, Y Bengio, ...
International Conference on Machine Learning, 5301-5310
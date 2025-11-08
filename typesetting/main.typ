#import "lib.typ": title-page, info-eng, info-est

#set par(justify: true)
#set page(margin: 25.4mm)
#set heading(numbering: "1.")
#set text(font: "New Computer Modern", size: 12pt)

#title-page(
  thesis-title: "Prefix Tuning Applicability In Language Models",
  author: "Mart-Mihkel Aun",
  supervisor: "Sven Laur, DSc",
  curriculum: "Data Science Curriculum",
  year: 2026,
)

#info-eng(
  thesis-title: "Prefix Tuning Applicability In Language Models",
  keywords: "LLM, fine-tuning",
  cercs: "P176",
)[#lorem(50)]

#info-est(
  thesis-title: "Prefix Tuning Applicability In Language Models",
  keywords: "LLM, fine-tuning",
  cercs: "P176",
)[#lorem(50)]

#pagebreak()

#outline()
#pagebreak()

= Introduction

= Prefix Tuning

Let $x$ be context and $y$ be output. Autoregressive language model
$p_phi.alt (y | x)$ parameterized by $phi.alt$. Consider the concatenation
$z = [x ; y]$, with $X_"idx"$ being the indices corresponding to $x$ and
$Y_"idx"$ corresponding to $y$. The activations $h_i in RR^d$ at time $i$,
where $h_i = [h_i^((1)), dots, h_i^((n))]$ is a concatenation of the
activations at all layers.

Model computes $ h_i = "LM"_phi.alt (z_i, h_(lt i)), $ where last layer of
$h_i$ is used to compute the output token distribution
$p (z_(i + 1) | h_(lt.eq.slant i)) = "softmax"(W_phi.alt h_i^((n)))$.

Fine-tuning
$ max_phi.alt log p(y | x) =
    sum_(i in Y_"idx") log p_phi.alt (z_i | h_(lt.eq.slant i)). $

Prefix-tuning prepends a prefix $z = ["Prefix", x, y]$ indexed by $P_"idx"$.
Initialize trainable parameter matrix $P_theta$ of shape
$|P_"idx"| times dim(h_i)$ and computes
$ h_i =
    cases(P_theta [i, :] &", if" i in P_"idx", "LM"_phi.alt (z_i, h_(lt i) ) &", otherwise".) $
Training objective is the same but $phi.alt$ are frozen and only $theta$ are
trainable.

Directly updating $P_theta$ is unstable. Reparameterize
$P_theta [i, :] = "MLP"(P'_theta [i, :])$ with a smaller $P'_theta$ matrix and
a feed forward network. Both $P_theta$ and $P'_theta$ have the same rows
dimension.

Above is plagiarism from prefix tuning @li2021prefix.

= Related Work

== Context based parameter efficient fine-tuning

P-Tuning @liu2024gpt, Prompt tuning @lester2021power.

== Weight based parameter efficient fine-tuning

LoRA @hu2022lora.

== Language models

BERT @devlin2019bert, Llama @touvron2023llama.

Few-shot learning @brown2020language.

= Conclusion

#bibliography("ref.bib", style: "annual-reviews-author-date")

= Appendices

= License

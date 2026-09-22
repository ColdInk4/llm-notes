# Math render probe 2 — E minimal repro

## P1 bare subscript-lt display
$$
x_{<t}
$$

## P2 sum + lt
$$
\sum_{t=1}^{T} x_{<t}
$$

## P3 log mid exact core
$$
\log p_\theta(x_t \mid x_{<t})
$$

## P4 backslash-lt variant
$$
x_{\lt t}
$$

## P5 sum + backslash-lt
$$
\sum_{t=1}^{T} x_{\lt t}
$$

## P6 E exact
$$
\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

## P7 inline ch11-exact
perplexity $L=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$ here

## P8 underscore text double backslash display
$$
\sqrt{\text{input\_dim}}
$$

## P9 operatorname -> mathrm
$$
D = \mathrm{mix}(R)
$$

## P10 nuclear norm star
$$
\lVert W_l\rVert_\ast = \Theta(\sqrt{n_l/n_{l-1}})
$$

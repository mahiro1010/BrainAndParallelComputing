# Statistical Theories for Brain and Parallel Computing -- quiz4

@import "../name.md"

---

## Formula

$$
\begin{cases}
    2x_1 - x_2 + x_3 = 2 \cr
    -x_1 + x_2 - x_3 = -1 \cr
    x_1 - 2x_2 + x_3 = 1 \cr
\end{cases}
$$

## Energy Function

### general Form

$$
\begin{aligned}
    E
    &= -\frac{1}{2} \sum_{n = 0}^{N} \sum_{m = 0}^{N} w_{nm}x_{n}x_{m} + c \quad (N = 3, w_{nn} = 0, w_{nm} = w_{mn}) \cr
    &= -\frac{1}{2} \sum_{n = 1}^{N} \sum_{m = 1}^{N} w_{nm}x_{n}x_{m} + \sum_{n = 1}^{N} \theta_{n}x_{n} + c
\end{aligned}
$$

### for this problem

$$
E = (2x_1 - x_2 + x_3 - 2)^2 + (-x_1 + x_2 - x_3 + 1)^2 + (x_1 - 2x_2 + x_3 - 1)^2
$$

## Network

```mermaid
graph LR;
    x0((x0));
    x1((x1));
    x2((x2));
    x3((x3));
    c[c = 0];

    x0 -. "&theta;1" .-> x1;
    x0 -. "&theta;2" .-> x2;
    x0 -. "&theta;3" .-> x3;

    x1 -- w12 --> x2;
    x2 -- w21 --> x1;
    x2 -- w23 --> x3;
    x3 -- w32 --> x2;
    x3 -- w31 --> x1;
    x1 -- w13 --> x3;

subgraph bias;
    x0
end
```

## Connect weight, threshold, constant

|$\theta_{n}$|1|2|3|
|--|--|--|--|
|n|-6|16|-5|

|$w_{nm}$|$m = 1$|2|3|
|--|--|--|--|
|$n = 1$|0|10|-8|
|2|10|0|8|
|3|-8|8|0|

$c = 6$

[mermaid reference](https://github.com/mermaid-js/mermaid/issues/39)

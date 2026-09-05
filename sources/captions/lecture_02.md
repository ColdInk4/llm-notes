# Lecture 2 字幕

I hope everyone is staying dry.
I'm not.
So, as I mentioned last time, the Marin project had a which was running and it finished and it actually matched the forecast.
So, remember we were running these each of these curves is essentially a iso FLOPs curve, which is a bunch of smaller model runs and you try to find the computer optimal point.
You fit a scaling law and this was the point where we predicted the loss and we ran the loss the model and it got lost within 0.05.
So, I thought that was pretty cool and if you extrapolate out to GPT-5 level performance, this is the loss you get.
Of course, your mileage might vary depending on how these scaling laws are.
Okay, just wanted to share share that news. so, last lecture I gave an overview of the entire class and we are talking about tokenization, which is going to be on the first assignment. today I want to talk about resource accounting, which is going to be more on the systems side of things. so, to recall, the main thing we're trying to do is train the best model we can given a finite set of resources, which could be a compute, memory, sometimes data, but that's not really going to be a limiting factor for us in this class.

And our goal is to simply to maximize the computational efficiency of our training.
So, before you can optimize the computational efficiency, we need to understand the efficiency of a given computation and for that we need to understand the compute and memory characteristics. just to give you a taste of the type of questions you will hopefully be able to answer by the end of the class. so, here's a question.
How long would it take to train a 70 billion parameter model on 15 trillion tokens on 1024 H100s.

So, how do you answer that?
Well, there's a formula which we'll talk about how you can get the number of FLOPs to be six times the number of parameters times the number of tokens. we can look up the spec sheet to see how fast the H100 is. we have this thing which called MFU, which we'll talk about 0.5.

then you can estimate the number of FLOPs that you need that you the hardware gives you per day and then you can compute the number of days.
So, the number of days is 143.

Okay. here's another question.
What's the largest model you can train on H1 eight H100s using Adam W?

so, you can look at Well, H100s have 80 GB of HBM memory. the number of bytes a per parameter, which is 2 + 2 + 4 + 4.

We'll explain where that comes from.
And then the number of parameters is that you can get is going to be about 53 billion.
So, there's some caveats here that we don't count the activations, which depends on the batch length and batch size and the sequence length.
So, this is all very rough back of the envelope calculations, but hopefully by the end of this class you'll understand where these come from and the point is not to precisely calculate every single thing, but just get the rough shape of things.

okay.
So, last time I talked about knowledge and what you can take away from this class. mechanics, which are how things work.

So, today that will be pretty straightforward.
The mechanics are just how PyTorch work, how tensors work.
It should be fairly There's no magic here. the mindset I want you to impart on you is that resource accounting is going to be very crucial and while everyone to get in habit whenever you write you know, a piece of line of code, think about you know, the performance characteristics.
And then finally, intuitions.
Here we're just going to get a sense of the resources, how they're spent.
There's going to be no ML magic today.
I'll leave that to Tatsu for the next lecture.
Okay.
So, let's get into things.
So, let's start bottom up and start building up.

So, what is at the bottom?
The bottom are tensors.
So, tensors are the building block of storing everything.
If you have parameters, gradients, optimizer states, data, activation, everything essentially is a tensor.

so, you know, for example, you can take a look at the DeepSeek-V3.2 model and you see that the model itself is a bunch of different tensors.
Each tensor has some shapes and also some precision, which I'll talk about later.
And so, as you know, tensors subsume vectors, matrices and can be generalize to any number of entities.
Okay.
So, let's talk about how much tensors take to store.
So, it depends on the type of tensor.
So, in general, we're going to be dealing with tensors that store floating point, but tensors can also store you know, integers and other types. so, for floating point, typically whenever you talk about float, I think the standard people refer to as float what is called float 32.

so, a float 32, if you break it down, has 32 bits.
One of the bits is a sign, eight of bits are the exponent, which gives you dynamic range and the rest is the mantissa or the fraction, which gives you kind of you know, variation.
Okay, this is also known as FP32 or you know, single precision. so, and the term single precision comes from the fact that back in the day when you're doing scientific computing, float 32 was sort of like a baseline.
It was just like you expect if you someone gave you a float, you would kind of expect it to be at least single precision and if you want more precision, then you can get double precision as that's you know, float 64.

But in deep learning, we're kind of going the other way because even 32 is a lot and the types of computations that we want to do don't demand the high precision that some kind of numeric simulations do. okay.
So, before we get to other types, let's just look at a float 32.
Okay, so let's construct a 4 by 8 matrix.
By default, the type of a tensor you create is float 32.
So, if you want something else, you should declare it.

and the memory usage is just the number of elements times the element size here, which is 4 bytes for a 32-bit number and that's going to be 128 bytes.

Okay. just to give you kind of perspective.
So, in GPT-3, which is a fairly old model, one of the matrix in the feed forward layer is about you know, 2.3 GB.
Okay, so these tensors can be you know, get quite big and this is by far even the biggest one that one can imagine.

Okay, so, since we're interested in efficiency, we want to generally reduce the amount of storage and we'll see that as you reduce the precision, you actually save my memory and you also save time because operating on you know, 16 bits is going to be faster.

let's say twice as fast, but not always it sort of depends, but and then memory by reducing memory, we'll see later that actually reducing memory can save time as well, which is maybe less obvious, but it will hopefully become clear.
So, the obvious thing is you say, okay, let's take away half the bits.
Now you have float 16.
So, float 16 says you have a sign, you have only five bits of exponent and then the rest is the mantissa.

so, fine, fifth float 16 is good, except for you know, its dynamic range is kind of poor.
Right?
So, even if you have, let's say, try to construct a 1E minus 8 tensor, then that is actually just zero.
So, you can't really represent very big numbers and you can't represent very small numbers.

and the reason is that this exponent is you know, it's only you know, you know, five bits of exponent from compared to eight.
So, if you train with FP16, which people did back back in the you know, day, you will get instability.
You will you get underflow, you get overflow, you'll get NaNs.
It's it was it's pretty challenging.
So, then so, BFloat16 was invented.
So this was developed in actually 2018 to address this issue.

And the observation was that well, let's not compromise on the number of bits.
It's it's it's the number of bits is going to be the same as FP16, but we're going to shift some of the bits from the mantissa to the exponent.
So that means it has more dynamic range than float you know 16. and it actually has the same dynamic range as float 32.
But of course the resolution is worse because there's no free lunch here.
But it turns out that in a lot of deep learning applications, this is well worth the trade-off.
You want the dynamic range to not overflow and underflow and because things are kind of sloppy and stochastic anyway, you don't need that much resolution.

Okay?
So let me actually skip over this part.
So okay, so to summarize what are the what are the implications for training?
So you can absolutely train with float 32.
And if you're training a small model, you probably just and you don't want to worry about it just you know float 32 is fine.
But it requires four bytes of sorry, four bytes of memory per you know float.
And that can take up a lot of memory. and if you train with float 16 then that's going to be too risky.
So float 30 16 a BF 16 is sort of a sweet spot. even BF16 can be kind of risky as well maybe see in a little bit. one thing that people have has become kind of common practice is to use mixed precision training. and actually let me I think these Okay, let me compile this again.

I think these are stale.
So mixed precision training is where some of the computations use some precision and for some other computations use other precision.
So in general as a general rule, BF you know 16 is what you would use for parameters activations and gradients.
And for optimizer states, you would use FP32.

We'll discuss a little bit more of that about that later.
And to do invoke mixed precision training, so PyTorch has a AMP library.

We're not going to talk too much about this, but you basically wrap your code and PyTorch has the library takes care of it automatically for you in the sense that it tries to cast things into BF16 when it's safe.
So for example, matmuls are generally safe, but if you try to do exponentiation, then it will try to leave things as FP32.

Okay, so we could do BF16 I think is probably where this class will end.
But if you're feeling very adventurous, you can go farther.
So FP8 this was introduced four years ago. it actually has been you know standardized.
So if you look at FP8, there's actually two versions because depending on if you need more dynamic range or more resolution. and there's there's you know two versions of them. and so we're not going to talk about that, but you know Nvidia has some you know the transformer engine supports FP8.
And well most recently, you can actually go down to FP4.
So last year Nvidia developed NVFP4 and there's only four bits per value.
So if you just to so we're on the same page, four bits is not a lot.
Like I can write all the values down on a single line here between minus six and six. so that's not very much precision.
Now there's a little bit of a cheat here, right?
Because if you just naively only use these values, you're you're not going to be able to train very very well.
So what actually this means is that you every value you can have this four bits of freedom.

But there are blocks and each block can be scaled up and down accordingly.
So you can actually represent more values, but you can't represent the full dynamic range for every single value.

And there was a model released actually this year, the NeMo-3 Super which was trained in FP4 which I think is a pretty pretty cool.

Okay.
So some of this you it's just good to know and some of this you can't even even touch.
It's not like you create a tensor and you call call it FP you know it FP4.
A lot of this is done under the hood by Nvidia's you know software stack.

Okay, this was a bit of a kind of a long digression, but it's it's maybe helpful to appreciate that the intricacies of you know precision here.
Okay, any questions before I move on?

Yeah.
Sorry there's a question like when you have a block you're scaling all those same values like you have a tensor by that block something like that.
The ratios of all of them will still be in FP4.

Yeah, so the question is just like scaling with the maximum value is to have more specificity.
Yeah, so the question is okay, just to explain the block a bit more.
So you have let's say a block.
Within that block you can vary up within the four bits.
And in addition, all those values can be scaled up and down according to how many of your bits your scaling factor is.
So if you look at an individual value, you actually get more than four bits of dynamic range, but it's just like you can't have this value be way over here and the next neighboring value way down Okay, so one question about like what about like one bit? couldn't BF16 be pushed to a bit one bit and all of that?
Yeah, so there's a problem with training obviously.
Right, right.
So maybe this is a good point.
So the question is about one bit because you can't really go lower than that.
So there's a difference between training and inference.
So a lot of the low bit stuff if we talk about quan- I think we'll talk about quantization later.
Is that you train a model at on and you know maybe even BF16 and then you quantize into like let's say one or two bits. and that is much easier than training a one bit language model which I don't think anyone has you know done.
Maybe it's possible, but I don't think anyone has trained anything credible there.
Okay.
Let us go on then.
So we talked about tensors and the memory calculus is pretty simple just the number of elements times however much memory each element takes up. so you know by default the tensors you create in PyTorch are going to be on CPU and of course you have to if you want things to go fast, you want to move them to GPU.
And actually so there's a bit of a I have a slight issue with the slides with that.
They were executed on my laptop which means that I don't have GPU.
So some of the code I'll just show but not execute. so to okay, let's see.

This is maybe not that the interesting, but I think everyone knows how to move tensors to GPU.
But just remember to do that otherwise you won't get your speed ups.

Okay, so we talked about memory of tensors which is very straightforward.
Now let's talk about computing with tensors.
So before talking about FLOPs accounting for the tensor operations, let's take a little bit of a digression to talk about einsums.
So how many of you have familiar used einsums before?

Okay, so like maybe two thirds.
Okay, good.
So the motivation behind einsums for those of you who might not be indoctrinated is it's very easy to mess up or I find it very confusing to look at code such as this and you have X and Y and then you have like transpose minus two and minus one and you're trying to figure out like what what minus two and minus one is.
And so this is kind of maybe the motivation for you know using variable and names rather than indices.
So einsums is a library for manipulating you know tensors where the dimensions are named.
And this is inspired by Einstein summation notation.
There's a nice tutorial which you can go through.
I'm just going to cover some of the you know basics.
So basically the way to think about einsum is a generalized matrix multiplication with you know good bookkeeping.
So here's an example.
So I have a matrix 3 by 4.
I have a 4 by 3 matrix and if I do the matmul, this is you know this is actually you know pretty nice.
It's pretty easy to understand. in einsums, basically you say X has two dimensions, the row and the column which I'm going to name seq1 and hidden.
I'm going to have Y that's a matrix which has also rows and columns, which I'm going to name hidden and seek two.
And I'm going to produce a tensor or here a matrix where the dimensions are indexed by seek one and seek two.
And anything that is not mentioned here, the hidden, gets summed out.
Okay?
So, the way this works is I'm going to sum over enumerate over all possible values of all the variables that occur here, which are seek ones, hidden, and seek two.
And I'm going to basically index into X, index into Y, multiply them, and dump them accumulate them into the result Z sub seek one seek two.

Okay?
All right.
So, you say this is Come on, this is much easier than this, right? so, Okay, let's try a more complicated example.
So, here's the example I showed you know, before.
Now we have a tensor.
It's 2 by 3 by 4 and another tensor 2 by 3 by 4.
And if I were doing things kind of the old way, basically I am what I'm doing is transposing these last two and then because at implicitly sort of batches the dimensions that are not the matmul dimensions, then you get the answer.
Okay?
So, you have to kind of reason about this a bit.
Einops makes this very clear.
It says there's a batch dimension, there's seek one hidden, seek two hidden, and I'm just going to produce batch seek one seek two.
Okay?
And notice that there's no transpose because I just you know, in some sense I've done the transpose by the naming.
If I had hidden and seek two, then it would be a no transpose.
I always get confused by transposes and the fact I don't have to think about transposing makes me happy.

Okay.
So, if you want to get fancy, you can say, well, batch I'm just going to replace with dot dot dot.
And this means that if I had, let's say, you know, a you know, rank 10 tensor with, you know, eight different you know, batching dimensions, I can just write dot dot dot without enumerating all of them. and this comes up in you know, language modeling because you might have a batch dimension, you might have a sequence dimension, you might have a head dimension, and you're trying to do like this matrix operation for all of them.
And you might not want to have to just, you know, worry about this. and the nice thing is that you can write modular code where you can write this dot dot dot without worrying about, you know, the even the shape of the tensor that, you know, comes in.

Okay?
All right.
So, that's ein sum, which is in the einops library. there's a reduce.

So, this is a generalization of sum, mean, max, and min.
So, for example, if you have, let's say, this tensor, and you want to sum according to dimension of dim minus one, okay, which means sum along the last dimension here. so, again, I don't like the this notation, but what you can do is called reduce.
And basically what you say is that there's some batching dimensions.
In this case, it would be this first these first two.
And then there's some, you know, hidden dimension, which doesn't appear on the right side, which means that it gets summed.
And here I put sum, which means that the aggregation reduction operation is sum, but you can replace it with mean or max or min.

Yeah.
Is there some speed up to this?

Is there some speed up to this?
This basically reduces to the same type of primitive operations.
You can think about it as just sugar.
So, it should be the same.

okay.
So, the final thing I'll talk about is rearrange.
So, this is I think a pretty powerful tool.
I think it will come up in the assignment once.
So, sometimes you have a dimension that actually represents you know, two dimensions, and you want to operate on one of them.
And the reason this happens is that sometimes you have a matrix and you flatten it. okay.
And then you want to maybe unflatten and flatten.
So, this is the way that works here.

So, imagine I have a matrix 3 by 8, but where the this the this dimension eight actually represents like a 2 by 4 matrix.
So, I want to multiply that 2 by 4 matrix by this 4 by 4 matrix.
So, what I'm going to do, actually, is to call rearrange.

And what I am doing here is saying Look, this is some number of batch dimensions, which here just corresponds to the first element here.
And then here I use parentheses to say that this H actually represents the product of heads and hidden one, where I have to obviously there's multiple ways to decompose this.
Could be 2 by 4, 4 by 2.
So, I said the number of heads is two, which means that hidden one is four.

And then I can break that up into you know, two dimensions, heads and hidden.
Okay?
So, this creates this So, before X looked like this, and now X looks like, you know, this.
So, cuz this might be a little bit hard to see.
So, then you can perform your operation transformation on W.
And this is sort of what we've seen before, where you have just a standard matmul, where there's on some number of batching dimensions for X here.
And this is hidden times hidden by hidden two. and then once you've done that transformation, you can rearrange it back. and this is straightforward.
You basically look at two dimensions, and then you can group them into to one dimension.
Okay?

Yeah.
Like if you take a two-dimensional thing and shift it into one dimension, are there two ways?
Like you could do it row major or column major.
Yeah.
So, the question is that if you have a one-dimensional thing and you shift it into Sorry, you have a two-dimensional thing and you shift into one dimensional thing, which way do you do it?
Well, the order you do it is specified in the order here.
Oh, okay.
Yeah.

Okay.
So, I Sometimes I find it takes a bit of time to get it used to, but it's well worth it because once you have einops, then you just think in a sort of a different way, and all the transposes and reductions all of them it's just like it's becomes more fluid.
Right?
You just have to think through these kind of more bespoke primitives.

All right.
So, Okay, we're going to use einops. we'll see it a little bit later. so, let Now let's go return back to resource accounting question.
So, I have tensors.
We've talked about how they take memory.
So, how much compute do they take?
Okay?
So, the thing we're going to use to measure computation cost is the number of FLOPs.
A FLOP is a floating point operation, and it's going to we're going to assume it's a basic operation like addition or multiplication.

So, now there's other things that you know, GPUs can do, but for the most part we're just going to ignore them because these are the bread and butter and are going to eat up most of your time.

So, one thing that is a kind of a pet peeve is that if I say the word FLOPs, it's actually ambiguous what I mean. so, there's FLOPs, which is saying the number of floating point operations, usually written FLOPs with a lowercase S.
This is a measure amount of computation done.
And then there's FLOPS, which is floating point operations per second. sometimes it's also very confusingly written as FLOPS with uppercase S, which but I'm going to always write /s to make it clear that this is measuring the speed of hardware.

So, if you go and see that H100s have 800 989 you know, teraflops, it's the second.

And when I say that, you know, GPT-3 took, you know, a 3.14e23 or whatever it is, FLOPs, that's the former.
Okay?
So, just to get that out of the way. so, just to give you an order of magnitude, so the number of FLOPS when I talk about 1e22 or 23 or 25, these are kind of referring implicitly to amount of compute or the scale of some of these models.

And so, if you look at H100s, you know, if you look at this glossy's spec sheet, actually, it's not on this page.
Okay, forget it. there is a spec sheet and it'll it'll tell you that for BF16, the number of FLOPS/s is 1979. and then you go and you benchmark and it's like, wait a minute, that's not actually what I'm getting.
And then you go read the, you know, the fine print and there's a footnote that says this is with sparsity, so sparse matrix, and for dense it's you know, over two.
Okay?
So, all you always have to take these numbers divide by two.
So, that's why you see these divide by two. okay.
So, so this allows us to you know, you know, just so our intuition, so if you have eight H100, so this one node for two weeks, that's you know, H times the number of seconds per in two weeks.

actually, this is looks like it's one week.
Okay, fine.
It's a one week times the number of FLOPs you get per second.
So, that's you know, about 5e 21.

Okay, so this is just kind of building intuition for the number of FLOPs that certain types of hardware have and how many FLOPs, you know, certain types of models require.
Okay, it's nothing fancy, it's just math and napkin math.
Okay, so now let's do some more something more mechanical.
So, you know, suppose you have a linear model. it turns out that a lot of this calculus is of counting FLOPs is actually going to be at the core it's like linear matmuls.

So, this is actually not without too much loss of generality.
So, you have M points, each point is D dimensional, and we're going to map each of these D dimensional vectors to a K dimensional output.
Okay?
So, so B is going to be the number of points, D is the number of dimension input dimensions, and K is the number of output dimensions.
Okay, so we're let's construct some an X, which is the data matrix B by D, the weight matrix is D by K, and we do a the matmul. the question is, you know, how many FLOPs that is.
And it turns out that this is going to be two times the basically the product of all the three dimensions.
And the way to see that is that we have one, you know, multiplication for each you know, triple, and then also one addition.

Okay?
So, there's a minus one because if you don't you have to actually add like, you know, in this case D minus one times, but let's ignore that.
Okay, I'll come back to this if this was a bit fast.
I think there's a there's another way to you know, derive this.

So, what about the FLOPs of other operations?
So, element-wise operations are just you know, the size of the matrix.
I think that's fairly clear, so addition also requires NM FLOPs. so, in general, no other operation you encounter is expensive as matrix multiplication for large enough matrices.
So, in general, we're just going to focus on what matmuls are doing with the important caveat of when we talk about memory.
Yeah.
I would just just like this is for interest, like there are some other algorithms for doing matrix multiplication.
Is this only for, you know, just doing it normally?
So, the question is that there are other algorithms for doing matrix multiplication.
You mean like sub cubic cubic algorithms. so, in general, that's the optimization the algorithms that people are going to explore for much of matrix multiplications are going to be much more about how you co-design with the systems rather than these more asymptotic algorithms.

Yeah.
Yeah. we're considering addition and multiplication in the same way.
It's not possible to do addition more efficiently than multiplication.
Yeah, so I think the way the hardware is built, the two are basically the, you know, the same.
But yeah, intuitively it seems like addition I can do addition faster than do multiplication, but the hardware the way hardware built is built they're kind of the same.

Okay, so you can think about this matmul as B is the number of data points, and DK is the number of parameters, right?
So, remember X is B by D and W is D by K.
So, another way to think about this formula is that the number of FLOPs in order to do a forward pass of this linear matrix is actually two times the number of tokens or data points times the number of parameters.
Okay?
So, this is a you know, it turns out that this actually generalizes to transformers, which if you remember the six times N times D formula, we can kind of see the shape of that you know, forming.

Okay, so unfortunately, these calculations are not going to be very meaningful because I'm doing this on CPU, but I'll just walk through the code here.
So, so the what we've so far done is measured FLOPs.
So, this is independent of hardware, it's just like the number of calculations you need to do for your model.
So, now the question is like how long does it actually take on hardware?
So, one way to find out is you just time it.

Okay?
So, in this class, I think in a few lectures we're going to talk more about benchmarking, but here's a little preview.
So, in general, when you time, especially on GPU, you have to call CUDA synchronize to make sure that because, you know, the GPU is like kind of running asynchronously, you want to make sure that you have this kind of synchronization point. and then you perform the operation, and after the operation, then you also have to have the synchronization barrier.
If you omit this, you're going to find that wow, your timings are really fast, and that's because these this is a non-blocking call, it just returns.

and often it's general good practice to try this multiple times and take the average. >> [clears throat] >> Okay.
So, so the actual FLOPs per second is basically the number of FLOPs you did times the time that you recorded on your hardware.
Right?
And so, remember there's a also a number number, which is the GPU has a spec sheet. let's see if I can pull it up on this link.
I feel like maybe I have the wrong link.
I'll have to fix that after class. that's it gives you some number of FLOPs, which was 989 teraflops.
And so, in general, the number of actual FLOPs per second is going to be different from the promised FLOPs per second.
Okay? and the Okay, oops.
Sorry.
And the way to direct think about the discrepancy is something called model FLOPs utilization or MFU.

And the definition of MFU is the actual FLOPs per second divided by the promised FLOPs per second.
And here this is, you know, ignoring the you know, the communication and other overhead.
Okay?
So, you basically take the actual and divide by the promised.

So, in general, you know, it's rare that you get more than what's you know, I mean, it's yeah, you just never get anything more than what you were promised.
Often you get less, and in general, if you get about MFU of 0.5 for modern models, you should be pretty happy with yourself. if you have only just like a straight-up matmul, you can get maybe like, you know, potentially like 80 0.8 even. but you usually can't get that high.
And sometimes if you're have really, you know, something's really wrong, you'll get something like 0.1, which means that you should do something.
Okay.
So, whenever you're you write your model, you can calculate you now know how to calculate MFU through a combination of counting the number of logical FLOPs that you your model is needs to, you know, do, and then looking at the wall clock time and essentially dividing.
Yeah.
Was there a question over there?
Oh.

Yeah, so is the promised What is the promised FLOPs number?
That is in the spec sheet, it is already divided by two of the 1989 number.

And then on top of that you only get like 0.5 of that in general, depending on your computation.

You're getting 50% Yeah, so why are you getting only 50% MFU? actually, that's a good question.
I'll come back to that when we talk about memory bottlenecks.

Okay, so to summarize here, matrix multiplications dominate the you know, generally the computations, and that's sort of by, you know, design.

so and the number of FLOPs per second depends on the hardware.
So, better hardware leads to better num more FLOPs, and also depends on the data type. which means that if you look at the spec sheet, different data types will have different FLOPs.
If you try to do float 32, nowadays it's going to be really really slow because they're not really optimizing for that workload, whereas now BF16 or FP8 are going to be much faster.

Okay?
And MFU, now you know what MFU is.

It's the you know, actual FLOPs divided by promised FLOPs.

All right.
So, to go back to the question of why is MFU 0.5?

And to understand that, I'm going to have to introduce this idea of arithmetic intensity.
And the re- reason is that, well, it's not just you know, doing a bunch of matmuls and then you're done and looking at how long it the matmuls take. this is my very cartoon version of what hardware looks like. you have high bandwidth memory and then you have where the compute cores or the cell accelerators chips are.
And then what how do you compute?

Well, you have to send your inputs, your matrices there's there's tensors are sitting down here, from the memory to the accelerator.
You do the computation, and then you send it back.

Okay?
So, if you want to measure how long this takes, this sort of depends on two things.
One is the accelerator speed, which is what we've talked about just you know, just now.
But the other thing that matters is the memory bandwidth of your hardware, which we haven't talked about.
And if you look at the spec sheet, we talked about how the FLOPs per second was 1979 E12 / 2.
And the bytes per second, which is the memory bandwidth, this is 3.3 terabytes per per second.

Okay?
And remember we why we were looking at memory how much things took to store, it's going to it's most obviously that, well, you just if you have a model that's too big it doesn't fit in your memory, that's not going to be fun.
But also, it turns out that memory you need to move this memory, which takes time.
So, actually the size of how large things actually influences speed as well.

Okay, so I'm going to talk through some operations, and I'm going to basically compute how long things are going to take, and introduce this idea of arithmetic intensity.
Okay, so suppose I have a million dimensional vector of BF 16, and I'm going to just compute a value on this.
So, re- remember ReLU is just max of X and zero done element-wise on the entire vector here.

So, I count two things.
One is the number of bytes that were moved.

So, I have to read X, in you know, copy it into the accelerators, and each of this is going to be two because a BF16 is two bytes per float times N floats.
So, that's two N, and then I'm going to write Y back.
So, that's another two N.
Okay?
So, that's the number of bytes that have to be moved.
And then, how many FLOPs were done?
Well, each of these elements, I'm just comparing with zero, and that's it.
So, that's N comparisons.
Okay?
So, now I look at the communication time, which is the number of bytes that were I needed to move, you know, divided by the speed of that movement, and that gives me the time, which is, you know, 1e-6 you know, seconds.
And what about the computation time?
That's the FLOPs divided by FLOPs per second.
So, that's 1e-9.

Okay.
So, there's also going to be another important assumption, which generally we try to hold is that we overlap communication and computation.
We're going to talk more about that when we talk more deeply about GPUs, but the idea is that in this case, as we don't sit here waiting for the you know, the things to move, as soon as they're there, we start computing them, and then we move them back.
So, that this movement and also the compute is happening at the same time.
So, mathematically we're just going to assume that the total time is the max of the two, because we're going to assume that we can perfectly overlap them.
In practice, it's not going to be perfectly overlapped, there's going to be some overhead, but this is a good enough for now.

Okay, so the total time, as we see here, is 1e-6.

Okay.
So, if you ask, you know, what is the bottleneck here?
So, when the communication time is greater than the computation time, then we call it the algorithm memory bound, because you're spending most of your time just waiting for bits to show up.
And when the com- computation time is greater than communication time, that's compute bound, because then you're actually, you know, your bottleneck is actually doing the compute.

And so, in this case, what is ReLU?

Rectified linear units.
Oh, sorry.
Is it memory bound or you know, compute bound?
Memory bound. >> [laughter] >> A memory bound.
Yeah.

So, and it's clear, right?
Because, you know, the compute is way less than the communication.

Okay.
So, here's another way to see it, and this is where I'm going to define the intensity.
So, what is intensity of an accelerator is essentially how much work can the accelerator do per byte transferred?

And for any given accelerator based on the spec sheet, you basically have the FLOPs per second divided by the bytes per second.
Okay?
So, how much useful work can you do per byte that's moved?
For H100s, that's 295.

Okay?
So, that means for every byte of you can do 295 floating point operations.
Okay?
So, that's an intuitive number to kind of have in your head, about 300.

okay, so now the arithmetic intensity of algorithm is how much actual work was done per byte for this workload.

And if you look at it for the ReLU computation, it's FLOPs over bytes, and it's actually it's so this is actually a quarter, I guess. not half.
So, the point is that it's very small, 0.25.

Okay, so now we can talk about bottlenecks through the language of intensity.
So, if something is memory bound if the arithmetic intensity is smaller than the accelerator intensity, and compute bound if it's greater than accelerator intensity.

All right, so these are equivalent, and if you look at the algebra, it's basically you have two fractions, and then you just kind of multiply and divide to you basically switch the terms around.
Okay, so in this case, you know, we're memory bound.

So, in general, we're going to find ourselves in a situation where we're memory bound because data movement is expensive, and so if you can get higher arithmetic intensity, that's good.

Okay.
So, 0.25.
If you see that number, if someone tells you your arithmetic intensity is 0.25, you should say, "Oh, this is this is really bad."
Yeah.
What's some typical arithmetic intensity for Yeah, so we'll get to that.
Okay.

All right.
So, so okay, so one way to think about increasing arithmetic intensity is that let's just try to do more stuff per unit of byte moved.

Okay, so the GELU is another activation.

it kind of looks like this.
It's some, you know, formula that is more doesn't you know, have zeros. and if you do this calculation, the number of bytes are moved back and forth, it's still 2N + 2N.

and the FLOPs here, it's about 20 FLOPs per element-wise GELU operation.
So, it's 20N.
Right?
So, the arithmetic intensity is, let's say, five.
This is a kind of a crude estimate.
And in this case, are we memory bound or compute bound?

Memory bound.
Memory bound.
Right?
Because five is still smaller than 295.
Way smaller.
So, even though GELU does a lot of a lot of work, more work than ReLU, in the way that things are structured, it's still it's still memory bound.
Which means that if you were just computing ReLU and GELU, you think that, well, GELU is like so complicated, it must be really expensive, but actually it's exactly the same.
Right?
Because that's not where the bottleneck is.

Okay, so now let's look at some linear operations.
So, dot product.
Okay, so dot product, you have a vector X, you have a vector W, size N, and you take the dot product.
So, how many bytes are moved?
So, you read X, which is 2N, read W, which is 2N, and then you write Y, which is a scalar, which is two.

Okay, and the number of FLOPs is you do the N multiplications and N minus one additions, so that's 2N minus one.
Okay, so what's the arithmetic intensity?
Oh, for this one it's about half.
Okay, so which is also pretty bad, which means that hopefully by you get the idea, it's memory bound.

Okay?

So, what about a matrix vector product?

Okay, so what I have this X is a vector, W is a M by M matrix, and you perform this product. how many of you think this will be compute bound?
How many of you think memory bound?
Okay.
So, let's see.
So, I'm going to read X, read, which is 2N, read W, which is 2N squared, and write Y, which is 2N.
And there's the number of FLOPs is basically you're you know doing N dot products, so that's N times the cost of doing a dot product.
And the arithmetic intensity is, you know, barely barely higher.

So, so this is, you know, also memory bound.
Okay.
So, now let's talk about matrix multiplication.
So, this is where things get interesting.
So, I have a M by M matrix, another M by M matrix, and I multiply them.
And the number of bytes that were moved was you know, 2N squared plus 2N squared, and then I have to write a 2N squared bytes back to Y.
And the number of FLOPs here is N squared dot products.

And then what's arithmetic intensity?
It's 300, woohoo.
Okay, so 340, and in general it's roughly N over three.

and intuitively this makes sense because you're sending N squared things, but you're computing N cubed things.

Right?
So, the number of things you're computing over number of things you're sending is order N.

And this gets better the larger you make the matrices.
So, and in general, this is why when you hear people talk about, "Oh, we need to make large batch sizes or have large matrices."
It's exactly this.
If you if you're under the accelerator intensity, making things smaller doesn't actually speed things up, right?
It's all kind of the same.
Whereas, if you get to the point where you're over this, then you're actually saturating your you know, GPUs.

Okay, so finally this is you know, compute bound.
Okay.
So, as long we have large matrices, we're actually pretty good.
We're in compute bound, saturating the accelerator.
And the question about earlier about, you know, what about transformers?
Well, it turns out we'll see in both your assignment, but also in the next lecture, that transformers are essentially big, you know, matrix multiplications with some things sprinkled in between.
So, the so that's good news from arithmetic intensity, and this is by design.
Transformers are designed in a certain way to have high arithmetic intensity.
And one comment here, just to foreshadow the inference lecture, is that matrix vector products is essentially kind of what goes on when you're doing transformer inference.
Right?
Because inference, you're generating one token at a time, and so you only get to it's like a vector that you're trying to dot product with a with a matrix.

and that's and that's where as we saw was memory bound.
Whereas in training time, you get this whole sequence, and you're processing all at once.

Okay.
So, note also that the intensity depends on the precision.
So, by default everything we're doing here is FP you know, 16.

Okay, so also the to tie it to the other question about MFU, and the reason you might be getting low MFU is that while you might be doing >> [clears throat] >> so, MFU is a promised FLOPs sorry, actual over promised.
Right?
So, if you have really large memory band bottlenecks, then you're not actually going to get very good you know, throughput even though you know, you know, the promise is like if you didn't have memory bottlenecks, you were just like going through and doing all the computations of your model.

Okay, final thing on arithmetic intensity, roofline plots.
So, there's a nice way to Yeah?

Yeah.
Generally speaking, these models tend to be memory bound, as they say, for most of the computation.
But yet, the accelerators are 50% is not as good, and then maybe you get 70, maybe you get 80, that's really good.
Mea- meaning that the accelerator is outsized compared to memory. bandwidth. why is it like that?
Why do these accelerators big just big Well, they're just idling or waiting for memory.
So, so the question is could you design maybe accelerators that had better you know, characteristics?

yeah, maybe when we talk about GPUs and understand a bit more how they work, we can talk about why this is.

but if you have an answer, you should tell Jensen, and maybe he can design a better better hardware. >> [snorts] >> okay, so let's visualize the relationship between arithmetic intensity and performance.
So, this plot basically plots the arithmetic intensity on the X axis.
So, every vertical slice here is corresponds to a particular, let's say, algorithm.
And then we have these lines here, and each of these lines corresponds to, let's say, a particular accelerator.
Maybe you know, H100s or B200s or so.
And so, what this shows is and then on the Y axis is the FLOPs per second that are realized.
So, if your algorithm has low arithmetic intensity, like value or dot products, you're going to be over here, which means that the FLOPs per second realized is going to be you know, not as high as your peak accelerator.
And as you increase the memory arithmetic intensity, that's going to things are going to be better.
You're going to be able to saturate your your hardware up until a certain point.

And after a certain point, you're compute bound, and you're obviously you can't exceed the peak FLOPs Okay?
All right, so let's go on here.
Okay.

So, now I'm going to go back and talk about memory and compute for the operations that we need in training.
Right?
So, so far we've done tensor operations, basically matmuls, and you know, we saw basically how much memory it took and how much compute it took. and the interaction between them.
So, now let's actually think about what it takes to train.
So, here is our running example here. actually it's not a linear network, just a deep network. so, I'm going to consider a case where I have a input, which is a B by D input. and a number of layers, where each layer is a D by D matmul.

pa- and then that produces some set of pre-activations, and then element-wise value, that produces some activations of the first layer, and then this is repeated again and again.

Okay, and the output is just going to be the same size.

Okay, so the deep network I mean, just to see what it looks like in PyTorch, just so it's basically a set of blocks.
Each block is has a weight vector associ- sorry, a weight matrix associated with it.
Okay?
And So, when you the number of parameters here is D squared times the number of layers, L.

And when you run the model on a batch of data, what we're doing is we're going through the layers, and each layer we basically apply the linear transformation, and then a point-wise value activation. and then we do that for all the layers.
Okay?
So, this is hopefully straightforward. very simple model. >> [clears throat] >> okay, so now let's talk about, you know, you know, gradients.

So, in let's use an actually a sim- even simpler example.
So, this is just a simple linear model regression.

so, we have a vector 1 2 3, a weight a vec- weight vector 1 1 1.

and we take a dot product and we form this MSE loss.
Okay?
And what happens when we you know, take the do the backward pass is that each of the variables involved in this computation graph has a gradient that is either set or not set.
So, the w.grad gets set to, you know, 1 2 3.

Okay?
So, this is just basic mechanics of you know, PyTorch.
I think it should be familiar to all of you.

So now, the question is like how much compute does gradients you know, take?

Okay.
So, let's count FLOPs for computing gradients. >> [clears throat] >> So, let's take a you know, simplified model where you have an X which is BID and times a W1 matrix.

well, we're going to it's sort of ignore this value for now just for simplicity times W2 which is the same D by D may same sa- shape D by D matrix.

Okay, I'm going to use einsum here for reasons that will become clear. so, the first thing you do in this linear network as you take X and you take W1.
X is batch by input dimension.

W1 is in by out and you get a batch by out.
Okay?
And H2 takes H1 and W2 and it's this kind of the same story.
These are just matmuls.

So, and then you form a loss.
I'm just setting it to some arbitrary just to get a number out.
Okay.
So, what happens in the in the backward pass?
I'm going to call these retain gradients for debugging purposes which you'll see later. you take the backward pass on the loss and the question is how much work did that how many floating point operations was that?
Okay, so let's zoom in on one layer.
Let's focus on the second layer here.
So, this layer.

and so, the second layer takes H1 and just multiplies by a matrix W2 to get H2.
And that's it.
So, if you look at the num- the FLOPs in the forward pass, this is just a matmul.
And we just take the three dimensions and we multiply them together and BF16 that's two bytes. sorry, that's that's irrelevant.
It's this is just two because it's a addition and a multiplication.
Okay. so, in the backward pass, you know, what does this look like?
Okay, so in the backward pass, if you remember your chain rule and your backprop algorithm, you have to compute two things.
You have to compute the kind of backward message the gradient of the loss with respect to your your kind of input and you also have to compute the gradient with respect to your parameters.

Okay?
And what do those gradients look like? this is just by you know, the definition of these I guess it's just the chain rule here written in einsum notation.
You take the D loss D H2 which is H2.grad.

That's a batch by out you know, matrix.
And you multiply it by W2 which is in by out.
And then that gives you H1.grad which is D loss D H1.

And which is batch in.

Okay?
So, I always get this you know, confused whenever you see if you see, you know, learn calculus there's like, you know, one of them has a transpose and like I always forget which order to put them in.
And einsum I think makes it very clear because if you we you can remember even just like looking at the scalar case that it has to look like H1.grad is H2.grad times W2 and the only question is, you know, what how do you index things?
And you index things by basically looking at the shape of these in the named dimensions.
And in this case, it's just a matrix multiplication where you are summing over the output, you know, dimension.

Okay?
And when you do that, we can check that so H1.grad was the thing that was computed by loss.backward and this is actually indeed the same thing that I kind of wrote out here just as a sanity check.
Okay, so the other thing you have to compute is W the gradient with respect to the parameters.
And that's D loss D H2 times H1.

So, it's the same kind of backward message that's coming back but multiplying with other thing that you're not taking the you know, the gradient you know, with respect to.
And you just write out the dimensions and in this case, you're summing over the batch dimension.

Okay.
So, but the you know, the nice thing is that if you just look at this, we know how expensive you know, this is.
The number of FLOPs is essentially the product of all the dimensions.
It doesn't matter which ones you're you're batching or not.
It's it's basically you enumerate over the IJK and you aggregate the way you aggregate is different but the FLOPs is the you know, the same.

Okay, so notice that the backward pass is exactly twice as expensive as the forward pass and this is because you have to take you know, compute two gradients.
One with respect to the parameters the for each parameter and then one with respect to the input the other thing that's not the parameter.

Okay?
Maybe I'll pause any questions about that.

>> [clears throat] >> Okay, so now that was just for W2.

And you just need to apply this to all the parameters in network and if you put it all together, you will see that for this network, the forward pass is two times the number of data points which is B times the number of parameters FLOPs.

And the backward pass is twice of that which is four times the number of data points times number of parameters FLOPs.

And so, the grand total is 2 + 4 is 6 and 6 times number of data points and parameters.
So, this is where the 6 ND comes from that you might have seen in various places.
It's just like counting forward and you know, backward.

So, we did this for just these deep networks. but it turns out that this is actually a good approximation for transformers as well as soon as long as the context length isn't too large.
If the context length is too large then you get the context length squared and that's more FLOPs that isn't in this kind of accounting.

Okay, so let's talk a bit about So, now we have gradients.
Now we have to the other piece when you do training is we have to do optimization.
So, like here's our deep network.
I'm going to just so that we're I'm not just giving you what's in the assignment one, I'm going to use the Adagrad optimizer which is from this is from 2011.
This is predates Adam.
It's where you can think about it somewhere in between SGD and Adam.

And basically it's SGD where you know, look at the you know, the second moments of the gradient. momentum is where you look at the first moment of gradients and Adam is where you, you know, combine the two of them.
I'm not going to have time to go into the optimizer details here since we're focusing mostly on the usage.
So, we can define an you know, optimizer.
And when you compute the you know, the gradients, and you compute gradients and then when you take an optimizer step, if you're defining your new optimizer, so in the assignment one, you're going to implement Adam.

for each of the parameter you know, groups which are, you know, in this case W1 or W2, we look at there's something called the optimizer state which is of you know, storage that you can the optimizer uses while it's it's it's running. and what we're computing in Adagrad is something is the squared gradients. the sum of the squared of gradients. so, here we're getting it from the optimizer state.
We're updating it with the current gradient and we're storing it back.
And then after you update the G2, then you update the parameters. >> [clears throat] >> Okay?
So, in Adagrad it's you basically divide by the square root of the gradient you know, average gradient squared or sum of gradient squared rather.

Okay, so let's see.
So, I'm actually going to maybe skip the you know, the training loop.
Actually, wait.
Hold on.
I think I actually skipped something I didn't mean to.
Okay, so that was the optimizer state.
And now let's look at how much memory the optimizer is using or in general what is the memory usage.
So, for parameters in this network, there's D squared parameters for each of the L layers, and each parameter takes two bytes if we're storing an FP16, so that's number of parameters.
Activations, this is you know two times the which is for BF16 you know, batch times D times the number of layers for every layer you have an activation.
You have gradients, which is basically a copy of all the you know, the parameters.
And this is all the BF16, and the optimizer state is for every actually this should be the I think this should be sorry, not this I'll fix this as a typo.
This should be the number of parameters, not parameter memory. and this should be number of parameters.
So, number of parameters times two, and then this is four times the number of parameters.

And the reason for this is that it's customary to use FP32 for the optimizer states for stability reasons. obviously people have tried using BF16, and what tends up happening is that now you're taking squares and you're averaging all those steps, and it doesn't you know, it's not very stable. and for AdaGrad it's here we have four bytes per parameter for storing the optimizer state.
Adam, you store the first order bit and the second order moments, so that's eight bytes.

So, if you think about the optimizer state is actually a lot of the memory you know, used. one note is that you know, remember memory serves kind of two purposes.
One is like how much you have to store this thing in your HBM.

But the other thing is that it has to be shipped to the to the [clears throat] to the accelerators, and in general the optimizer state is not really the bottleneck for compute.
So, the amount of memory here is not really so important for you know, performance in terms of speed, but it is means that you can't fit large models in your your memory. >> [clears throat] >> Okay.
So, to put it you know, together as we mentioned, the number of parameters here is D times D times L, and the number of FLOPs is D times number of tokens or number of data points times number of parameters.

So, for transformers, this is going to be a bit more complicated, but you're going to do that in assignment one, and you do it more carefully.

Okay, I'm going to skip the training loop.
This is just a sort of a general kind of review.
Two things I want to quickly touch on before we conclude.
One is that as we see, memory does have an important effect both on the ability to store large models, but also sometimes your speed.
And so, in general you want to reduce your memory usage.
So, there's two things that people you know, typically do.
One is gradient accumulation. so, in general you want to use batch sizes that are large enough to improve stability up to a critical batch size which Tatsu will talk about later. and as we saw that activation memory scales with batch size.
So, you might going to run out of memory at some point if you have too large batch sizes.
So, gradient accumulation says, well you compute, you have these micro batches, and you compute the gradient on the micro batches, and then you just accumulate the gradients.
You don't zero out the gradients, and every batch size over micro batch size steps, you update the parameters and a zero out the gradients.

Okay?
So, this is actually a very simple code change which allows you to save you know, on compute.
So, the other thing I'll quickly mention is activation checkpointing.
So, in training we need to store in general the activations of all the layers.
This is done by you know, default.

actually interestingly for inference, we don't need to compute the gradients, so we only need to store the current layers activations. but for training the memory usage is B times D times two [clears throat] times L.

And the question is can you reduce this?

So, the activation checkpointing also known as gradient checkpointing or rematerialization.
The key idea here is that in the forward pass, you just keep the activations for only a subset of the layers.
And in the backward pass, you recompute the missing activations from the last checkpoint.
That's why it's called checkpointing.
And this is a general trick that you see in you know, systems which is trading off if you want to reduce memory, you can just recompute things. so, here we're going to define this to operationalize this.
This is actually fairly you know, easy actually.
I didn't Okay, maybe I didn't see the Okay, so basically what you do here is you have the same model except for you just add torch.utils.checkpoint on a layer.
So, that means do this computation, but don't store any of the intermediate you know, activations.
Only store what is needed.
So, this means that if you store all the activations, remember in your deep network, you have the thing that's pre the ReLU, and the thing that's after the ReLU.
Right?
So, that's a lot of storage, and instead if you do activation checkpointing on those blocks where each block has a linear and a ReLU, you don't store the pre ReLU, and you save basically half. you can get away with half the memory.

And then at the when you're doing the backward pass, you need G3, but you can compute G3 easily from H2.

Okay, you can go farther than this. you can say well, I can store all the layers, but you know, in extreme case, I can just store no layers. and so, that will be memory maximally memory efficient.
The only thing is that the compute is going to be L squared because every for every one of these layers you have to start from the beginning. and maybe a sweet spot is if you store the checkpoints at square root L layers, that means your activation memory is square root of L, and your your com recomputation overhead is also you know, square root of L.
So, that's that's balanced.

Okay, so to summarize this lecture everything is operating on tensors, parameters, gradients, activations, optimizer states, data.

we introduce this Einops, which hopefully you guys can embrace as a way to think about tensor operations. six times number of data points times number of parameters is a formula which now we have demystified as number of FLOPs per for training step.
And actually if this is training step, this should be batch size. and then we talked about arithmetic intensity and roofline analysis, which allows us to diagnose whether a computation is memory bound or compute bound. matrix multiplications are compute bound.
Basically everything else is memory bound. and then finally, gradient accumulation activation checkpointing are ways to reduce the memory and by reducing the memory, that allows you to use bigger batch sizes.
Okay, so that's it for today's lecture.
So, next week Tatsu will talk about architectures.

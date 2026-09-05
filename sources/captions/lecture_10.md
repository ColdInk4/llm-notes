# Lecture 10 字幕

All right, let's get started.
So last time Tatsu talked about scaling laws and we're going to take a little bit break from that and talk about an inference. so the problem of inference is very simple. you've trained a model and you're given a prompt and you want to produce the response usually as accurately and as quickly as you can.
So inference turns out to be only one lecture but it's actually of growing importance and it shows up in many places.
So clearly once you've trained a model, you don't just sit there and if you're a researcher, you put a plot in your paper of here's how my model works.
But for everyone else, you actually want to use this model.
So what does use look like?
It could be chatting with an AI assistant or chatbot.
It could be doing code completion. these days agents are very popular and that requires inference. batch data processing as well. it's also useful for evaluation for evaluation that requires generation in particular and it's also used inside training if you're doing reinforcement learning you need to have a model you generate rollouts and you score them and you update the weights appropriately.
So inference shows up all over the a place and you know efficiency as the is the you know theme of this class really matters more than ever here and even if you just look at the actual use kind of app vantage point you know training is a onetime cost.
It could be very expensive.
It is very expensive but once you're done with it that's it.
But inference is a repeated cost. you kind of incur them every single day. so you know OpenAI is estimated to produce 8.6 trillion tokens a day. and just for reference DeepSeek v4 which came out earlier this year was trained on 32 trillion tokens.
So in less than four days the number of tokens that OpenAI has to produce and therefore the com compute is you know at least DeepSeek v4 and of course you know even the frontier models might be trained on more tokens but still inference the number of tokens generated is going to be higher.
So that's just a you know some perspective and moreover I think this the importance of inference has grown in the last year because you know once upon a time we thought of language models as primarily as chat bots or assistants where you put in a prompt you get back a response and presumably the goal of the human is you read the response.
But now as we move more into a gentic world, what it looks like is that a query goes in and the agent is going to do a bunch of stuff.
It's going to think, it's going to reason, it's going to call some tools, it's going to introspect and at the end of the day, it will produce some output for a human to read.
So the most of the tokens that the agent produces is actually not for reading.

And so you should really think about the number of tokens generated as really the compute spent.
And there's no sort of limit.
If you have an ambitious enough problem, you're going to need much much compute and a lot of tokens, right?
So if you were in the chat world, maybe inference you after a certain point it's fast enough because humans can only read so fast.
But for an agent, you know, there's no kind of limit to how much value you can get out of squeezing more out of inference.
So there's a lot of people doing inference. on the commercial side, of course all the closed API providers have to serve their models.
So inference is a big deal for them.
And there is also a bunch of providers serving open weight models and providing inference as well.
And in the open source community, there's a bunch of packages. so vLLM is probably a popular one, kind of a go-to.
SGLang is another one that's u particularly good for agentic workloads, but maybe not as popular yet.
And then there's TensorRT-LLM from Nvidia, which is really, you know, fast, but it's more narrow. and then llama.cpp if you want to run inference on your on CPU this is a popular you know package.
Okay so hopefully I've argued that inference is huge. if you can make inference you know twice as fast or even 10% faster that is a big deal.
So now the question is what does fast mean?
So there's a few metrics that capture the notion of fast and they're going to be applicable and in different settings and have different trade-offs.
So one metric of fast is time to first token or TTFT.
This is essentially how long a user waits before any generation happens.
So you put in a query into TGBT and some number of you know milliseconds passes by and then and then from the point the first token comes in that is that is the time and this is generally useful for interactive applications because that latency where you're just waiting and doing nothing is the longer that is the worse the user experience and as soon as tokens come in it doesn't maybe have to be that fast because if you're going to read it anyway, you can't read that fast.
So the second metric is latency. and this is from the standpoint of an individual user.
How fast are tokens appearing for one query?
This is also important for interactive you know applications. so basically this is how fast the tokens are kind of streaming through.
And then a second related concept is throughput measuring tokens per second.

This is how fast tokens are appearing for many queries.
So latency or one over latency and throughput are clearly very related, right?
It's it's very kind of aligned of the you know in general many interventions will make latency and throughput you know better.
But as we'll see later there's actually a trade-off here.
And throughput is useful for if you're doing a bunch of batch processing, right?
You just want you have, you know, a pabyte of data and you want to process it with a language model. you just want that job to get done.
It doesn't matter if one query is coming in faster, sooner or later.

Okay.
So, what determines the efficiency by either any of these metrics of inference? so the high level bit is as follows and this is kind of a you know maybe one high level bit to remember from the lecture is that when you're doing training you see all the tokens at once because in supervised fine-tuning you see all the tokens and you can parallelize over the sequence remember think about in the transformer the calculation for attention MLP the sequence is just a dimension so it's a big tensor that gets multiplied And so you essentially process all the tokens at once.

In inference, you can't do that.
You have to generate because of the auto reggressive nature of inference token sequentially one at a time.
So this is a fundamental problem of why inference is a very different workload than training because you can't paralyze across the sequence dimension.
And therefore, as we'll see later, it's going to be harder to have high arithmetic intensity or fully utilize the compute.
Okay.

All right.
So, let's let's go on.
So, first I'm going to just maybe of a preview of what's in this lecture. first, I'm going to do some math to understand the arithmetic intensity and throughput and latency. how to think about that for a transformer. and then I'm going to talk about various techniques to reduce the cost by you know making the KV cache smaller, quantizing model pruning and then finally I'll talk about speculative decoding and then some kind of practical concerns.
All right, so this is going to be a bit of a review mostly kind of swapping in make sure we're on the same page with respect to notation. much of this lecture is based on the scaling book from Google on you know transformers and inference.
So I do recommend that people go check it out.
It's very nicely written and some of the picture the figures are you know generously lifted from that book. okay so just to establish a bit of notation to understand this diagram. so we're going to use symbols think about in ops to denote dimensions but also their kind of length.
So B is going to be the batch dimension and also the number of sequences.
T is going to be the sequence dimension which is also the number of tokens.
D is the model dimension.
H is the head dimension.
Okay.
So if I'm going to write this is taking a product between a tensor with dimensions B, T and D. and another matrix of D and H and I produce BT th.
So what is going on here is that there are some dimensions marked red which are going to be contracting.

These dimensions appear in both operands and disappear from the results and then there's going to be other regular dimensions in black that appear only in one operand and stay in the result.
Okay, so this is kind of the general form. another case is if you have blue dimensions that means they're they're batching dimensions and they appear in both but stay in the result they do not get contracted or reduced.
Okay.
So the interpretation here as we'll see later is that we have this tensor for every sequence and token position you have a vector and you want to multiply that by a you know a matrix. and then here you're basically taking a bunch of dotproducts between two you know tensors.
Okay.
So with that in mind this in all its full glory is the description of a transformer block.

so it looks a little bit like a circuit diagram.
There's a lot of details here but I actually find this probably the most kind of crisp definition of what a transformer is. you know you see a lot of description of transformer and frankly a lot of them are kind of you know it's hard to understand what is exactly happening this one basically tells you exactly the shapes of the tensors and allows you to reason about the dependency so for a transformer block you take the x which is the activations of one layer you feed it through attention and then you feed it through mlp and just as a kind of a quick refresher the x you projected using a query matrix a key matrix and a value matrix. you'll see here that we have n which is the number of heads and h is a dimension of each head.

So the query matrix is batch time sequence time number of heads times the head dimension. and k and v instead of having n we have k which we'll see later. which we have a potentially smaller number of key value heads. and then this is the tension operation where notice that there's batching dimensions so B appears in both and we're contracting over the head you know dimension and we'll come back to why this B being in both makes inference hard why attention is kind of a bottleneck there.
Okay, so just kind of hold on to that thought. and the MLP is kind of very you know simple and straightforward. you just do some you know map malls.
This is the gating you know matrix.

This is the up projection and the down projection.
So I think you guys have implemented this so I don't want to belabor it but just to swap in the notation. by convention we're going to assume that in the MLP the MLP up projects the D-dimensional model dimension into four times that.
So f is always you know think about as 4D whenever you see f and then the model dimension gets split across the n heads.
So the model dimension is always equal to number of heads times the head dimension. and then the number of heads gets split in the case of group query attention into the number of groups and the number of heads per group. and finally we have two variables S and T which both represents the kind of the sequence dimension and the difference is that S is going to be represent the number of input tokens to pro and T is going to represent the number of output tokens.

So in training time these two are the same because we're predicting all the out the same inputs as outputs. at inference time t is going to be one and s is going to be the input.
Okay, so another review for arithmetic intensity.
So I talked about this in the second lecture.
So just as a warm-up, suppose I'm multiplying two matrices, a B by D matrix and a D by F matrix.
So intuition B is the batch dimension, D is a hidden dimension or the model dimension and F is a projection dimension in the MP. so remember what is arithmetic intensity?
I have to count FLOPs and I have to count the amount of memory moved.
Okay, so we can we can do this as follows.
So what do you have to do to multiply these matrices? now remember the systems you know lectures you have to read x from hpm and if you're storing everything in bf16 which is the case for inference we're going to have 2 * b * d you're going to read w that's going to be another 2 * d * f you're going to do the mold so that's number of FLOPs is 2 * b * d * f and then you're going to write it back.
Okay, so every time you read and write it's basically the basically number of entries.
It's a kind of a quadratic like term and mammals are cubic and remember this is going to be important because that's how you're going to get high arithmetic intensity.

So number of total number of FLOPs is you know the cubic and then the bytes transferred is the sum of the reads and writes.
Okay.
So remember arithmetic intensity is how much compute we do per bite transferred and we want this to be high.

So the intensity is going to be equal to you know FLOPs over bytes transferred.
And I'm representing all these things symbolically because it will be easier to see and work with rather than numbers. one kind of just you know simplification we can make is that suppose that the batch dimension here is much less than d and f then we can simplify this.
So in particular what this is doing is saying d= c * b and f= c * b and let c go to infinity.
So both D and F are really large and B is smaller and this just reduces to intensity of B.

Okay.
So the upshot of that is that remember when we did arithmetic intensity for you know matt mole it's was something like n over n over three and this is the kind of the analog where instead of just having you know one variable for dimin having square matrices we have non-square matrices.
Okay.
So just to compare so the accelerator intensity of hardware is you look at the FLOPs per second in the in the spec sheet.
You look at how much memory bandwidth how fast memory gets moved between HPM.
You divide and that is the accelerated intensity.

And now you compare your computational computations intensity with your accelerator intensity.
And if it's greater then you're compute bound which is good.
If you're less then you're memory bound and that's bad.
So in this case for H100 for this particular map operation you're compute bound if the batch size is greater than 295.

So here's an extreme case.
So suppose you just have one example.
Okay. what happens? you have one example, then your arithmetic intensity is going to be one and this is going to be memory bound. and you can kind of see what's going on here.
You're reading this DF matrix.

but B is only one.
So you're effectively doing only two * D * F FLOPs. so you know that's and this is basically kind of the workload that you'll see in inference.
You don't get these like full matrices. you get these like very thin matrices or tensors.
Okay, let me stop there in case there's any questions.
Yeah, in the transformer I just want to confirm sorry across across.

Yes.
Can I search cross?
Across across across across you can't just cross. [clears throat] Yeah.
Should that be across K moves instead of G?
I'm having trouble with the connection. [snorts] >> let's see.
So, the number of groups here is wait, I'm just making sure.
So, K is the number of key value heads and so you have Q sorry G is that G per KV head. so >> my understanding is like each key had >> yeah correspond if we break down the n into a g then that means g is the query i that means we have ks that's my >> okay so I think yeah I think you're right so k should be the number of groups and g should be the number of the you know heads per one of those groups. >> Yeah.
Okay.
Thanks for that.
I'll fix that later.
Okay.

so let's talk a little bit more about the arithmetic intensity of inference.
Okay.
So here's how to think about what's happening in inference.
Let's say you just do it naively.
So here is a prompt never going to give you going to the transformer and the transformer is u does you know generates keys and values and activations and at the end of the day it generates logits over the output vocabulary and you sample a token and that token gets concatenated with the prompt and then you just do this again. you generate never, you attach that to the prompt and so on and so forth.
Okay, so this is the most naive thing.
If you have a black box that takes a sequence and outputs a distribution of a tokens, which is exactly what transform does, you can just apply this repeatedly.
Okay, so this is works but this is really bad because each time you generate one token you actually take order t² time where t is the number of you know tokens that you've generated so far.
So generating t tokens is actually key cubed.
Okay, because the attention is already t squared and you have to do that you know one for every once for every token.
So this is you know pretty bad but the observation is that you don't actually have to do this. a lot of the work can actually be shared across prefixes.
So for example, if you're over generating never versus you're generating up, you're actually computing a lot of the same key values for you know never going to give.
So these tokens they shouldn't change.
Okay.
And this is because it's a causal you know transformer.
If it's birectional then if you attach a token then everything changes.
But if it's causal then the activations here don't change based on any tokens you append.

So with that observation, the first obvious thing is to store a KV cache in HPM so that between successive token generations, you can just reuse the KV cache.
So that means when you are trying to you know generate GANA, you don't actually have to compute all the keys and activations of these previous previous tokens.
So this is what it looks like with a KV cache.
Okay.
So there's going to be two stages prefill.
So you get the prompt and you populate the KV cache which is the set of key and value pairs that the transformer computes. and then you generate the logic just the same computations we did before.
And now now you have the KV cache here and then this distribution which you sample a token.
And now you feed that through the transformer which you know uses this cache and then produces a both the distribution of the next tokens but a new KV which is the basically activations corresponding to the token up and then this gets fed in.
So now you have the commmented KV cache, you have this logits, you sample and then that feeds to the transformer.
You output the distribution of next tokens as well as the next the new token which you add to the KV cache and so on so forth.

Okay.
So the KV cache formally is for every you know sequence so there's B of them for every token there's s of them and for every layer in every head you store each dimensional vector.
Okay.
And just to reiterate for inference, there's prefill.
You're given a prompt.
You encode the prompt into these you know vectors.
And this is parallelizable just like in training because you see the entire prompt, you can compute the KV cache and then in generation you generate the new response tokens, you know, sequentially.
But at least you don't have to pay for generating the KV cache of the tokens that you already looked at.
Okay.
So, is everyone comfortable with a KV cache?
Any questions about this?

All right.
So, let's try to figure out the FLOPs and memory IO for both MLP and attention layers. so remember S is the number of tokens we're kind of conditioning on and T is the number of tokens we're generating. so we're going to do this sort of abstractly and then later we'll specify to the prefill where T equals S and generation T equals 1.
So for the MLP layers and through all of this I'm just going to look at the mammals because the mammals are the thing that actually require a lot of work. everything else can be is not that many FLOPs and can be potentially fused into the mammal too. okay, so let's do this calculation.
So it's the same as the matrix calculations.
I'm not going to maybe be labor step through every single detail here, but just the form is you read X from high memory. you read all the parameters, you compute the up projection and then you know you write it to HPM, you compute the gate and you write it HPM and then you compute the basically the down projection of that and you write it to HPM.

Okay, so the number of FLOPs is you know depends on P B and T and DNF.
So batch size, sequence length, model dimension and the fe forward dimension MLP dimension and the bytes transferred is you know this expression.

Okay.
So now you can compute the arithmetic intensity which again is FLOPs divided bytes you know transferred. so this is some expression. we're going to assume that just like before B * T is much smaller than DN DNF. and with that then we see that intensity is you know B * T.
Okay.
So this is analogous to just the ML case.
There's more matrices and there's like more dimensions but fundamentally it's kind of the same thing and it makes sense.
MLP is basically a big mammal.
Okay.
And it also makes sense because the batch dimension and the sequence to length dimension are sort of everything is independent right in for the MLP they don't interact.

Attention's different story and so you know B * T so as long as you have you know in the prefill if you make B * T large enough large batches long sequences you'll be fine.
Now let's look at generation.
So remember generation there's two problems here.
One is that gen generation t equals one.
Okay, you're only generating one token at a time.
And so that means your arithmetic intensity is going to be B.

But B what in generation is the number of concurrent requests.
And so if you're in a sort of batch setting, you kind of can control that.

But if you're let's say serving a chatbot, it's the number of concurrent requests is essentially you know how many users concurrent users there are which can be high can be low.
So it's a bit unpredictable it can be changing over time.
So that's something we're going to address when we talk about you know continuous batching.
But overall, this is not bad, right?
Because as long as there you have large batches, the sequence length isn't going to really help you, but you have large batches, you should be good.
So now let's look at the attention layer. so again, S is in the previous tokens already generated.
T is the number of tokens you want to generate logits for. so in attention you read the QV you know matrices from HPM you compute the tension you know compute the softmax which doesn't really matter and compute the value you know matrix and then you write the result to HPM okay so the FLOPs is B * S* T * and the bytes transferred is this expression.

So this should you know all the everything is a mammal right so this is should be always you know one degree higher of a polinomial than the bytes transferred because it's a mammal.
The only question is you know what is that factor look like?
So that factor for attention is s * t over s plus t.

Okay.
So, so let's look at the prefill.
So, prefill what this is saying is that the pre-fill intensity is s over two when t equals s.
So, this is good, right?
Because as long as you have long sequence lengths for attention, you're going to maintain high arithmetic intensity. notice that the batching dimension doesn't happen here.
I'll explain a bit you know why later but for generation this is bad news right so the generatedation intensity is s over s + one and that's less than one or even let's just call it one and arithmetic intensity remember one is bad we want it to be something like 295 for h 100 to be to saturate the compute and so you know this is really the bottleneck.

So we did all this analysis for you know MLP attention you know pre-fill generation and we've found this to be a bottleneck.
So let's try to contemplate why this is a bottleneck. so unlike MLPS so MLPS for generation is actually okay as long as your sequence is sorry is your you know batches large enough.
So the problem with this ML you know attention is that you know let's look at MLPS every sequence hits the same MLP weights.

Okay, so these don't depend on B. whereas in attention layer each sequence has its own KV cache.
So these all depend on you know P.

and so you can think about this as what's what's going on here is that you know in the MLP case having B being big actually is helpful because you kind of get to load these MLP weights you know kind of once and you can kind of use it for all your sequences right that's how you get high arithmetic intensity because you use a simplifying a bit you load it once you do all your batch processing and then that is that is good whereas in attention you can't you know all these depend on B so increasing B doesn't help it's like every for every sequence you're basically doing a matall so they're all in independent so you know doing more matt malls isn't helpful just like remember in the very beginning I showed you this example, this also has pretty bad arithmetic intensity.
It's not not a mammal because we're essentially batching by a coordinate.
And this is essentially basically the same as doing a dot product, which has horrible arithmetic intensity.
And remember this is in the attention.
This B blue B is like the cause of why the tension is arithmetic intensity doesn't scale with B and why that is a bottleneck.
Okay to just summarize here so prefill is compute bound generation is man memory bound.
So if you look at the MLP intensity for the for prefill it's B * S great prefill for in for attention intensity S over2 not as good but workable generation MLP intensity also workable requires long concurrent requests but it is really the generation intens attention intensity which is a fundamental bottleneck and you just that's it.
You just have to if you're sticking with a transformer, you can't really, you know, improve this.
Okay. pause there for questions.

So now whenever you hear people say, oh, inference is memory bound, you know why.

Okay.
So let's now use these intuitions and calculations to think about our you know inference metrics throughput and latency and also TTFT.

All right.
So inference is memory bound. the now the main I mean in some ways this simplifies a lot of things because when we think about how long things take you just look at how much memory needs to be transferred okay because assuming you overlap communication and computation the bottleneck is going to be just the amount of memory that you have to deal with.
Okay.
So in some ways it's nice because it's simple but in other ways it's frustrating that your accelerators are sitting there not doing anything. okay so let's walk through example.
So for Llama 2 13B on a H100.
So what is the latency and the throughput?
Okay so remember llama 13b has a particular shape.
So oops this is a sequence length the model dimension fe forward dimension number of query number of key value heads there's no GQA here so the nn equals k head dimension number of layers vocap size and the memory bandwidth for h100 okay so now using this config let's compute the to transformer performance you know statistics okay so these are inputs here and just what the statistics I'm going to compute are going to be number of parameters the memory usage latency and throughput okay so first of all you know what's what does takes memory Okay, so the parameters take memory.
So we compute the number of parameters. and so you have look at the embeddings the you know the MLP layers that the projections of the KQV at the end of the day you get some number of parameters. assuming that you know we're doing a bf16 which always case for inference then parameters take up this many bytes also in memory is the KV cache and the KV cache is the number of tokens in your sequence times the number of your heads and the KV heads times the head dimension times the number of layers ers you have one for the key and one for the value and then you have two multiple two for bf16 okay so that's the size of the KV cache and the total memory usage is this is for each sequence you have B sequences so that's B times that plus the parameter size and that's the amount of memory you need and now what's the latency the latency is determined by the memory a IO.
Okay. because inference is memory bound most assuming you overlap communication and compute all the memory is it's it's going to be on you know shuffling parameters you know back and forth between HPM and SRAMM.

So that's how long it takes to you know move memory around.
Throughput is the inverse of latency but we're generating B tokens in parallel.
So this is you know this is tokens per second.
This is seconds per token.
This is tokens per second as well as having an additional B because you're processing a batch of B.
Okay.
So now let's compute for this config what are these actual values.

Okay.
So num parameters is you know 13 billion.
So good sanity check that is a advertised as a 13 billion parameter model. memory is this term which is about 838 I guess million times B.
So it's basically you know a linear function of B plus some other term.
So this is a KV cache which grows as B.
This is the parameters which is double the number of params.
Now latency is just this scaled by the memory bandwidth.
So it has the same form as the memory. and throughput is B over that.
Okay.

So notice that latency as you increase B grows because the KV cache grows and in order to process stuff you have to copy the KV cache back and forth.
And now throughput is more interesting because as you increase B you know throughput does improve but up to a limit right because throughput in improves because now you're advertising the cost over a larger batch but also you know the speed at which you're processing this obviously is also increasing over time.
So this sort of asmmptotes it's not going to you know through can't possibly go to infinity.

Okay any questions about this so far?
So basically you just have to remember latency is a linear function in B.
This is the where the constant is the size of KV cache and then this is the number of parameters and then throughput is you know some B proportional to B over B plus something.

Okay.
So now let's instantiate this for a bunch of situations.
Okay.
So if you have batch size one so this is what you would get. you get a latency of 0.08 you know seconds per per token and the throughput is 124 tokens per second.

So if you what happens if you increase the batch size you'll see that the latency goes up but the throughput also goes up.
So latency gets worse but the throughput improves.
So this is kind of an interesting thing because we think about oh we just want to make it go fast but fast actually has two meanings here which actually depending on which one you care about is you know complete opposite.
If you want to tune your batch size that's going to really determine you if you want latency or throughput and what happens if you increase the let's increase we're in a we really want high throughput because we're processing lots of documents.
Let's increase the batch size even more.
Okay.
So the latency gets even worse.
Not not I mean yeah it gets worse.
Throughput gets even better.
But the main problem here is that your memory you run out of memory.
Okay.
Because the memory for storing this KV cache is ex going to exceed your H100 memory.
And if you have B200s you can increase the batch size more but eventually you hit some sort of limit.
So there's kind of limitations to how much you can in through your throughput.
You'll never get to sort of the asmtote because you'll hit the memory.
Okay.
And also your your throughput you know is getting your gains are kind of diminishing as well.

So increasing batch size worsens the latency because now you have a larger KV cache to read and write. and remember it's batched.
So if you're an individual query you have to wait for everyone to finish.
So you have to basically you get you're like waiting for a bus and the latency is you know pretty high.
You wait and then but and then you go whereas the throughput of a bus is pretty good because you can move everyone at once. so the throughput improves as batch size increases because remember the parameters are shared and you load that you know once into memory and you can process a lot of sequences.
Okay so trade-off between latency and throughput just to make sure everyone's aligned.
Smaller batch sizes use better latency but worse throughput.
Large batch sizes yield better throughput but worse latency.
Okay.
I'm not going to really talk about parallelism too much.
There's also another dimension of inference which is you can you know shard your your model across multiple you know devices. you can look at the scaling book chapter on inference if you want to know more. just as a kind of a very trivial example, if you launch five m copies of the model, the latency is the same and the throughput increases by m. and then the other metric we didn't talk about is time to first token.
And this is essentially the time it takes to do prefill because after you finish prefill then you can basically start generating.
Okay.
So if you want you know to fast to if you want you know faster TTFT you should use smaller batch sizes and you want larger batch sizes to improve throughput.
Okay so hopefully that is clear.
Any questions about throughput and latency and how they're in tension with each other?
Okay.
So now now we have a conceptual framework for thinking about how efficient inference is in terms of arithmetic intensity and you know throughput and latency.
Now let's try to make it faster.
How do we make inference faster?
There's a bunch of different techniques which are quite varied ranging from changing the model architecture to doing systems optimization everything in between.
So inference in some sense is a fairly rich crosscutting topic.
Okay.
So the most kind of d maybe now in hindsight obvious thing you can think about is hopefully beat it into your head that you know memory is the bottleneck for inference and the KV cache takes up a lot of memory and it could even be larger than the number of you know parameters if for a large enough batch size.
So let's just try to reduce the size of a KV cache.
Now you have to be careful about how you do this because you want to make sure you don't lose too much accuracy in the process.
So here is one thing you can do which we already talked about which is grouped query attention and just as a kind of a reminder here.
So multi-headed attention basically for every token you have a key and a value and a query.
And if you do grouped query attention, you basically compute the same number of queries, but you have only a smaller number of groups and you compute key and value for each group. okay.
So K is a number of groups here. and so in the multi-headed attention K is N.
No reduction. there's something called multi-query attention which no one uses because it's really bad.
K equals 1 and somewhere in between is hopefully where we'll find a balance between accuracy and speed.
So there's this is the paper that introduces GQA from 2023 and they show that if you look at you know time per sample which is related to both latency and throughput you know you see that the MHA so multi-head attention full attention is has this high time whereas if you start k equals 1 it's it's much faster And then you can actually keep on increasing the K to like you know K equals 8 and it's still pretty good and eventually your your time goes up quite a bit.

Okay.
So you know why does GQA improve latency and throughput?
Well, it reduces the KV cache by a factor of N over K. and you know and re just as a friendly reminder re reducing memory usage leads to speed up because we're memory bound.

Okay.
So let's revisit our friendly llama model here. remember in the initial configuration we're just using k equals n.
So there's no multi-headed there's multi-headed attention.
No reduction in number of key values. and so for this one remember the using a batch size of 64 we get this throughput and latency.
Now if you do GQA we're going to put in let's say a kind of a sparity of one to five.

that reduces the you know the memory quite a bit which in turn improves the latency and also improves the throughput.

So it's not that latency and throughput are always at odds.
If you reduce the amount of memory then it improves both.
It's mainly the batch dimension that allows that is the point of tension.
Okay.
So this is this is great. and let's actually just imp increase the batch size even more and see what happens.
So before if we had a batch size of 256 we ran out of memory but now it fits in memory and we see that you know the latency you know suffers a bit because we're increasing the batch size but the throughput you know goes up proportionally.

Okay.
So sometimes you kind of play with these parameters jointly like you can reduce the KV cache but that allows you to increase the batch size and allow make making other trade-offs.
So the final thing you have to do whenever you do some you know lossy change is that you make sure the accuracy doesn't drop and this paper shows that for GQ GQA the you know the time is better but the across a bunch of evalu basically it works well.
So now with these accuracy evalu I think you always have to take it with a grain of salt because this is like for particular model later the deepseek paper was to show that it actually does hurt.
So you know I guess take everything that's not just kind of math with a grain of salt here.
Okay so speaking of DeepSeek here's another idea to reduce the KV cache.

Okay, so the theme is reduce the KV cache and latency and throughput improved. so this is multi-headed attention same number of queries and keys and values for every token and GQA remember we have reduced the number of keys and values for every token sorry not for every token we basically have reduced the number of values and keys and now the multi- latent attention for DeepSeek says we're actually going to leave the number of keys and values this the same one for every you know token essentially but I'm going to parameter I'm going to compress these so normally you have some how do you compute your keys and values you have your your activations and you multiply them by some matrix to get k and some other matrix to get v and these are generally n* h you know dimensions like your model dim which is which is big.
So MLA says I'm going to actually project this these activations down into C dimensions.
So DeepSeek-V2 reduced it from you know 16,000 to 512.
So this is quite aggressive compression here.
And then I'm going to compute the K and the V from this compressed representation.
So now I can just store C.
This is much smaller.
And then when I need my keys and values, I can just materialize them.

So there is one wrinkle here which is that MLA is not compatible with rope which operates directly on the keys and values. so what they do is add additional dimensions for you know for handling the rope. but more or less it's it's still a pretty big you know reduction.
So the latency and throughput improvements follow by just you know simple math.
The smaller the KV cache the you know the faster you go okay it's almost kind of linear scaling up until some point and then remember you need to check whether your model is you know accurate.
So first of all this is sort of the result that you know kind of contradicts the or is intention with the GQA paper. they show that GQA actually isn't that you know great it you know it's you know it's much so this is MHA this is GKA these numbers are smaller than these numbers but they show that their method MLA multi- latent attention works even a little bit better than MHA but let's just say it's about the about the same so This column and this column are much better than I guess there's no GQA on this table.
It's yeah compare over here.

Okay.
So I'll show you an yeah >> how does this compare to reducing the number of dimensions the model?

>> So question is how does this compare with reducing the dimension of the model? so that's a good question.
And I don't these oblations don't show that.

U my guess is that reducing the model dimension just makes things you know worse if you because you're sort of like indiscriminately just like reducing everything.
I think the trick in all of this kind of is to find places of the model where you can squeeze. and this aperary I don't think you can necessarily know for sure.
You just have to do a bunch of experimentation and see what works.
So here's another idea to reduce your KV cache.
This is called cross layer attention. so the idea is that normally every layer has KV K's and V's.
But you know, let's say instead of doing that, we're just going to compute KVS for a subset of the layers and then just for the this layer, I'm just going to use the previous layers KV cache.
Okay, so this is kind of another way of sharing.
Just like GQA shares KVs across heads, now I'm sharing KVS across layers. and empirically this paper shows that you know doing this improves a paro you know frontier.
So each of these models is better than you know in given a method you can always kind of sweep this the size of the KV cache by changing the K and the head dimension. which I guess kind of relates to your point about changing model dimension and then but if you do this CLA cross layer attention it's it's better okay so let me give you moving on whirlwind tour of different techniques for reducing KV cache there's local or sliding window attention this is a fairly kind of old idea and a very kind of natural idea Okay, so if you look at the full attention matrix is n squ and instead of doing that if you're going to generate a token you just look at the last k tokens.
Okay, so you essentially have a sliding window for every token you generate.
You just depend on the last k. and now if you do that the effective so the kvcast is now independent of the sequence length.
It's just you know the number of the batch times the other variables which is great and this is especially great for long context. now you because of number of layers the effective context length as actually larger than the number of the stated context length because information can propagate you know farther in if you go down the layers. now you can do fancier things like you can maybe not do a dense selection of layers but you can space it out.
You can also do this global plus sliding window where you [snorts] have attention to a fixed you know grid of different you know token points plus a local sliding window.
So you can do various things.
Now the problem with this is that it actually still hurts accuracy.
So this reduces expressivity. there's no you know free lunch here. or at least this was an expensive lunch. so the solution that people come up with is that they interle local attention with global attention.
So these hybrid models have you know full attention for some of the layers and some local layers for you know some of the other layers and you're basically always trying to and then so that allows you to essentially you know reduce the KV cache a little bit and you're always trying to balance you know accuracy with speed.

>> Yeah. the trade-off between like a linear attention variant versus like a sliding window. >> Yeah.
So question is what about linear attention versus sliding window attention?
So I'm not going to talk about linear attention but very quickly there's a bunch of methods that where instead of you know storing a KV cache is you basically compute some sort of like compressed representation of all the history.
So linear the most naive linear attention is you just sum the KV values up into a single vector.
So that's definitely independent of the sequence length.
You can do fancier things.
There's like gayet, delta nets and mamba which allows you to try to compress but not forget you know as much. now the question is you know how do those compare?
Those have also been used in place of sliding window attention and people have gotten good results with them.
You can also use a combination of full attention, sliding window intention, and the linear attention because they sort of capture different, you know, aspects.

If you care about kind of local kind of high resolution stuff, then sliding intentions better.
If you just kind of want like broad summaries of the past, then the other, you know, linear attention might be better.
So then for like a long context sentence would you say linear attention would be a better setting for that? >> so the question is for long context would linear attention be better. let me talk. so there's no free lunch, right?
Like I think let's say you have a very long context and you're sort of solving a needle in a hay stack problem, right?
If you have to compress your entire history into like a small context, you're just going to lose information and you might just, >> you know, not be able to retrieve it.

because I guess I feel like what it seems like people are going with hybrid art tickers that you'll always need some sort of longer attention.
But I'm I'm just trying to understand like in my head like what the trade-off between using a sliding window versus like some sort of mamba delta net layer would be that like is using a delta layer like better than using sliding window consistently or like what is the actual like maybe representative tradeoffs?
Yeah, I guess maybe I'll say that the mamba and delta net are more powerful than the sliding window attention.
Maybe you can think about the mamba as it'll probably can like it certainly can represent some of the aspects of sliding window attention because like you can just as you're doing the recurrence it can just look at the last you know state.
So yeah maybe yeah maybe you can say think about the linear attention or it's like extensions as being you know you know better at least there they have more room like once you do sliding into attention like you're done there's nothing else you can >> yeah okay so let me just quickly go through this just highlight this you know deepseek deepseek continues to you know innovate different types of attention mechanism S so remember they came up with the multilaten attention which compresses the key values. there is this thing called now they have compressed sparse attention DeepSeek Sparse Attention and heavily compressed attention.
I never kind of remember all these acronyms and what they mean but let's look at this diagram.
So normally you have your KV tokens and your your query token.
So compressed attention is going to basically compress every M tokens into you know one token and then there is this thing called DeepSeek Sparse Attention which basically selects a subset of those to kind of keep and the way you select a subset is that you actually compute some light lighter weight queries and keys and then you do like a you know a smaller attention to get these index scores. so you know what to keep.
So like a kind of a lightning fast way to figure out what tokens you need to keep and then you use those. and then there's a more compression that you know happens. so in the interest of time I'll move on. the goal of this section is to reduce the KV cache. because KV cache is related to memory and we saw that inference is memory bound.
So that directly translates to improvements in throughput and latency and the key is to do this without hurting accuracy.
So you can do lower dimensional KV caches across layers by across heads and across like you know head dimension.
You can do local attention, you can do linear attention which was discussed earlier. and there's much more and also there's also diffusion models which is a non auto reggressive way to generate which can be much faster.
Okay.
So let's talk about a few other ideas which are important.
So quantization is more of a I would say less of an architectural and much more of a systems u perspective on how to make things you know smaller.

So the key idea here is just reduce the precision of numbers.
Less memory means higher latency and throughput and the obviously you have to worry about accuracy.
So quantization is you know there's many options here all the way from FP16 all the way down to in four. and so one thing you can do if you're kind of scared that quantization is going to mess you up is that you train a model with quantization in mind.
So this is called quantization aware training. and during the fe forward pass during training you quantize and dequantize and you're basically simulating these quantization errors as you train. so then you generally now the weights are adapted towards the quantization things will work better but the con is that requires expensive large scale training and typically what so what typically people do is they train models and then they quantize after the fact.
So this is post-training quantization.
It's much you know cheaper often. you know there's a naive way to do it is that you basically for every layer or tensor you basically determine the scale and the zero point for each let's say tensor and then you quantize that you know separately. this is generally you know doesn't work as well. you can use this idea called GPTQ which uses some hashing information to you know quantize layer by layer and then the you keep track of the errors which get propagated into the non-quantized weights and so it kind of allows you to correct for the errors.
And now activation aware quantizing is you know more s kind of a sophisticated way where the observation is that some activation channels are large and those the weights that interact with those matter more.
So let's allocate more precision to these weights.
So let's look at this picture. so normally if you take FP16 weight matrix and you quantize it you get let's say you quantizing down to N3 but what you're going to do is you figure out which of these you know activ so these are maybe activation channels and some of these activation channels are large so as if they're large in general then you basically allocate like let's say FP16 for this channel you keep everything else as in three and for a few important channels you use higher precision.
Okay.
So another idea here is to do model pruning.
So here you just take a large model and you rip out pieces of it and you fix it up.
It's kind of a very crude way but it turns out to work.
So there's this paper from Nvidia where essentially let me actually let's see so you kind of have first have to estimate the importance of the different parts of the model and you know choose the most important parts and then you basically remove different you know hidden units and different even you know layers And now now we have a model.
It's not going to be very good. and so what you do is you post train it.
You train it some more on the data or the tasks that you care about to kind of heal it.
Okay.
So this is in some sense a training p way to you know reduce the KV cache but where you sort of initialize it with parts of a of a good model.

Okay.
So this seems to work pretty well.
So they were able to take a 15B model and reduce it to a 8B model and it doesn't really hurt accuracy by too much and the amount that you use to kind of train the model or through this process is much less.

Okay.
So just to summarize here you know the game is to reduce the inference complexity without hurting accuracy.
You can think about this is mostly reducing either number of parameters or KV cache. you can define a faster model architecture and train it or you can define a faster model architecture, initialize the weights from original model which might have a different architecture but you kind of just make this Frankenstein thing and then you repair the faster model with distillation.
Okay.
Yeah. >> Can you say more about how you distinguish the important layers from the unimportant layers?

so how do you distinguish the important layers from the unimportant layers?
So you know in general you have you know a calibration you know set and you pass the inputs through the model and you're basically looking at you know the magnitude of the activations and the ones that are you know some of them especially if they're dead units will be kind of close to zero and the ones that are large you want to keep.
So that's a high level idea >> I guess like why does it matter if the activation is like high it could just be like what if it's just always high for instance >> like is that >> so in g this okay so the question is why what if all the activations are high so in general this is a kind of an empirical observation that some of the channels will be much higher than others if this weren't true then these techniques wouldn't necessarily work but it happens to be true because you know that's how these models ended up being you know trained and then you can exploit that >> or I just made up like say a neuron which is always like value 100 for like across all the samples >> would that mean that it's necessarily meaningful or like maybe it's just an artifact of training just like >> I see so the question what if a neuron is always 100 I mean if that's the case you can look you can also look at kind of variance related questions.
Like if it's 100, you can't just like remove it because then everything is going to be broken.
But you can if it's high mean and low variance, maybe there's another, you know, a way to just incorporate the bias essentially.
Okay, let me quickly go through this other idea.
So, so far we've looked at lossy methods which basically kind of really crunch down the KV cache, but it could hurt accuracy.
There's a very elegant way of doing this in a lossless way.
This is called speculative sampling or speculative decoding.
So remember that if you're doing prefill you can code all the tokens in parallel and this also gives you probabilities.
This is fast.
This is computebound and all nice things.
And in generation it's one at a time.
Okay.
So checking is faster than generation.
If I give you a sequence it's fast to tell me how good it is.

much faster than it is to generate one at a time.
Okay.
So, you can exploit this asymmetry using the following idea.
So, what we're going to do is use a cheap draft model to basically generate from the guest a few tokens.
Let's say four tokens and then we're going to use the model we actually care about the target model Q which to basically review these tokens and accept or not accept them.
Okay.
So, so things are kind of chosen to be balanced, the draft model is smaller and cheaper.
And so even though it's memory bound and it has to generate one at a time, it's not too bad.
Whereas the target model is big and expensive, but what we're doing is asking it to process a batch of tokens in parallel.
So it won't also be too bad.
Okay.
So here's the video that shows kind of how things work.
So if you use a big model token by token that's it's going to be pretty slow.
But if you are doing speculative decoding then then you can see the small model kind of generating a bunch of tokens and then the large model basically you know critiquing them and then so you basically can get this sort of burst of tokens and then maybe another burst of tokens and so on and so forth.

Okay.
So, here is the algorithm for there's a few papers that came out with specular decoding around the same time.
This is, you know, one of them. and so the idea here is that in order to generate we're going to generate K tokens and we're just going to sample from this draft model.

So Q is a draft sorry P is a draft model.
We're going to sample K tokens and then in parallel we're going to compute the logits of these draft tokens using Q.
And then now we have to determine whether we accept or not.
And this is where you do a bit of math and you know probability and statistics.
You basically are going to accept with probability min over this ratio of Q over P.
So if Q is you know much larger than P the larger the Q is the more likely we want to accept it and otherwise you kind of sample from this you know residual distribution and exit.
Okay.
So this is basically rejection you know sampling except for in rejection sampling sometimes you just when you reject you get nothing but here we always are guaranteed to get an exact sample from the target model.
I'm going to skip this you know kind of simple proof. you can it's basically the same arguments as kind of rejection sampling to show that it's the exact you know probabilities from the target model. and the initial paper shows that this is this fast and generally if you look at there's a you know if you have too few draft tokens you're not really leveraging the batching on the target model side and if you have too many then you're going to reject more often.
So there's a sweet spot around you know in this case three or four.
Okay.
And in general the draft model is much smaller than the target model.
And ideally you want the model draft model to be as close to a target which means that you want to kind of distill it.
So which means that actually a lot of the same ideas that we just talked about are applicable to speculative decoding as well.
So basically the idea is that let's try to reduce your KV cache via all the different shenanigans and if you end up with a model you're happy with just serve that.
If you're not happy with it, then it at least can be a draft model and you can use your main model to fix things up. there's a bunch of whole literature on speculative decoding right now that improve over the original u which I'll kind of skip for now.
Okay.
So very quickly now dynamic workloads.
Okay.
So this is the use case is your you know survey live website and come users come and chat with your model. the requests arrive at different times. they have different shared prefixes and they have different lengths. so it's kind of pretty messy.
It's far from this kind of very simple training where you have these blocks of you know the same number of tokens all at once.
So what do you do in this case?
So there's this idea there's a system called Orca that was built.
This is actually very early on which introduces this idea of continuous batching.
So the idea is that you get a bunch of requests that look like this.
So here's a prefix of the first request and you're generating this token.
Here's a second one. here's a third one, a fourth one.
It's jagged because every prefix has a different length.
And what we're going to do is we're going to decode step by step.
So every step you decode one token for all the sequences.
Next step you decode another token for all the sequences.
And then if you end you just, you know, eject that se sequence and then as new requests arrive to the batch then you basically put it in the batch and then you just kind of continue.

so that's why it's called continuous batching because it's sort of this your batch is dynamically u being updated with either old finished sequences being evicted and new ones coming in.
So now one problem here is that you know everything we've seen batching works when all the sequences have the same dimensionality right you have tensors everything every slice has the same dimensionality but in each request here has a different length so what do you do about that so there's this idea called selective batching where you let's say you have you know a three length three length nine and length five so here in the tension computation you can't really do anything about this because attention sort of depends on the length of your your sequence.
So if you have a 3x3 you know computation a 9 by9 computation you can't really you know share the tensor effectively but for the nonattension the MLP layers which is you know takes up a lot of you know FLOPs you can actually just concatenate all the sequences together to form a mega sequence and you process that okay so final idea is PagedAttention this was introduced in the vLLM paper of course, vLLM has many other bells and whistles now, but this is kind of the core idea at that time.
So, so the question is how is the KV cache stored, right?
So, if you think about requests coming in the you have to put them in memory somewhere and in general there's this problem that you get fragmentation.
This is what happens to or used to happen to your your your hard drive. and you have to defrag your hard drive u you know back in the day. so there's two types of fragmentation.
One is that you have to allocate enough buffer so that because you don't know a period when you're going to stop.
So you might have a max token limit of like you know 1024.
So you have to allocate all this memory and you can't put anything in there because you don't you're going to just generate until you hit the max tokens and that's very wasteful.
That's internal fragmentation.

And then there's also you have there could be space between different requests and that space is maybe too small to get used effectively.
So that's just wasted space.
So the solution here is just you know these are systems people so they know their operating systems.
They said, "Okay, well, we've solved this problem once before, so let's just use the same idea here."
So, we're going to divide the KV cache of a sequence into non-ontiguous you know, blocks.
Okay, so if you have this sequence four score and seven years ago, our fathers brought forth, we're just going to chunk it up into these blocks of size four. doesn't matter where the blocks go, but they're going to be kind of aligned according to the blocks.
So, there's some uniformity there. so when two requests share the same can actually share the same KV cache. so you might have this block and this block.
So this block might go here and here and this block might be over there.
So they're they're in kind of interspersed, but as long as you have the indices and keep track of where everything is, it's fine. so in particular if you have you know system prompts then you can actually just you know cache these system prompts the KV cache and the system prompts once right and that can be useful for all the queries okay so this is very useful because if a lot of people are using the same system prompt then you don't have to compute the KV cache for every you know request also there are many applications where you have the same prompt and you actually want to generate multiple responses.
So in that case you can also just share the KV cache for all the pro for the prompt and just have unique responses coming out.
Okay.
So for example if you were to generate let's say multiple generations from four score and seven years ago R blank and so u what would happen here is that you would have four score and seven and you would start by having years ago or hour and then this is called copy-on-write semantics. you keep this and then we have these two samples and if they had happen to sample the let's say the same token you just you know just continue with that but if they sample different tokens you split the block and then you can kind of continue there.
So you're basically sharing as much c of the prefix cache as you as you can. there's a bunch of other optimizations like you know kernels that I'm not going to have time to go over. but the general idea is that you're using kind of these operating you know systems metaphors to manage kind of your inference.
Okay so summary here inference is really really important. it's very different from training even though it's the same model but you're asking the model to do something very different ends up being very memory bound and it's also you know dynamic if you're are kind of in a live chatbot use case we saw a variety of different techniques to improve inference you can quantize you can come up with new architectures you can prune and distill you can also use speculative you know sampling but all of these are driven apply the same you know principle here which is reduce your KV cache but don't hurt accuracy too much and then there's ideas from you know you know systems like paging and speculative execution that can be brought to bear for actually live kind of inference servers one thing we didn't really get a chance to talk about which was discussed briefly is that I think that new architectures have actually a huge potential for you know improvement things like you know state space models or linear attention or diffusion these can you know at some level the KV cache and the way that attention is built fundamentally makes it an inference unfriendly arch kind of architecture.
So if you can come up with a new architecture that is sort of designed for inference in the way that the transformer was not this can maybe unlock a lot.
Okay.
So I will stop there and next class Tatu will return and talk about SC laws part two.

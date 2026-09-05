# Lecture 8 字幕

So, last lecture Percy covered some of the, you know, underlying mechanics, let's call it, of parallelism.

Today I'm going to do my favorite thing of talk about all sorts of sort of things you can know, like knowledge details and trivia about how modern parallelism for language model training works.

and what we're going to try to do is really get to grips with the many, many complexities that are involved when you are having to train huge models on huge clusters. it's going to turn out that there are many different parallelization strategies that you can use at once. at the end we will get to a slide where I say 4D parallelism because there's four different things they can do at once.
Actually, even more that I'll talk about. and it turns out that to parallelize at the largest scale, you'll need to use kind of all of them or most of them. and we'll sort of discuss why. part of your assignment will be figuring out given a particular sort of network topology and a particular model, what is the optimal parallelization strategy for that.
And then finally, I'll walk through some examples of recent large-scale training runs with details that are published and kind of show you this stuff in action.

Okay.
So, the very first part I'll I'll go through somewhat quickly.
It's partially a review of basic collectives, basic networking primitives, right? but the reason why we need all of this parallelism stuff is because there's going to be two bottlenecks that will solve by going to multiple machines and multiple GPUs.
So, one of it is compute.

We need more compute than we can bring to bear on a single chip.
And the way we solve that is just to have many, many machines and link them up together, right?
The fastest supercomputers in the world have like exaflops of compute, whereas, you know, a single GPU is nowhere near that number.

the other reason why we need to continue to scale in parallel is because of memory, right?
Models are big.
We can't fit them all into one GPU, so we need to shard them into smaller pieces somehow.

so, this is going to lead to complexities of having many GPUs across many different machines.
And an important conceptual object that we will deal with throughout our lecture today is the difference between intra-node parallelism, where your connections are very fast, and so you can do things that are very communication expensive, and you're going to have slower inter-node communication.
And because these are slow, we will have to use sort of processes that respect the communication constraints of our channel more, right?
So, we're going to have to do both of those.
Now, remember, you know, all of the things that we will do today, we're not actually going to be, you know, sending packets or anything like that.
We're going to be dealing with this at the level of collective communication primitives, right?
So, whenever we do accounting for, you know, how much communication is done, I might say, "Okay, you know, we're going to have to do a all reduce or we're going to have to do a all gather to implement some of these operations."
So, in order to implement some of these efficiently, you're going to have to go all the way to the hardware level, but our discussion today is going to be algorithmic and so all of the networking primitives I'll talk about is going to be at this level of a of a collective.

And remember, Percy talked about this, but this is important for very important for one part of the talk, which is going to be that the equivalence between all reduce and reduce scatter plus all gather, right?

So, there's going to be one algorithm where on one hand you do a all reduce and on the other hand you do, you know, two of these different steps.
And the fact that these two have the same sort of cost in some sense, means that, you know, we can do the algorithm on the right for free.
We'll we'll see that again later.
This is just a quick refresher just to make sure you remember what Percy said.

Okay.
So, that's, you know, it for the really basic review. the other thing I want to touch on briefly before we get into the algorithms, which are going to be kind of hardware and substrate agnostic, is the difference Oh, sorry, yes.
So, is there any particular reason why you're asking only all reduce and not breaking to reduce scatter and Okay, the question was, is there any reason why I'm talking about this specifically?
We will You can also say that all reduce is equal to reduce plus broadcast.
Yes, yes, there are other decompositions.
This one in particular will be useful for A algorithm, which is why I'm emphasizing this particular decomposition.
That's right.
There there's others that you could do, of course.
Yes.

Okay. so, the other thing I will talk about before I start, most of our discussion today will be hardware agnostic.
Like for the most part, once you understand the basic concepts, you'll be able to apply it to any hardware substrate. but I do want to sort of briefly, before we start, just talk about the low-level hardware stack, both because it's kind of interesting, but also because I think it leads to different sort of parallelization strategies across different model training companies.

So, remember the TPU, right?
TPU is Google's accelerator. and I mentioned during the GPU lecture, a TPU is kind of like a lightweight GPU, but what's different is the networking.
So, how is the networking different?
Well, in sort of one slide, the TPU network is what's called a toroidal mesh, right?
So, you can think about a grid and then sort of like allowing the ends of the grid to communicate with each other.
It's a little bit more complicated.
It's a 3D picture, which, you know, Marcel, you have the visualization, right?
Can you can you post that in the in the lecture channel?
Yeah.
So, there's like a fun web visualization that you can look at that like visualizes the toroidal mesh in 3D, which is very cool.
But your your mental model, I think, can roughly be this picture.
You can forget the more complex one.
All the chips are networked kind of to neighbors, and the neighbors wrap around, right?
So, what does this let you do?
This lets you have a very simple networking topology that can be indefinitely scaled up very simply, right?
You're the number of neighbors you have is kind of the same no matter how large your your network gets. in contrast, the GPU networking philosophy is very different.
It's really much more of a all-to-all philosophy.
And the way that it's networked is it's kind of like a tree.

it's often called a fat tree.
You have, you know, your GPUs connected very, very quickly at the lowest layers, and then you have groups of GPUs that are connected in a pod, and then the pods might have sort of spine switches that allow them to communicate to each other.
And this means that as the number of nodes grows, you know, kind of the tree gets bigger and bigger, your communication costs is going to get higher or your communication sort of topology will get more complicated because you're sort of connecting all of these all-to-all as you go.

Now, this difference in philosophy means that if you're communicating only to neighbors, TPUs are great.
They're cost-effective and you can make each connection sort of more beefy, right?
With the same power consumption.
But if you're connecting in very random ways, like very stochastic, unpredictable ways, then this GPU networking will be more flexible, right?
So, there's this very different design philosophy that you see between TPUs and GPUs, and I'll try to mention where this shows up as we go through the parallelism lecture. and I posted this into the lecture chat last lecture, you know, because someone asked, you know, what's the difference between these two and so on, right?
But just to back up my point, you know, Bill Dally and Jeff Dean were sort of talking about the different sort of hardware and like what are what they're good for, right? you know, GPUs are good for things like mixture of experts, where the communication is much less clear cuz tokens route to different experts.
TPUs are great if you have this dense model that you're doing sort of very predictable partitions of.
I'll talk about that in a in a idea called tensor parallelism later in this lecture.
And so, these hardware things are good for different workloads.

And this was kind of where I was going to leave it for the lecture until this morning when Google decided that they were going to announce the TPU AI and T this morning. and then I looked at this as I was, you know, preparing my lecture and I said, "Oh my god, TPU AI, that's a tree topology.
They've like switched to like a much more all-to-all connection." and you know, if you think about it, it makes sense. you know, modern-day language models are MoEs, and if you're going to serve a MoE, you're going to have all sorts of bandwidth just flying around as you route tokens to different experts.
You probably do want much more of a all-to-all connectivity, where, you know, at inference time these communications are real bottlenecks.
You want to kind of deal with them. and even at sort of training time, the TPU 8T, which is a training chip, you know, it's sort of the cross-rack connectivity like looks a lot more like a GPU.
Like there's much more of like a all-to-all style switch that's being built out called the Virgo network.
That's a higher-level networking stack.
So, I think there's kind of really interesting things happening.
It's a little bit of a convergent evolution across both TPUs and GPUs, where really the workloads are kind of defining the network that we need to have.
Okay.
And then the final thing about hardware, you know, I just love talking about kind of the what's happening in the Chinese AI scene, is kind of this idea of, you know, I think in the GPU lecture someone asked, "If SRAM is so good, why don't you just all use SRAM?
Like only SRAM chip."
And that's Groq, right?
It works really well.
Now, some of you might be thinking, you know, if all-to-all connectivity is good, why don't you just connect everything and connect them with really fast fiber optic chips, right? and then you would kind of get the Huawei Ascend 910, right? so, if you look at the specs of the Huawei Ascend, it's actually really interesting because each chip is a lot worse than a H100.
Like it's a lot slower at matmuls, you know, it's, you know, kind of slower in many ways.
But then, you know, if you're willing to dedicate kind of a giant rack of sort of fiber optic switches to connect a much larger number of chips, like in their case, 384 in a big rack, then maybe you can kind of deal with the fact that each chip is slower by just connecting a much, much larger number of these guys, right? and then so, I think what this shows us is a really interesting trade-off because once you look at this architecture, you realize the power consumption of this thing is insanely high.
It's four times that of the equivalent Nvidia system. and so it's something like, well, you know, if you're willing to pay the power cost, you can solve a lot of, you know, kind of communication problems brute force, and you can scale out much more aggressively to 300 chips. but you're paying a power cost, right?
Similar to that SRAM story of if you want to design a hardware that is efficient to use both power and manufacturing-wise, you end up in a certain place.
If you want to brute force things, you end up in a very different place, right?
So, this is this is I think just a very interesting place to learn from about sort of these hardware design trade-offs.

Okay, good. so now, you know, just to put this all together, the first thing you should think about is the new unit of compute is not the GPU, it's the entire data center.
And we want to have several things.
We want to be able to control the amount of memory we use.
We want to be able to control the amount of compute that we can bring to bear.
And we want all of these to kind of be lossless, right?
We want to use all of the resources that we have.
And then, you know, we're going to, you know, talk about the algorithms that will enable this through all these like communication primitives components.

Okay.
Good.
So now, the second part of this lecture is going to be the bulk of the content, and these are kind of the algorithms that we will use to parallelize language models.
And there's going to be a couple different important ideas.
Actually, these should probably maybe even be grouped.
You know, data parallel is one set of ideas for essentially taking your data that you're going to train on and distributing them across GPUs.
So, the data is the ones that move around. and then model parallelism, you're going to chop up different parts of the model and split them around. turns out that the boundary between these two is a little leaky because one of the algorithms here will actually cut parameters up too.
But we will see kind of why I put all these algorithms into these different bins, right? and conceptually, we're trying to both scale out compute and memory, and that's why we will need a whole bunch of these different tricks.

Cool.
All right.
So, data parallelism, I think is the easiest to understand.
It is conceptually very simple, and I think it's the, you know, standard way that you would try to parallelize an algorithm like LM training.

So, forget that we're doing Adam.
We're going to do just naive SGD for the moment. and we have a capital B batch-sized batch, right?
So, each update step, I'm going to take capital B elements, and I'm going to sum up the gradients, and I'll take my update, right?
So, the most naive form of parallelism that one can do for this is to take my batch, B.
I can cut it up across M machines.
If they're divisible, each machine will get a smaller, you know, B over M-sized batch.
And then, what I'm going to do is each machine computes gradients, and then I will synchronize all the gradients to get my, you know, sum.
Okay, great.
So, compute scaling-wise, you know, this is perfect.
As long as there's enough examples per GPU, we're good.
Communications overhead, how much do we have to talk to each other?
Well, every batch, we have to basically communicate two times the number of parameters, right?
Because what I have to do is I have to basically communicate my gradients back and forth.
Okay.
And then, for memory scaling, there's absolutely none.
Every GPU has the same copies of the models.
You need to have the same sized activations.
So, you don't save any memory at all in this scheme.
So now, what is the problem?
Clearly, we've addressed compute to some extent, but we have not addressed memory at all.
So, let's take a look at kind of this memory, you know, problem and just do a little bit of accounting.

initially, it might seem that the memory situation is bad, but it turns out our memory situation is actually just terrible. and the reason why the memory situation is terrible is if you just kind of do some parameter accounting, right?
Like it just turns out you need to store a lot of stuff, right?
And not only do we need to store a lot of stuff, it's actually more than the parameters that we need to store.

So, roughly the rule of thumb, it depends on your precision, but the rule of thumb you might say is that we need something like five copies of weights and 16 bytes per parameter to store a model. so, you know, we will need to of course store our model parameters.
We need a place to put our gradients, right?
You know, it's kind of the accumulator before we put them into the model parameters.
So, we need to store those guys.

but actually, you need to maybe store more stuff. you might need to have like a higher precision accumulator that you accumulate into in SGD.
This might be temporary, but you might need this. and if you're doing Adam, and this is really where the bad stuff comes in, you're going to need to track both the first and second moments of your gradients, and you need to track them over time, right?
So, you need two of these.
And unfortunately, you might need to store them in high precision depending on your stability characteristics, and these might be quite expensive, right? and we generally call these things optimizer state, right?
Like the thing that you're accumulating into, the first moment, the second moment, right?
And these are actually quite expensive.
If you look at the accounting, this is most of the cost memory-wise of doing an SGD update. and so, I think one of the really, you know, elegant and nice, you know, things that you can do in this space is to start cutting up the amount of memory you need and distributing them across the accelerators, but doing so in this structured way that gets you these gains for basically free.
So, as I kind of said in the last slide, the optimizer state is most of your memory.
So, here, you know, the optimizer state is green.
That's most of your memory. orange is your gradients, and blue is your parameters.
And notice how parameters and gradients are the same size, and your state is bigger, right? if you do naive data parallel, you're just going to replicate all of these the state across the GPUs, and this is not so good, right?
Because you've in some sense consumed way more memory than you need, and your memory consumption is linear with your number of accelerators.
Not so good, right? what we can do instead is perhaps we can shard the optimizer state onto different GPUs, and this would be a big savings.
If we shard different other parts of, you know, what we need to track, that would be also big savings.
Like maybe we can also shard the gradients into different GPUs, or in the extreme, maybe we can shard everything into different accelerators.
And then you can see that, you know, the implied memory gains on the very right would be quite dramatic as we go all the way down from 120 to 1.9.

Now, the main thing that we will sort of study now is what is the cost of doing this, right?
You might imagine there's no free lunch.
You know, baseline is going to be the fastest, and as we go downwards to here, we're going to have to pay a big communication cost in order to achieve these memory savings.

I think the remarkable thing is going to be that a lot of this will be free.

but I'm going to walk through this step by step. so, let's start with what people call zero stage one. so, there's going to be three stages corresponding to each of these steps.

And so, stage one is going to be the simplest.
The only thing I will shard is my optimizer state, and I will shard it across the GPUs, okay? everyone is going to have both the parameters and the gradients.

and so, what you can think of is each worker is going to be responsible for updating one slice of the parameters.
So, if I'm GPU number zero, my responsibility is updating this very first slice over here, and I'm not responsible for any of the other updates.
So, how would it work, right?
So, what we're going to do is everyone computes everyone has a different data item, right?
So, they're going to compute a full gradient on their data item.
That's the first step.
Now, I'm going to take the gradient I computed, and I'm going to reduce scatter the gradients onto all the other machines because everyone else needs my gradients, right?
But they don't need all of the gradients, they only need the gradients associated with their update, right?
So, every worker is going to get, you know, from me, my gradient, but their part of the parameter space, right?
So, this is going to be equivalent to a number of parameters communication cost because I'm sending my gradient, but I'm cutting it up and sending it into each GPU.

Now, each machine is going to be able to now do their update.
They have their parameters, they have their gradients that they got from everybody else, and they have their own state that they've tracked, right?
And now, the parameters have been updated, and now I'm going to all gather them back, right?
I was responsible for, let's say, like 1/4 of the parameters.
So, those updated parameters, I'm going to send back to everybody else, right?
So, this is a all gather, right?
So, this is where the equivalence is kind of useful.
How many sort of operations did I do?
Well, in the naive data parallel case, you know, what I had to do was I had to do one all reduce, right?
I had to sort of exchange all the gradients of everybody, and then do the updates, right?
So, the communication cost is two times the parameters.
Now, in zero stage one, you know, I did two operations.
One was a reduce scatter to send the gradients out, and then an all gather to collect the updated parameters.
But we already said a reduce scatter and an all gather is equivalent to an all reduce.
So, zero stage one is going to have the exact same sort of communication characteristics as naive DDP, right?
So, this was free.
Free memory savings.
Wonderful, right?
And we look at the memory, you know, we've basically taken the parameters and anything that's not optimizer state, I've divided by N GPUs, right?
Wonderful, wonderful, wonderful.

Now, can we do more?
Well, zero stage two is now going to say I'm going to split the gradients up, and then I'm going to you know, shard that as well.
Now, this is going to be trickier because before I was relying on the fact that I can compute the entire gradient.

Now, I can't even materialize the whole gradient.
So, what do I do? well, this turns out to just be kind of a systems trick.
You know, as I sweep backwards through my network, you know, I don't need to compute the full gradient.
I can send out the gradients to the associated workers as I kind of go.
So, the key step here is now, instead of materializing the gradient vector all at once, I'm going to walk backwards through the compute graph.
And after computing a layer's gradients, I'm going to immediately just reduce that to send that to the right worker, right?
And once gradients aren't needed in the backwards graph, just immediately free it, right?
So, you're kind of now incrementally computing and sending gradients as you go.
And as you can see, you know, doing it incrementally or doing it all at once, same thing, right?
So, it doesn't really make a difference.
So, now you can at the end, once again, all gather the parameters I have sharded the gradients in some sense without incurring any additional cost, right?
Okay.
So, the last and the most hairy thing, and this always, you know, when I first learned about this, seemed like total magic, is, you know, we've gotten something for free so far, so let's keep pushing it.
I'm going to shard the parameters as well, all the way across all the states. and so, we're going to use the same kind of idea that we used in zero stage two.

And the idea, remember, was I'm going to do things incrementally, right? and so, what I'm going to do is I'm going to send and sort of receive parameters on demand while stepping through the compute graph.
And this is really tricky because each GPU only sees a slice of both the parameters, gradients, and optimizer states at any one time, right?

and this is zero stage three, also known as FSDP.
If you've ever paralyzed, you know, anything in a torch library, you've probably dealt with this before. so, the way that it works under the hood, is what you're going to do is you're going to do two all gathers and one reduce scatter as you go.
So, we're going to step from left to right here, and this is kind of going to be the entire sort of update process.
So, what you do is you're going to load your model, you know, you have your two GPUs at the top and the bottom.
You're first going to gather your weights for this one layer, I'm going to forward this one layer, and then I can free my weights cuz I've already done the forward, right?
I don't need these parameters anymore.

Now, let's say I only have one layer, you know? now I can move to the backwards process.
Now, to do the backward step, I need the activations, which I've kept, and I need the parameters for this layer, which I can all gather kind of on demand.
I all gather it, I do the backwards, and then I scatter the gradients back right when I get them, right?
So, I got the gradients, I reduce scatter the gradients out, and then I free the weights, I don't need them anymore, and I repeat this process, right?
So, I grab the parameters as I need them, I compute my backwards, I send the gradients out, and I repeat this process, and then, you know, I update my weights, all done, right? so, what's the communication cost, right?
How many all gathers did we do?

We need two all gathers, one reduce scatter.
Also, like, you know, so, we have one extra all gather, which seems bad.
That's one thing.
The second thing is this seems a little crazy in terms of the overhead because we're doing communication all the time.
Like, every layer, we're doing this communication.

you might think that this is extremely expensive and bad.
But I think, you know, one of the things that's counterintuitive and important is you know, zero stage three uses kind of two really important ideas to make this essentially overhead free. so, the first one, which we've already seen, is this idea that we're going to kind of sweep through the compute graph, and then we're going to do the necessary communication, and then immediately free the memory, right?
That's already an idea we've experienced.
The other idea, which I haven't talked about yet, is we're going to overlap the communication and computation cost of this process, right? and this is kind of hard to explain initially, so I think I've taken this plot from this paper, which I think was explaining the PyTorch FSDP sort of architecture.

so, the way that this kind of works is, you know, in order so, we have different streams here.
This is kind of what's happening in the CPU, this is what's happening in the GPU for computation, and then what's happening in the GPU for communication at the very bottom here.

Now, you know, the way FSDP works is I have to first get my parameters before I can do my computation.
So, I'm going to all gather zero, layer zero, right?
And once I've all gathered, I can now do my forward.
But as I'm doing my forward, notice that I'm requesting the next layer's parameters.
I'm all gathering one while I'm doing this computation.

And now I all gathered one, I can now do forward one, right?
And I can start all gathering two while I'm starting this process.
And so, I'm overlapping computation and communication as I'm kind of doing this.
And, you know, I might spend some time freeing my parameters, I might also spend some time sort of computing another forward.
For example, if you reuse that W naught, you know, I'm reusing this forward zero sort of twice, right?
And so, what this allows you to do is if you have enough computation and your network's fast enough, by overlapping communication and computation, right?
FSDP can basically be free, right?
You do see some bubbles, these are overheads, but if your comms is very fast and your computation is very big, right?
You can easily see how the computation can run faster than the communication or sorry, the computation can take longer than the communication, and you end up having this really great improvement in memory usage without paying for almost any amount of memory use.
Okay.
So, you know, when we look at the different zero stages, right?
DDP, the naive data parallel, costs two times parameters, right?
It was a single all reduce. the two zero stages, one and two, right?
The ones that shard gradient and optimizer state, this is free in the literal sense because it has the same amount of communication cost, right?
It's just the identity of all reduce being equivalent to two operations, right?
Now, zero stage three is technically a little bit more communications.
You need an extra all gather, right?
But it's not so bad because of the really clever ways in which all these libraries can hide the communication cost underneath the computation.
So, in practice, you know, FSDP, if you run it, you're going to see GPU utilization that's very close to, you know, just the single GPU performance.
Like, it is actually quite remarkable how good FSDP is.
And it's also very conceptually simple.
You will actually have to write an FSDP implementation as part of your assignment. and, you know, you can just kind of like write a wrapper that will wrap any module and turn it into its FSDP version, right? conceptually, all you're going to do is, you know, do a bunch of all gathers, compute, free, and then repeat on the backwards pass.

Okay, good. yes.
So, the assumption is that, like, we are taking the gradients from the next GPU to calculate the gradients for the previous right? when you say "gradients" So, so, we're not I think what you just described, which is like taking gradients from one GPU to the next, that's closer to pipelining.
Pipelining would be like if one layer was on one GPU and another layer was on another GPU.
That's not the case.
So, in all these cases, you know, every GPU kind of goes through the entire model, right?
From the start to finish. but the difference is no GPU is going to hold the entire parameters at the same time.
And so, what we're going to do is, you know, if I only have part of the, you know, network that doesn't have layer zero, I'm going to sort of demand the layer zero parameters first, and then do the full computation, and then I'll move on to the next layer, right?
So, it's the same computation, you know, it's the same single GPU computation as naive data parallel, but you can just kind of think of it as request parameters, free parameters in between every computational operation.

Yes.
So, are you saying that every GPU has a part of every layer, and that's what the all gather is for?
Yes, that's all.
So, so, the picture is here, right?
Parameters, gradients, and optimizer states are all sharded across different GPUs.
That's right.

Yeah.
Good.
Oh, yes.
Why is the communication cost not multiplied by the number of layers in small Yeah, because these are So, so, the number of operations multiplies by layer, that's So, that's correct.
But each of these communication pieces is smaller, right?
Cuz this is just one tiny MLP or something. and I'm comparing that to the cost of all reducing a whole network, right?
So, so, it all adds up in some sense.

Yeah, good. all right.
Okay, good. so, in practice, you know, if you look at, you know, the maximum size that can fit on, you know, let's say a 100 GPU, you know, if you go from like the baseline, you can not even fit a 7B model, and you go to zero stage three, you know, you can now fit, you know, 50 billion parameter models and so on, right?
So, it becomes a lot easier to fit these big models using FSDP, right?
The improvements that you get in terms of the memory improvement is quite significant.

Okay, yes.
So, can you go back to the previous slide? so, regarding this on index E E100, what would be what would be the limitation on H100 or P200?
Okay, I can't do the mental math of like multiply by 141 and divide by 80, but if that would be the number for like an H200, and so on and so forth.
Yeah, it would it would be linear, right?
Like, it wouldn't really change the math very much to go from one accelerator to another.

okay.
So, FSDP, I think is very clean, it's very elegant, I like it a lot.
I wish I could end the lecture here, but unfortunately, we will have to proceed into uglier and more hairy things.
And there's, you know, two reasons why we have to do that.
One of them, and this is an important idea for parallelization, is that in some sense, data parallel is consuming an important resource, and that resource is kind of the batch size, so to speak, right?
If you have a batch size of eight, you can never have more than eight accelerators, right? and you might think, okay, like, let's just make the batch size bigger and bigger and bigger as we have more accelerators. you can't do that because at a certain point, you know, there's something called the critical batch size where the basically gain that you get from the additional batch element is less than if you had taken another SGD step on that single element, right?
So, there comes a point where, you know, at small batch sizes, adding extra elements is, you know, perfectly helpful.
It's the same as taking more SGD steps.
But at a certain point, diminishing return hits in and an infinitely large batch size is not infinitely better than infinite steps, right?
Infinite single steps.

because of this, now we have faced a hard trade-off.
Like, do we want small batch sizes and let our GPUs be more idle?
Do we want big batch sizes and sort of take the hit from optimization?
Sort of hard to say, right?
There's another issue that we have, which is that we still have memory issues, you know, zero stages one and two don't really let you scale memory at all.
And zero stage two lets you cut up the parameter memory, but unfortunately, you know, there's activation memory and other kinds of, you know, memory use that this does not split up or this does not reduce at all, right? and so we need better ways, more fine-grained ways of cutting up the model in order to really start pushing down the memory use beyond these points, right?

and so this kind of takes us to various model parallelism ideas. and so model parallelism is going to be this idea of essentially, like FSDP, we're going to split up the parameters across GPUs, but I think the important conceptual difference between what we talked about and what we will talk about next is that now we are going to communicate activations, right?
In FSDP, we cut up the parameters, but in some sense, it was just kind of a wrapper.
We were still doing the normal computation, we were just sending parameters back and forth before doing Right?
So parameters were the things flying around.
Now what we're going to do instead is we're going to communicate activations back and forth, right?
So if, for example, one layer lives on GPU zero, the next one lives on GPU one, right?
What I have to communicate is the activation of the layer in between, right?
So that's going to be the big difference.
And this will, you know, involve pipeline parallel, which is cutting up layers, tensor parallel, which is cutting up matrices, and expert parallel, which is sharding experts into different places, right? and Percy's talked about, you know, tensor parallel already, so you have roughly some mental model of this.

I'll start by talking about pipeline parallel, or layer-wise parallel. you know, this is I think conceptually very simple.
Like, you know, back in the days when I first learned about parallelism, I was like, oh yes, it's simple, just cut up the layers and put them on different GPUs, right?
Very very simple thing. and so you're going to pass activations forward in the forward pass, and then you're going to pass like partial gradients backwards in the backward pass.

Now, okay, if I do this, what will happen?
You will get a terrible-looking picture like this, very very depressing.
I have four accelerators, each handling a quarter of the model.
And so what computation happens as a function of time?
So time is sweeping from left to right, and the different accelerators are the different rows.
So first, my zeroth accelerator at the bottom here, that's handling the first layer, will do some computation, and then it will stop, right?
And then it will hand off the computation to the second one, and then it will do nothing for the rest of time, until the very end.
And then I will basically passing things back and forth, and one GPU will be active at a time. and then coming backwards, one GPU will be active handling the backward pass at a time.

and so most of the time my GPUs are idle, my utilization is truly truly terrible. and this is called a bubble.
It's the part of the pipeline where you're not doing anything.

and the solution to this is to essentially do kind of a batching or pipelining. so instead of doing a simple sort of one layer after another, you're going to have a pipeline, right?
Where in this case, in this picture here, you have four elements to be processed, and as soon as you process one, you know, you start on the next element, and then you pass off the one that's already done to your next layer, right?
So you can do this to essentially pass elements upwards and then back downwards. and so if you want to make this reasonable, then what you need is, you know, the ratio of essentially this bubble time to the useful compute, the utilization in some sense, is going to be the number of stages divided by kind of this microbatches.
And so we need a huge batch size in order to reduce this bubble towards zero, right?
Like so if we want our bubble to be zero, it's going to go down as one over the microbatch size.

This is kind of why I said batch sizes are useful resource, because now we can spend it in a different way, which is the pipeline more to reduce the amount of idle time in pipeline parallel.

Okay.
So pipelines are terrible, and by folklore, you know, people say things like, you know, parallelization code looks reasonable and people can understand it until you implement pipeline parallel.
So why do we do pipeline parallel?
Well, you know, pipelines can save memory compared to DDP, cuz we've cut up layers, and you can also compose this with data parallel, so you can do both.

but the other reason why this is a very important idea to learn, and you will always have a pipeline in certain kinds of topologies, is that it has very good communication properties, compared to both FSDP and compared to, you know, any other kinds of things you can do. it only depends on activations, like B * S * H.
It is also point-to-point, you know, one layer is going to communicate to another, it's not all-to-all.
And so, you know, depending on how you sequence these things, they can be a very efficient way to utilize your network.

So in practice, what we're going to do is that pipelines are going to live on the slowest networking links.
So if you have, you know, multiple data centers or multiple pods that are slower to talk to, you're going to parallelize across those two using pipeline parallel, because that's kind of the most communication-efficient parallelism that you can kind of bring to bear here.
Okay. and as I said, you know, pipeline performance is really dependent on batch size.
So I'll I'll point at this paper repeatedly, but I'll mention which one this is now.
This is the Megatron paper from Nvidia.
They have a bunch of really lovely sort of parameter sweeps of what happens to utilization as a function of different sort of configurations you can do. so as you if you have large batch sizes and you pick a large pipeline size, you can get pretty good utilization, close to, you know, not doing pipeline parallel at all, but you need the big batch sizes in order to be able to keep utilization up.
Otherwise, pipeline parallel is going to rapidly degrade your utilization and performance.

Actually, I'm going to pause for a moment here to ask questions, cuz the next slide is complicated.
Yes.
So you mentioned on the previous slide where that like pipelines have like good communication properties compared to FSDP.
What is the specific mechanism for why that is? it's a smaller amount of stuff that's being communicated.
So it's batch times sort of sequence length times hidden, and that's almost always a smaller amount of data than like attempting to communicate a whole parameter matrix, right?
So people have tried to figure out much more clever ways of, you know, reducing the amount of pipeline sizes, and you can kind of do clever things that cut up sort of the different stages of what you're doing for different layers or different microbatches.
So you can sequence different forward pass elements in between different backwards pass elements, and by doing sort of clever scheduling, you can further reduce the bubble size.
This was taken from the DeepSeek paper.
So by really cleverly manipulating when you're doing forward and backward computation, you can get a little bit better in terms of the bubble size.

But I think the even more clever thing that you can do, which is called like zero bubble pipelining, is not really about scheduling per se, but it's actually about kind of thinking carefully about the structure of your backward pass. so in the backward pass, there's really two things that have to happen. if you think about it, you're kind of going backwards in your computation graph, and as you do that, you're going to do two things.
One is going to be you're going to propagate your sort of partial derivatives further down your computation graph.
So like, remember, you know, backpropagation, you start at the end, and you're propagating these like partial derivatives back, right?
So you have to do that.
But you also want to compute the gradients for the current weight, right?
So you have two things that you do at sort of every node of your computation graph.
You're going to propagate backwards, and you're going to compute the derivative of your weight.

The important thing about, you know, thinking about this compute graph is actually, you know, one of these two is really important for pipelines, like, you know, and the other one you can kind of do whenever.
Propagating the partials backwards, right?
That's very important, because your next stage can't do any work until you've propagated that signal back.
On the other hand, the derivative with respect to the weights, that is kind of a leaf node, so to speak, on this graph, and I can do it whenever, right?
And so the clever thing to do now is you kind of separate these two, right?
Like so you got your Bs over here.
These Bs are kind of the backwards, right?
We're propagating backwards.
And then these Ws, these Ws are computing the derivative of the weights, and they can happen whenever.
So what you do is you compute the Bs as quickly as you can, right?
So those come first, and then you can defer the Ws until later and do it when you have a gap in your computation.
And when you do this, you can basically almost all completely fill up the pipeline.

and this zero bubble pipelining thing is very complicated, or much more complicated than you would normally like to deal with, but it allows you to almost deal with these like pipelining issues almost entirely, depending on, you know, essentially the workload involved in the W versus the B, and so on and so forth.

Good.
But I think, you know, having having seen this, I was like, oh this is this is so clever.
Such clever things that you can do with systems.

Okay.
So that was pipelining. any questions about that?

Good.
So I think there's a natural sort of analogy or like a parallel you can think of.
So pipeline parallel is like depth.
So you have your depth, you cut up your your network along the depth dimension. but we have two ways that we scale, right?
We got a depth axis, and we have a width axis.
So, how What if we start cutting up our model along width, right?
So, this is tensor parallel, as Percy explained before. and it's really the simple idea that, you know, matrix multiplies can kind of be cut up into smaller matrix multiplies, which you then add the partial sums back together.
This is really the same idea as tiling, as well.

So, this core primitive just appears many, many times.
And so, when you do this in practice, right?
What does this actually look like? so, this is a, you know, GeLU that's happening.
So, what do you normally do?
You have your X coming in as input on the on the very right of this slide here. and then I have a matrix A that I multiply with.
I apply my non-linearity GeLU, and then I multiply with B to down project, and then I output my Z, right?
This is the normal sort of serial computation that I do.
Now, in tensor parallel, what I would do is I would cut up my A into two sub matrices, my B into two sub matrices, and then I would, you know, run the computations in parallel, and I would sort of gather them together at the end.

Now, one thing that is, you know, kind of important is there's kind of a duality between the forward and backward pass.
So, in the forward pass, you know, all I do with the X is I, you know, copy them twice, and then I run the computation forward.
So, F is the identity, this F function.

And then, once I get to the end of my parallel component, the G function, that's a all reduce that brings them back together, right?
But in the backwards pass, as I'm doing my gradients, this is going to flip.
My G is going to be an all sorry, my G is going to be an identity, right?
Because the partial derivatives are coming backwards.
And then, my F are the partials, you know, that I need to sum up in order to get my, you know, partial derivatives at the start of this module, right?
So, they're going to flip and this duality is kind of important if you're going to, you know, write tensor parallel.
Okay. and so, you know, you might have already noticed, right?
But when we split things, you know, some of the ways that we cut are going to be column-wise.
So, column-wise tensor parallel is going to happen at the kind of the inputs, right?
So, the inputs of the MLP, the projections of the attention are going to be cut up by the columns in each transformer block.
So, this is column-wise cuts.
And then, there's going to be row-wise cuts in sort of the corresponding second stage.
So, in the down projections of the MLP, as well as the outputs of the attention, you're going to have a row-wise cut.
And then, everything that's kind of a small layer, like a layer norm, more like a non-linearity, or routers in ML sorry, in MoEs, those are all fully replicated across the machines.
You don't really want to bother with the overhead of cutting those guys.

Okay.
And then, like so, when do we do tensor parallel? tensor parallel is extremely computation hungry. what you're doing is every time you have a matmul, you're going to be doing some communication, right?
You have a all reduce, or you have a, you know, in the backwards pass, you have a you have a all reduce here. one in the in the G when you're going forward. and these are kind of activation size, and you're doing them very frequently.
So, this is very communication hungry, and so you only want to do this within a node.
So, for GPUs, you know, eight GPUs are going to be networked tightly together in a single box.
And so, up to eight, you might be willing to do some tensor parallel.
Once you go beyond that, you're going to be going cross node, and those inner connects, those connections, are much slower, and so you're going to get a big drop in performance as soon as you sort of hit that point of going beyond a single machine.
So, tensor parallel, you should kind of remember as very communication hungry.
You usually use it in the fastest inner connects that you have at hand.

And this is kind of the point at which I, you know, I'm going to sort of go back to the thing I said about TPUs, right?
Remember for TPUs, you don't really have this distinction between fast eight GPUs and next machines.

You just have this big mesh. and so, one of the big advantages that TPU people will tell you is, you know, you can tensor parallel very large numbers compared to the GPU world, because they're able to sort of have high bandwidth in this very regular communication pattern over this mesh, right?
So, that's a big difference if you're parallelizing between TPUs versus GPUs, the amount of tensor parallelism you use versus pipeline parallel will actually be quite different.

Okay.
So, how do we compare these two?
They're both model splitting strategies.
They'll both save on, you know, both parameters and potentially activations. the pros here of tensor parallel is that there's no necessarily pipeline bubble. and if your network's fast enough, you know, you might be able to get full utilization. it's also low complexity.
It's just cutting up some matmuls. but the con is that the communication is much larger.
It used to be that you were doing B * S * H point-to-point communication.
Now, instead, I'm going to be doing something like 8 * B * S * H times roughly one all reduce communication, right?
So, this is not even just point-to-point.
This is, you know, all-to-all communication that needs to happen.
So, tensor parallel is great whenever we have high-speed inner connects.
Every other time, we probably want to be using pipeline parallel.

Okay.
So, the last thing I want to talk about about sort of, you know, we're talking about memory use, is to, you know, really think care more carefully about memory.
I think the most naive view of memory in kind of deep learning and language modeling is something like, well, memory is just parameters.
Like, that's this green bit over here, right?
You have a certain number of bits you need to store for your parameters, you know, that's it.
I've already talked about optimizer state, which is important, which is this yellow box over here. but if you look at this plot, this is, you know, actual memory use as profiled in one of the PyTorch profilers, you'll find actually that this is not really the whole story.
There's a lot of dynamic memory that needs to exist in a computation.
You need to store all the activations, you know, these big red bumps over here.
And then, as you compute your gradients, you kind of need to store them in your backward sweep, right?
And so, really maximum memory usage happens a little bit after, you know, the maximum activation point.
Like, after you start sweeping backwards on your gradient, but you still need to compute a lot of keep your activations, those are usually the maximum memory points, right?
So, to reduce this, maybe we need to deal with this big red hump, this activation.
And if you start looking at sort of more modern sort of workloads for language models, you start to find that as you get to sort of bigger and bigger parameters.
Forget the column that says present work, I'll talk about that in a moment here.
But if you look at just the baseline columns of this plot, as you go to bigger models for moderately large sequence lengths, you know, your activations are just kind of going to dwarf the parameter memory that you need.
And so, any sort of memory saving strategy has to sort of reason about activations in order to be fully effective.
Like, we can't just deal with parameter memory.
That's not really sufficient for us.

So, if we're storing everything, and this is kind of a good, I don't know, statistic or rule of thumb to start with, if we need to store everything, the amount of activations that we need is going to be something like 34 * S

* B * H, and that's going to be sequence length times batch size times hidden dimension size, plus 5 * AS over H.
So, that's AS is attention heads times sequence length divided by hidden dimension size.
And this SBH dependence, right, is kind of fundamental because, you know, we expect to have to store something for every element of the sequence, we expect to have to store something for every batch element, and we expect to have to store something for every hidden dim size, right?
So, we're going to see this SBH term repeatedly

as we try to go through different ways of reducing activation memory and accounting for the total amount of activation memory. and in case you're curious where the terms come from, you know, this odd-looking term, the 5 AS over H, is going to come from the quadratic attention terms including the dropout terms. and we can drop this term via recomputation if we do flash attention.

Okay.
Now, what if we did tensor parallel?
Like, you know, one of the promises that I made to you in talking about model parallelism was that we could reduce activation quite a bit.
So, now let's think about what happens.
Well, tensor parallel is going to split out the matrix multiplies in attention and MLPs.

So, which parts are those?
Well, the MLPs are 24 of those of those 34 that I talked about before. and the attention heads are 5 AS over HT, right?

The second term over here.
Now, I can reduce that by T, my tensor parallel size, and that's kind of good.
This is not bad, because I've taken, you know, most of the memory, and I've made it scale linearly with the tensor parallel size T.
So, that's good. but it turns out that there are other things that you do.
Like, you might have a layer norm.
You might have a dropout.
You need to keep the inputs to attention and the MLP, right?
Cuz the inputs to each layers need to be stored as residuals for the for the backwards pass.

And unfortunately, these are not reduced with your tensor parallel size, right?
Tensor parallel does not split the layer norm, and therefore it does not split the activation for these guys, right?
So, this is unfortunate, because you were hoping that if you had a thousand different GPUs in tensor parallel, cuz you're Google, you might be able to reduce this activation dramatically.
The unfortunate reality is that you're still going to suffer a 10 times SBH penalty for doing all of this.

and so, the last part that's usually used in composition with tensor parallel is this idea of sequence parallel. this is an extremely misleading name, because usually when I say sequence parallel, actually there's a thing called context parallel that is actually more natural for this name, but sequence parallel is kind of the following idea.
So, what you're going to do is we have these remaining terms, 10 SBH, and these terms are very lightweight.
They're like layer norms, they're stuff that doesn't have much computation. and what I'm going to do is I'm going to split them up somehow, and I'm going to split them up over the sequence axis rather than over the hidden axis. and this is going to then involve all gathers and reduce scatters before every operation where we need these. and if you think about it, this is very reminiscent of FSDP, right?
We have this thing that we need.
We need attention, I'm sorry, we need activations at some point.
We need to compute the activations going forward, and we need the activations when we sweep backwards for the derivatives, right?
But, we don't need them now.
So, we're going to store them by sort of split them up splitting them up across the sequence axis and materializing them on demand.
So, I would say this is like conceptually very similar to the FSDP-style idea where we're kind of storing them in this sharded format, and then we're gathering them as we need them. and then as I said before, there's a duality to this.
So, in the forward pass, you know, G is an all gather, G bar is a reduce scatter.
In the backwards pass, this is going to be reversed, right?
The all gather and reduce scatter are going to be reversed between G and G bar.

So, if I do this, now what do I get?

Well, remember with the no parallelism setting, the very first part, you know, what I told you was you get this activation memory.
You use 34 * SBH with an attention head component.
With tensor parallel, you know, I get to divide the MLP part and the attention part by T, but not all these other sort of pointwise ops that I have.

if I do sequence parallel, I've split these up by my accelerators as well just in the sequence axis, which gives me now fully linear dependence. if I do activation recomputation, I can even drop the second term because remember this is just the storage that I need for the softmax, so I can drop this through recomputation, which gives you SBH * 34 / T. this is nice to remember because this is kind of the lower bound of what you can achieve, reasonably speaking, for normal training for activation memory.
And so, if you want to do things like compute by hand whether your model will fit into a GPU, this is a good thing to remember.

You need at least this much for activation memory, you're going to need some memory for your optimizer state, and you will need some memory for your parameters and your gradient, and once you have those things, you have a rough sense of how much memory you're going to need for your model.

Okay, I'm going to stop here for a moment.
I have talked through both pipeline and tensor parallel at this point. and so, people might be a little bit confused, and we can talk through a few things before I move on to expert parallel.

Good.
Oh, yes.
MLP 24 over T is the MLP part, yes.

And MLP part cannot be optimized through the recomputation.

You can recompute the MLP as well.
Like, you can do a lot more recomputation than what is listed here, but recomputation beyond what is listed here is very computationally expensive.
Like, a recomputation for MLP involves running the MLP again in the backward pass, which you probably do not want to do.
But, recompute for attention is much cheaper. you recomputation for attention is generally cheaper, yeah, cuz you do it tile-wise as well.
Yeah, and also you don't want to pay the quadratic cost.
That's quite expensive.
The quadratic cost Okay.
Cool.
All right. so, I'm going to talk now talk about the last ingredient in sort of the standard parallelism toolbox, which is expert parallelism, right?
Now that MoEs are very standard part of the toolkit, you know, most big models are MoEs these days, and that allows us to take advantage of one of their big benefits, which is expert parallelism. and expert parallelism, you know, I think of myself as being something analogous to tensor parallel, right?
In the sense that you're taking the MLPs, right?
The FFN components of your network, and you're going to split them up in various ways, and you'll incur a communication penalty.
It's in the same way that you're maybe splitting up the matrices, except here you're splitting up the whole MLP across different devices.

and the system's behavior of tensor parallel is oh sorry, expert parallel is roughly like tensor parallel as well.

In the sense that this is a high-bandwidth parallelism primitive that also reduces activation. but, one important thing is that if you're doing a ML MoE, you almost always prefer using an expert parallel sharding strategy over tensor parallel.
I've taken this screenshot from the parallelism guidelines from Megatron, which is Nvidia's like parallelism library.
It's a really wonderful library and document, and one of the guidelines they say is, "Look, if you're going to do some sort of parallelism, either EP or TP, you know, you should be using EP over TP."
Why is that?
Well, this is actually a good list of drawbacks for a tensor parallel.
If you have a matrix, and then you cut up the matrix too finely, well, the matrices are going to start to get small, and your GPU utilization will suffer, right?
So, you want your matmuls as big as possible, and tensor parallel kind of reduces that. for MoE layers, it's a lot easier to sort of route sparse token activations rather than to sort of potentially route these dense big tensor parallel matmul activations. and then you know, you can sort of do things like route tokens exactly to the places that they need, and you can sort of skip some computation or sorry, some overhead in the case is that if you're going to have MoE anyway, you might as well have the number of experts sort of split up over all the devices.
Now, this Sorry.
Yeah, okay.

This might make it seem like, "All right, EP is great.
Everyone should be doing expert parallel all the time." unfortunately, expert parallel is still really complicated and still very difficult.
I think everything that I've said after FSDP, you know, tensor parallel, pipeline parallel, system-wise is quite non-trivial to get working really well. and I'll just point to two different libraries, not to tell you the details of what happens inside of them, but to tell you this is like kind of seriously complicated business.
DeepSpeed has a library called DPP, which I think is one of my favorite things from the DeepSpeed V3 era, which, you know, is basically their own library for doing expert parallel sort of routing and dispatching, where they're like looking at really low-level GPU networking primitives in order to try to, you know, basically combine different operations or like have the hardware handle certain operations in order to make MoE routing as sort of efficient as possible.
Similarly, Nvidia has a library called Hybrid EP, which, you know, is a similar, you know, effort of having very efficient low-level hardware implementations of expert parallel dispatching. and just to like give you the high-level thing, right? the reason why this stuff is hard is, you know, you are doing a lot of all-to-all dispatches.
If you If you look at the communications pattern of a MoE, much like tensor parallel, every time you have a MLP, you're going to have to route tokens all sorts of different places, and you need to do this in a very latency-sensitive way because your computation is waiting for your tokens to arrive.
So, reducing sort of the latency of this dispatch is extremely extremely important.

And sort of one final thing that is like a fun trivia about the DPP library is, you know, the DeepSpeed folks are so intense that, you know, in order to optimize like every last bit of performance, they basically looked and found sort of undocumented sort of PTX, which is like, you know, GPU machine code, or machine-code-like things, and they found how to use these undocumented instructions to further accelerate their networking communication.
So, really this is like the level of stuff that you kind of need in order to get to the frontier of, you know, parallelism efficiency.

Finally, you know, up until now, I've sort of talked about all the parallelism primitives in isolation, like data parallel and so on and so forth, with the implicit assumption that you can just kind of like put them together like LEGO blocks.
And this is true for the most part for basically all the other parallelism strategies.
Like, it's very easy to combine like, I don't know, data and tensor parallel.

Expert parallel has sort of additional constraints that you want to be somewhat respectful of.
Usually, the naive way to do data and expert parallelism, which I think is in many of the old libraries, is that the sort of replicas for DP and EP are the same.
So, in other words, like let's say I have a data parallelism of eight, you know, I'm going to shard my eight experts my experts across those eight replicas, and then my data will be split across those eight replicas and routed in various ways, right?
So, in other words, you split your GPUs according to DP, and then your EP is sort of a subset of that split, right?
And this is like a very natural thing to do because you're routing tokens. but, if you do this, there's a maximum bound to how far you can parallelize with EP, and it constrains sort of how DP and TP interact. and so, I think this is the final complexity.
I apologize that expert parallel is a little bit, you know, like really messy in many ways, but because of these complexities, there is actually this one interesting sort of systems thing that I'll I'll tell you about before I stop talking about EP, which is this following fact.
Remember, MOEs only change the MLPs, right?
They don't change the attention at all, right?
So, because we're cutting up the MLPs but not the attention, this means that expert parallelism kind of applies unevenly to the model.
So, when I parallelize at the expert level, I'm parallelizing the MLPs but I'm not parallelizing the attention, right? and so in order to parallelize the attention, you know, maybe I want to also use tensor parallel, right?
So, I want a high tensor parallel level in order to cut up my attention, but, you know, this is going to then affect, my my, my MLPs because if I have high tensor parallel and high expert parallel, I've cut up my matrix matrices into really tiny pieces and my utilization is very bad.
So, there's this weird trade-off where you want high tensor parallel for the attention, but you want low tensor parallel for the MLPs.
So, you know, there's this conflict. so, what do you do?
Well, in the last few years, people have come up with more complicated system solutions that kind of decouple the tensor parallel, in the MLPs and the attention.
And so, the attention layers get one kind of tensor parallel and the MOE layers get another kind of tensor parallel, and so now you can fully decouple the parallelism, and this leads to, you know, more complicated but more effective combinations of EP and, tensor parallel as well as data parallel.
Cool.
Okay.
So, the last thing, in this space that I kind of want to end with, is, context parallel.
So, context parallel or, ring attention, is an idea of basically splitting activations in a very long sequence across different accelerators. and you can do things like, you know, pass the, activations to the, sort of device that's needed in kind of this ring-like way following the mesh topology, of TPUs.

So, ring attention, which I think was the original paper that did this, showed that this worked really well on TPUs. context parallel is a standard parallelization strategy that does this.
Both of these are used in long context extension stages and in, sort of model serving.
I'm not going to talk about it, mainly because I think it overlaps in concept to a lot of what we've talked about already.

Okay. so, this is the end of the second part and I think to put everything together, I've made a big table of, parallelization strategies including data parallel, which are the top two rows, and the various model parallelism strategies below them. and I think, you know, I've colored in red what I subjectively feel like are the drawbacks of the different methods. and the reason why I've colored things in red is to try to convey to you that there is no one strictly dominant parallelization strategies.
It's all like a whole bunch of trade-offs that you somehow have to manage gracefully in order to get a good outcome.
So, you know, for example, FSDP is great.
It's a wonderful strategy, does not help you with activation, it sort of consumes your global batch size in order to do parallelization, right?
So, it has limits.
And to address these, you know, maybe you want to compose them with some of these other tricks that you have at hand.
You know, maybe you want to do some tensor parallel to cut down on activation memory. it also doesn't touch global batch size, but if you do this, now you need fast networking and you need, sort of high bandwidth.
And also then to leverage your slower connections, maybe you want to use some of the other parallelization strategies, right?
So, you kind of see how all of these different advantages means that there's a place for one of these parallelization strategies in, you know, not every architecture but in many large-scale architectures use a large combination of these.
Yes?

Yeah, so, the question was, you know, is pipeline and, what?

FSDP applicable to MOEs?
Yes, yes, they're absolutely applicable. if anything, like the standard recommendation, is, you know, expert parallel is kind of like tensor parallel, so you want to go up to like eight or like, you know, fast connections. people don't really follow that anymore, but even then, FSDP and or data parallel in general and pipeline and tensor are used extensively in all the frontier models.
Yeah.

Good.
Okay. and you will do this, or something like this, you know, as part of your assignment, but one of the things that's cool about parallelism and I think is nice is that you can do some math, right?
You can actually say, "Okay, how much compute do I do per layer for my different sharding strategies?
How much communication do I do per layer for my different communication strategies?
And if I have different devices allocated to different communication strategies, how do these things scale?"
And then I can figure out, you know, for a combination of these strategies, you know, how do they scale?
And you can start to sort of make plots of, you know, if I have a certain batch size and I have a certain parallelization strategies, I have three different parallelization strategies here, you know, how much am I sort of utilizing my GPU, right?
The ratio of compute time to communications time.
And as long as my compute time is longer than communications time, in principle, you know, you can just hide communications underneath your computation, right? and so you can make plots like this where, you know, this is the dash line at the very top.
That's kind of the boundary of where you're efficient.

Anything above that is sort of fully utilizing your computation.
Anything below that is on the on the bad part of your roof line, right?
You're waiting for communication to do your computation.
One thing that's pretty interesting here, right?
So, I think this should feel intuitive to you based on what I've told you.
If your batch size is big enough, FSDP only is good, right?
Like if you have a batch size of 2,000, right? per chip. so, then, you know, how how far can you go?
Well, you've got FSDP doing extremely well.
You're fully compute bound, right?
But, the important thing is as the batch size goes down, FSDP only will hit a point where you're now communication bound.
Now, what do you do, right?
Well, what do you do is now, in order to keep pushing efficiencies, you have to incorporate MP, in this case is tensor parallel, to the mix, right?
Now, you can sort of push that curve out a little bit more and be compute bound even into smaller batch size regimes, right?
And you keep adding strategies to push this curve out in order to get something better.
And this is, you know, what people call 3D or 4D, I don't think there's 5D, parallelism, which is when you put all these strategies together to keep your, compute units fully utilized, under different communication topologies.
There's a very simple prescription, you know, I think up until now I've been telling you things in generalities, so it might feel complicated, but I think actually the strategy for how you parallelize is very simple, actually, which is until your model fits into memory, you're going to cut up your model by whatever means necessary, right?
And how are you going to cut up your model?
You're going to use either tensor or expert parallel for your fast interconnect.
And so, if you have eight GPUs per machine, you're going to use a tensor or expert parallel of eight.
And then you're going to pipeline parallel or zero three, you know, which is FSDP, the rest of the way to make your model fit.
And once your model fits, then you're going to just data parallel the rest of the way, right?

They used up the rest of your GPUs with data parallel, right? if your batch size ends up being too small, you can do gradient accumulation, to get better GPU utilization, right?
Very simple strategy, and this is borne out in, you know, the actual kind of like large-scale parallel training libraries that you might use. so I've added the link here to, Megatron's like guide to sort of parallelizing MOEs, but this idea actually applies to the dense models as well.
And you see exactly the thing that I said, but kind of in reverse order.
So, guideline number one is minimize model parallelism, maximize data parallelism, right?
So, use most of your GPUs to shard data if you can.
Now, if you're using GPUs, you know, stay within NVLink.
So, that's one machine, so one one box, right?
Keep expert parallel and tensor parallel within one box, right?
And then if you're going to go multi-node, use pipeline parallelism to go multi-node.
And then, you know, if you're, MOE, then prefer expert parallel.
And then finally, if you're doing long sequences, use context parallel, right?
Ring attention. so, exactly kind of what I've been trying to get across to you through but, you know, in kind of practitioner form that you see here.
Okay. the final component, actually, maybe I can pause here, in case anyone's curious about like 4D parallelization stuff.
Yeah.

Oh, yeah, yeah.
Well, I mean, sequence parallel is like often just like, you know, you do it with tensor parallel in order to reduce activations.
It's kind of more of like an add-on that you put in order to reduce activation.
It's not its own like stand-alone thing in many cases.

Yeah.

Okay.
I don't know how much I buy the loop transformer mythos, rumors.
Maybe it is.
It'd be interesting. okay, but to the question of like what how the system's structured would change, it's a good question.
I mean, I guess, it might mess with things like FSDP.
Like you can't really discard the weights.
It does have the advantage though that like, you don't have to keep getting weights.
Like it is much more parameter efficient and so much maybe much of the model parallelism stuff wouldn't be as important. it would be interesting.
It would be definitely different, for a lot of the FSDP style things cuz the FSDP strategy is really predicated on like get weights, discard weights, get weights, discard weights and in a loop you're not discarding anything.

Good.
Okay.
All right.
So, this is an old paper, but it is actually a you know, probably one of the best resources.
And also it's not like the networking fundamentals changed that much.
So, I think you know, the lessons from here are very relevant today.
And so this is a paper by Nvidia and formerly unfortunately Stanford folks Matei and Deepak who did a bunch of runs of large-scale parallel training across many different configurations to basically tell you, you know, as you're scaling up, how should your parallelization strategies change across different scales.
And you see exactly this prescription that I've been talking about before, which is you know, you have data parallel sort of maxed out and then your tensor parallel kind of increases until it hits eight and then you stop because you don't want to go any higher and then from that point on your pipeline parallel increases and increases and increases, right?

And with sufficiently large scale data parallel kind of decreases and at the very end you have a DP of six and this is because you just need this much tensor and pipeline parallel to fit the stuff, right?
And you're using the rest of your budget on data parallel.

And if you sort of put them all together, right?
Even as you know, go to you know, ludicrously large numbers of GPUs, you know, your utilization stays very flat and very good.
And this is why I think you know, you see this gigantic data center build outs.
Like the parallelization strategies people have come up with are kind of so effective and the hardware for communication is so good that even with the gigantic data centers or even cross data center training, you can get really good utilization across you know, these different regimes.

They have really nice quantitative evidence showing you know, for different batch sizes for tensor parallel, you know, what is the right place to stop and it's clearly you know, tensor parallel size of eight as you go you know, beyond that you're going to get into trouble and pipeline parallel you also need big batch sizes in order to be able to utilize the machines effectively.
That's kind of why there's a blue to orange gap.
There's pipeline parallel and the larger the pipeline parallel, you know, the larger the gap between orange and blue.

And then I think another thing that maybe you have already internalized from flash attention, but maybe not is you know, maybe you should do a bunch of activation recomputation because if you do activation recomputation, you know, very cleverly, then you can get a bigger batch size and the bigger batch size then allows you to get better utilization.
This is kind of a initially I think very counterintuitive observation that actually you should do more computation via recomputation in order to you know, end up getting better utilization because it allows you to save memory and memory can be turned into batch size.
Okay.
So, that was kind of a more quantitative view of how to pick parallelism strategies.
The last thing I want to end with is just to look at just a very few training runs and looking at the training runs maybe we can get a sense of how modern parallelization is used in practice and how it's evolving.
So, Dolma by AI2 was a 7B open source model.
Sorry, Olmo which was trained on the Dolma data set and this is in the Dolma paper for the details was a 7B model trained fully with FSDP across a whole bunch of accelerators.
I forget exactly how many. but you know, this is one example of you know, FSDP scales surprisingly well even to like many many GPUs as long as you're training a small model, you know, FSDP is actually a pretty good strategy for parallelization.
I think many 7B-ish models are trained purely with FSDP.

DeepSeek because of their systems you know, skills does do some pretty exotic and useful stuff.
So, DeepSeek V1 was trained just with data parallel zero stage one with tensor sequence and pipeline parallel as I kind of talked you before.
DeepSeek V3, you know, that's a MoE model if you remember my MoE lecture.

So, they have you know, not only pipeline parallel, but they also have expert parallel instead of tensor parallel, right?
And their expert parallel actually is a little bit exotic because they have 64-way parallelism.

So, they group eight different machines together and that's their expert parallel sort of domain and to enable this large expert parallel, they basically use the same tricks that they have from their pipeline parallel pipelining in order to try to make sure that expert parallel doesn't sort of like have low utilization period.
So, this is actually a quite a complicated trick that they do to get these very large expert parallel splits.

Another example Yi from Chinese open weights training setting, they use once again zero stage one and tensor and pipeline parallel, right?
This very much makes sense.
It's a it's the classic data parallel tensor parallel pipeline parallel combo.
And once again, once you go to MoEs, you replace the tensor parallelism with expert parallelism, right?
They serve similar sort of goals, but expert parallelism is just a little bit more efficient.
On the other extreme of a really big dense model, you have Llama 3.
The Llama 3 report I like quite a bit because they're one of the few reports that have like literally a full breakdown of their parallelism strategy, not just like in the abstract, but across all the different phases.
So, the cool thing that you see is they have a pre-pretraining stage like a little warm-up phase at the very start.
You can ignore that first row.
And then they have their main sort of pretraining phase in the middle row here and this is a very standard parallelization strategy.
Tensor parallel of eight, one sort of context parallel, 16 pipeline parallel and 128 data parallel, right?
Like big data parallel splits.

And then at the very end because they want a long context model, they do long context extension and for this they crank up the context parallel, lower the data parallel and that's kind of how they parallelize this very memory hungry stage for long context extension.
And notice you know, we're in tensor parallel not expert parallel then because Llama 3 405B is a gigantic dense model.
It's not a MoE model.

And as a side note, right?
Like one thing that I don't talk about, but if you for example become an infra engineer at one of these companies, you'll notice is you know, GPUs fail all the time.

Apparently in during Llama 3 405B training, GPUs failed 148 times and so you need you know, lots not just like fast parallelism, you need redundancy in order to be able to deal with all these like kind of horrible things that can happen to your training.
So, there's a distributed systems challenge as well.

Gemma 2 which is a Google open source model has FSDP plus tensor parallel and sequence parallel and uses these two strategies only and the reason why I sort of brought up Gemma 2 is as far as I understand this is basically a realization of the Google claim that you know, for TPUs really you don't need to do pipelines.
You just take a really big toroidal mesh and you do tensor tensor parallel over that big mesh network that you have.
Unclear whether this can like scale out forever.
That has always been a little bit less clear to me, but at least at you know, the Gemma scales, this is definitely the case.

Finally, I took a look at some of the models that were trained this year like Mistral 8x22B and during my investigation, you know, one of the things that I discovered was actually you know, if you're interested in like model training and parallelism configurations in the wild, Nvidia has a repository called Megatron-Bridge where they sort of release a lot of you know, recommended training configurations for a lot of different model sizes and model settings and you can kind of take a look at what they end up saying is the right model configuration for like 8x24 22B MoE or DeepSeek V3 model.
And so for Mistral, you know, they do expert parallel of eight, pipeline parallel four and an additional tensor parallel of four.
This one for the attention layers and you kind of see you know, this follows the prescription of keep your expert parallel roughly around eight.

Nemotron 3 Super follows kind of the DeepSeek V3 model, has a bunch of expert parallel as well as context parallel because they were doing extensions at least in this article.

And then the last one that I want to talk about is Qwen 3 which follows the DeepSeek recipe, which is fairly large amount of expert parallel 32 and they have eight pipeline parallel and two tensor parallel to split up kind of things like the attention matrices.

It is actually kind of interesting to see additionally the different kinds of benchmarking that Nvidia folks have done for you know, parallelizing different models.
And you see I won't go into details of what this is.
This is in the Nvidia Bridge repo if you want to take a look.

There's all sorts of different like sub configurations that you can pick even within tensor parallel that can significantly affect the performance of these.
So, the systems integration is quite significant.
So, to sort of put everything together, you know, I kind of decided maybe I would build an overview.
Really the thing that's common to this is you know, as much as you can, all the models use data parallel.
They maximize the data parallel domains to the extent that they can.
Tensor parallel almost always remains below eight and expert parallel can sometimes be big now and I think that's partially because of DeepSeek V3 and a lot of the infra that they built for large-scale expert parallel training.

Good.
Okay.
So, to put everything together, right?
The highest level point right of this lecture is we need to think about multi GPU multi node, maybe even multi data center parallelism.
And to think about efficiency in that regime, right?

We need to really bring to bear all the approaches that we have because you've got fast links, you've got slow links, you've got techniques that make use of batch sizes and all these different resources that you all want to consume as much as possible.

But given all of these, it turns out that there's fairly simple rules of thumb for combining all these different forms of parallelism to get, you know, in the end what is effective effectively like full utilization of your compute hardware.

Great.
And next week I think we're talking about scaling loss.

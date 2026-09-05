# Lecture 6 字幕

So, welcome back everyone.
So, on Monday, Tatsu did an excellent job of describing a high-level overview of GPUs and how to think about performance and all the quirks that come with GPUs.
This lecture is going to be a continuation of that where we're going to dive more deeply into the code, write some Triton kernels, and as well as do some benchmarking profiling.
So, just to start, just to refresh what's going on with GPUs, here is a simplified diagram of what a typical GPU you know, looks like.

There's memory and then the actual GPU chip.
And just to keep in your head what are the characteristics of a GPU?

So, of course, we're talking about Nvidia GPUs, but Tatsu also talked about TPUs and there's AMD and so on.

But focusing on Nvidia GPUs, we've had multiple generations from A100s to H100s to B200s.
And in each generation, you have a GPU, it has a number of streaming multiprocessors or SMs.

And the number of SMs is, you know, about 100 between 100 and 200.
So, that hasn't really changed that much.

And then within individual SM, there's a set of registers.
So, B200s have 65,000 registers for a total of 256 K per SM.
And this also hasn't changed that much.

In addition, there's a L1 cache as well as shared memory.
So, remember these are the same memory, shared memory you can control L1. you can't.
This is per SM.
And this is like in the same order ballpark size.
And then there's a L2 cache.

And this is not per SM, this is for the whole chip.
And this is a bit larger.
And then finally, you have a high-bandwidth memory HBM, which is large.
And you see that this is the number that's actually going up quite a quite a bit.

Okay.
And then you can in addition to size, you can think about what's happening with the bandwidth.
And essentially, it's you know, kind of inversely correlated.
So, registers are very fast, L1 is slightly less fast, L2 is less fast and high-bandwidth memory is the slowest.
Although 8 terabits terabytes a second is still, you know, not not that slow in the grand scheme of things.
So, this is the kind of main hierarchy you should have in your head.
Memory, large memory is slow and far, but big.

And fast memory like registers and L1 resides on SM, it's local and it's fast, but small.
Okay?
That's the what the environment we're dealing with.

Okay, so then how do you program a GPU?
So, the program model is as follows.

So, there's notions of threads, which each thread executes a piece of code on a small part of the data.
So, each of one of these arrows, you can think about it as a thread.
These threads are organized into thread blocks.
Also known as a concurrent thread arrays or CTAs.
And this is a group of threads.

And then finally, you have you have a collection of thread blocks, which forms the grid.
So, when you launch a kernel, you're basically launching a grid of threads and thread blocks to all parallel you know, simultaneously do some you know, computation.

Okay.
So, this is a simplified, there's some other things that are happening.
For example, there's H100s and B200s also have thread block clusters, which are clusters of thread blocks that enable some amount of distribution memory.
Also, B200s have tensor memory now for tensor cores, which is somewhere between the registers and shared memory.
Some of these things are kind of invisible to the programmer, but they are in and in the hardware.
But we won't worry about them for this intro lecture.

Okay, so one thing you might ask about is why do we have thread blocks?
Why can't we just have a grid of threads and each thread just takes a piece of data and does something?

And this would be normally fine if all you want to do is element-wise operations, which we'll see.
For example, GELU is a activation function that applies element-wise.
And threads are pretty natural.
Each thread processes one element.
So, it's like FI for I equals you know, ranging over your data set.
But for operations that are involved communicating with you know, threads such as softmax or matrix multiplication, this view isn't really enough.

And the reason is that well, it could be enough if you were willing to pay the cost of writing and reading HBM.

Because then you can still have, for example, in a matrix, we'll see later, you can just have every element in a cell, it goes and computes that element of the matrix.
And it can just read and write HBM.
But as we've noticed, HBM is very slow, so this would not be a good strategy.

So, instead, what we should do is use shared memory.
Which is local to a SM.
And what the thread block allows you to think about is a collection of threads that are all going to access the shared memory.
So, So, consequently, you can think about a thread block, one of these thread blocks is being scheduled on one of these semantic not semantic streaming multiprocessors and it does its thing.
And what it's going to do is it's going to read a bunch of data from HBM and then process it.
Where the processing might involve communication between you know, the threads via the shared memory and then writes it back out.

Okay?
So, this is kind of a critical piece that comes up.
Later, we'll talk about tiling and that's the whole kind of game here.
Okay, in Triton, in fact, which we'll see as the primary way that we're going to write kernels these days, we're going to think natively in thread blocks.

And you know, I think once you get the hang thing of hang of thinking in thread blocks, this makes a lot of sense and makes your life a lot easier as we'll see.

Okay, so the programming model is fairly you know, simple, I think.
There's threads, thread blocks, and grids.

Okay, so where I think things get a bit more you know, complicated is the interaction between the programming model and the hardware.
So, the programming model is actually very nice.
It provides abstraction of the hardware.
You when you can write a kernel, you just all you have to know is that there's a bunch of thread blocks, you define them, and then define what computations each thread within the block or the thread block has to do.
And those are just like writing it's like writing Python.
So, that part isn't hard.
And in fact, if you don't care about if you just care about correctness, that's all you need to know.
But in practice, the performance is very sensitive to the hardware.
And so, you need to really deeply understand the hardware to obtain high performance.
In fact, the whole reason we're talking about GPUs and kernels is that you're trying to squeeze out performance.
So, there's sort of these two levels here where you're trying to understand the computation and that is only a part of the programming model, but you know, how fast it runs is going to be strongly dependent on the hardware.

Okay, so I'm going to give you some examples of why this matters.
And some of this will be review from what Tatsu you know, talked about, but hopefully it'll reinforce some of these concepts.

So, I think there will be around five different examples.
And you know, this is just to give you a flavor of the considerations.
So, there's something called warps, which I didn't talk about.
So, in the very simplified view, warps are not really part of the this clean picture of the programming model.
You have threads and thread blocks or grids.
Technically, you can also exceed the warps, but you don't you know, have to when you're just you know, programming.

So, what is a warp?
So, each thread blocks, remember, is a collection of threads.
But these threads are actually grouped into warps.
And you know, there's basically 32 threads per warp.
So, for example, if you have 64 threads in a thread block, there's two warps, the first 32 and the second 32.
And as Tatsu mentioned last time, all threads within a warp, all of these Ts, must execute the same instruction in lockstep on SM.
So, every cycle, they have to execute exactly the same instruction.
And control divergence is when different threads within a warp need to actually execute different instructions.
For example, if you have branching, if A if something then A else B.

And then what happens is that you can only do the A's in your threads in your warp, and then you do the B.
So this gets kind of sequentialized.
So this is bad and inefficient, and that's why branching is something you want to generally avoid.
One thing that's I think really cool about warps is that a SM actually is running multiple warps.
There's a warp scheduler, and there's a bunch of you know resident warps that are about to run.
Each warp has some you know and threads have registers.

And you this SM can switch between them with you know zero cost.
Right?
This is not true in general like on a on a CPU, but this way that it's designed is to hide latency.

So this is important because one of one of the warps is for example reading from HBM, which remember is very expensive.
Can take like a you know 100 cycles or something.
You don't want to just sit around waiting for that warp to do nothing.
You switch immediately to another warp where it can actually do some tensor core operations.

Okay.
So that's one thing to note about the warp.
And another reason why warp comes up is this idea of occupancy, more specifically warp occupancy.

So each thread just you know the hardware constraint says that each thread can use at most 255 registers.
Okay?
And so what happens is that the SM has a fixed number of registers.
So the more registers each thread is using, the fewer threads you can have.
That's just you know math.
And that is called that can reduce your occupancy.

But that's not necessarily bad because if you have fewer threads, but each thread is doing more work, that can actually be good.
So you know occupancy is something that you can measure, but it's not necessarily the larger the better because there are some other trade-offs here.

In fact an example where you might want to you know have fewer threads is this idea of thread coarsening where let's say you have a you know element-wise operation, and you're going to you can have one thread just perform work on one element.
Okay?
So then you have a lot of threads.
But you can also say each thread processes multiple elements like a constant maybe eight.
So that gives you fewer threads, which makes scheduling and these things easier, but each thread is doing more work.
So if your threads are very light, maybe you want to fatten them up a little bit.

Okay.
So just as an example of what might happen here.
So suppose you have a block a thread block with 128 threads in it.
And each thread is using 160 registers.
Okay?
So there's a hardware constraint.

So the B200s have a maximum of registers of 65,000 per SM.
And it also has a constraint that says you can't run more than 65 64 warps at a time.
So then you can go and do some simple math to compute how many what is a you know occupancy here.

So you know number of registers per thread, this has to be less than 255.
That's fine.
The number of registers that you're using per block is the number of threads per block times number of registers per thread.
That's 20,000.
That means you can run at most three blocks on your SM concurrently.
And then that corresponds to 12 warps.
And because the number of maximum warps is 64, that means you have a occupancy of you know 18%.
So you're only using running 18% of the total number of you know warps you have.
And this is because you have a lot of register use per thread.
Okay?
So this is an example of where I guess this is sort of memory is kind of constraining you in terms of how much compute you can do.

Okay.
Here's another example, something called bank conflicts.
This applies to shared memory.
So remember shared memory is like L1.
It's on a SM.

And just the way the hardware works is that shared memory is divided into 32 banks, each one 4 bytes wide.
Okay?
So these are my 32 banks, and this is my memory here.
Each bank has many elements here.
So there's a constraint that says every you know clock cycle, each bank can only be accessed by one thread, at most one thread.
Assuming that's not a location.
So for example, if you can't access this location and this location you know at the same time.
Which means that if you have multiple threads that are trying to access the same bank, the accesses have to be serialized.
And this leads this is called a bank conflict.
In the worst case example, imagine this were a matrix that was kind of laid out like this, and you had some operation where 32 threads, you think wow this 32 is so great I can massively parallelize this, and you try to all access this first column.
They're just going to just wait in line.
And this is a 32-way bank conflict, which is the worst possible setting you can be in.
And now you could say well okay, why do you do that?
Just like access rows of course, but you know this is unavoidable because for example, if you're doing a matmul, you have to access rows of one matrix and columns of the other matrix, and sometimes you do transposes.
So you can't always just get away with choosing the order in which you go down. you know you know for element-wise operations it's fine.
You can go in any kind of order.
But if you're doing matmuls, you can't control like for every matrix which row major or column major you're doing.

There's some solutions here, which I won't get into.
Something called swizzling, which arranges your shared memory so that when you're going through, you can kind of avoid bank bank conflicts.
Okay.
So that's you know another you know consideration.
And when you're you know profiling, you can look at the bank conflicts.
You can look at occupancy, and you can kind of see you know what's what's happening.

A final note, which are actually two more things.
So memory coalescing, which talked to you talked about, so I'll just really quickly remind people what this is about.
So when you have 32 threads in a warp, and they try to access HBM, the memory accesses actually get combined into transactions of 128 bytes, which are called cache lines, and it goes and fetches it all at once.
So imagine you have memory laid out like this.
This is 32 wide.
And you know in the in the best case, which is called full coalescing, all the threads are accessing the same cache line.
So you have thread one accessing M00, thread two accessing M01, and so on.
Which means that you know at all at once you're going to grab this entire cache line.
Right?
And whereas if you go down the columns, then then you know you're going to fetch a lot of memory here, which you're not going to use.
Same for the second row and so on.

And so this is similar to feel similar to bank conflicts, but it's a very different constraint.
And this is about shared memory, and this is about HBM.
Okay?
Okay.
So final thing, block occupancy.

So thread blocks are remember scheduled onto SMs.
But you know we live in a you know finite you know logically you can define as many thread blocks as you want, but physically on chip you only have a certain number of SMs.
For example, 148.

So if you launch 160 thread blocks, then you can only schedule 148 of them, and then you have to wait until they're done, and then you schedule the 12.
But what happens if you schedule 12 is that there's a bunch of the SMs are just not doing anything.
So that's called you know low occupancy when the last wave of thread blocks gets fewer than the total maximum number of thread blocks.
Okay.
So in general, you know maybe it's a good idea to make the number of thread blocks divide the SMs.
So maybe it's just just to summarize here.
You know there is a very elegant programming model where you have a grid of thread blocks, and the thread blocks have individual threads in them.
And you know at and in terms of the memory, HBM is global to everyone.
Shared memory is local to a thread block, and registers are local to a thread.

But again, all the details of the hardware, warps, bank conflicts, memory coalescing, occupancy really determine you know performance.

So I think you know and many of these details are sort of hard to know because you don't I mean, the profiling tells you a bunch of information, but you know, you have to know exactly how many SMs there are and exactly the sizing of everything and that's gets you know, sometimes the scheduler does something you don't really have control over what it's doing.
So, it's a lot messier than the programming model.

Okay.
I will stop there for questions.
Yeah.
Yeah, so quick question.
So, can is there a scenario in which the kernel can share an SM or like for example like in the problem we see like 148 SMs and 165 blocks, right?
And we have to like launch in the two different ways.
Is there no way in which a block can like share an SM?

Yeah, so the question is can a block you know, share an SM?
I think the issue is if you're doing things right then the you know, the I guess it depends on the block.
This is if you have a block that's for example using most of the tensor cores on the SM then putting another you know, block there isn't really going to you know, speed things up.

I think fundamentally there is this sort of jagged you know, problem here of unevenness.
The because the blocks have to stay together.
So, you can't like take this and like spread it out over here.
You can define your you can I think the thing the thing you would do is to change your block size.
So, you should change the number of blocks.
So, you don't get this kind of tail here.

Any other questions?

Okay.
So, let's move on.
So, that was hopefully you guys are you know, getting more comfortable with GPUs now.
So, now I'm going to talk about benchmarking and you know, profiling.

I'm not going to actually maybe say that much in terms of content, but I do want to emphasize the philosophy here which is here's a recipe for success.

You benchmark and profile your code.
You make changes and you benchmark your profile your code again.
And the reason I'm doing benchmarking and profiling as opposed to teaching you try and earlier is because you know, you should always just measure what's going on in your code and find figure out what the bottlenecks are before you start you know, writing kernels.
Okay, so benchmarking is basically how long things take.
And it gives you just this end-to-end time.
It doesn't tell you where things are you know, spent, but nonetheless it's pretty useful because ultimately that's the thing that you care about how long you know, things are running for.
And it also gives you because it distills things into one number, you can see how things are scaling.
Let's say with dimension.
So, there's a you know, nice tool for benchmarking, but because this class is language models from scratch, I'm going to do it from scratch.

But you know, I think I'm doing it just to highlight a few gotchas with you know, benchmark.
So, let's say we have this operation matrix multiplication.
So, run operation two is this wrapper that basically instantiates two random square matrices of size dimension by dimension and then it returns a function to perform the operation.
Okay, so matmul if you call this basically does the matmul of those two random matrices.

The matrix random matrices are already generated it's just like multiplying the two.
Okay, so how do you benchmark?

So, the naive thing to do is just start time, run it and then stop time, but you know, there's a few things.
One is that always remember to do you know, warm up.
I think I mentioned this you know, on the first on the second lecture, but I think it's worth emphasizing.

And this is because if you have some things are lazily compiled and you want to just make sure that thing is that is that time doesn't factor into because most of the time you care about how fast something is because you're going to run it over and over again.
So, the initial conditions don't really matter.

And then you often want to time it multiple times because there is some variance.
The kind of the proper way to time things is to use the these CUDA events that a start event and an end event which you can kind of call record on, actually do the computation and then hit this the end event record.
And then remember to you know, synchronize to wait for the CUDA threads to finish because everything on the GPU is happening kind of async and this is a synchronization barrier.

And then you record the time.

Okay, and then you can do this again and then again and then you know, here we're just taking the average.
I think you might want to do you know, if you are being very particular, you want probably want to maybe look at the whole distribution the P95 or whatever, but we'll just do the average here.
And then one thing that you can do with benchmarking is you can let's say scale up your your matrices and see how the time you know, changes and you can see that in this case matrix multiplication you know, as expected it should grow you know, cubically, but you know, notice that there is this floor where up until you get up to like almost 2000 dimensional matrices things are basically kind of a you know, constant, right?
And this is because as we've kind of discussed the shapes of the you know, these you know, GPUs are built for fairly large matrix multiplications and if you have like a 2 by 2 matrix [clears throat] it's going to be very kind of inefficient.

Okay.
So, really quickly on profiling profiling tells you where time is actually spent.
Hopefully all of you are familiar with profiling and are doing it.
Maybe less obviously profiling even if you don't care about the time helps you figure out what's actually happening under the hood because especially with these high-level languages you write some code and then it runs and you get some result and sometimes it can be just like good to understand what's actually going on.

So, PyTorch has a built-in profiler in your assignment you're using site which gives you more details, but we're going to skip that for the in the interest of time.
So, let's start with just you know, the if you add two numbers and or sorry two tensors in PyTorch.
Again, run operation creates two random matrices and applies operation.

So, in profiling I'm warming up and then I'm just you know, putting this context and the profiling context and doing the run.
And then let's look at what the profile looks like for just A plus B in PyTorch.

So, if you're not you know, normally doing PyTorch you probably don't think about it.
It's like okay, well, these two tensors just get added.
So, what's actually going on underneath the hood?

Right?
So, if you look at the things that it's it's calling there is this you know, long name kernel at CUDA functor add.
So, this is basically you know, a kernel that adds two two you know, tensors and you know, the times aren't going to be that interesting because I'm only adding.
So, that's going to take 100% of the time.
But this tells you that well, underneath the hood there is this thing called add.

What about matmul?

So, I'm going to you know, in PyTorch I'm doing A at B.

Similarly, there is this you know, long name that describes this particular matmul you know, kernel.

color F32 F32 64 64 16 and so on.

Notice that if you change the dimensions.
So, now I'm doing a 128 by 128 matmul.
You get actually a different one.
If you look closely that this is 64 64 16.
This is 32 32 16.

Okay, so underneath the hood you know, PyTorch it looks like you're doing just add, but underneath the hood there could be all sorts of things that are you know, happening.
So, observations.
Here you can see which CUDA kernels are actually being called.
These are generally the ones that are have the long names and different CUDA kernels are invoked depending on the you know, tensor dimensions.

The name actually tells you something about the implementation as well.
So, this name So, CUTLASS is the NVIDIA CUDA library for linear algebra.
SM100 corresponds to the Blackwell you know, architecture.
So, this is a kernel that's specifically designed for Blackwell.

This is f32 and then 64 64 16 is the shape of the tile, which we're going to talk about later when we talk about tiling in the context of matmuls.

Okay.
So, uhm last benchmarking profiling, just remember to do it.
Uhm I think we make you do it on the assignment, so you have no choice.
Uhm okay.

So, let's apply this uhm to another example here on the GELU.

So, uhm So, remember the GELU activation you know function is uhm this function, which is a typical non-linearity that's used.
Often it's approximated with this uhm you know tanh approximation, which is more compute-friendly.

Okay.
So, uhm so naively, uhm if you implement the GELU, you can do it in PyTorch uhm just like this.

So, you take a tensor, you just you know take this equation and you just put it into PyTorch, okay?
Uhm that's fine.
And you get some number.
Uhm PyTorch also has a built-in uhm you know if you call NN functional GELU, uhm you can get that version as well.
So, uhm and you can check that they're they're the same.
You can run uhm the two you know GELU versions and make sure that the answer is the same for a random input.
Uhm there's also something that uhm you know people probably some have discovered.
It's it's a if you know haven't encountered this, this is a important thing to know, is that you can take any PyTorch function, you call torch.compile it on it and it generates another you know function uhm and it does the same thing.

Uhm okay.
So, we have three uhm horses in this race.

We have the naive uhm implementation.

We have the built-in implementation.
We have the compiled implementation.
So, let's we can benchmark them.
Naive takes three uhm I guess this is 3.75.
Built-in is much faster and uhm compiled is much faster as well, but not maybe as quite as fast as built-in.
Okay.
So, so what's what's happening here?
Like why are why are these things you know different?
They you know all compute the same answer, but they have wildly different performance you know characteristics.

And so, here we can pull up the profiler and see what's actually going on underneath the hood.
So, if you look at the naive uhm GELU here, uhm and just do a profile, uhm this something's wrong with this view, so I'm going to go to uhm this.
So, you see that uhm the profiler shows uhm you know how time is being spent.

There's a bunch of different kernels on this binary functor, unary, add, uhm a tanh is a is a you know kernel here.
And this corresponds to the fact that in PyTorch, when we write the PyTorch expression, you look at the computation graph and each you know primitive in the computation graph is actually realizing a kernel.
Right?
And the reason this is slow is that you know when you launch a kernel, you know the kernel has to read from HBM, pull it all the way over to your SM, do the computation, and write it back.
And then the next kernel picks it up from HBM, you know, and then writes it back and so on and so forth.
So, you're doing a lot of reads and writes back and writes back and forth because between kernel invocations, you know, things have to go back to you know HBM.

>> [clears throat] [snorts] >> So, and if you look at what the built-in is doing, this is actually not that interesting.
There's this GELU CUDA kernel implementation, which is a single kernel that just implements the GELU.

And so, why does this exist?
Well, well, because you know, people use GELU, so someone wrote a kernel for it and put it in the standard library.
Uhm I mean, this is how you know, there's no nothing magical.

So, com- compilation is really interesting here.
I'm not going to say too much about uhm how it works, but the is really I think uhm a fascinating uhm you know topic where you can take a naive implementation, which you know remember in PyTorch it has a computation graph, and run a compiler, which basically if you look at what's underneath the hood, it's just a single kernel.
And this is because it's it's figured out to uhm you know, look at the computation graph and essentially write that kernel in uhm in Triton.
So, you can see that this is actually a Triton kernel.

Okay.
So, naive implementation multiple kernels requires multiple reads and writes to and from HBM.
There's no you know kernel fusion here.
Uhm this is slow.
The built-in and compiled version, there's one kernel.

The basically all the operation in the GELU have been fused together into one kernel.
So, you read from HBM once, you write to HBM once per you know element.

Uhm and you see that the compiled kernel is a you know is a Triton kernel.

Okay.
So, that maybe is a good segue to talk about what this Triton thing is all about.
Yeah.
So, what you know what is kernel is this the built-in?
Is it also in the indices go straight out of CUDA and So, I don't let's see.
I mean, I guess this says CUDA kernel implementation, so I imagine someone wrote it in CUDA.

Okay.
Any other questions?

Yeah.
Why is why is Triton kernel faster than this one? is CUDA kernel What So, why is a Triton kernel faster?
So, a Triton kernel is actually not faster in this case.
Oh, sorry.

The compiled kernel is the is one Triton kernel and this is slower than the built-in.
Yeah.

I think last year when I did this, it was actually closer.
Uhm or but you know, these things change and it's very hardware-dependent and you know, I think none of this is like terribly optimized.
This is just giving you a the general idea here.

Okay.
So, let's write some Triton kernels.
Uhm So, remember our programming model.
You have a bunch of threads organized by thread groups by thread blocks and there's a grid of thread blocks.
Okay.
And uhm if you were to you know, write in CUDA, which was originally developed by NVIDIA and that's been for years the thing that you do when you write kernels, uhm the mental model is what does each thread do?
So, you write a bit piece of code, which essentially has some ID that identifies which thread you're talking about and it just like executes the code.
So, the nice thing about this is that it's very closely related to what is actually uhm happening underneath the hood.
Uhm and it's you know, fine-grained gives you fine-grained control, uhm but you know, there's cons here, which is that remember all these threads uhm uhm are in a thread block you know, some operations require them to communicate.
So, what has to happen is that they have to synchronize and uhm like they read from HBM all at once and they have to synchronize and do the computation and then or uhm and you have to basically do that you know bookkeeping.
So, if you were doing all element-wise operations, you know, this CUDA is just fine.
It doesn't really matter, but as you get more complex operations, then Triton provides some value uhm of the abstraction.
So, Triton developed by was developed by OpenAI.
I think by now it's been pretty you know standard.
You basically specify what each thread block you know does.

Uhm generally it's powerful enough, especially for you know, this class and you're getting started.
If you really go and want to exploit every single new feature of the latest hardware, you know, it might not give you the full flexibility, but you know, uhm you know, let's not worry about that.

And the conceptual framework to think about in Triton is that you're think about what does a block do?
A block is going to load data into shared memory, operate it on, and write it back to you know global memory.

So, in some sense, these blocks are intermediate point between uhm thinking about what individual elements are doing, as well as thinking about the general operation.
And so, in PyTorch, you basically define these huge matrices and you say multiply them together and that's the sort of the atomic operation and a lot of what you're thinking about is how do I get things into big matmuls, right?
And at the at some level, Triton is sort of you know, hybrid between that and you know, the individual elements.
As we'll see.
Okay, so let's start with a the value example here.
So, let's define a 8,000 dimensional vector and start writing some Triton.

Okay, so, Triton is basically going to be you know, you write you know, Python.
How many of you have you have written Triton before?
Just as a show of hands.
Okay.
Okay, so hopefully this I'll try to not that many, so which is good because then you won't be bored.

okay, so you know, so this is normal this is normal PyTorch, right?
There's no Triton here, but I'm just preparing.

And so what I'm going to do is I take this tensor and I'm going to allocate an output you know, tensor.
Okay?
Because in Triton we're not thinking functionally anymore, we're just thinking about moving you have to explicitly read and write.
So, there's no like returning value.
So, I'm going to allocate the output tensor which I'm going to the kernel is going to write to. and then so this tensor is can be you know, arbitrarily big, right?
And I can't generally fit this all into one SM because you know, it's just it's just too big. so, I need to break it up into blocks.
So, what I'm going to do is you can think about you know, this X is being this array.

I'm going to chop it up into blocks, so the total number of elements is 8,000.

I'm just going to set the block size to 1024 you know, for now.

And then I have eight blocks.

Okay. so, then I'm going to use this sort of weird syntax to call the kernel. so, Triton gelu kernel this basically in brackets tells me the essentially the shape of the grid.

So, basically this says the grid has num blocks you know, blocks. and I'm going to for every one of these blocks invoke this function Triton gelu kernel which I'll talk about in a bit.
I'm going to pass in an X, Y, num elements and the block size.

Okay?
All right, so unfortunately I'm not going to be able to trace through this, so I'll just show you that what the code looks like here.
So, let me get rid of that.
Okay, so this is probably the kind of the simplest kernel you can imagine.
So, when you are looking at the Triton gelu kernel.
Now, this is you know, before we had X and Y.
So, now these are pointers, right?
You can think about these are just like integers, they're addresses. so, you have to get comfortable with that. and then we have the number of elements and the block size which are basically passed and actually passed in from here.

Okay, so the way to think about is that for every block we're going to have this function being you know, called.
And what happens is the first thing I'm going to do is I'm going to wake up this you know, block wakes up and say who am I? well, the program ID is basically the PID is basically identifying the block.
So, PID would be zero for this block one, two and three here.
Okay, so now I have to figure out what data I'm going to operate on.
So, that's PID times block size.

So, this is the this is the offset into X pointer.
So, if PID is zero, then start is zero.
If PID is one, then start is block size.
If PID is two, then it's two times block size and so on.

Okay?
So, now I'm going to figure out the span I'm operating on.
So, offsets is start plus this is the you know, the Triton you know, library A range zero block size which gives conceptually gives you the integers zero through block size minus one.

So, offsets is going to be essentially you know, for let's say block one, it's going to be block size, block size plus one all the way to blocks two times block size minus one.

Okay, so you know, in this case num elements divides block block size, but in general that's not the case.
So, you'll often see in Triton code there's this masking which says well, sometimes if the let's say the tensor only goes up to here, then I'm going to form a mask which is going to be true up until that point and false after that point.
And for blocks that are don't are not the final block, it's going to be just all ones.

Okay, so now I've done the setup.
What I do is I read and this is basically pointer arithmetic.
So, X pointer remember is the integer that specifies the memory location of where X is.
I'm going to add offsets to that which gives me the first block size number of elements there. you know, according to the mask if I'm masked out and I don't you know, I don't I don't read it. and then now I think you know, you can just think about this as a you know, a vector. and then you do your normal computation.

And then you get you know, Y.

And Y is the same size as X and then you do TL store Y pointer plus offsets on Y and the mask.
Okay?
So, this load loads from HBM does some stuff and then it writes back to HBM.

Okay, I'm going to stop there and take any questions about this first kind of Triton kernel. yeah.

Yeah, so the difference is what is the difference between CUDA?
And that it's for this element wise it looks pretty much the same, right?
And in fact, CUDA's even I think simpler because it really is element wise.
You wake up and you I identify the thread and then you just operate on that element. now, the only thing here is that it's sort of like the vectorized version where you have a block and you operate on that block. later we'll see that operating on blocks is doing more than if you do more something more than element wise, then CUDA's going to be a lot more annoying to work with. there's a bunch of questions.
Yeah, back there.

So, how is this related to if you want to use the tensor units? so, I'll later show you what this code actually compiles down into and maybe we'll I'll get back to that.

But the short answer is that you don't control that.
Like the hardware figures out what where to put things.

Yeah.
Can you actually walk through what's happening at the level from HBM to shared to register?
What's like the step-by-step where things are actually going?
Yeah, so the question is what is actually happening when this the same end is executed. you know, so the short version is that this is in some sense a lie, right?
It's not like the GPU actually calls the you know, the Triton library on the GPU and it's like actually executing this code.
This is basically for our consumption to you know, specify the computation.
The compiler takes this as we'll see later and writes it into something called PTX and then it will actually do the you know, the work.
But conceptually like conceptually so that's mechanically what's happening which I'll get to later. if you're asking about the conceptual question the way to think about this is that you know, this ish X pointer is you know, a memory location in HBM.

And this basically specifies a range of memory locations and load takes those memory locations and returns the data associated with them.
And it gets you know, here I have a local variable I called X.
In practice this is going to be generally you know, a register or shared memory.
Actually, you know, you don't actually you know, Triton kind of figures out what to what to do there.

the thread level and this is like so when this is invoked, it's not really clear like did it did I did I line up what is going to be in shared and going to be at the register level beforehand or is it happening now?
Like, which is like way late.
And so, that's just we're all the threads are sitting idle to get memory to flush in.

execute so that it's not we're not sitting there twiddling our thumbs just waiting for data to come in from HBM.
Yeah, yeah.
So, the question is like, when does this actually get executed?
Doesn't this like block? let me try to come back to this question while I show you the PTX and maybe hopefully it'll provide a bit more context on what's happening.
Okay, any other questions?

Okay, so this is a all the kernels are going to look something like this.
So, I do want to make sure people understand you know, the just the general form, which is that you have generally your inputs, your outputs, you wake up, you figure out which index you're going to look at, you read, you do some stuff, and then you write to HBM.

Okay.

So, Okay, so let's talk about PTX briefly.

So, let's see, how does this work?
Does this link work?
Okay. so, when you know, write Triton, the compiler generates PTX code, which is this intermediate assembly language for new GPUs.

You know, I am obviously not going to go through all this, but just to give you the flavor of what this you know, you know, looks like. let me actually start with obser- observation.
So, a few things.
So, you know, this is now what a thread is actually doing. not a thread block because that's been kind of compiled, you know, away.

And so, you know, if you look at a few few notes here.
So, this ld.global is basically saying load from HBM into some registers, and the registers are denoted as like the %r* are integer registers, %f* you know, floating point, you know, registers. and then you have statements like, you know, move zero into R5, move zero into R6, and so on. and then you have multiplication, you're going to multiply this register by this, you know, constant, and then put it in you know, this.
So, this code gets executed, and at the bottom you should see you know, global store.
So, this is writing back into the HBM.

So, this gets executed.
This is actually the code that gets executed on, you know, a thread. you know, and Triton is basically a you know, a layer above you know, above that. one other thing to notice is that, you know, you see all these you know, kind of blocks, and what's going on there.

And what's going on there is what you know, I'm alluding to thread coarsening, which is that this is one thread, but rather than processing a single element, it's actually processing eight elements.
So, the compiler decided that, well, actually this thread is pretty lightweight, it doesn't do that much, so let's just try to thicken it up a little bit.

so, you know, looking at PTX can give you some sort of appreciation for you know, what's what's going on underneath the hood.
Yeah.
So, it makes this for each individual So, all of these are So, probably the same one for the whole thing. so, the question is, does it make this for each thread?
So, this is compiled you know, once, and it's the same piece of code that each thread runs.
And the way that the thread distinguishes itself is that this piece of code gets passed basically the you know, the thread you know, ID. so, here the %ctaid.x is the block index, and %tid.x is the thread index.
So, this basically if this piece of code is running, this tells you which block I'm in.

And T- %tid.x tells you which thread inside that block I'm in.

Okay.

Okay. any other questions?
And I'm trying to figure out a Okay.

Okay, so that's a generally a flavor of you know, your first Triton kernel kernel load from HBM, compute, write back to HBM, and then we saw the PTX, which is like kind of the grungy what actually happens underneath the hood.
There's still a lot of things that are not specified in the PTX.
Like, for example, which SMs things are operating on, and the you know, the warps and everything, and that's a lot of those are kind of hardware controlled, so you don't even you know, see.
Yeah.

So, >> [snorts] >> PTX code is generated by the compiler, not something that you would normally go write in.
It looks like it's assembly code.
Yeah, so the question is, do is PTX this is generated by the compiler?
So, there are people who do write PTX. if you really, you know, think you're better than the compiler.
And I think, you know, the NVIDIA compilers are generally pretty mature, but some other, you know, accelerators that are less developed, I think, you know, sometimes you just have to reach in and actually hand-hold a bit more.

But generally, yeah, you shouldn't need to do that.
Yeah.

So, when I look at the PTX, so I when a warp will get scheduled onto an SM, I'll get to that TF.load, and it's almost like a CPU trap call, where like I'm just waiting for like, you know, some you know, something to happen, and like now I'm just I am twiddling my thumbs, so some other warp will get scheduled over me, and then they will do operations.
Now that the TF.load is done, then I'll reschedule that warp, and then I'll continue on.
Is that kind of Yeah, that's that's right.
So, just to repeat the question or the comment. so, if you look at this Triton code, these statements like the this load is going to block for some number of cycles.
And then so, this is running on some thread on some warp on some SM.

And remember the SM is running multiple warps at the same time.
So, you know, when you get to that point, it can just like, you know, find another warp to run.
And then when this is done, the warp scheduler comes back and takes over.

Why would four warps be So, I'm sorry.
Why would there be four warps in the same kernel on each SM if four warps are scheduled Yeah, so what the question is, why four warp schedulers?
I don't know exactly the reason behind that.

Okay, so now let's go through some other examples.
Maybe just as a kind of a quick preview.
We're going to do three more examples.
Gelu Gelu is this kind of the simplest you know, form, even though the computation is kind of like, you know, has a lot of messiness.
It's just element-wise, so conceptually in the context of this lecture is actually very simple. now we're going to look at softmax, where now you're going to have to do reduction. but in this case, we're going to think about the case where row fits on a block, and then we're going to consider the case where row doesn't fit on a block, and then we're going to do go up to matmul.
And then hopefully by that point, you'll have all the ingredients that you need to do the assignment and implement flash attention.

Okay.
So, so far we looked at element-wise operations. now let's think about other operations that aggregate over multiple values.

So, remember what the softmax does.
It's ex- takes a let's just think as a matrix.
You exponentiate and normalize each row of a matrix.

Okay, so this is used in attention, it's used in, you know, generating probability output probabilities. generally a good good thing to do.
So, let's just start with a naive implementation, and just keep track of what's what's happening.
So, here I'm defining this tensor, and here's the naive implementation.
Actually, I think this is it's on assignment one.
Okay, anyway.
So, here I have a M by N matrix, and so I need to for every row you know, compute the you know, I'm going to compute the max of each row, and this is for numerical stability.
I'm going to subtract off the max, and then I'm going to, you know, exponentiate element-wise, and I'm going to sum and compute the normalization constant for every row, and I'm going to divide.

and then that's it.

Okay.
So, so if you count the number of reads and writes, remember this is just plain PyTorch.
So, this is a different kernel, this is a different kernel and unless you call a torch that compile, these are going to be different, you know, operations and each operation is going to read and write, read and write from HBM.
Okay, so you have, you know, five MN you know, reads, three MN writes, and in principle you should really only have you know, much fewer.

Okay, so this is you know, a piece of code kind of makes sense.
Everyone should be familiar with what a softmax is doing.
Okay.
So, let's let's write now the Triton kernel. and you know, in some sense the form is going to be very similar.
Just like in a GELU, right?
Once you do the scaffolding, the core computation looks very much like the naive version.

so what we're going to do is we're going to say each row is a block.

Okay?
And why do I make each row a block?
Well, because now remember each row I have to normalize and sum.
So, it's not element-wise.
Softmax is not element-wise, but it is sort of sort of row-wise.
So, the blocks don't interact.

so, they don't need and the blocks don't have, you know, shared memory, so that's that's fine.
A There's no shared memory across blocks. so then within each row, I'm just going to do some stuff.
Okay, so let's see what happens. so again, I'm going to in Python, I'm going to allocate my output tensor.
I have this M by N, you know, matrix.

I'm going to define the block size as you know, basically the number of columns. you know, go to the next power of two for good luck. and then the number of blocks is just the number of rows.
Okay, and then I'm going to call this kernel.
So, how many blocks are there?

M, right?
One for each row.
And each block I'm going to pass the input pointer, the output pointer.

and then I'm going to also pass these, you know, you know, strides which tell me how how far to move down. okay, so Okay, let me actually go and show you the softmax kernel.
Okay.

okay, so what does this look like here? so I wake up.
I'm on a particular row.

Okay?
And this is going to give me the all the columns from zero to the number of block size, which you know, I'm assuming that's all the columns.
I'm going to you know, read.
So, so basically I need to figure out which where to read from memory and that's going to be the start of my data plus the you know, which row and the row stride basically gives me, you know, every row is basically this is the number of columns essentially.
It's going to get tell me where how far to go down.

and then this is X pointers is basically the addresses of all the data I need to load.
I load them up. and here I do this thing where if it's you know, if it's masked out, then I, you know, put minus infinity because I know those are that's going to be kind of the equivalent of a zero for the soft softmax operation.
And then this part is essentially the same as the naive softmax.
I'm just going to compute the max, subtract it off, exponentiate, sum, and divide.

And then I write it back.

Okay.
Yeah.

Yeah, so this is the version where each block can just span the entire row.

So, yeah, this is sort of the and you can see that Triton makes this very easy, right?
Because this is as if you were just normal writing normal PyTorch code.

So, if everything fits in a block, you basically the thing is the thing is if you can fit anything through a block, you can just like write normal PyTorch almost.
Yeah. what if my number of columns and number of rows are bigger than block size?
Yeah, so we'll get to that.
Now, if the number of columns, rows, in general, it's going to be much larger than that block size, so we'll come back to that.

Yeah.
So, if you wanted to do like a softmax like so, the question is if you wanted to softmax by column, I think that should be fine because here we're tracking these pointers, right?
So, the pointers are can be anything and here I've basically I think all you would have to do is change the stride here.

to basically access the columns.

Actually, it would be here.
The column offsets, you would just make like the multiply this by like the row stride, I think.
Okay, let's let's move on.
All right, so now you know, warming up to the mammal, suppose that your row doesn't fit into a block.
Okay, so what do you do in this case?

So, for example, if you have, let's say, a row that's 400 4,000 columns, but the block size is only 1,024.

So, you can't cons you know, you have to do something here.
So, here's the strategy.
We're going to break up the row into tiles.

So, in this case, there's going to be four tiles and each thread is going to iterate over the tiles and accumulate a sum.

And then finally at the end, we're going to do the kind of the reduction. by you know, summing everything that each thread produced.
So, I'll I'll show you an example of this.
And now I'm I'm sort of switching from softmax to row sum because it's just easier to think about.

Okay.
So, So, here Okay, well, this is not very interesting.
The built-in row sum does what you would expect.
So, the row sum operation just basically takes a matrix and then computes the sum of each row.
Okay?
So, just no surprises there.
Okay, so conceptually what we're going to do here is as follows.
Okay, so each block is still in charge of one row.
So, that part hasn't changed.
So, suppose I'm block one, row one.

I wake up and what do I do?
I'm going to remember now there's tiles and suppose tile zero is columns zero through three, tile one is columns four through seven, call and then tile two is columns eight through 11.

Okay, so what I'm going to do here is that I'm going to iterate now.
So, first I basically process this all the threads.
Basically, each thread keeps a kind of accumulator and processes the first tile and it's going to move on to the next tile and it's going to add the current element to the accumulator.
So, here I'm putting 3 1 4 1 and then the second loop iteration, I'm going to add five to this, I get eight.
I add nine to this, I get 10, two to four, and six to one, right?
So, each of these four threads is going to be kind of accumulating its own thing.

And then at the end, you know, I do another for tile two, I add five and three to their respective accumulators.

So, at the end of the day, I have this you know, a vector of accumulators and then that is actually you know, that I can just like sum up.

Okay, so let's see what the code looks like here. so I'm just going to call the row sum kernel and let's see.

Okay, so here's what it looks like.
So, wake up.
I am on a particular row and this is what one row looks like.
There's one tile, a second tile, a third tile, and so on.
So, N is the number of elements of that row and block size is the size of this tile.

Okay?
So, remember block size is the number of threads.
I'm processing more data than the larger than the block size.
I just have to do it iteratively now.
So, I'm going to loop over all of the tiles.
So, start is going to go from zero to block size, to two times block size, and so on and so forth.
So, each time I'm, you know, jumping across, I am doing you know, calls basically get the, the offsets at the particular tile. and then I'm going to load the data from HBM and I'm going to add it to the accumulation.
So, this accumulation is going to be, either in registers or, shared memory.

Okay?
And then finally after I loop over all the tiles, I process a whole row, I get basically for every thread I have a cumulator of what that the thread has picked up and I can do just do a sum to get a scalar and I write it out.

Okay?
So, this is a little bit more complicated than before because now we have a for loop within a thread and this is, necessary when your data doesn't fit within a block.

Yeah.

Yeah, so the question is, I think, if you can control where the cumulator there resides.
At least in this Triton program, you don't explicitly say and this is up to the, Triton compiler to, figure out where to put it.

But in general, if the block size is large enough, it has to go in shared, shared memory.

Okay.
All right.
Does this, kind of make sense?
So, you know, you it might you know, just just to, I think, make sure people on the same page.
Remember in gelu, we also split a row into a bunch of, pieces, but those were blocks, right?
And so, each of those pieces was a block that was processed independently.
These are not blocks.
These are tiles.

The block corresponds to this whole, row.
And has to basically process all the tiles and so, this is where it starts to not look like PyTorch because, you're not able to process all your data in one nice kind of not everything fits into shared memory.
Yeah.

Yeah, I let's say the cumulator is stored in shared memory.
Yeah.

Okay.
All right.
So, let's now maybe go on to our finale, which is matmul.
Okay?
So, matmul of course is the bread and butter of deep learning.
It's been optimized to death, and, and it's, you know, in some sense, very, you know, fundamental operation.
So, you take two matrices, you multiply them.
I'm going to add a little bit of a twist here.
I'm going to do a matmul followed by a relu just because just for kicks.
Okay? and this is, well, this happens, right?
Because if you have, you know, one linear layer, it's a matmul and then you apply, a relu activation.
So, this is not like completely out of nowhere.

but I'll show you later why I did this.
Okay, so how do you build a matmul kernel?
All right.
So, here's the naive approach.
Okay, so here's my, let's say, a matrix.
I'm multiplying A times, B and I'm trying to write out, the matrix C.
And A is M by K, B is K by N, so C is M by N.
Okay?
So, what I'm going to do is I'm going to fix an, a one of these elements.
Let's say M equals 1, N equals 2.
Okay?
So, I'm processing Actually, let's do M equals 1, N equals 1.
So, I'm doing C5.

And then basically for every K, I'm going So, I'm going to, yeah, I'm going to iterate over the K, the rows of A and the columns of B.
I'm going to read from HBM.
I'm going to multiply them, accumulate that and at the end, I'm going to write out to, this element.
So, that is a valid matmul kernel.

Okay?
So, what's wrong with that?

It's correct, but, you know, if you look at how many reads and writes it's doing, this is not good, right?
So, basically for every M and N and K, I have to read from HBM.
So, it's on the order of M times K times N reads.

number of writes, I mean, it doesn't matter, but this is a bottleneck, right?

And, if you remember from the second lecture, if you look at the number of, operations you're doing divided by the number of bytes that were transferred, that's the arithmetic intensity, which you want to be high.
So, the number of operations is M times K times N order, and then the number of reads is also the same, so arithmetic intensity is a constant, which is, not good.

Okay, so if you look carefully, you know, you notice that there is a lot of redundant reads.
So, imagine computing C4. you needed to read A4, A5 and A6 and if you compute C5, you're going to have to read those over again as well.

So, so if you can just read those once, then you really save on reads and that's that's great.
So, let's try to use, shared memory to do that.

So, here's the idealized approach.

I'm going to load all of A and B into shared memory and then I'm going to just compute C.

So, if I can do that, then I get, now I don't have this cubic number of reads, I only get quadratic number of reads, which means I get the arithmetic intensity of order N, which in the second lecture I sort of, you know, said was a I kind of ideal thing you could hope for.

Okay, so this if you can do that, that's that's great because you basically there's no redundant reads.
You read everything once into shared memory, you do the computation and you write it back.
Okay, but what's the problem with this idealized approach?

the problem is that A and B are usually too large to fit into shared memory.

Right?
So, so then what do you have to do now?

Okay, so the idea here is this, you know, very classic idea of tiling.

And the idea basically is to, well, fit as much you can to shared memory as you can, essentially.
So, in some sense, it's it's going to be look like the naive It's going to globally look like the naive approach, but locally look like the idealized approach is the way to, you know, think about it.

so here's the, you know, the picture.
I think Tatsuo showed this as well.
So, what we're going to do is we're going to take this matrix C and instead of the Remember the naive approach just said for every element, I'm going to do compute it.
But now I'm just going to say for every tile, I'm going to do it.
So, I break it up into tiles.

and each of these tiles is going to be a thread block.
I'm going to have a bunch of threads that is responsible to for computing this.
Now, a different tile is going to be computed, you know, completely separately by another, thread block.

Okay, so I'm going to, you know, imagine I'm in Triton, I'm a I wake up, I'm I'm looking at this thread block.
So, what do I have to do?

Well, in some sense, it's the same as the, you know, naive approach where for every, you know, row tile of A, I'm going to go scan across the rows and for every column tile of B, I'm going to load the corresponding tile A and the corresponding, so say I'm here and the corresponding tile from B into shared memory.

I'm going to multiply these two together and that's going to be, you know, you know, like the kind of the idealized approach and I'm going to accumulate that in the partial sum and this is all sitting in shared memory.
And then after I finish all the sweep of the row and the column here, then I can finally write this output tile to HBM.

Okay?
So, that's, you know, the conceptually what's happening. and then the arithmetic intensity here now goes up to order tile size.

So, you can generally not reach order N because that would require you to fit everything into shared memory, but, you know, if your tiles are big, then that's, you know, still not too bad.

Okay, so just as kind of a bonus, you know, while you're doing all writing a kernel for this anyway, sometimes if you want to apply an element-wise activation function, it's very easy to just like put it on at the end.
And this is, you know, kernel fusion.
Okay, so very quickly the implementation here, so oops.
Okay, so, just as kind of a reminder about strides since that's just going to show up.
So, a tensor is, you know, multi-dimensional array, but in memory, it's linearized and strides of a tensor tells you basically how to map from a multi-dimensional index such as a row a comma column into an actual index.
And basically, what you do is multiply the row by the stride the plus the column by the stride of the column.
Okay, so in this case, every time you go advance to the next row, you go four positions in your memory.
And every time you advance a column, you go one.
In the if it were a transpose, it would be the flipped.
Okay, so what does this a kernel look like here?

So, there's Okay, so the launch is not interesting. so, you wake up and you are on tile M and N.
So, it's like, "Okay, I'm responsible for computing the C matrix, but for the M / comma N tile." there's a bunch of like index manipulation, which you can kind of I'll just gloss through, but it's it's it's straightforward, but a little bit. you know, just have to track the indices.
So, this basically tells you which rows of A the matrix A I'm looking at, which rows of oh sorry, which columns of B I'm looking at, and then this is just the numbers one through one through K.

then I going to get the pointers into A and B at my tile location.

and then I'm going to set up this accumulator matrix.
This is going to be in shared memory. this is M by N.
And then this is going to look kind of like the row reduction, right?
You have this sum over all of the tiles, but instead of just going across the tiles, I'm now going across the row tiles and also simultaneously down the column tiles of B.
I load A the small matrix, I load B the small B tile, and then I perform this dot.
So, remember, whenever things are in shared memory, things look like PyTorch, and I can just say say "Matmul it."
And you know, it'll do the do the thing.
And then and then and then I advance to the next row tile of A and the next column tile of B.

and this is like you know, just the bonus is that if I wanted to apply element-wise non-linearity, I might as well, you know, do that here.

before I write it out to HBM, I can do some any sort of operation on it.
And then the finally, I just write it out.

Okay.
So, there's some sort of indices that you have to pay attention to, but hopefully the form of the algorithm is clear.

Okay.
Any questions about that?

All right, maybe I'll just summarize, and you can ask me later.
So, today we talked about the there's a programming model, which is you know, you're talking about either PyTorch or Triton or PTX.
Even this is the you know, what the kind of programmer can control.
Right?
And even PTX, you can write PTX, and you can control every, you know, specialize however you want. but this is not the full picture, because the reality is that your code has to run on hardware, and there's only a finite number of SMs, a finite number of banks, and you know, sizing of memory and registers all are finite.
So, you come in with your big matrix and transformer, and you want to you it has to kind of fit in with the constraints of the hardware.

so, that's why, you know, benchmarking and profiling are really important to understand how the messiness of the hardware translates to, you know, performance.

we talked about Triton, which is I think a pretty nice neat language to think about thread blocks.
Hopefully, by now you can appreciate that things are easier to think about thread blocks than individual threads, because you don't have to think about explicitly synchronizing threads or doing shared memory.
And the way to think about it is that you figure out you have your computation, you break it down into these thread blocks, where you just need to read from shared memory, do some stuff, and write it back into HBM.

And then we saw some examples of increasing difficulty.
Element-wise is the easiest, and then reduction over a row, reduction where the it doesn't fit into a row, and that's sort of we introduced kind of the baby tiling, and then Matmul is the kind of the canonical example, where you actually do tiling.
Okay, so that's all I'm going to say about how to program a single GPU.
Next time, we're going to go to more GPU and talk about multi-GPU programming. yeah, question. so, if I'm able to write my own kernels, what are the alternatives that I have to Triton, and what are the range of my choices that I have?
Like how close can I get to the optimal performance? so, the question is what are alternatives to Triton?
So, you know, there's a trade-off between every language has sort of inductive bias, right?
That makes certain things easier and certain things harder.
So, most of what we do, Triton was built, you know, by people who, you know, train transformers.
So, anything involving transformers, I think it's going to be relatively you know, easy there. of course, in the extreme, you can always go to PTX and write that, but you know, wouldn't advise that as a first step.
There are a bunch of other language sort of libraries.
There's you know, ThunderKittens, there's you know, cute you know, various DSLs that allow you to give you know, they're not necessarily necessarily comparable either up or down The Stack, they just give you different you know, characteristics.

Yeah.
When you have a high-dimensional tensor, you know, multiplications or a high-dimensional tensor processing, what is the best approach?
Is there is there like what I'm thinking is can I load unload the whole tensor both of the tensors on you know, bunch of threads or thread blocks on my GPU at the same time and compute them, or would it be better to sort of, you know, take each element, just what we did here, right?

For the particular particular component of my tensor, then process it, and then write it back into HBM?
Yeah, so the briefly the question, and we should probably wrap up, is it better to kind of read all at once or maybe process individual elements at a time.
I think it's hard to answer this in the abstract.
It kind of depends on the nature of the computation. maybe I we can talk offline about that.
Okay.
All right, see you next time.

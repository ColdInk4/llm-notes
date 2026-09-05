# Lecture 5 字幕

Okay. we will get started. now we're going to start the systems portion of the class. so we've been doing a bunch of neural network stuff for the last two lectures. and now we're going to start getting into GPUs, parallelization, and inference. and systems I think is going to be I think part of The Stack that is most reasonable in a way in the sense that you can like reason through all the pieces and there will be kind of like logical steps that you can go through to get the result so it's very different and satisfying but when you first start interacting with GPUs it will feel a little bit like both magic and like a very strange device that you're interacting with I don't really know as a kind of class survey how many of you have ten like a GPU programming class of any kind.
Okay, so this is this is pretty good.
Like that's like a fourth, maybe a little bit less. but I think when you first interact with these things, it's actually a little bit of magic.
So it's good to understand. and I think you know deep neural networks feel like magic, but also GPUs can too. sometimes people will show you plots like this. part of this lecture I think my promise to you is that by the end of this lecture you will understand what this plot is or why this plot looks the way it does. and this is you know how much throughput you're getting on a matrix multiply where the x-axis is your dimension.
And you might think well you know maybe bigger matrix multiplies are always more throughput because you have more work to do.
Well you've got all these kind of crazy patterns happening where like at certain sizes your matmuls are much much slower than others.

Why? you know, you'll need to understand that you also need to understand some basics of how to sort of mentally think about a GPU in order to build fast algorithms.
And even if you're not a systems person, I think probably not all of you or even a majority of you are systems people, but if you're thinking about architectures or just generally reasoning about these things, you kind of need to understand the hardware model of how your, you know, model is going to execute in order to make an effective model, right?
Thinking back to Percy's initial lecture, right?
Big part of scaling is making sure you use your resources effectively.
And without understanding your systems, you will never be efficient at using your resources.

and you know, I am not a systems person, surprise. and so before I start, I want to, you know, make sure to point you towards resources that I think are very good and very useful and that I used in making this lecture.
Horus he has, you know, really incredible blogs and explainers of, you know, how GPUs work.
There's an enthusiast group CUDA mode that loves to talk about GPUs and CUDA kernels and so on. and there's a TPU and now what? >> They're GPU mode now.
They changed their name. >> Okay, fine.
GPU mode. and the TPU book, but now it's also a GPU book by the Google folks.
Actually, this is a really incredible resource.
So, if you're really excited or interested by any of the stuff I talk about today, I would strongly encourage you to look at that book. can they also have exercises that you can work through and your assignment will have things that look a little bit like those exercises so you can get a little bit of a preview of what we will send you to later.
Great.
Okay.
So this lecture is going to be in three parts.
The first part is really I would say just like broad introductory material on GPUs.

This is going to be just the hardware model just how the programming of the GPU kind of works. because the GPU is kind of a foreign device and I want you to understand philosophically how they're different from CPUs. and once you do that, we're going to talk about I think six different tricks important tricks to get a GPU to go fast.
These are just kind of building blocks of how you can start to you know manipulate your code in order to make it run efficiently on a GPU.
And then finally, the last part where we put all of this together is going to be we're going to walk through flash attention and this will be kind of our victory lap. we will say actually at this point we understand everything we need in order to you know in reinvent our own flash attention and to understand all the different components right so that's going to be kind of the goal that we're going to be building towards by the end of lecture okay so we haven't gone to scaling yet but you know from Percy's introductory lecture or you know just generally following language modeling you know we kind of know that if we have more compute on hand we can get you know better language models right this is kind of the currency by which we operate.
So if we have faster hardware or if we have better utilization or you know we have more chips and then we have improved parallelization all of these can drive progress right. and so in some sense the game for the last few years has been how can we bring more compute to bear in training models and I guess in the last year or so how can we bring more compute to bear in the inference of these models right so compute is just extremely important and systems is part of that so if we were talking about this in the '9s I think the things that we would be focusing on is clock speed right we would be saying things like well you know how we make computers go fast is we have a CPU that executes instructions serially and we make those instructions go faster and faster, right?
I mean, this type of scaling where transistors got smaller and clock speeds got faster, you know, called Dennard scaling held quite well for a while.
Really the 2000s was about when this kind of scaling tapped out. and one sort of interesting side note, right, the number of transistors of course has continued to go up, but making transistors smaller does not necessarily mean that your transistors go faster, right?
They don't move at a faster clock speed. there's fundamental physical reasons why you know you don't actually continue to get benefits of smaller and smaller transistors at a certain point.
So this old approach of like making your clocks faster making instructions execute faster is not going to work right.
So how do we you know get LMS to continue to scale in this world where we can't just make things faster in sort of this frequency sense right and that's you know where the GPU shows up and sort of parallel scaling in general right Percy's going to talk about sort of more advanced topics next lecture but after that's going to be parallelism right parallelism and GPUs are both kind of part of the same paradigm instead of making things go faster serially with your your clock going faster and faster and faster instead you're going to you know scale horizontally right?
Have more things executing your your instructions in parallel.
And so this is kind of the story of GPUs.
This one slide really I think or one picture summarizes you know the state of parallel scaling which is you know in the early days of you know K20s and M40s you had you know respectable but somewhat pitiful by today's standards FLOPs and all of a sudden around P100s and V100s this curve takes off and you get super exponential scaling in the amount of floatingpoint operations that you can do sort of yearbyear right this is kind of a remarkable plot in the amount of compute that we can bring to bear and so you So GPU scaling has really driven almost all of the compute scaling that we've seen. and we'll talk about all the different components that we have here. you know I'll highlight them here just to put them into your head.
You know at the V100 era in 2017 we start getting tensor cores.
That's one of the big hardware innovations.
And then in the in the next few years we're seeing you know structured sparsity and FP8 and all these sort of lower number formats also driving our our FLOPs forward.
And we're going to see sort of what that means in detail in a little bit.

So how is a GPU different from a CPU?
Those that have, you know, interacted with a GPU already know this.
We'll move on to more advanced things in a minute.
But for the rest of you, I think it's very important to understand the model like the philosophy in some sense of how they're different.
So a CPU is designed for quick serial execution where you have sort of complex branching logic, complex conditionals, complex control flows.
Because of this, a CPU generally speaking has really big control units and it has a few alus that can do sort of mathematical or other operations, but this is meant to run very quickly, right?
So you want to go as fast as possible and sort of have low latency.
The time between when you get an instruction and when you complete it should be very very short, right?
That's kind of the general CPU design philosophy.

In contrast, the GPU is really about throughput, right?
And you know the plot on the right I think is a great pictorial illustration of the when you change these philosophies what you end up getting.
So in a GPU, maybe you dispatch a task.
It takes a long time for it to finish, right?
It might be that you process a task and then you stop and then you go work on another task and then you go back and work on another task.
But in general, right, even though your tasks don't finish quickly, you know, you're getting large aggregate throughput across all of your tasks.
And what enables this is having tons and tons of lightweight cores, right?
So a GPU has hundreds and hundreds of sort of compute units that are sort of packed into a chip and all of these can execute in parallel, right?

Okay.
So, this is a very different mode of programming and thinking compared to a CPU.
Okay. and I want to demystify what a GPU is.
I don't want you to think of it as like a, you know, set of pieces of text on Nvidia SMI.
I want you to like be able to sort of visualize what the hardware is of a GPU.
I think this picture is, I think, usually used as a pictorial illustration of what a GPU is.
I think this is a great logical picture of what's happening inside of a GPU.
So a GPU the basic unit of a GPU is SM, a streaming multipprocessor.
And a streaming multipprocessor is kind of like a core.
It's its own independent compute unit that has, you know, subcomponents that it can use to accelerate its computation and it has access to certain kinds of memories that it can connect to.
Right?
So each SM, you know, building on this idea of parallelism, it's not a single sort of compute unit.
It has streaming processors inside of it that can then execute different threads in parallel.
But SM is kind of the discrete compute unit.
We'll see why later when I get to the programming model. each SM you can think of as kind of a core almost. and GPUs have tons of SM like a A100 will have 128 SM.
Each of these can be independently programmed and they can independently execute different jobs and they have access to sort of global memory outside of it.

and the other thing that we will see that's kind of the sort of compute story of a GPU. but we will see throughout this lecture that a really really important part of this story is not just the compute part but the memory part.
Right?
Modern sort of hardware and LM optimization is really defined by the memory.
And so we need to understand what the memories are and where they live.
So when we talk about memory, there's different kinds. there's different levels of caches that are connected to a GPU.
There are registers which are even more limited and even faster.
And then you have global or high bandwidth memory. and as you can kind of see from this table, this is a benchmarking that some folks did of I think this was the A100 latencies. you know the L1 and shared memory these are very fast right in something like 20 to 30 cycles you can kind of read and get data from them if you're hitting the L2 cache much slower and if you're hitting global memory you know that's 10 times slower than the L1 cache this is a big difference in your speed right and this global memory is kind of what you normally think of as a you know highle programmer as a GPU memory right when an H200 tells you I've got 144 gigs of memory you're talking about global memory right you're not talking about shared or L1 cap Now physically where do these things live?
Well, global memory, you know, usually are these kind of memory chips that live outside of your sort of compute unit, whereas L2 and L1 cache, they kind of live physically in the chip itself, right?
So, this is kind of the picture of a GPU chip. and you see all these SM, these green blocks are SMS. and you see L2 right here, and the L1 is inside of each of SMS, right?

So, so they're physically much closer to the compute units.
That's why their latency is so much better, right? and you might say, okay, then if shared memory is so nice, why don't you build a whole chip out of shared memory, it's much more expensive, hundreds of times.
It's also much more power hungry, right, from the way that these memories are built. and so this is the reason why if you're building a practical accelerator, often you don't want to just, you know, have a giant amount of shared memory.
You want to have the sort of hierarchy of memory, right, and use it effectively.
I think Grock right which was recently acquired by Nvidia was a design where you just had these giant amounts of SRAMM right and for certain workloads like inference this is very helpful but still for most accelerators you're going to have this hierarchy that you're going to really have to respect in order to make anything run fast >> yes the shared memory L1 cache >> yeah so the shared memory is you know and the L1 cache I guess kind of operate differently.
The cache is really just a cache thing that is storing recently accessed data elements.
Shared memory is a programmable element that you'll actually interact with that you can like put things in and out of.

Okay, cool.
Right.
So that's the memory story and keep that in mind because it's going to keep coming back.
The memory hierarchy, that's a really important object.
Now, okay, we have the hardware mental model in place and now we're going to start talking about the software and programming model of a GPU.

so there's three different important players that live in the execution model of a GPU. so there's a thread and if you're familiar with multi- parallel programming of any kind, you know what a thread is, right?
A thread is a sort of lightweight thing that's running in parallel and doing some work, right?

in a GPU.
An important thing about the threads is they follow the SIM key model, which means that all of the threads that you have are executing the same exact instructions, right? they're going to have different inputs, which means that the threads can actually do separate useful work, but the instructions they execute have to all be the same, right?
And this is this has important implications for both the efficiency of your code as well as the programming model, right?
If the threads can do all sorts of different stuff, it becomes very hard to program, right?

GPUs are this trade-off between programmability and efficiency.
Now, a block is a group of threads.
And the reason why a block is a relevant object is that a block is guaranteed to run on a single SM.
And remember, an SM has a shared memory.
And so, you can think of a thread as doing work in parallel.
Blocks have a access to a shared pool of memory. that will become very important later when we talk about this idea called tiling where these blocks are going to be reusing the same pieces of memory.
Finally, a warp is essentially kind of the scheduling unit of what is happening inside the GPU.

Threads are going to execute in groups of 32 consecutively numbered threads.
That is a warp, right?
This decreases the overhead of a scheduleuler deciding which threads to run.
But essentially, these are kind of the important three terminology items for a GPU.
I might say something like, oh, you know, we're going to have different warps executing.
That's what I mean here.
That's a basic unit of 32 threads that execute together.
Yes, you mentioned that all threads execute the same instructions.
So is that all threads in a block or all threads in a warp? >> All threads in a warp are going to be executing.

>> Cool.
Yeah. >> So theul decides which warp the next instruction goes to. >> That's right. or sorry the scheduleuler decides which which warp is going to be executing next and on what SM and so on.
Yeah, cool.
Okay. and so in the memory model of a GPU you know device code can read from registers.
Registers are some of the fastest and most local items.
This might be storing things like memory addresses of you know where you might need to your arrays start and end.
That's a place where you might store things in registers. then you've got local memory.
This is local to within your your single SM.
You've got shared memory that can be accessed sort of within your block.
So if you have something you want to transfer across threads or you want to have information that is going to be reused across threads, those are going to live in shared memory.
You can of course then read out or sorry read from global memory, right?
The DRAM that's far away as long as you're willing to pay sort of the latency hit of waiting for data to come to you from global memory.

and there's also sort of constant memory, which I don't think I've seen very much used in various programming things. that you can use as well.

Now, you can even sort of transfer to and back from host memory, right?
When I say host memory, that means sort of CPU memory that lives on the machine. you can do this if you need to offload beyond the size of your GPU.
This is kind of the set of things that you can do in your memory, right?
And really the key thing the most important part of this sort of memory model is that as soon as you go outside of your shared memory things are going to be slow.
So grouping the blocks to sort of decrease the amount of global memory reads is going to be the name of the game throughout this lecture.

Okay.
Now I'm going to talk for two slides two slides about TPUs. the majority of this lecture is about GPUs mainly because I think you know if you all go out and you know write your code and you build your models is probably going to be on GPUs for most of you. but I think it is very important that we talk about TPUs both because I think TPUs are cool and becoming more popular but also because TPUs are kind of like the alternative evolution of GPUs and you can kind of see what's the same what's not. it's very interesting to see how TPUs work as well, right?
So actually what's very interesting is that TPUs are very similar to GPUs at a high level.
This is kind of conversion evolution.
If you want to build a energyefficient accelerator that does machine learning, you kind of end up at the same place. so the core mental difference between a TPU and a GPU that you should kind of keep in mind is that TPUs are in some sense simpler.
They're more optimized for the machine learning workload. they have light or weight control units and they have much bigger matrix multiply units and so you know they have the same basic structure that I'll talk about for sorry for GPUs where they will have a specialized circuit to multiply matrices they will have components that can do parallel vector operations and they will have you know some control system they have a very similar looking memory structure where they have slow memory which is called high bandwidth memory and then they have a even faster local memory called SMEN or shared memory.
Right?
So this looks very very similar to the architecture of a GPU.
Really the difference is in is in how these things are sized and the flexibility of these operations. and I would say oh and the main difference that I won't get into today is the networking for these accelerators.
So if you're thinking about what is the big difference between a TPU and a GPU, actually the biggest differences come in networking, not in the individual chips because really the chips, you know, they live to just multiply matrices.

and this is from the Jax book, you know, which is a really lovely resource. but they have these two tables.
They have a section on GPUs where they describe how to essentially map GPU ideas to TPU ideas.
And whatever I say today about what's on the left here, you know, you can map directly to the TPU.
But the important thing you know at this point in the lecture to notice is that for every sort of concept in the GPU there's a corresponding concept in the TPU and this you know mapping is actually pretty precise like a tensor core in a GPU which is the matrix multiply unit corresponds almost exactly to an MXU which is a matrix multiply unit in the tensor core and if we're getting into really nitty-gritty details they actually use the same underlying you know like circuitry right they're both what's called a systolic array which kind of streams data in and out to do a matrix multiply.
So the architecture is very similar.
Now, how are they?
Oh, sorry.
Yes. >> Do you know like what a typical system size is on the GPU versus >> I do not know that off my head.
I'm sorry.
I'll have to look that up.
Yeah. >> Okay. so the main difference between these two is the number of different units that we have. so if we look at Oh, actually I'll mention one very very confusing thing before before I move on.
Sorry.
TPUs call their streaming multipprocessors tensor cores.
GPUs call their matrix multiply units tensor cores.
They are named exactly the same thing.
So you will have to disambiguate them based on context. if someone says tensor core for a TPU that means a processor.
If someone says tensor core for GPU they mean a matrix multiplier. so if you look at a GPU a typical GPU like a H100 might have something like 132 streaming multipprocessors.
That's a lot of processors. a TPU will only have two units, right?
There's only two separate units running in parallel.
Now, the GPU, if we just look at the matrix multiply units, will have 528 matrix multiply units, whereas a TPU will have just eight, right? and these are, you know, roughly scaled to be comparable looking objects. why is it that it's this way?
Well, the TPU is relying on much bigger, smaller numbers of matrix multiply units, whereas the GPU is relying on smaller but much more numerous matrix multiply units.
And you can kind of imagine that if you have a small number of many matrix multiply units, it's going to give you more flexibility.
You can program them to do this and that.
Whereas a TPU, you're locked in to big big matrix multiplies, right?
And so there was a funny situation in a recent paper that I wrote where you know we did a batch size sweep and it goes all the way down to 64 and it stops at 64.
Why is that?
Because the tensor core refuses to accept anything smaller than a 64dimensional input there.
Right?
It has to be big matrix multiplies. so this is the main difference.
But as you can see GPUs and TPUs share a lot of similarities.
They rely on a memory hierarchy between fast memory and slow memory.
And they have a matrix multiply unit.
And because of that, the core concepts that I'm teaching you today is going to transfer over between GPU and TPU, right?
The basic ideas are always going to be the same for that reason. and this is really convergent evolution.
There are only so many ways to sort of cost effectively allocate memory and there's only so many ways to multiply matrix very fast, right?
There's some design decisions that are different, but otherwise you end up at the same place.
Okay, cool.
So that's TPUs and Right.
Good. and GPUs I think have been enormously successful because it allows for you know easy scalability for things like matrix multiplies.
If you want more throughput you just keep adding SMS right as long as your memory bandwidth can handle those SMS you get more and more throughput. the programming is you know something we don't think about but it's deceptively easy right if you were to try to write you know lower level machine code looking stuff like ptx it's actually doable because it's all simp right it's not like you're programming every single thread you know you're saying okay here are my instructions here are my inputs now run them on all of my inputs right it's something that maybe you all are already familiar with things like applying applying or using functional programming finally I haven't talked much about thread threads.
But whenever you're doing GPU programming, these threads are always very lightweight.
They can be stopped, started at any time.
The scheduler can decide that, okay, I have another warp that's, you know, better to run.
So, I'm going to stop you and start this one. and it's very sort of quick to sort of swap in these jobs.
That allows for much higher throughput if certain jobs are stalled or have some property that's lowering throughput.

Okay, so that's the benefit of GPUs, you know, as a hardware model. it's been enormously successful. and I think even in the early days before tensor cores sort of were implemented, you know, people realized that, you know, commodity graphics hardware, this massive parallelism would be very effective for, you know, scientific computing. so, you know, this is one of the original papers on using graphics hardware for fast matrix multiplies. and this is a really really cool sort of hacker work. you know, people figured out, well, we have these shaders that are implemented, but we can program specific shaders that will actually give us sort of matrix multiplies and different rendering settings actually give us faster matrix multiplies.
And that's that's a sort of really cool computer science things you can do.
But nowadays, you do not have to do this.
You do not have to manually program your GPUs to do matrix multiplies because as of the V 100, Nvidia decided, you know, here is a piece of hardware that will do your matrix multiplies for you. and once that hardware sort of existed, matmuls became the sort of one privileged operation in machine learning.
Right?
So at this point there's a gigantic gap in sort of the throughput that you're going to get between sort of parallelizable but non matrix multiply operations versus amount of [clears throat] operations you can do in a matrix multiply.
Right?
So this is one of the reasons why sort of any near future machine learning architecture that we see that scales with compute is going to have a matrix multiply in it.
Right? because this is the one way that you can really effectively get a lot of compute throughput. more than 10 times faster than any other floating-point operation that you can do in a GPU.

Okay.
And then the last sort of hardware thing that I'll end with I think this is the last slide of this section is that different components of the GPU are scaling at different rates over time. we know that the GPUs are getting faster and faster.
This is just like a thing, right?
As well as CPU, you know, overall compute throughput is getting faster and faster.
So that's the gray line, right?
That's pretty fast.
We're getting faster and faster improvements in the total floatingpoint operations.

green is our memory bandwidth.
This is how fast we can transfer data.
That's growing comparatively slowly.
Blue is parallelism.
This is how fast we can connect different GPU devices.
That's also going very slowly.
And what does this mean?
This means that if you're out here kind of in the early days of GPU programming, probably you weren't thinking much about memory because you were you compute wasn't that much faster than your memory transfer.
But now as we go more and more to the right of this plot, we're going to see a bigger and bigger gap between compute and memory.

And that's why a lot of the optimization I'm talking about today are memory optimizations.
Right?
As we go further to the right, we're going to see memory and communication bottlenecks more and more.
And so it's going to take more and more work to fully utilize the hardware that we have.
Okay, so that was our sort of very basic overview of GPUs.
I can't possibly, you know, give you a 20inut lecture that fully helps you understand what the GPU hardware is, but hopefully you have a flavor, right?
The mental model should be GPUs are massively, massively parallel.
They execute instructions all at the same time. compute scales really fast, really, really fast and even faster than memory, which means memory is really important.
And because memory is important, everything we do has to respect the memory hierarchy, right?
As much stuff that we do goes into shared memory and not global memory, right?
So, this is kind of hopefully your understanding of GPUs.
And if you have this understanding, at least you'll understand all the basics necessary to internalize the next part of this lecture.
Okay, I can pause for a moment here in case people have hardware questions or anything else before I move on to the next part.
Yes. >> So before V100s, there was no matrix multiplier unit. >> Yeah.
Yeah.
There was no tensor core that like specialized to a map.
Yeah.
The tensor core is a V100 edition.
Before that you would be programming manually.
Of course you have so many SM and so many like ALUS that you can still do quite good throughput.
But you know a specialized circuit for matrix multiply really you know changes the game for them.
Oh yeah back there >> what was the difference between the share memory and the L1 cache?

>> the cache just operates on its own like you don't get to control the L1 cache >> is O2 slower than L1 just because of the physical distance or does it also differ some other way? >> Yeah I mean physical distance is a big part of it.
I mean they are if I believe if I remember right they're both SRAMM like that's one of the reasons why global memory is much slower but the physical distance and the interconnects that are used are the reasons why it's slower.
Yeah this is not inherent by the way one additional side thing is that the equivalent L2 cache in TPUs is much faster because they make different design trade-offs on silicon to enable that.
That's one of their selling points.
Yes.
If the compute memory gap keeps growing, do you think that affects the way future chips will be laid out?
Like just >> Yeah.
Or maybe the future chip kind of budget splits. one of the things that I really didn't want to talk about, but now you forced me to, is kind of the inference landscape for hardware is actually even more crazy. so one of the things that people started doing I think last year was something called prefill decode disagregation where the prefill part which is very heavy matrix multiplies is one chip and then the decode part for your for your inference which is more memory bandwidth limited is another chip right and some of the models I think this was Step-3 which is a Chinese open source model even has a thing where they like try to disagregate different layers so an attention will go to one chip optimize for one thing or one accelerator and then MLPS will go to a different accelerator, right? and so you're starting to see like if this gap grows, any sort of clever tricks to, you know, utilize your very precious fast memory becomes very valuable.
And you start to see this very heavily in inference, which is even more memory bound than training.
Yeah.
Cool.
Oh, yeah. >> How many warps are in a block? >> How many warps are in a block? >> do I know that?
I don't think I know that outside my head.
I think it's also hardware dependent.
Okay, good. okay.
So, the second part of this lecture is now you all have a general sense of what a GPU is and how it works.
Now we are going to talk about all of the basic tricks to make workloads fast on a GPU.
These are kind of basic components that you can kind of put together to make things fast.
And to keep us kind of motivated, one of my goals by the end of this section is to explain this plot to you.
Right?
So this plot, as I said before, has a very simple picture.
It's the size of the matrix, a square matrix that I'm multiplying.
I'm m multiplying two square matrices. and this y-axis is how many items I'm processing per second.
So this is the throughput, right? and as I'm going to the right, in general, throughput is going up because I have bigger matrices.
I have more work to do.
And so, you know, if I have a lot more work, I can keep my hardware more utilized, right?
But there's all sorts of strange patterns here.
And we kind of want to understand why are there strange patterns here?

Why?
And how can we be at the very top of that curve rather than the bottom, right?
So, that's what I'm going to be talking about in the second part.
Okay?

And you know, Percy already mentioned this.
This is the roof line model, but the plot I showed before looks a little bit like this classic roof line model, right?
This looks like there's a diagonal part and then a flat part, right?
And this is what we expect right the roofline model says up until a certain point we will be memory limited right we're memory limited so no matter how much you know compute you give me my sort of throughput is not going to go up at some point you have enough work per unit sort of memory movement that I can keep sort of my compute units completely saturated and past that point having bigger and bigger sort of units of work does not help me anymore right no matter how much sort of extra work you give me that doesn't help because you know I fully saturated by matrix multiply units.
So this is what we expect out of the classic roofline model where you know as we sort of have faster memory we can kind of push this upwards but there's a memory bound region which is diagonal and then a computebound region which is flat right this is kind of what we expect now if you want to make sure that your code runs efficiently on a GPU really what you need to do is you need to be on this flat part right as soon as we're at this flat part there's nothing else to do we're at maximum throughput right so we want to prevent ourselves from being in this diagonal part which means we want to increase the operational intensity.

The amount of computation we do per memory read needs to be high enough.
So that's what we need to do and there's six different tricks that we will talk about.
Actually some of these are not sort of memory tricks.
The first one only there are six different tricks that are maybe necessary to understand to make GPUs go fast.
The first one's a little bit different in flavor but then the rest of them will have this flavor of trying to minimize memory movement as much as possible.

The first one is kind of a GPU specific thing and it relates to one of the things I said before the SIM key model.
So in a GPU every thread is executing the same thing.
Now what happens if you write an if statement on a piece of code that executes on a GPU right on a CPU it's quite clear what happens on a CPU you know you will either pick one branch or the other you will execute that branch and you will move on with your life right in a GPU you will do something actually quite different and slightly counterintuitive every thread has to execute the same instruction so actually every thread will kind of execute both the if and the else but if you're not part of the if branch you will just mask out that computation you just sit idle, right?

And so whenever you have a conditional, your mental model for the execution kind of looks like this.
You know, part of your threads will diverge based on this if statement.
Some of them will go to the bottom branch, some of them will go to the top branch.
And the bottom branch ones will execute first, top branch ones will wait, and then the top branch ones will execute, and the bottom branch ones will wait.
Right?
So in order to execute an if statement, you're actually sort of walking through both of these.
And then some of the threads are just waiting doing nothing if they're if they're on the wrong side of the branch. and this is called control divergence. and this is nice in some ways because it simplifies programming.
It's not nice in that you now have these big compute gaps where your GPU is doing nothing, right?

this is one of the reasons why you really do not want to write if statements. you know if you look at GPU code like if you have a relu you'll often find you're like multiplying with zeros or multiplying with masks instead of doing like an if statement where you're updating half right because because a multiplication is going to all kind of go at once and an if statement might actually have to wait two clock cycles or something like this two operations okay so that's the first thing but you know that's a sort of separate story from all this memory stuff which is kind of the main narrative or the main story of this the first one which you don't really have to do too much for, but I think is one of the most important ones.
And hardware-wise, I think where Nvidia is investing the most effort is low precision computation. and if you look at this plot, I've brought, you know, Bill Di back to the slide. this super exponential growth, where does this really come from? actually a pretty non-trivial part of this, you know, is number representations.
You start at FP32, you go to BF-16, and then you go to int8. you just keep decreasing the precision of your numbers.
And of course, if you have half the bits, you have half the bits to move.
And so you've have your memory movement, right?
And so you've solved the memory bottleneck for at least a little bit.

you know, this is a dumb example, but you know, I'm just going to really drill it in there.
You know, let's say I have a rel on a vector of size n.
If I'm on float 32, I need to do one read, right? to read my number.
I need to do one right after I've computed my value.

That's going to be four bytes per operation.
Total of eight bytes for my memory access.
I only have one FLOP worth of operation.
So that's eight bytes for a FLOP.
You know, if I have the precision, of course, I've haved the number of bytes per up.
I've, you know, haved the number of bytes that I need to move per FLOP.
Sorry, this is the inverse of arithmetic intensity. arithmetic intensity should be reverse of this, right?

Now that's a very simple example, but actually how low precision works in a modern GPU is actually very complicated and actually either very cool or very crazy depending on how you sort of think about, you know, how cool things are. so when you do a matrix multiply, let's say you have a tensor core that's operating in lower precision.
How does that actually work in practice?
Well, you're actually not going to keep everything in low precision. you're going to downcast before you take your product.
And when you sum up your partial summons in your matrix multiply, you're going to do this in full precision.
And then you might emit your outputs or your your results in FP32.

And part of I think the reason why low precision is actually kind of a black art is not because you can just like decrease precision.
Anyone can do that. you have to figure out which of your operations has to be in which number format, right?
So, you know, matrix multiplies, maybe both your weights and activations can be low precision.
If I'm doing a soft max, maybe I need to do that in FP32.
Or if I'm doing an exponential, maybe that also needs to be FP32.
Or maybe I can get away with BF-16, right?
So, you know, I think it took several years really to get low precision training to be the state that it is today.
And a lot of that work was you know slow incremental empirical work figuring out which of these operations we can sort of downcast how we can downcast them and how we can preserve stability of training while making as much of our training and inference low precision as possible.

people have been pushing this further and further.
I think, you know, once you think about how hard it is to pay for fast memory, you realize if only I can cut my precision by another factor of two, right?
Like people really really want to do that. and so, you know, FP8, you know, was kind of the next frontier, I would say, of low precision. once you get to FP8, there's no longer a standard canonical format that everyone uses.
Like if you're using FP or sorry 16 bits, you're probably using BF16, right, for a lot of what you're doing.
But with FP8, there's going to be E4 M3, which stands for four bits of exponent and three of mantissa, or there's going to be E5 M2, which is more exponent, fewer mantissa, right?
Because these are used in different places.
There's no oneizefits-all once you start to get to so few bits.
But I think the thing that's really interesting is the more advanced low precision number formats that are coming out with the new hardware.
So I'll talk briefly about MXFP8 because it's both cool and what it does to training is strange to say the least. so in FP8 the traditional version of training what you have is that all of your let's say activations are in 8 bits, right?
So this is up here and then you're going to have a scaling factor in 32 FP32 that's going to allow you to scale this up and down.
Why do you have a scaling factor?
The scaling factor is necessary because you only have four bits of exponent let's say right that's so few bits of exponent you're going to quickly over underflow.

So you need the scaling factor to kind of keep you in the right range so you can keep some information right. well people realized this is not optimal because a single matrix might have very different magnitudes right one part of my sequence might have much bigger sort of activations than another so instead of having a single scaling factor why don't we have many scaling factors right so MXFP8 and sort of these multi-caling factor formats that are sort of much more advanced now have many scaling factors each of these scaling factors corresponds to the relevant colored submatrix, right?
So the first four elements in this matrix corresponds to this first scaling factor and so on and so forth.
Now I have more scaling factors.
These scaling factors are actually going to be in lower precision.
These are going to be eight bits as well.
They're going to be E8 M0.
So they only have exponent bits.
So they're all powers of two you know in 8 bits.
So you have to make these trade-offs.
Now what was going to say?
Okay, so MXFP8 is going to use more mantissa in the elements.
Scale factors are themselves FP8. and you're going to have one scale factor every 32 elements in your matrix.
Okay, I'm going to pause here.

don't read the third thing.
Now, think for a moment.
What is the problem with this design?
Like what is the big issue if you do something as clever as this?
So, the big issue here is let's say I want to transpose this matrix. the transpose of this matrix does not have the same pattern so to speak scaling pattern as this matrix right and so now it used to be that a transpose of a matrix was simple I just transpose this matrix I don't have to re-quantize I don't have to do anything now transposing this matrix is very complicated because I now have to potentially re-quantize this whole matrix to fit this 1 out of 32 pattern that I have and so what happens internally when you actually use MXFP8 And I found this to be crazy but also kind of cool is that when you're training your model with this thing, it actually creates two copies of every quantized matrix.
One for your original matrix, one for the transpose.
So if you ever want to transpose, you have the transpose version waiting for you, right? and that's kind of how you get around this, which I think is a is a crazy and cool thing. the point here is maybe not to emphasize the GPU trivia, but the point here is that once you start getting to these FP8, FP4 formats, things start getting pretty exotic because the amount of exponent and sort of mantissa loss that you have are pretty extreme.
So, you have to be very clever about how you do this.

and as I said, you know, if you were to go and really train a model FP8, which I think a lot of frontier labs are doing for the compute benefits, you know, you have to be very careful, right? this plot does show kind of what I was saying.
So, every time you quantize, you're going to quantize one way and you're going to transpose and quantize.
So, you're gonna have both copies.
And then the other part, which I told you about a few slides ago, is you're only going to quantize certain layers that you think are safe to do so, right?
And this is another trial and error thing that you have to figure out, right? and but once you have all this together, you have the big benefits of training an FP8, which is that when you're doing those matrix multiplies, you're probably going to get 20 to 30% savings, maybe more depending on the size of your matrix. unfortunately, it's not going to be a 2x speed up because you have to do all these quantization operations, right?
There's a big overhead doing this.
So, it's not free.
Yes.

So, I remember from the like reading that like the first layer and the last layer hard to quantize.
Do you have any intuition about that? >> That's right.
That's right.

I don't know if I have intuitions for the first layer.
The last layer I think being hard to quantize is because you know it is kind of a first order factor.
It drives a ton of the loss and so I think you end up getting like both instability and big loss increases if you attempt to quantize the very last layer.
Okay.
I put this up there almost because I think it's so ridiculous.
You know the next frontier this is already implemented MXFP4. the entirety of MXFP4 can be shown to you in this slide. it's -6 to six.
You know, you have all the possible values. but now that you understand MXFP8, you can also understand how this thing works.
You know, you're going to have a structured block where each block is going to be one of these four bits. you're going to have every 16 of these numbers, you're going to have one scaling factor and that scaling factor will itself be FP8.
This is a E4 M3 scaling for those scaling factors. these are there has been a paper that has trained with FP4.
I don't think I've heard of anyone successfully training in the wild, you know, real big serious models with FP4. but I think this is coming, right?
Like the next generation of models will probably be FP4.

Okay, so that was low precision.
I think it's really cool that you can really push the limits so far that, you know, you're ending up with just plus 6 to minus 6 and yet you can do interesting things with matrices in that precision.

Now, oh yeah, there's a question. >> So, is this only for like matrices that you could apply this like not just symbols? >> Yeah.
So, you can you can quantize anything.
You can quantize the activations after a relu for example, but there just isn't as much of a of a need to.
So, like a matmul, if you quantize that, you can get really big throughput improvements, right?
And that's why if you look at the MXFP8 training one, you know, these are all matmuls that are getting FP8 quantized. the overhead is probably not worth it for most like hardware parts. >> Well, it's not exclusive to that.
Yeah.
Like you can always do lower precision stuff and then it will decrease your memory movement.
This is true.
But for the others, it doesn't quite make sense.

>> This isn't any like hardware limitations.
It's not a limitation, but there is hardware support for all of this, right?
So, it is baked in, a bunch of this is baked into the hardware to accelerate it. yeah, >> you've positioned this as a as an improvement for memory bandwidth or well requiring less memory bandwidth, but is there a computer as well? >> Yes.
Yes.
For sure.
For compute, you get basically linear improvements for just multiplying the quantized numbers.
But the fact that you have to quantize and dequantize means that the benefits are more diluted. >> Yeah.
Do you have to train on the scaling factor?

>> the scaling factor is an implementation dependent choice.
Like depending on the library for example, you might get different ways of quantizing and dequantizing. training is a is a strong word, but you might have quantization factors that like scan through and pick the max min or you might have scaling factors that look at historical running statistics and then use those.
I wouldn't really call them training per se. at inference, you might fit them more.

>> So there are gradients. >> At inference time, you might. at training time, you would not.
Yeah. >> Yeah. >> Is there has anything been done with structured sparsity?
Like >> I guess we had it but >> yeah there's a lot that's been done with structured sparsity and I mean are I think one successful example of a structured sparse operation that people use.

>> Yeah.
Right.
So I mean a lot has been done.
Chris Ray here has done a lot with you know lots of different kinds of structured matrices.
I think the benefits of that like you know in terms of the compute benefits versus the representation losses I think has kind of washed out.
I don't think it's a bad idea but empirically it hasn't I think necessarily panned out.
Oh yeah.

>> A lot of like squeeze that we've been talking about decad what is current approach is it like train in full precision or half precision and then like distill into like a quantized model that you can run on your Apple computer things like that like what what is like set up for this? >> Yeah I mean there's a combination of like you probably want to train a bigger model and then quantize down. there's various kinds of like quantization scaling laws and papers that people have written like precisely characterizing the trade-offs here.
I think the state-of-the-art involves combining both like quantization aware training to some extent and like model scaling with u post- training quantization of various kinds like yeah there's a lot that I think is not fully well understood. even industry teams that I've talked to are sort of talking about still doing like science of quantization.
So yeah.
Okay.
So the other trick this one is much more of like a conceptual programming thing is operator fusion.
It's it's fairly simple idea but you'll be surprised at like how much this doesn't already happen. so think of a GPU as a factory.
You've got your memory warehouse and you've got your little compute factory and you've got a little bell conveyor going back and forth, right? if you scale up your compute, you're going to be bottlenecked by your little bell conveyor.
And this will be worse if you have many operations, right?
Like imagine I've got lots of different operations I need to do and I'm going to ship my you know raw materials to my factory and back and forth and back and forth.
This is a lot of memory bottleneck that you're paying for, right?
And lots of sort of duplex birectional memory bandwidth.
Now wouldn't it be better if I just had a one giant factory that would take all the raw materials and ship back my finished products, right? this way I only have to pay for memory bandwidth twice.
Very s very simple idea.
This is operator fusion, right? and this appears very often, right?
If you write a very simple let's say computation of sin^ squar plus cosine^ squar, what does the pitor torch computation graph look like?
It looks like this.
You got your x, I do my sign, I do my cosine, I square it on both sides, I add them, and I get my output, right? naively, if you don't do anything smart, what this will do is each of these will be a factory, so to speak, right? you will read and write from global memory each one of each time one of these is called, right? and you will incur quite a bit of memory cost every time you do so.
So, this is pretty painful. and you can imagine instead of doing that, why don't I just write GPU code that reads once from global memory, does the whole thing right, inside the inside the SM as I talked about before, and then after it's done inside the SM, I'm going to write it back to global memory, right? single sort of kernel that does all these operations.
This is called fusion, right? and anything that's easy like this like this is a very easy fusion.
It's like a graph that I can sort of just you know squish down into a single unit. a compiler can automatically do this for you.
So torch compile which is a compiler for PyTorch or Jax compile which is you know Jax's compiler which is more deeply integrated into Jax.
Both of these will fuse these kinds of basic things into a single CUDA call.

I'll talk a little bit more about fusion later but I think this is you know the advanced versions sometimes require sort of manual intervention but the simple versions you can just think of as you know compilation will solve your problems.
Okay the third idea is recomputation.
So we know in back propagation what what do we do?
Well we go forwards upwards up on the tree.
This is from Percy's 221 class.
You know you go upwards on the tree and you keep track of activations right? you store these little activation numbers and then when I go backwards I'm going to use the activations that I stored to compute sort of the backwards signal down my my tree right we all know back prop to some extent so you know we know what this is but also now you know we're looking at this not from the perspective of like mathematics how do we compute a gradient we're thinking about this from the perspective of systems right like how do we minimize our memory well if we think about what's happening let's say I have something simple I have x sigmoid sigmoid sigmoid note right each of these sigmoids is going to generate in the computation graph some sort of residual right or some sort of a activation that I need to track so I after my first operation I have S2 second operation S1 and then when I need to do my backwards pass it will consume S2 S1 and my output or sorry the gradient at my output and I'm going to push this backwards towards the derivative of my input right so if I count the number of memory reads and writes it's going to be one memory read for X three memory rights for S2 S1 one out and then on the backwards pass this is sort of flipped right so this is a total of eight memory reads and writes very low arithmetic intensity because I'm just stacking a bunch of sigmoids now what can I do instead well what I can do is I can just throw away all of the activations normally you wouldn't do this right like why would I want to recomputee something but imagine you're in a world where computation is super super cheap you just have abundant computation you're not using all of your units but your memory is very expensive like any read and write has a lot of penalty.
Then you might prefer this computation graph where I read an X and then I'm going to do sigmoid sigmoid sigmoid output.
I'm going to store this output.
I have to write this because that's the output of my graph.
But now in the backwards pass, right, my backwards graph is going to get in D out and sort of on the fly I'm going to reforward the X and then compute the activations on the fly right where I need them.
Right?
So now this is going to be two memory reads for the D out and the X and one memory right for the DX.

Right.
So I've gone from eight total memory accesses to just five.
Right?
So now I have 5/8 the total memory accesses.
Same computation, but I've spent a little bit of extra compute and saved a little bit of extra memory.
Right?
This is the another very important trick.
We don't normally think about this because usually we're concerned with like just computing the gradient, but once you have a systems mindset, you start to think about recomputation.

Cool.
Okay.
Trick number four. this is another sort of memory trivia thing but surprisingly becomes important. and also why sometimes it's important to understand the hardware model of what you are you're interacting with. so global memory which you know is the big memory that lives outside the chip is DRAM and DRAM because of the way that it is structured actually comes in bursts.
What that means is when I access let's say the very first position, I don't actually get the just the first value.
I can read off all four of these first values for free.
And that's because the way the memory cells are lined up, there's kind of these blocks of memory units.
And then when I sort of activate an amplifier, I can just read out all of these almost for free, right?
The way that it's sort of structured in parallel allows me to read blocks of memory.
That kind of makes sense, right?
As a hardware model. and so this is called a burst section.

And so whenever a section is accessed, all the other locations can also get delivered for free, right?
And you won't pay for them, so to speak. and so, you know, you might have something like 128 bytes worth of a burst.
That means a single read will return potentially 128 bytes worth of data as long as it lives in the same contiguous block.
So this means that if you have your reads arranged contiguously where they're all together, it's going to go much faster. and to use a term you say that a memory access is coalesced.
So that word means if all of the threads in a warp fall in the same burst.
So what that means is let's say I have you know three threads and all three of my thread or sorry four threads I have four threads and my four threads live inside of a single burst section right or they're reading from this burst section then this is coalesced right because they're sort of using up all of this entire burst section and I'm fully utilizing this very nice property of DRAM memory and you know so if I have you know multiple warps you know I'm going to have different multiple burst sections but you can efficient read out of memory by sort of grouping together your memory reads.

why oh yeah why does the pattern like this sort of shape >> which one like this >> like this one?
Yeah. >> Yeah.
So, so as far as I understand, you know, the memory cells are arranged in a grid and your amplifiers for like deciding which cells to read sort of operate at a column-wise level and like activating the voltage for this is often a thing that takes a significant amount of time.
So once you've like selected the column, readouts of multiple elements within that column are much comparatively cheaper.

>> But this is this is to like select to read that upper upper red cell, right?

Or >> or yeah, right.
So if I want to read this for example, right?
Like you know I'm saucing this or maybe it's columns that are cheap in this diagram at least.
But you know if I'm trying to read this, you know, I might get this whole row for free, right?
Because I've addressed this whole row.

>> Okay.
So I don't get a T for free.
I just >> you don't get a T for free.
Yeah.

Okay.
Good.
Okay.
Okay.
So you're saying like Tatsu, why are you telling me DAM trivia? the reason why I'm telling you DAM trivia is because you know think about matrices right matrices you often want to read in big blocks right because you're going to do mat moles and certain kinds of memory accesses on mat moles are sort of privileged they're they're much more nicely structured than others and sometimes you can take advantage of this to get very very fast memory reads so the pneummonic or you know the thing that I remember is if you have threads and you have a row major matrix so you know it's sorted where you know each row is sort of how you your indexes are increasing. threads that move along the major axis are not coalesced.

And to really make make this visual actually I like this diagram on the right.
Imagine I have this 4x4 matrix where this is row major.
So it's sorted in the order of you know the yellow block comes first, the red block comes next and so on and so forth.
This is the linear ordering sort of in memory address.
This is kind of the matrix that you know I'm trying to store.
Now imagine that I have threads that are trying to read each column each clock cycle.
So I read a column, I move to the right, I read a column, I move to the right.
If I'm doing that at the very first iteration, I hit this element, this element, this element, this element.
Notice that each one of these limit a different burst window and I've read the entire vector so to speak, right?
Or the entire matrix, sorry, so to speak, even though all I wanted was a single column, right?
Now imagine if I was reversed, if I was flipping this matrix, I could read the entire sort of block.
Every thread will get its read in a single sort of read of the memory, right? so coalesced can give you significant improvements in sort of reads from large contiguous structures like matrices.
So this is one reason why you might want to think a little bit about row versus column versus other kinds of ordering.

Okay. the last one that I want to talk about and this is the big one. and maybe the most important idea, I've like saved it for last, is the idea of piling.
And I think this one has probably the big impact on performance. tiling, I think more than any other idea is this idea of trying to do as much as you can with the memory hierarchy.
So what we want to do is we want to group our memory accesses as much as possible and sort of shared memory accesses should be done together and they should be done on shared memory.
So the way that we the reason why it's called tiling is I'm going to cut up the things that I need.
I'm going to tile my space and each tile will be loaded into my shared memory.
And once that tile is loaded, I'm going to repeatedly read from that tile and do as much as I can.
So let's think about the matrix multiply case.
I want to multiply you know this matrix and this matrix and this matrix on the right bottom.
This is my output, right?
This is sort of where I'm writing my matrix out.
This is a pretty standard notation for this. now if I want to do this right I'm going to be reading you know as I'm reading through the threads multiple sorry the same entry will be read multiple times and to be precise each entry will be read n times in an n byn matrix multiply right naively so you might think right if I'm reading this entry let's say m0 multiple times instead of hitting global memory each time why can't I sort of group all those memory accesses together and the nice thing about matrices is that there's a very natural grouping, right? and this is to essentially cut the matrix up into submatrices and then load them and then operate upon them, right?
So, here is a very simple algorithm.
I'm going to cut up my matrices into sort of 2x two, you know, submatrices and then I'm going to load different tiles.
So, I'm going to first load the m00 tile, the top left here.
I'm going to load the n00 tile, the top left of my top matrix, and I'm going to load those into shared memory.
Right?
So I'm going to pay the cost of transferring those slowly from global memory once.
But once they're in shared memory, I can read and write very fast with those tiles, right?
This output tile will also be for now in shared memory.
And I will operate on these submatrices entirely, right?
So I'll multiply these two submatrices and I'll put the partial sum into my output submatrix.
So that allows me to do a bunch of reads and writes all in the shared fast memory and then I can write it back out when I'm done. what does the math look like?
Well, in the non-tiled case, I already told you, right?
Each element of my input is going to have to be read n times when I do this big n byn matrix multiply. if I tile them with a tile size of t, then each input is going to be read n over t times from global memory and then t times within each tile, right?
And this kind of makes sense the extreme, right?
If t is equal to n, I read it once from global memory and access it n times in shared memory, right?
And this is great because it's a T times reduction in global memory access.
If you have big big shared memory tiles, it means that you can dramatically reduce your global memory access on something like a matmul.

cool.
Tiling.
Okay.
Yes. >> So, are you doing this because you can't >> Yes.
Yes.
Yes.
Yes.
That's right.
Yeah.
Like you know if you were building one of these like very fancy alternative accelerators like rock or something which is all SRAMM then yeah you just put the whole matrix in there and you're very fast for the whole thing but very expensive.
Very very expensive.
Yeah. >> And then this also [clears throat] happens.

So like and then you just keep adding that cell again and again.
Is that kind of >> you mean like are you asking whether this kind of like splitting happens at like the accelerator level >> for a tile across >> yeah GPU this will be one tile >> that's right yeah so data parallel you can think of as this like that's you know if I have data on this axis and weights here if I cut it on one of the axes that's kind of data parallel tensor parallel could be another kind of parallelism all these parallelism look a little bit like tiling in some ways yeah cool okay oh yeah.
So for this trick and the previous one, can I assume that you know a library like Python will just do that? >> Piling.
Yes.
Yes.
Yes.
Yeah.
So like any like matmul kernel that you call will do tiling underneath.
Yeah.
That's right. if you're doing some sort of exotic operation, you might want to think about tiling like things happening, right?
You want to be able to locally reuse your your reads. but I will say even PyTorch or something is not perfect.
And so, you know, we will talk about some of the imperfections that will happen. so tiling is very very powerful as an idea, but it's also complex.
You can actually end up with some like really subtle things happening with tiles.
So imagine I have a tile size of 1281 128.

You know, if my matrices are 256 x 256, life is good.
I've got four tiles, right?
Perfectly cut.
Now imagine I increase my coordinate size by one.
Right?
Now I've generated two very skinny tiles with basically nothing inside of them.
Right?
So now you kind of realize, you know, actually this tile sidings sizing thing is actually pretty complex.
Depending on my matrices, I might need to adjust my tile sizes to somehow, you know, trade off my memory bandwidth versus, you know, this gigantic empty space where I'm doing nothing, right? so you know, you have to tile sizes is actually a thing that you optimize.
Okay, so like coalesce memory accesses, the size of your shared memory and accelerator and the sizes of your matrices all affect you know which tiles are optimal and if you know about different options in PyTorch you know there's a compiler option called max autotune.
I don't know how many of you have used this or know about this.
You know, you turn on that flag for the next 15 minutes, PyTorch is going to like, you know, run an endless series of benchmarking tests and you'll realize actually what it's doing is it's like trying all sorts of different tile sizes on your matrices and trying to figure out which one is actually faster. and you'll find that you'll get, you know, non-trivial benefits from actually optimizing these things. the other thing that you'll find related to coalescing is if your tiles or if your matrices are the wrong size.
What you can do is that your attempt to read a tile might actually trigger a lot of memory reads.
So imagine I have this nice world where you know I have let's say this blue one is the row and I want to read this nice tile and my burst sections are sort of the width of the tile.
And in this case, if I do sort of four reads, you know, I'm good to go because because of the burst sections, I've read this entire submatrix, right?

So good memory reads from coalescing.
Now imagine I shift the matrix size by one.
Now I've bumped this first row over.
So this blue box now lives over here. and there's no way for me to coalesce my reads to read a single tile in one go, right?
Because now all these things are sort of ragged and misaligned.
So I need to do at least two tiles worth of reads to read out this one tile, right?
So, you know, there's actually a lot of complexities involved in optimizing tile reads once you start thinking about coalescing and all these other things.
And so, in this case, what you have to do is padding in order to line up essentially the boundaries of the burst sections with the tile sizes that you have.
Okay. and this is why you end up getting stuff like this, which seems very mysterious.
I think this is one of my favorite weird ones, right?
Andre Karpathy when he was doing nano GPT speedrun said well you know when I increased my vocap size to 5257 to 5304 you know I got a 25% speed up right and it's kind of weird to have to pad in order to get a speed up but maybe it makes sense once you realize things like this right padding can start to shift the rows and columns so that you actually get very efficient reads whereas at other times you do not.

>> Go ahead.
Is there a is there usual like tile size that you should be taking into account? >> You don't want to think about tile sizes per se because tiles you should normally think of as like a lower level systems thing unless you're writing your own kernels.
But you do want to think about the size of your matrices.

>> Okay, but you want your size of your matrices to be divisible by the file size. >> You want to be right.
So, so you want it to be I'll talk about that in a moment.

Right?
Generally powers of two ideally divisible also by like 32.

Right?
Okay.
So now we can talk about this matrix mystery over here. we have essentially all the pieces that we need to understand this.
So we'll walk through this one by one. we've already talked about compute intensity, right?
The roof line plot.
You need to have enough work for each memory read in order for you to be able to saturate your compute unit.
That's what's driving the sort of diagonal upward trend.
So that one is done.
Now let's talk about some of the other ones.
First one's tiling, right? if we look at each of these lines and we color them based on essentially the divisibility of the matrices, you will find that you know the blue ones are essentially only divisible by one.
They're not even they're not even even. yellow ones are sorry orange ones are divisible by two, right? green is divisible by eight.
K red is 16 and then purple is 32.
And you see, you know, that the ones that are not divisible by, you know, one or just one or just two actually have very bad throughput.
But once you start to get to sort of higher multiples, you start to essentially get no sort of diminishing or sorry, no penalties, right?
Like 16 or 32 perform equally well.
And this is really it's not because of like somehow powers of two are magic or something like that.
It's because 16 and 32 are sufficiently large that it has the right sort of burst window property.
It sort of fits the whole burst window in now.
And so you can get coalesced reads into your entire tile and it has a major impact on alignment. now if we look at sort of what's remaining if we look at this orange line we'll sometimes see this weird periodic behavior where suddenly performance will drop right and this is very weird because I'm just multiplying matrices.

Why does adding a single extra dimension suddenly kill my throughput? and this is you know I think using all the things that we learned is kind of fun to get to this end.
Percy calls this GPU trivia.
So I think he hates this one, but I love this one. so why is it that going from here to here I get such a big drop in throughput?
Well it happens between 1792 to 1793. and why does this happen?
Well, I'm using a tile size of 256x128.

So that means how many tiles are there?
Well, there's 98 at 1792, but once I bump myself up to 1793, well, actually, I've increased both tile sizes by one, so I now have 120 tiles.

Okay, so I have 98 tiles before and I have 120 tiles now.
That's bad, but it's not this bad, right?
This drop is gigantic.
Well, it turns out the reason why it's so dramatic is because a A100 has 108 SM, right? and 98 tiles fits in 108 SM, but 120 tiles does not fit in 108 SM, right?
And so you finish doing 108 SM and you're like, well, I've got 12 more of these guys.
I got to I guess process them and most of your your GPU is sitting idle, right?
So you can end up with these very weird effects once again where, you know, by bumping up your tile size by one, suddenly you need to wait a whole another cycle for you to process a bunch of tiles, right?

that's called wave quantization.
Okay, so that was all for part two.

how do we make machine learning workloads go fast on a GPU?
Well, a lot of the story is really just about memory, memory, memory, memory, right?
So, how do we reduce the amount of memory access we have to do?
Well, let's coalesce our memory reads together.
Let's fuse our operations together so we don't have to read as much.
And for the things that we have to read no matter what, well, let's reuse the memory, right?
So shared memory move things to shared memory and reuse it as much as possible which is tiling.
And then finally maybe you can trade other resources that you have for memory and that's quantization and recomputation right like so all these ideas very much relate to this idea of like memory is our bottleneck how do we deal with that right cool okay any questions for the last finishing yeah >> we just bump up >> very expensive very energy I mean there there's two reasons one is it's you know just expensive to build SRAMM also very difficult because it has to be physically close like you know signal propagation is hard and I think a third reason is that SRAMM and DRAM are just kind of different storage mechanisms like SRAMM needs to be powered the whole time to hold value very energy hungry and so you know if you're thinking about total cost of ownership energy consumption maybe you don't want a giant SRAMM chip you really want to be using it where you need it >> yes those things scale up like 32 or like if you offer it in terms of like 64. >> You mean like does it help if I go higher? >> Yeah. >> No, because I think you know this is really it's kind of like as long as it divides by your burst size then you're good to go, right?
It's a divisor property. there are other phenomena where you might get benefits from you know larger divisor but not not here.

yes [snorts] >> that's just the number of SMS.
You will have to ask Jensen for that.
I do not have an answer for you.
Yes.
So yeah, going back to the to the hardware side, sorry. not this related but I it just occurred in my mind.
Have you studied GPU versus CUR versus wafer scale engine? >> Yeah, I mean I think sitting over there knows much more about wafer scale stuff than I do.
We were we were talking about last after last lecture how difficult the compilation was because if you have something wafer scale then you have like wave interference effects and all sorts of horrible things that happen.
That's a world that I'm not familiar with but I think is much more complicated than what we have here.
Okay, cool.
All right.
So now I think we can talk about in some sense the finale which is we've learned a bunch of things about GPUs.
It would be nice if we can tie that all together to essentially be able to reproduce or understand something that I think is really important in advance which is flash attention.
Okay, so flash attention is one of the big I think improvements in attention capabilities in a while and it's all systems, right?
So it's going from just PyTorch's naive implementation of attention to a single very cleverly fused kernel you know this is from the original paper but you've got this dramatic improvement in the latency of serving these things and as I showed you in I think two lectures ago right flash attention is a nice memory efficient way of computing things so you can do bigger and bigger attention and you can see you know a lot of the gains are going to come from improvements in the amount of memory that is transferred from HBM which is the global memory, right?
Okay.

So, what what do they actually do?
If you read the paper, you know, they'll say, okay, we just do two things.
We apply tiling and recomputation, right?

And then we do it in a way that allows us to do things sub quadratically in the number of memory accesses.
I think we understand all of those components now.
So, let's see what's actually happening.

So, remember attention computation, right?
Has a couple things.
It has a kq v and we do a you know multi matrix multiply with the k's and the q's and then we have a matrix multiply with a v at the end and have a soft max in the middle.
Right?
So those are the operations that we need to deal with.

Three matrix multiplies and one soft max.
So we have matrix multiplies.
We know how to do them now.
Right?
So we're going to do a tiled matrix multiply. figure one from the paper is literally just tiled matrix multiplies.
Right?
We've seen this before.
We're taking our inputs our k's and q's.
We're cutting them up and we're just multiplying them sort of chunk by chunk on a tile and then after we do the computation we write them out.
Nothing fancy or special here.
The thing that is tricky about flash attention and you'll see this in the assignment is there's a big global softmax on the whole thing, right?
So we can't just do this naively because the softmax is a global operation that ties everyone together, right? [snorts] So it connects all the different tiles together. it might not seem obvious how to tile attention.
The key trick and once you see this, this is really the core I guess idea or observation.

Once you see this, the rest is very simple.
Normally to comput a soft max, you would do it kind of this way as shown in this plot here, which is you would exponentiate all of your elements.
You would subtract the maximum for numerical stability reasons and then you would normalize that, right?
Standard softmax.
Now there's a different way of computing this online and I think once you sort of realize it's kind of simple which is you're going to sort of as you go compute the normalized softmax and every time you encounter a bigger number than what you've seen before you're going to sort of swap out the maximum right and so you have an online way of computing sort of the soft max as you go.
The only tricky thing of course being the max and the way we deal with the max is we keep track of the max and then we sort of swap it out and correct for it every time we see a new maximum.
Otherwise, this is really just an online running sum of an exponential from everything that we've we've seen before.
And of course, at the end, you'll just divide everything by the accumulator that you've computed online.

Now, the important thing about the online softmax is because it's online and it goes kind of block by block or piece by piece, we can compute this tile by tile, right?
I don't need to see the rest of the tiles in order to compute this.
So I can compute the softmax for my tile and then I can sort of save the partial results maybe write them back to global memory and then just sort of continue on with my day.
So this allows us to now break up sort of all the different interacting pieces into something simpler.
So this I think is a I think this was from flash attention two or maybe three I think it's two.
I like this visualization which shows you know which parts are being tiled.
So the dashed blocks are being done tiled in SRAMM.
The blue boxes are the ones that are in high bandwidth memory, the global memory.
And what you see is let's say we do the KQ you know inner product.
You know these are tiled matrix multiplies and they're stored in SRAMM and we keep them in the SRAM and we compute this exponential.
And then similarly in this tile we're going to compute this running softmax.
And I'm going to move to my next tile and I'm going to do the same thing, but now I'm going to keep sort of the partial running sums of my softmax as I go.
I can keep them sort of in shared memory or in registers or elsewhere.
This is a very small number of things to keep track of.
At the end, after I've computed all these exponentiated quantities and my normalizers, it's very simple to just multiply it with a v once again in the tiled form and then divide at the very end.
Right? this allows me to essentially take the attention operation, cut it up into these small tiled chunks, and then go.
Finally, you've got, fusion.
You're going to do both of these together, right, as much as you can without going back and forth. and then I'm not going to cover this or talk about it. but what you can do is you can, in the backward pass recomputee everything that you need.
Right?
So the last piece of flash attention is if you saved your activations, you still have this n squed sized activation that you're going to need to save for your attention.
Instead of saving that, throw it all away and recomputee everything tile by tile on the backward pass.
Right?
So it's got the recomputation piece as well.
Okay, that gives you flash attention.
Now we've seen all the different tricks in action. and that's essentially all there is, right?
A lot of the tricks that you see for efficient implementation and how to use GPUs well boil down to those kinds of ideas.
So to put it all back together, right, important to understand the hardware.
The hardware in our case, the GPU is powering scale and compute.

And we need to understand really all the way to the low-level details to fully grasp why we make certain decisions.
Like I think, you know, I don't want you guys to be the kind of people that cargo col, you know, 32 multipliers for your matrices.
I want you to really understand you know why we do those things and that goes all the way down in my opinion to the hardware level. the other thing is that you know the current you know way that compute is growing means that we really want to think about matmuls those are the core operations that are very arithmetically dense and because of the gap between compute and memory we need to think very carefully about data movement right and then finally things like flash attention I think are really cool you there's also like architecture work that's like very hardware aware thinking carefully about the hardware when we do things like architecture is, you know, critical to good performance for future systems.
Thanks,

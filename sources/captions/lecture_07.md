# Lecture 7 字幕

Okay, let's get started.
So, welcome back everyone.
So, today we're going to talk about parallelism.

And remember in the last week we introduced how to make a single GPU go fast by writing kernels. and we really looked inside his GPU.
And this week we're going to talk about how to leverage multiple GPUs to make your code go even faster. so the picture you should have in your head is something like this.
So for the last week, we focused on one of these boxes where you have your GPU. remember you have your high bandwidth memory, HBM, L2 cache, L1 cache, registers and a bunch of streaming multip u streaming multiprocessors. so now the picture gets extended because instead of having one GPU, you might have four, you might have a thousand GPUs and those are going to be connected. and I'll talk later about how those GPUs are going to get connected.
And then you're going to have to figure out how to leverage all of this compute to train models.

So in both cases meaning both in the single GPU case and in the multiGPU case as we'll talk about today the situation is kind of similar if you zoom out which is that the compute the arithmetic logic units the tensor cores so on is far away from your data and far away for a single GPU means all the way over here in HBM and now if you have multiGPUs the thing that you need here might be all the way on a different GPU you and you're going to have to shuffle that over somehow.
But the same principles are going to be the same because the game is to orchestrate the computation to try to avoid data transfer bottlenecks.
It's very easy to use a ton of GPUs, but it's hard to use them effectively.
So taking a little bit of liberty we can think about the generalized hierarchy where at the at the sort of local level next to the SMs is a single node single GPU where you have an L1 cache shared memory.
This was the fastest and then you had HBM which we lamented was so slow but in this lecture HBM is going to be considered fast. now we're going to think about the single node multiGPU setting where the GPUs are going to be connected via NVLink and NVSwitch. and then finally the multi- node multiGPU where we have to resort to infiniband and Ethernet depending on what network you have.

Okay.
So last week we talked about various tricks for imp reducing memory accesses fusion and tiling read into shared memory do everything you as much as you can and then write it back out and this week we're going to talk about how you can reduce amount of communication across GPUs by replicating and sharding appropriately.
So, so why do you do GPU multi-GPUs? the obvious answer is well, you want to scale, but to put a finer point on it, there's really two reasons.
One is that your parameters or activations or gradients and optimizer state don't fit on the HBM memory of a single GPU.
So, B200 has 192 gigabytes.
If you're training a one trillion parameter model, that's not going to fit on a single GPU.

And the other reason is that even if your model could fit on single GPU, you might want to leverage more GPUs by splitting everything up to train faster.

So sometimes there will be kind of some decisions to be made because you could fit everything on GPUs, but you have fewer cores. but if you spread out, then you're going to have to pay the communication bandwidth, right?
So that's some calculation you're going to have to do to figure out how to paralyze.
Okay.
So just one note here is that so far this lecture is Python.
You execute it and you can just show everything.
Now in this lecture if you run this directly it uses multipprocessing. but when I trace through it I'm putting in some sort of like special single process mode.
So if you want to see the standard out for this lecture run in the multiprocessing setup you can click here and I'll show this as we go through the lecture.
But just remember as we step through this lecture we're not actually doing multiprocessing because we're just stepping through single lines of code.
Okay.
So this lecture is going to include two parts.
One is we're going to learn about the building blocks of distributed communication and computation. starting with the programming model, talk a little bit about the hardware, start to implement things in torch which you're going to do on your assignment too.
And then the second part is we're going to look at actual training. we're going to look at three types of parallelism. data parallelism, tensor parallelism, pipeline parallelism.
Each is going to cut up our model in different ways.

We're going to do this for MLPS. rather than the full transformer but it's really sort of the core computation is going to be shown here.
Okay.
So let's dive in.

So the first thing to talk about is these things called collective operations.
So collective operations are these primitives from distributed programming that go back to the 80s.
So the idea of parallel programming is very old. it's wasn't invented for LM training. and it's still the case that these primitives are the ones that we use today.
And here collective just means that you're specifying a general communication pattern or a template across multiple devices rather than managing pointto-point how this GPU is going to communicate with another GPU.
And this is going to be much easier and the systems can do a lot more work for you.
So this is a very tried andrude interface for doing parallel programming.
Okay.
So the general setup is as follows. the terminology here is a little bit I find it a little bit strange but this is standard in parallel programming.
The idea is that you have a bunch of ranks where rank corresponds to a particular device.
In our case, a GPU could be a TPU.
But the point is that you have let's say four ranks here.
The world size corresponds to the number of devices.
So the world size here is four.

Okay.
So there's a few operations we're going to go through. broadcast, scatter, gather, reduce, all gather, reduce, scatter, all reduce, and all to all.
And each of these operations is going to specify how this set of ranks or devices is going to transfer some amount of data slashcomputee with it to some other set of devices.

So the first three broadcast, gather, gather, reduce are just warm-ups.
I would say they allow you to get a sense of how these collective commission collective operations work but they're not really going to be the ones that are driving most of training.

All gather reduce scatter and all reduce are the ones that are going to show up again again for distributed training of language models.
And finally all to all I'll just mention here which is important forees but we're not going to actually spend too much time on this lecture.
Okay.
So let's dive in with the simplest operation which is broadcast.
So in broadcasting you have a rank zero.
It could be any rank but let's just say for sake of picking one rank zero has some tensor 0 one 0 one two three and it broadcastes to all the ranks.
So at the end of this operation we have that each of the ranks has the same tensor on it.
Okay, that should be pretty straightforward. and this again doesn't really show up in the like kind of the core path of training.
Generally broadcasts are used for kind of initialization where let's say you initialize a load initial checkpoint and then you br broadcast it to all the ranks.
So something that's done like once.
Okay.
So the second operation is a scatter.

And a scatter basically says I have a tensor at rank zero whose is split up into the world size and I'm going to basically scatter my tensor onto the other ranks.
So rank zero gets zero the zero component rank one gets this rank two gets this and rank three gets that.
Okay. so again this is not directly used but scatter is a important stepping stone to understand reduce scatter.
So, as the name implies, scatter just takes a big tensor at one place and spreads it out onto multiple places.
And you can see how this might be helpful because you want all the GPUs you're scattering to do some local computation on the different parts.

Okay, so the inverse of scatter is gather. so this should be very predictable. the input is you have a dump for a bunch of pieces each of which reside on a particular rank and then when you do gather that's with respect to a particular rank rank zero it's going to just concatenate all the pieces together.

Okay, again gather isn't directly used but it's going to be a stepping stone to understand all gather.

So next is reduce.
So those of you who do you know functional programming are probably familiar with reduce is.
It's exactly the same.
The idea is that you have you start in the same starting point as a gather and where the denser is split up across multiple well you have some piece of data on each of the different ranks and then you're going to apply your reduction operation to all of these and get that in put that on rank zero.
So in this case if you do a reduction with sum then you add these all up you get six.
Okay.
So you can think about gather as a reduction where the operation is you know concatenation if you will.

Okay.
And of course reduce is important to understanding all reduce.

Okay.
So let me pause there. those were just kind of the warm-ups just in case people have any questions about what a collective operation is. broadcast scatter gather reduce.

>> Yeah.

Or is it? >> So the question is this related to broadcasting and nonpie?
I mean I think it's the same I conceptually the same idea where you have one thing that goes to many things like in numpai if you have a scaler get broadcast to a tensor but the instantiation this is for collective communication so it's a bit different okay so let's move on to something more interesting so all gather is what you do to basically you perform gather to all ranks, not just rank zero.

So remember what gather does, it basically takes all the different pieces and it just puts it on one rank, rank zero.
And now all gather just does it for every single rank.
Okay, that's what the all is for.
All means do it for all the output to all the ranks and gather is what you're doing to all the ranks.

Okay, so this is going to come up a bunch. it's not important that you understand this statement precisely, but later we'll see that each rank holds part of the parameters.
And then what you need to do is all gather the parameters to get the full parameters for the full forward pass.
So in general as we're doing training we're going to see a lot of this gather to do something and then scatter and then gather and scatter again.

Okay.
So reduce scatter is performing reduce on each dimension and then scattering you know the results.
Okay.

So you have let's say you have you know four devices and each of them has some you know vector and so when we did a reduce before we just had 0 1 2 3 and that got reduced to six but now reduce scatter says that for each component of this tensor I'm going to do a reduction and then I'm going to put it on a different you know you know rank.
So the first dimension I'm going to add these up I get six. now for the second dimension I'm going to add these up.
I get 10.
For third dimension I'm going to process these.
For the fourth dimension I'm going to process those.

Okay.

Okay.
So, so where this is going to show up just as a to foreshadow things after the backward pass when you sum the what you're going to do is each GPU will be dealing with different data right and what you're doing is you need to sum all the gradients from the different you know shards and then you're going to distribute redistribute this you know storage Okay.
And then finally all reduce. if you understand all reduce scatter and all gather, it's basically you do one and then you do the other.
Okay.
So what this does is reduce scatter the same input as we had you know before.
And remember in reduce scatter we had 6 10 14 and 18 sit on different ranks and the all gather part of all reduce just puts them all on the same everything on the same node.

Okay.
So all reduce is in some sense the easiest to understand.
You have a bunch of tensors you reduce in this case sum and then you replicate them on all the nodes.

Okay, so we're going to see this one actually first when we do a data parallel where we sum the gradients and then we replicate the full you know parameters.
So that's where we're actually going to start.
So maybe just focus your attention on all reduce.
Later we're going to see how to get to fancier things like ZeRO or FSDP we need to break the all reduce into reduce scatter and all gather because then you can intervene and you can manage things a bit more but for the basic version all reduces is fine.

Okay, finally all to all.
This one is the in some ways the most general.
You basically specify how each rank sends a particular message to another rank. and so here's a you know simple example where you have the same input as you know before.
And what this is saying is that I want to send zero to this element to rank zero. meaning keep it myself.
I'm going to send one to rank one, two to rank two, and three to rank three.
And then for if I'm rank one, I want to send four to rank zero, five to rank one, six to rank two, and seven to rank three.
So basically the position here is going to denote where which rank is going to be the ultimate destination.

And so if you look at the output, what happens is that the first rank is going to receive everyone who sent everything in column zero because all these ranks are sending these things to rank zero. similarly for rank one, all of the ranks are sending this column to rank one and so on and so forth.

Okay, so this is going to be useful when for training and the intuition here is that each rank has both a split of the data and also a subset of experts and basically you in the key idea of thee is that it's sort of dynamic routing. you have to look at your data to figure out which experts are you need to route that those activations to.
So it ends up being a allto communication. so if you look at if everything were balanced meaning that every rank sent the same number of you know bytes to this to every other rank then all the wall you can think about as essentially a transpose.
If you think about this as a matrix, all you're doing is transposing that matrix.
But in general, alt also handles unbalanced splits. and I'm not showing this here, but you can configure it to send any number of you know bytes to any other another rank. but in general, you want the splits to be as balanced as possible.
So remember Tatsu's lecture where we had load balancing to make sure that things were as balanced as possible.
So morally the ideal goal is to have thing all to all look kind of like this.

Okay.
So just to summarize maybe a few helpful tips to remember the terminology because I just went through like quite a few different operations.
So reduce is well it's reduce it performs un some sort of associative commutative operation could be sum could be max could be min scatter is the inverse of gather scatter distributes gather centralizes and all just means that the destination is all you know devices so that explains all reduce and all gather.

Okay, let me pause there to take any questions about collective communications. >> Yeah, for operations such as like gather reduce where we like where we like basically like these wall ranks to like zero does that rank zero like is it like a particular GPU every time or can that rank zero change?
Yeah.

So the question is when you do a gather or a reduce the target where you write the output right now I said rank zero. you'll see later in the code you basically specify the GPU ID or the rank and it goes there.
So it doesn't have to be determined like way in advance, but it has to be determined basically when you execute the call.

Cool.
Anything else?

Yeah, these are just like conceptual.

They're not like actually like are there like these the actual Yeah.
So the question is are these just conceptual building blocks or are they code?
So I'm showing these right now as just conceptual building blocks but we'll very quickly see how these are implemented in code.

Okay.
So before getting into code I want to talk a bit about the hardware in particular how GPUs are connected because we already know what's inside a GPU. so you know let's talk about networking. in general.
So this is kind of a very classic picture.
You can tell from this very old looking image that you know this is how computers used I mean generally work.
So you have this you know server and then you have a bunch of CPUs.
There's a PCIe bus which you connect things like your or used to connect things like your mouse and keyboard and then you have a bunch of GPUs sitting off of them and you have some RAM and then this computer is connected to Ethernet to another computer and so on so forth.

Okay.
So this is a particular setup because a particular topology and you know GPUs on the same node you use PCIe to communicate and GPUs on different node you have to go all the way through Ethernet.
Okay.
So this is like if you bought your like gaming GPU and you had you know hooked it up with your friend and he's like I'm going to train some big model.

that's what you would have to do.
But you know if you're really serious about training then things look more like this.
This is the picture I showed in the very you know beginning where you have the GPU and there's something called NVLink and NVSwitch and infinipad.
So the typical you know setup is this and these numbers eight is typical but this 256 is kind of made up.

So typically you have eight GPUs per node. and these are connected via Nvidia's NVLink to a switch. and just for kind of calibration if you use NVLink five then you're getting 1.8 terab it's terabytes per second of total you know bandwidth.
And remember HBM was for B200 was 8 terabytes per second.

So it's, you know, about four about 4x slower.
So I mean this is still pretty fast if you think about going between devices, but obviously not as fast as high bandwidth memory, which is much slower than you know shared memory or L1 cache.

Okay.
So basically a NVLink connects to the switch and which means that from a programming perspective you can think about GPUs as connected to any other GPU right you go GPU to any other GPU and the hardware takes care of transmitting that to the switch and the switch routes it.

Okay.
So typically what you will also have is that at some point you can't have NVSwitch and NVLink and you have because as your clusters you know number of GPUs grows then you're going to have to put these nodes into pods or which are connected by infiniband and the way infiniband works is that now there's a bit you know more the so Now the GPU doesn't connect directly to the other GPU. it has to go through PCI E and goes through this kind of special Infiniband cable and you see that the you know the speeds are much much lower and then finally if you run out of Infiniband and you have these like huge pods you know then you need to connect them via you know Ethernet and in Ethernet you know you have to go through PCIe and actually goes through the your CPU which as we'll see is you know even even slower.

Okay, so it's kind of analogous to the kind of memory situation.
The more nodes you have then the slower it's going to be.
You can't have like a NVSwitch handling like a 100,000 you know GPUs.

So one note is that this I mentioned alluded to this bypassing the CPU which is going to be an important thing from a hardware perspective. so if you have traditional Ethernet you know what happens is that the GPU has to talk to the CPU to get its data copied.
So it basically has to copy the data to this the CPU has a kernel socket buffer here is kernel means you know not the GPU kernel but this the CPU traditional notion of a kernel and then has to build some network packets copy to the network interface and then ship it over.
So this generally introduces a lot of latency and so there's this technology called remote direct memory access RDMA which allows a GPU to directly write or read from another GPU's memory without using the CPU at all.
So obviously if you're in NVLink land and NVSwitch land then you have RDMA.
Infinibad also supports RDMA.
So if you're connected via Infiniband then you can directly have GPUs connect to each other without access involving the CPU.

but standard Ethernet does not.

there's two notable advancements I will mention is that so Nvidia has been really pushing the limits on what you can do with larger and larger you know pods.
So they have for the basically the B200s and the B300s they have something called NVL72 which means that they have these trays of eight GPUs but have nine of them and so basically at the end of it you have 72 GPUs that are all kind of NVSwitched into one NVLink domain and if you remember the NVLink you know speeds are very fast right so normally if you're if you're you know mortal you think well okay I have eight GPUs that are interlin really fast and then outside of that you know then things get slowed down a lot but you have a lot of money you can buy this really fancy hardware and you can get really fast interconnects up to 72 GPUs and the other thing I'll mention is that you know I said standard Ethernet doesn't support RDMA but there has been progress on the Ethernet front as well.

So there's something called ROCE RDMA over a converged Ethernet where the Ethernet actually bypasses the CPU.

and this is sort of their their answer to infiniband.
So Infiniban generally is very expensive as is you know a lot of Nvidia products. but you can get you know pretty good performance by using using RDMA over converge Ethernet and Meta had some papers showing that they were use exploring this.
So, llama may or may have been trained over, converg.

Okay.
So, that's just a kind of brief overview of what the hardware looks like.
You have GPUs.
They connect via NVLink to NVSwitch over some domain, maybe eight, maybe 72, and then it's infinite band from there.

And now let's talk about, you know, how do you program this?
So at the very lowest level there's something called the Nvidia collective communications library n NCCL or nickel which translates the collective operations the all reduce reduce broadcast into the actual low-level packets that are sent between GPUs.
So what nickel does is it when you when you use nickel it's basically like saying I want to all reduce and then nickel goes and figures out what is the topology of the hardware figures out the path between different GPUs and then it actually launches the GPU kernels to send and receive data because at the end of the day remember everything that runs on a GPU is a kernel.
So there are communication kernels as well that actually do you know communication with other GPUs.
Okay.
So we're not going to look too too much more in at nickel but just know that it exists.

and then we're going to actually go to PyTorch.
So maybe before that any questions about you know hardware? >> Yeah.
Could you like describe it like a rack and tray? >> can I describe physically a rack and a tray?
Like what a rack is and what a tray is. so for the NVL72, so I'm not I'm not a hardware expert, but a rack I mean is literally like I mean if you've seen data centers like literally a rack and each tray is something that has so G the G stands for grace.
So there's you know two CPUs and each CPU is connected to four GPUs.
So each tray has eight GPUs on it and they're stacked and everything, you know, is connected to this, you know, MB's switch. >> Yeah.

>> Yeah.
So the question is what is the difference? >> Yeah.
So RDMMA you can think about it as a more of a diterata.
RDMMA means that one GPU can read and write from another GPU's memory.
And there's multiple ways to do RDMA.
One is to use NVLink and NV switch.
And another way is to use Infiniband.

So, Infiniband and NVSwitch and NVLink are more the hardware like what what pieces like what cables are and switch is are there and RDMA is more operationally like what what happens when you're communicating >> so we just like one week there will be other ways not Yeah, for example, this advancement RDMA over converge Ethernet is another way to do RDMA. >> Okay.
So, so we draw >> okay yeah >> is nickel optimized for multi-node clusters cuz obviously like it seems intuitive for but like for like RDMA based processes would nickel be for that?
So the question is nickel optimized for multi- node clusters? so my so I don't know the details of how you know they have or have not optimized it.
All I can say is that you know Nvidia has been basically you know optimizing their entire stack for you know inference and training of these large models because their main customers are the kind of major you know providers of language models.
So I would be surprised if they haven't thought of optimizing for those type of workloads.

>> Yeah. >> What's the most common way you have workloads that So, the question is what happens if you have nine GPUs?
How do you distribute the workload across those?

I suppose I guess it kind of depends on how the nine lands in your your setup.
For example, a lot of times you'll have you know, let's say you have eight GPUs per node.
So then the ninth one would be on a different node.
And if you don't have you know, NVLink connecting them, then that's going to be really bad because that's going to be one node which not providing that much compute and also very expensive to communicate with. but if you let's say had a everything were connected by MB's you know switch then it would be much more reasonable. >> Okay. one more question I'm going to move on.
Yeah. >> TPUs. >> So how is this different from TPUs?

so Tatu to describe TPUs a bit more.

So I mean the let's see what can I say quickly about this? so TPUs are generally kind of much you know simpler objects.
I'm not too familiar with the details of how this like what this each of these components corresponds to but maybe we can talk about it offline.

Okay.
So let's actually get down to some code to take advantage of this hardware.
So PyTorch conveniently has a torch distributed library that provides a clean interface into these collective operations. so you don't have to explicitly think about NCCL. and in fact this library also supports different backends for different hardware.
So if you are on GPUs then you would use the NCCL backend.
And if you're on CPUs then there's something called gloo which still allows you to you know like I said parallel processing has been around for you know a long time before GPUs and you can do these collect operations on CPUs as well. this library also supports higher level models and algorithms such as FSDP but we're not going to use those in the course because we're building things from scratch.
All right, so let's walk through some basic examples of collective operations.

so there's this function called spawn which takes another function.
I'm going to call and it says I'm going to run this replicate it four times where four is the world size.
So let's see what this does. this is actually a wrapper I wrote just to sort of hack around the fact that I can't do multiprocessing in this lecture.
So normally what you would do is you call you know torches multipprocessing.spawn and you call the function but you know I'm going to do this branch which disates disables distributed.
So let's just go through that.
So now now I'm in this function that it's supposed to be running on asynchronously for each process.
So remember world size is a number of processes and the rank is either zero or one or two all the way up to world size minus one.
And so there are worldsized number of these functions that are each running on a process at the same time.
Okay, so I'm on rank zero right now.
So what do I do? there's a setup. you basically configure the master address and port. notice that this is not actually how the GPUs are going to communicate.
This is more for just general metadata and coordination.
The actual data goes through nickel otherwise it'll be very very slow. and if you have CUDA available then you can use the NCCL backend.
I am on my laptop so I'm going to use the gloo backend.

Okay.
So now I'm here I'm running you know pretend I have four of these processes running. so there's this barrier function which is useful as it's a synchronization barrier.
So basically if I see this then it waits for all the processes to get to this point. okay so basically you can think about all the processes are running asynchronously so I don't really control one could completely finish before the other they might finish be interled in any way so if I want to make sure that certain there's some code that is executed before other code I put these synchronization barriers in.
So now the downside of putting more barriers in is that well you end up kind of waiting potentially unnecessarily.

Okay.
So let's try an all reduce.
Okay.

So I'm going to create this tensor 0123.
And I have my rank just to make it more interesting.
Each rank is going to have a different tensor.
I'm going to print out what I have before the all reduce.
So, now I'm going to skip over here and see what's gets printed out. so, rank zero before all reduce has 0, one, two, three. rank one has 1, two, three, four, and so on.
It's the same example as I showed before.
Notice that the print statements are coming in whatever order the hardware feels like it because it's running async, but all the data is there.
Okay.

So now if I do a all reduce and this you pass in this is a PyTorch function.

It pass in this tensor.
You pass in the reduction operation which is a sum and I'm going to say don't do async.
And what this does is it calls in this case it would be glue but it could be nickel.
And which in which case it would spin up the CUDA kernels. it would do the communication.
It takes care of everything for you and then it basically writes in place to the data.
So after the all reduce I have the all reduce operation remember which is the sum of each of these columns but replicated across all the ranks.

Okay.
So, so that's, you know, all reduce.
And if you wanted to be, you know, fancier and do async, then you could say async equals true.
But then it would screw up all these print statements.
So, I'm trying to put more barriers than I would normally would.

>> Yeah.
Question.
In this case the rank is >> so the question is the rank the GPU for this class the rank is the GPU.
Yes.

Okay.
So let's try another example here. so I'm going to do a reduced scatter.

So here I'm going to create an input which is zero through the world size and I'm going to have the output I'm going to allocate the output and so what does this look like before I do the reduce scatter it looks like let's see this where each it's a kind of the same you know input and the output is happens to be zeros but it could be you know just anything and then I do the reduce scatter tensor here instead of writing in place I have a output tensor and an input tensor I say I want to you know do the sum and then afterwards I get basically the input is not touched but the output I get the reduction of each component written into the respective ranks.

Okay.
So yeah question.

>> Yes.
So the question is how does it work to do the all reduces asynchronous?
So what this means is that this is sort of like a monolithic operation.
You say go do the all reduce and it's spinning up you know CUDA kernels. that's going to you know do the communication and remember CUDA is already kind of async with respect to the process processes and now we have all the processes being async and the point is that this code just would return and then you could do kind of other things.
So a typical thing which I'm not going to have talk about this class is overlapping computation and communication. so for example you can do this operation and then you can go ahead and load some other data for the next step which is independent of this operation.
And then when you want to make sure that you actually are done then you can call a weight or a barrier.

Okay.
So let's do the final one which is all gather.
So by now I think you kind of get the idea here.
I'm going to set as the input the output of the reduce scatter.
I'm going to allocate an output. so before the all gather it looks like this.
I have my results from the reduced scatter here. the output is you know it just allocated happens to have some values in it but don't you know worry about it. and then after I do the all gather into tensor and then I will have you know all the different inputs you know gathered onto all the different ranks.
Okay.
And you can see here that indeed proof via example that all reduce is equal to reduce scatter plus all gather.

Okay.
And then just to wrap things up just as I started by a setup then I clean up which you know just it's good practice to clean up.

Okay.
So that was your first example of a torch distributed you know program.

Okay.
So let's do some you know benchmarking. this will be quick since I want to actually move on to the part two.

so how fast does communication happen?
So let's do an all reduce. so here I'm going to all reduce with you know 100 million elements and oops okay let me sorry mess this up okay so I'll reduce so I'm going to create you know this tensor with this number of elements call and remember just like before when we do benchmarking we warm up first and here I'm going to call the CUDA synchronize and also the barrier just to make sure that because there's two forms of asynchrony here the CUDA kernels and the different processes and just want to make sure everything is kind of not running in the is done before I start the time and I'm going to do the all reduce and then wait again with the synchronize in the barrier and stop the time.
Okay.
So remember this is running for every single rank.
So if I look at the output here for rank 0 2 1 3 I have a different time potentially because they're all different processes. each of them is going to report a certain you know measurement and if you want to report one number you can take the average for example.

so now one thing that is useful to do which is analogous to when we were computing MFU is to compute measure the effective bandwidth and the idea here is that well this took 1.6 six milliseconds, you know, how much you know, is that good or bad?
So to compute the effective bandwidth, what we're going to do is to compute essentially how much how many bytes were sent and should be sent kind of during this computation. and then you divide by the total time then you get the B effective bandwidth.
Okay.
So the size of what I'm sending around is the size of each element times the number of elements.
So that's the basically the number of bytes of this data tensor. how many bytes get actually sent.
So this needs some unpacking.
So for all reduce if you think about let's just say for the simplicity you do a rank zero plus rank one plus rank two plus rank three you need to iterate this world size minus one steps because there's a world size minus one you know addition operations though so that's this factor there's a two because you need to both kind of send and reduce so and then you multiply by the size of the payload code.
So that's the total number of sent bytes and the total duration is the time that the wall clock time that it took and then you multiply by the world size because it's like the total amount that the all the ranks have waited and the bandwidth is the bytes sent divided by the total you know duration.

Okay.
So in this case you get something like you know about 400 gigabytes you know per second.

Okay.
So one a few notes here.
One is that the effective bandwidth if you look at this expression size time 2 * world size minus one divided by world size times duration.
So as world size increases this world size minus one divide by world size essentially converges to one.
So you're effectively left with two times the size by over the duration.
So this is essentially the bandwidth notice that this is independent of the world size which is which is good.
So if you grow the number of GPUs you have you are still you know the bandwidth doesn't you know change.
It is also independent of the topology which is something that kind of nickel you know figures out whether you're going to pass the messages in a kind of a ring or a tree topology.

so yeah okay so that is all reduce and for reduce scatter this is very similar so I'm going to create the inputs and the outputs warm up perform the operation and time it so notice that the reduce scatter has this these timings and you can also measure the effective bandwidth here.
The number of bytes that were in the input.
You have the number of bytes that were sent.
Here there is no you know 2x here. and then the total duration you divide by that you get the bandwidth.
So here the bandwidth is it should be very similar.
I guess sometimes there's some stoasticity but it's on in the kind of 400s.

So a few notes here.

All reduce is remember as we stated reduce scattered plus an all gather.

And so all reduce naturally is moving twice the amount of data.
U because reduced scatter is like has some cost.
All gather has some cost and all reduces doing twice as much work. but and it takes twice the amount of time. but you know the two cancel out so you get the same kind of bandwidth.

Okay.
All right.
So that is the end of part one.
Maybe any questions before I move to part two.

do that synchronize for.

>> So why do we have to do the synchronize for the CUDA kernel?
So at the end of the day we are still doing CUDA operations, right? we have just multiple processes each with a GPU doing some CUDA operations and then so if you're doing a CUDA operation remember by default it's async so when you reach the next line in Python that CUDA operation might not be done >> so we need to wait always until that's to make sure it's done by synchronizing >> yeah the other way >> so do you have to do barrier first and then synchronize. not sure. >> Yeah.

Yeah, I think one problem is if you barrier first and then the CUDA might not be done running and you just immediately go to the barrier and then you are still kind of each independently synchronizing the different CUDA kernels which means that you're not really synchronized on like the barrier might if all those operations just return the barrier doesn't really do any thing.

Okay, let me move on.
So now let's actually start thinking about how you train models.
Okay, so we're just going to walk through a very barebones implementation of training MLPS, multi-layer MLPS.
I guess that's redundant.
It's multi-layer perceptrons already. and remember that MLPS are the ones that are the actual compute bottleneck and transformer.
So this is actually pretty representative of what you'll see.
Okay.
So, data parallelism, tensor parallelism, and pipeline parallelism.
And the picture I want to you to have in your head is this picture.
So, this is a little bit of a schematic, so don't think too deeply about this, but it's more of a way to conceptualize how you're cutting your data and parameters.
So data parallelism says I'm going to split the data into pieces and then each of the GPUs is going to be responsible for part of the data and I'm going to do normal you know I'm going to keep track of all the parameters and do model normal model you know training and then I need to synchronize.
So let me explain how this all this works.
So I'm going to generate some sample data.
So there's a batch size of 128 number of dimensions 1,024.

So this is a batch size by num dim matrix data matrix.

And let's jump into this data parallelism.

Okay.
So the way that it's going to work is that you have this data matrix and I'm going to break up the rows into a bunch of into worldsiz pieces and this case four and each u rank is going to get a piece okay so the number of dimensions the batch size so each I'm going to call the local batch size basically you know the batch size divide by the world size which is every row is going to that GPU sees is going to have 32 you know data points and this is just indexing start index data start to end gets you the slice of that data and I'm going to just put it on that on that rank.

Okay, so at this point each GPU has now a distinct data tensor which is the part that they're responsible for.
Now in practice each rank should probably load its own data rather have this you know this bottleneck but this is just for illustrative purposes.

Okay.
So let's instantiate the MLP.
So here I'm assume we have numbum layers and num layers and for each of the layers I'm going to have just a num dim by num dim matrix so I'm just going to initialize a random you know set of parameters and then I'm going to feed that into optimizer.
Okay, so here's the training loop.
So in the forward pass, I take the data.
So remember data is not the all the data.
It's just if I'm rank two, then I only get B2 part of the data.
I'm going to just you know go through the number of layers and do a forward pass and then I'm going to do a backward pass and then normally this would be it.
But now remember every rank has different data.
Therefore the gradients are going to be you know different as well.
So this is the key step that makes data parallelism work.
We're going to synchronize the gradients across all the workers.
This is the only difference between standard training and DDP.
It's actually pretty nice and elegant.
So basically for all the parameters I'm going to do a all reduce of pram.grad grad and I'm going to average. and then after this all reduce is done.
Then now at this point each of the ranks has the exact same gradients and then I'm just going to update the parameters.

Okay, so it's kind of really kind of elegant I find because it's basically standard training where you apply it to your local batch but you just insert this after the backward pass.
Let's just you know average all the gradients via this all reduce.
It's a oneline you know code change and then the parameters get updated.
So as you're training each rank is basically performing parameter updates as if it were had all the data on it but it's only actually processing a part of the data.

Okay.
So that's basically DDP or the first type of data parallel.

Any questions about this? >> Yeah.

So the question is, can you only do this with batch size greater than one? yes.
So your batch size has to be at least world size for this to really make sense.
And usually it should probably be quite a bit larger. >> Yeah. >> Yeah.
So the question is, should the batch size be a multiple of world size? then that's also would be nice.
Yes.

I mean if it's not then you can pat it with zeros or something.
So there's ways but it's just easier for everyone if it is. >> Can you talk about transformer?

What what would it look like for a transform? >> yeah.
So the question is what would this look like for a transformer?
It would actually be basically the same.
The DDP has a nice thing that is very modular.
You do the forward pass.
I mean DDP just averages the parameters here.
It doesn't care what your forward pass looks like.

Okay, let me move on.

Okay, so that's DDP. so just to summarize the losses are different across the ranks. the gradients are also you know initially different but they are all reduced to be the same across the ranks and therefore the parameters all remain the same across ranks.
Okay.
So next lecture Tatsu is going to talk about fancier data parallelism FSDP and ZeRO and the idea there is as I've alluded to here we use all reduce it's like a very simple monolithic operation but it does require holding all the MA models parameters in memory but what if the model parameters don't fit in memory then you're going to have to be more clever and that's the topic for next class okay so let me talk about tensor parallelism.
So here the idea is we're going to cut this way.
I mean we're not going to cut the data.
We're going to cut the essentially each layer.

And so each rank is going to get part of each layer.
Okay.
And generally this rem means that we're going to have to transfer a lot more more data.
We'll discuss this a bit later. so what does tensor parallel look like?

Okay.
So, we're just going to assume we have this data.
Every rank has all the data just for simplicity.
And here, remember the data is batch size times num dim.
And I'm going to define a local num dim to be for this rank.
I only am responsible for a subset of the dimensions.
So the kind of the picture here is that each model still has all the layers here for all the layers but the parameters now are num dim times local num dim.

Okay so if this were one of the parameter matrices for one layer I would be doing splitting down the columns.
Okay, so this is also known as column, you know, tensor parallel.
You can also do it by rows, but we're not going to talk about that right now.

Okay, so now what does a forward pass look like?
So go through all the layers and I'm going to compute activations.
So I start with you know x the data and I'm going to access the parameters at layer layer and notice that this is only a slice of the parameters right so if I'm on rank one I only get this part of the matrix but you know I can still proceed I can apply this nonlinearity because this is elementwise anyway Okay. but now what I'm going to do is I'm going to you know communicate activations.
So if I have a data matrix, rank one has activations for part of the activations for this part of the matrix. rank one has part of activations for this matrix and so on so forth.
And I need to basically put all the activations on all the ranks and but we know how to do that.
We introduced all gather as a collective primitive.
So I have this is all the activations.

so this is you know the batch size times local num dim which is the shape of the of the activations. and then I'm doing an all gather sorry so this is the allocating memory for the activations.
So X is the actual part of the activation.
So this is batch size times local num dim and all gather says each rank has x and each rank is going to allocate activations which is a list one for each world size and then after the all gather x is going to be copied into each of the respective location activations.

Okay.
So then once I gather all the activations, I concatenate them to form the full dimensional X which is batch size times numbed in.

Okay. any questions about column tensor parallel?

So this is done for every layer. we now notice one difference between data parallel is now we have to kind of muck around with a model.
Data parallel is very elegant because it's splitting by data.
The model is treated as a as a module. but now we have to muck around with a model. and you know this is sort of strongly leveraging the fact that if you want to do a matrix multiplication you can split it up into a set of small matrix small smaller matrix multiplications.
We can do those on different ranks and then we can gather the results.

>> When it comes to propagation, it's sticky, right?
Because your radian still >> Yeah.
So, so now the question is what happens in backrop?
Now in the back prop you have your your activations and you have to reduce scatter to all the different you know gradients.
So in some ways all gather and reduce scatter have this kind of duality where in forward if you're all gathering in the backward you're reduce scattering.

>> Yeah. >> So question is does that done automatically by autograd?
So none of this well okay so if you just call backward it's not going to do it because there's no parallelism in that but you know pietorrch has all these things that are done automatically for you so >> what would I have to what do I to my So in this so what hack is done for you and versus automatically here we're managing things fairly explicitly.
So which means I'm not doing the backward pass but you would have to manage and call the reduce scatter yourself. >> Yeah. >> Yeah. >> And that's by design because this is 336 you know building language models from scratch.
In practice, you probably wouldn't have to do that.
Okay, so let's do pipeline parallelism quickly.
So the idea behind pipeline parallelism is we're going to split the network this way.
So each rank is going to get a subset of the layers.
Now within each layer, it's going to get all the dimensions and it's also going to get all the well the one of the ranks is going to get all the data. we're it's gonna every rank is going to see all the data in some form. so this is the way it's going to work. okay so let's we have all the data which is again the batch size times numbum dim. and I'm going to split up the layers.
So local num layers is going to be the number of layers that a particular rank is going to handle. and so now I'm going to have local params which is basically only there's local number of layers number of them but within each layer I'm going to do the you know numbed in by numbum in okay so one thing I maybe tatu will talk more about this on Wednesday is this idea of micro batches so maybe I'll just present this and explain why I'm doing things this way.

So, in addition to splitting up the layers, I'm also going to split up the batch into a bunch of micro batches.

so if I'm rank zero, then I get the data. and I'm going to chunk it up into number of microbatches. and for each you know microbatch what I'm going to do is receive it from the previous rank and then do the feed forward pass only on the layers that are assigned to this rank and then I'm going to send to the next you know rank.

Okay.
So here I'm actually using these receive and send which are pointwise operations.
I so I didn't cover those before but they're fairly explanatory.
This basically says I'm rank and I'm going to send this tensor over to you know sorry I'm going to receive this tensor from rank minus one. and this says I'm going to send tensor x to rank plus one.

okay so basically the reason why I'm talking about micro batches is and Tatu will talk more about this on Wednesday is that in pipeline parallelism so you have one rank that gets the data it processes some of the layers and then it sends it to the next you know GPU and it processes some of the layers and it then sends it to the next GPU right so this is a very natural way of dividing out deep network but the problem is that you get these what called pipeline bubbles where while you're not sort of processing you're kind of waiting around for other tensors to process and this is ends up being quite inefficient So the idea behind micro batches is that you break it up into smaller batches so you can kind of process it quickly, send it on to the next one.
So this can reduce the number of pipeline now bubbles.

Okay.
So the other thing I will mention that is not handled in this very kind of naive version is the idea of overlapping communication and computation which is actually very important to pipeline parallelism. basically gives you the right structure.
If you put a I before these then it becomes kind of async.

you have to add more things to kind of manage the code. and the idea here is that you want to be while you're computing here you should you know you can be receiving data or sending data.
So computation and communication should be overlap.
So that reduces amount of time u you're actually spent waiting.

Okay.
So a few things are kind of you know missing here which we'll hopefully fill in next time.
So the communication versus computation overlap which is especially crucial in pipeline parallelism.
I didn't mention in data parallelism this also happens because I just did a forward pass and then at the end I'm just doing all these all reduces but if you're clever then on the backward pass as soon as the gradients are done you can start sending that and that's something you're being explore in the assignment too and this again allows you to just you know overlap communication and computation more We they had some questions about what about general models.
Again, I think this MLP gives you essentially all you know, mostly of most of kind of what you need for understanding the you know the basics. some of the larger models just require a lot more bookkeeping.
So it's harder to kind of see the core core algorithms. there are other types of parallelism that we haven't covered. so sequence parallelism takes a whole you know sequence and chops it up into pieces and that allows you to paralyze attention computation.
Expert parallelism which allows you to paralyze the experts forees and this is where the all to all that I mentioned comes in and then also you know you know different combinations of different paralyzes techniques which also will show up in the assignment. so one thing to note is that the which parallelism technique you choose is going to be strongly dependent on the hardware.
For example, tensor parallelism there's a lot of communication because for every layer you're just you know you need to send all these activations which are fairly big.
So generally tensor parallelism happens within a node on NVLink or NVSwitch where you have high bandwidth whereas you wouldn't do tensor parallelism passed to kind of a NVLink you know domain whereas a pipeline parallelism you'll see people using it and generally this is can tolerate much you know slower you know interconnects.
So some of the decentralized training work uses pipeline parallel because your your nodes are GPUs are actually across halfway across the world. but you wouldn't want to do tensor parallel in that setting.

so sometimes you when you look at these combinations it will be tensor parallel within a node and then and then you know pipe data parallel or FSDP and then and then you know pipeline parallelism if you if you need it. there's other effects such as if you do data parallel, you might be able to do data parallel quite a bit, but then you start hitting to something called the critical batch faces where if you start increasing the batch size too much, it doesn't actually help you in which case you're just kind of wasting your compute and then you're better off using tensor parallel.
So there's a bunch of these considerations which u we'll talk more about as we go through the class. so final note so TPUs kind of came up a little bit. one thing to note is that on purpose we are using PyTorch and not just using PyTorch but really using the collective operations in a very primitive way so you can see mechanically what's happening.
Another approach especially if you're in your Jackson TPU land is that you can simply define the model and the charting strategy and the compiler actually handles a lot of the decision of how to basically what kind of communicated operations you need. you basically say well this piece of data needs to be here and here and here and then the compiler does some magic to figure out what so that's appealing but you know obviously it would take a lot of the you know the joy out of actually building things from scratch okay just to summarize so there's many ways to parallelize you can cut by data cut by tensor or expert cut by pipeline or sequence. we looked at data parallelism. we only did DDP.

Next time we'll do FSDP and ZeRO. tensor parallelism as I mentioned requires very fast interconnects.
Pipeline less so u but you need to really work hard to reduce these pipeline you know bubbles. and then maybe at a high level we see this kind of you know pattern come up a lot right so you can either recomputee or store in memory when we were talking about things like activation checkpointing or doing you know when we're working with GPUs or in this case you can think about an extension of this is that you can store on a different you know GPU right from that perspective If you look at the data parallel, right, you're doing redundant work in some sense because every  you know rank is actually updating its parameters and keeping track of all the parameters.
But the reason you're doing that is that you don't have to move the optimizer state across. so you know one thing is that you know hardware is getting you know faster but in some sense we'll always want you know bigger models.
So this idea of having a hierarchical structure will you know always be there.
Okay.
So that's it for today.
So next Wednesday Tatu will do more of a deep dive on more parallelism techniques.

Okay.

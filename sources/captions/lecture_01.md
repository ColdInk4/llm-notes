# Lecture 1 字幕

Welcome everyone to CS 336 language models from scratch. this is a teaching staff.
I'm Percy, this is Tatsu, Marcel, Herman, and Steven, and we're bringing you the third edition of 336.
So, we'll just do a quick round of introductions. so I'd like to say that I've been doing language models for 20 years, but most of that time was of small language models. actually this is still small in the grand scheme of things.
I think when Tatsu and I started teaching this class 2 years ago, we weren't really sure of what we were going to expect, but we were very pleasantly surprised that so many people wanted to learn how to build language models from scratch, especially in these days when a coding agent could probably zero-shot a language model.
I'm really glad to see all of you here to actually want to learn how they work.

Yeah, thanks.
I'm Tatsu.
I'm one of the co-instructors.
I'll be, you know, talking to you once we get to architectures and scaling and all these other very fun things.
I'm really excited.
This is the most fun class I've taught in my time here.
I think every year, you know, Percy makes fun of me because he says you have to redo all your lectures because you took on architectures and everything changes every time. but it's actually pretty fun for me to do it and it's kind of the first time that I've had that experience.
So, I'm looking forward to going through this experience again with you all.

Hello, I'm Marcel.
I've seen this class once before and I'm I'm returning cuz it was so much fun last time. it's it's a lot of work though. in my research I do architecture stuff.
I do higher-order gradients and I do training. yeah, looking forward to work with you guys.
Hello everyone, I'm Herman. a year ago I didn't really know how LLMs worked.
So, I think for me tokens were the things you collect in video games and attention was the thing you had like the attention economy. then after spending a lot of time on this course last year, I'm now doing LLM research and I'm really excited to TA this year.

Hi everyone, I'm Steven.
I am a first-time CA for this course and I'm very excited about it.
I think it'll be a lot of fun.
Broadly in my research I work on language models, theory, and some data efficiency stuff and I'm excited to meet you all.

All right, so let's let's go into things. so this is the third time we're offering it. last year we decided to put all our lectures on YouTube, so some of you will have seen it. so what's new? well, what hasn't changed is the from scratch philosophy.
We still believe strongly that by building everything from the ground up, you really learn how everything works. of course, we don't actually build everything up from scratch because that would be, you know, that wouldn't fit in a quarter. so over the last 2 years we've been refining our our recipe and figuring out what are the things that you build up from scratch that are most highly value.
And then finally, as Tatsu alluded to, even in a year a lot has changed.
I think this year we're we're going to spend maybe a bit more time on mixture of experts and of course agents are very popular these days, so getting a handle on long context and what is needed for that are going to be important.
So, why do we make this course?
And I think the problem through 2 years ago was that researchers were becoming disconnected from the underlying technology.
Used to be the case maybe 10 years ago, all AI researchers would just implement and train their own models.
And then even 8 years ago, people would download pre-trained models such as BERT and fine-tune them.
And I think a lot of today you can get by simply prompting a model.
And of course, there's nothing wrong with prompting a model.
You know, I think you can do really amazing things with it.
I think moving up the abstraction in general is a great thing, but abstractions are leaky and sometimes if I'm sure all of you have prompted models, you run into situations where something is you wanted to do something, but you it just can't do it and there's no recourse. and I would argue that if you're really interested in fundamental research, by simply prompting a model, you're sort of vastly constraining the set of options, the design space you're looking at.
And by looking at fundamental research, you really need to tear up the whole stack.
So, I would argue that full understanding of how language models work is really necessary for fundamental research.
And the way that we're going to get our understanding is by building.

That's the philosophy of the class.

But there's one small problem, which is that industrialization of language models has happened.
Frontier models are really really expensive.
Even this is 3 years ago, GPT-4 was supposedly costing 100 million dollars to train and now costs probably are on the order of you know, a billion, although that's speculative.
And the number of GPUs that all the big labs are building is just kind of immense.
Furthermore, there's no details on how many of these models are built.
So, even back in 2023, the GPT-4 paper explicitly says that due to competitive landscape and safety implications, we're not going to share anything about how the models are being built.
So, these frontier models are in some sense out of the reach for us.
Now, we could build small language models and we will build small language models, but this I think it's important to remember that these might not representative of the actual frontier models.
And I'll give you two examples why this is might be the case.
So, here's one.
Of course, this is from actually quite a long time ago. think back in 2021.

If we're going to spend more time on FLOP counting and looking at where the computer is being spent, but if you look at small scales, the a fraction of FLOPs spent in the MLP layers is around 44% and if you scale up to 175B, then it goes to 80, you know, percent.

So, what you optimize and what matters at large scale is going to be different from small scale.
So, if you did a bunch of small scale stuff here on the attention, you might not experience the same benefits at large scale. the second I example is that we know of emergence of behavior with scale.

So, small models this is again for a while ago, but even back then, if you have zero-shot or few-shot learning of various tasks, it basically seemed like nothing was working.
And only when you reach a critical scale, do you suddenly see a lot of improvement.
So, again, if you're working at small scale, you might not see certain types of phenomenon compared to if you were working at full scale.
Okay, so that might be a little bit disheartening, but fear not, we're going to learn something in this class.
And the question is what can we learn that actually transfers?
Okay, and I think it's important to break this down into three types of knowledge.
First, there's the mechanics of how things work, what a transformer is, how model parallelism works, right? there's mindset, which is how do you go about approaching building a language model. we're going to talk about how you want to squeeze the most out of your hardware, taking scaling seriously.

And in finally, intuitions, which data modeling decisions are going to yield good performance.
So, now in this class, I think we can do a pretty good job of teaching the mechanics, which is how things work, and the mindset.
And we're going to really emphasize that you profile and benchmark everything and try to optimize for efficiency.
These things do transfer to a larger scale.
Now, the intuitions about what modeling decisions and what data decisions do work might not necessarily transfer across scales. for that you actually have to go somewhere where you can have do things at scale.

So, on the note of intuitions, it is worth remarking re- remarking that some design decisions are just not justifiable. and just purely come from experimentation.
Whereas mechanics, you can kind of by construction see how the parallelism and how kernels are going to speed things up and so on. but for intuitions about what modeling changes work, I think you just have to run experiments.
There's this famous Noam Shazeer paper that introduces the SwiGLU activation, which we're going to look about.
And in the conclusion section, he very honestly has this final sentence which says, "We offer no explanation.
We attribute their success of these architectures all else to divine benevolence."
So, that in some sense is something that you just have to gain from experience.
Final note about the bitter lesson, which I think has been, you know, circulating and people talk about it.
I think there's a sort of a common mis- conception here what it means.
I think the wrong interpretation is that scale is all that matters, algorithms don't matter.
But that's not correct.
The right interpretation is that algorithms that scale are what that or what matters.

Okay?
And you can think about very simply that the accuracy of your model is basically your efficiency times the resources.
So, efficiency is output over input and resources is the input. and you know, efficiency is actually in some sense way more important at larger scale, right?
If you're doing a small scale experiment, if your run takes twice as long, maybe you just wait twice as long and then you come back later.
But if you're doing things at scale, you know, that could be, you know, hundreds of millions of dollars and you know, definitely don't want to do that.
Even like a 5% improvement might be a big deal.

So in fact, efficiency is actually really really critical and I hope to bake that into your your kind of mindset as one of the consequences of this class.
And empirically, if you look at there's this paper from OpenAI in 2020 that showed there's a 44x algorithmic efficiency on ImageNet between 2012 and 2019.

And it's surely the case that hardware did get a lot better. but that also comes with algorithmic improvements and of course, when you multiply them together, that's when you see like a huge bump in efficiency and accuracy.
So the framing with all of that said is what is the best model one can build with a certain data and compute budget? for pre-training, it's mostly we're going to talk about compute budget because we're going to try assume that we have a lot of more data than we have compute, but if you're in a setting where you're data limited or you have you know, stash away actually tons of B200s, then you might be data bound.
So in other words, maximize efficiency and we're going to see this theme come up throughout the class.

Okay.
So next I want to spend a bit of time talking about language models and a bit of a history and just contextualization before we jump into the more technical details.

so language models have been around for a while. you know, Shannon back in the 50s was using language models to measure the entropy of English. and for a long time, N-gram models were used actually in machine translation and speech recognition systems. and they weren't the whole system, but they were important part of making sure that you generated fluid text.
I would say that the lineage of the modern language models comes from neural architectures.
And so there is a bunch of ideas, I think, which are important to this development.
So in the 90s, there was LSTMs.

Joshua Bengio actually you know, wrote the first neural language model in paper back in 2003.

This was actually not LSTM.
This was just a feedforward network that looked at the small context. and then there was a seek-to-seek modeling which boldly said we can actually compress a whole sentence into a vector. the Adam optimizer, attention mechanism which was developed for machine translation, the transformer architecture which built on top of that which was also developed for machine translation.
And then scaling up to mixture of experts, model parallelism, you see a lot of different architecture and also systems and optimizer ideas in developing the 2010s.

And then by the late 2010s, I think things were starting to get really interesting.
So there was the ELMo and BERT.
These are language models that were trained on lots of text and then they you could fine-tune them on some downstream tasks like question answering and it would show a huge improvement on that.
So the model there was, you know, take one of these models and then fine-tune. and then Google had a paper that was really I think foreshadowing this view of, you know, prompt in response out.

So it was really I think OpenAI that really opened up the floodgates here by embracing scaling.
So they had a GPT paper back in I think 2018 or so that they scaled up to GPT-2 and then and then they really figured out you know, or embraced the idea of scaling laws which we're going to talk about in a in a second which enabled them to train GPT-3 which was a much more massive, almost more than 10x larger large model at the time.
And it could show emergent behavior like in-context learning.
At that point, you know, Google was saying, "Okay, we need to do something too."
So they trained a massive model.
It turned out to be kind of under-trained and turned out that their DeepMind which was not integrated with Google at the time had you know, figured out you know, optimal compute optimal scaling laws.

so all this was happening and after GPT-3 came out, I think for a many folks, this was sort of kind of wake-up call.
And at that time, there were a lot of early attempts to let's try to replicate this.
So there was a grassroots organization called EleutherAI that created some open data sets and models.
They weren't very large because they didn't have much compute.
Meta's first LLM was you know, it was you could see tell that it's a replication because it was like 175 billion parameters.

but it was not a very good model.

they ran into a lot of hardware issues. and then there was another Hugging Face BigScience project.
So these models were I would say not a very strong.

Then in the last 3 years, I think the open model ecosystem has changed quite a bit with Meta kind of leading the way with the Llama series of models, Llama, Llama 2, Llama 3.
Mistral got in the game.
And then a whole set of Chinese models I think I'm missing some like ByteDance has something and I think Tencent probably has some other stuff.
So it's hard to keep track of everything, but you know, everyone's heard of Deep Seek and Qwen. so I think I got the main ones.
But I think what's interesting and exciting about these is now now we have open weight models that are approaching closed models.
So depending on who you are asking and how you benchmark, they might be a little bit behind or comparable, but they are definitely very very credible models that are being widely used in industry.

Now there's an another line of work which is going beyond just releasing open weight models.
AI2, Nvidia, and the Marin project which I work on, we try to provide not just the weights, but the paper and the code and the data so that we can understand how these models are built in a more thorough way.

Why do I emphasize this open ecosystem so much?
Mostly because this course would not be possible I think without these models.

By the fact that there are still many papers that are being published about how these big MOEs and RL systems are working enables us to at least glimpse into how these frontier models are being built and trying to triangulate the pieces.
Now I think of course a lot of the details even with these like you know, Qwen papers, I think they're missing a lot of details.
You can't reproduce them.
Notably the data mixture is something that we don't know, but I think it's better much better than nothing.

Okay.
So in the last decade, I think the idea of what a language model has changed.
Right?
It used to be something that you fine-tune. and then it was something you prompt.
And now in the ChatGPT era, it was something you talk to and that you can have a you know, conversation with.
And now okay, I guess I don't have internet. that's fine. and now we're in the era of agents.
If you click on this link, it basically shows you a giant agent trace. and I'm I'm still kind of it's mind-boggling how strong some of these models are.
You give it like a page of text and it does have some really complicated agentic coding task.

so you know, what we demand of our language models today is probably beyond the imagination of anyone back you know, 10 years ago.
That said, I think the fundamentals I think haven't changed that much.
Largely, we still build on GPUs and kernels.
We still optimize using gradient or stochastic gradient like approaches.
We still have the transformer and attention and you know, we'll we'll talk a little bit more about architectures, but it hasn't changed that much.
I think the specs are different.
Now we demand greater context lengths which means that inference efficiency matters even more.

So the good news for us is that we didn't have to completely change this class, only touch two sections on the latest you know, Chinese architectures, but the fundamentals I think are here to stay at least for now.

Okay.
Maybe I'll pause there in case there's any questions or thoughts.

Okay.
So let's let's go on.
So brief interlude.
So what is this program? so this is what I call an executable lecture.
This is a if you think it looks like a Python program, it is because it is actually a Python program, but it has been rendered for your viewing pleasure. so when I step through it, it is actually executing the lecture and it makes us it possible to basically step through code and which will be hopefully interesting and important later.
And you can also see the hierarchical structure of this lecture.
For example, we're done with this function now we go back to main.

Okay, so let's talk about the course logistics and the syllabus which I think it's going to take the good chunk of time.
Okay, so all the information is online on this website or cs336.stanford.edu.

This is a five unit class.

I think this probably class has a certain reputation so I don't need to belabor the point too much but you know, this is we have five assignments.
They are pretty intense even the first assignment according to this one review was actually you know, equivalent to the five assignments for cs224n.
I've been told also this is like wait this is exaggerated but better to I guess be conservative in your estimates.
So why should you so why should you take this course?
Okay, so first you have an obsessive need like me to understand how things work.
That should be your primary objective.
I think just pure curiosity on how language models work and then in doing the class you'll develop a much stronger research you know, engineering muscles and have the confidence to go into a new setting and be equipped to deal with whatever comes up.

So I think when I started at Stanford I screwed this class called you know, statistical learning theory which was basically teaching the theoretical side of machine learning and that was nice because it equipped people so when you read a paper you can understand all the math and now since the field has kind of shifted a lot more towards this more systems and empirical side of things.

This is sort of the kind of an analogous class which gives people enough depth that they feel that everything else seems kind of easy.
So why should you not take this course?
This is important because there's reasons you shouldn't take this course.

First you actually want to get some research done this quarter.

You should probably talk to your advisor.
They should know you're taking this course otherwise they'll be probably there might be some surprises.
You're interested in learning the hottest new techniques in AI.
I think there are many other great courses you know, seminar courses topics courses that are good for that.
We don't do all many of the things we don't do multi-modality.
We don't talk about you know, agents in any depth.
So if you want to learn about this that stuff this is not the right course for that.
If you come in and say like I have an application domain I want to get good results on it.

Probably this is not the right course at least to start.
I always recommend just prompt the model fine tune the model and then as a last resort you pre-train your own model because it is a pain and it's also expensive.
But it's a lot of fun.

Okay, so if you're not taking the class you can soft follow along at home.
So all the lecture materials will be posted on the website oops and they are also recorded through cgoe so thank you cgoe for doing that and later they will be published to YouTube.

Now of course following at home watching the lectures is great but you learn really by doing the assignments so you'll have to figure out how to motivate yourself to do that.
Okay, so speaking of assignments we have five assignments.
The philosophy of the assignment is you know, how how do we do the from scratch but not just you know, say build a language model and that's the assignment.
So we don't provide scaffolding code but we do provide a bunch of unit tests to make sure that whatever you're building is actually correct so that it's you don't get the sparse rewards setting where you submit a homework and it's either correct or not.
What I recommend is that you can the assignments are structured so that much of the assignment can actually be done locally on your laptop.
You can implement it and check for correctness and then we are providing a cluster so that you can do an actual training run to see what the accuracy is or get a bunch of GPUs to actually benchmark the you know, performance of a some kernel.
And then for fun we will have you know, some leaderboards for most of the assignments at least and they will look something like well, now that you've learned about this topic try to minimize the perplexities given some sort of budget.
And both Marcel and Herman were masters of leaderboarding back in the day which was last year or the year before.
So if you want tips on that I'm sure they would be happy.
Actually I don't know if they would tell you their secrets but you can try.
Okay, so You know, last year we were thinking about well, what can AI do is like okay, well, I mean yes everyone can use AI but you know, just try to do your best.
I think now I think coding agents have gotten so good that they can just also solve all the assignments, right?
But to state the obvious obviously you're not going to learn anything if you just feed in the PDF assignment one into Claude code.

At the same time AI can be very useful for answering questions and tutoring.
So we have to find a way to you know, leverage AI.
So what we've decided to do is we provide you agents.md file or of equivalently a prompt which asks the AI to be pedagogically minded.
You can read more about in our AI policy guide and the requirement is that if you're going to use AI use it with this prompt.
And so it will answer questions about code.
It will clarify any understanding but it won't accidentally generate like the transformer for you when the homework is to implement the transformer.

Okay, and this is the first year we're trying to do this so please try it out and give us feedback if it's working or not working.

Okay, so compute.
So this year we have thanks to modal.
They have provided us with a compute credits on their platform.

It's actually quite quite nice.
You can get a number of unlike last year well, I guess you guys didn't take the class last year.
Last year we had a cluster you SSH into.
This is using more of an API to which I was initially skeptical of but you know, looking at it actually is pretty pleasant to use.
So again try it out and give us feedback on how it is.
We've written a guide on how to access and use the compute.

Okay.
Any questions about course logistics?

Okay.
All right, let's talk about what we're going to cover in this class.

So there's basically five parts mirroring the five assignments that you'll have.
Basic systems scaling laws data and alignment.
So I'm going to now go through each part and just give you a taste of what you will learn.

Okay, so in the basics this is basically the first two weeks.
The goal is to just be able to train a language model and build it from scratch.
So the components here are we're going to tokenize the data.
We're going to define architecture and then we're going to implement optimizer and trainer.
Okay.
So then you wonder like what are the other the rest of the class is for?

Well, we'll get to there.
So let's start with tokenization.
The tokenization is really about what are the atoms that the model operates on.

And you know, formally a tokenizer converts between raw inputs which are just bytes and a sequence of integers which are represent the tokens.

Conceptually it's a segmentation of the of the text.
We're going to talk about the byte pair encoding BPE tokenizer which intuitively breaks the input into frequently occurring chunks.
And then remember this class is about maximizing efficiency.
So through an efficiency lens tokenization is good because it takes a long sequence if you just think about the raw byte stream and reduces it into a smaller number of tokens.
But more subtly but maybe more importantly it allows you to do adaptive computation.
So some maybe some places are actually a lot of bytes but actually it should be compressed into a one token where some of the more rare or interesting parts of input should be left as multiple tokens.

Okay, we'll talk more about this.
I just want to mention that you know, every year I'm hoping that I don't have to talk teach tokenization because the dream is to really have an end-to-end way of that directly operates on bytes.
And there's been a number of work including you know, recently there's been this H net work that seems promising but so far these have not been scaled to the frontier and since the frontier models are still using tokenizers felt like it would be still wise to teach tokenizers.
Okay, so, now after you tokenize your input, you have a bunch of tokens.
Now you define a model on top, and everyone I think has a familiarity with the rich the transformer. and if you take in 224N, the NLP class, then you've seen transformers.

since then, I think there have been a lot of improvements or refinements to transformers, which I think are important, and Tatsu is going to talk more about this in a bit.
But just to run through a set of types of things that one might have to think about.
So, the activation functions have evolved.

how do you positional encodings has evolved? how you normalize has the different layers to prefer blow up has evolved. you know, instead of doing full attention, there's many ways to basically reduce the attention computation because attention is n squared and it's and where n is a sequence length and that gets really expensive. so there's a bunch of ideas around that. if you're more ambitious, you can look at these state space models or you know equivalently linear attention. like Mamba and gated DeltaNet.
These have become been popular in the last you know few years, and usually some hybrid between these models and attention seems to work you know quite well.
So, we'll be exploring some of that. and then within the MLP layers of the transformer, the original transformer was just a dense MLP, and now mixture of experts has become the sort of a dominant paradigm for building compute efficient your transformer.
So, we're going to talk about that.
And of course with M MLP a mixture of experts, you know, it's not just defining your architecture, but we'll see that we'll also need different techniques for training the model.

And then finally, perhaps somewhat boringly, but an important question is, you know, what is the shape of your transformer?
How many layers? how many heads? what is hidden dimension?
Number of experts? this might come in more as we talk about scaling laws, but setting these is actually seems kind of you know, almost trivial.
It's a hyperparameter, but actually in the context of scaling language models has a huge huge implication.

Okay.
So, once you define your model architecture, how do you train the model? and here there's a bunch of design decisions around the loss function.

There's next word next token prediction, which is the default, but people have found that looking at predicting more than one token seems to be helpful for improving the model. and there's optimizers.
People used to use Adam, but increasingly Muon has been used especially with some of the latest open models such as the Kimi K2.5 models. initialization, which is again sounds kind of you know boring, but turns out to have a huge impact on how your ability in the training stability of larger models, learning rates schedule, regularization, you know, batch size.

and then MoE specific things.

So, you know, you look at this list and you might think, well, these are just hyperparameters.
I'm going to try a bunch of different options out.
But it turns out that, you know, really being very careful of about setting these hyperparameters in a principled way will make the difference between a run that just blows up and is useless between a run that is achieving state of the art.

Okay.
I'll come back to that point when we talk about scaling laws.
Okay, so then in assignment one, what you're going to do is you're going to implement the BPE tokenizer, implement the transformer, the loss function, optimizer, the whole training stack.
We're going to make you do a bunch of resource accounting, so you understand where your FLOPs are going.
You're going to train some models on these beta sets like tiny stories and open web text.
And then there's going to be a leaderboard that where you're going to try to drive down perplexity as fast as you can.
So, for those of you familiar with the NanoGPT speedruns, it's kind of similar to that. >> [clears throat] >> Okay.
So, by the end of assignment one, you should be able to walk away and build a language model you know, from scratch.
So, that's very exciting.
The you know, if you have a if I have a high-level takeaway here, it's that you know, while the tokenizer and modeling and training are sort of presented as distinct pieces, it is actually you know, everything is about kind of balancing the following.
So, you want expressive models because you want to represent the complexities of the data, but at the same time, you want to your training to be stable.
And we're going to see talk a lot about how do you keep the parameter and gradient norms in the sort of Goldilocks zone, so they don't blow up and they don't vanish.
Turns out a lot of training language models is about just you know, stab a stability. and then finally efficiency, which is somewhat more straightforward.
You just make it run fast on hardware.
But you're going to see interesting things like, you know, if we change the architecture, a lot of the architecture decisions are, well, we can make it faster by let's say reducing the projecting our low dimensional space.
But then the question is, does it work as well?
And so, making those tradeoffs is something that is going to is sort of the name of the game here.

Okay, so, as in assignment two, we're going to dive more deeply into you know, systems. and the goal here is just to get most out of your hardware. so we're going to talk about kernels, how you parallelize across multiple GPUs, and how you do inference.
So, the basics, which we're actually going to start on our next lecture.
I mentioned resource accounting, and I think you know, you've all probably you know, built models but this is really about kind of keeping track of all where all the FLOPs go.
And where all the memory is being spent.
So, we're going to spend some time basically doing the resource accounting. we're going to see, you know, this formula that comes up, how do you how many FLOPs does training a 7B on model on 1 trillion tokens?
Well, it's 6 * n * d roughly.

and where does that come from? and then we're going to look at you know, the hardware.
And here's a very cartoon I'd you know, picture of what what to remark about hardware is that your memory is not where your compute is, and you have to move your either your parameters or activations into from the memory to compute, do the compute, and move it back.
Right?
And that often is the bottleneck. so, you know, for example, a B200s, which we'll have the opportunity to play with, has 2.25 petaflops per second if BF16, and has 8 terabytes a second of memory.
So, you know, what does that mean?
I think in when we I think I'll do this next lecture.
We're going to break this down and use this information to do some calculations and see how long different types of algorithms will need to take.
We're going to talk about roofline analysis, which allows us to understand whether a computation is bottlenecked by either compute or memory.
In general, it is you know, memory. and then talk a little bit about benchmarking and profiling.
Okay, so, so here's what you know, DGX B200 looks like.
You have eight GPUs. they're connected via you know, NV link. and then if you have many if you have a thousand GPUs, then you would have multiple of these, and they're you know, connected either InfiniBand or Ethernet.
So, the next two parts when we talk about system is, you know, kernels.
So, kernel is basically a function that runs on the GPU, and when you're just using plain PyTorch, you know, the all the PyTorch primitives actually correspond to launching particular kernels, which are built in. so you're already using kernels this you know, whether you know it or not. but the point is that if for certain types of computation, if you look at it, you can actually write custom kernels to make the GPUs go faster.
And the main principle here is organizing the compute to minimize data movement.
So, remember this picture, moving data from memory is expensive, so you want to try to minimize that.
So, just as a kind of a simple example, suppose you wanted to compute A and B.
So, often you would have to read from your high bandwidth memory HBM, compute it, write it back, and then read again, compute it, write it back.

Right?
So, then you're basically sending the data back and forth twice, and there's this idea called fusion where you read it once, do both of the computations and you write it back.
And that will save you a lot of time.
So, that's operator fusion.
Tiling is kind of a more sophisticated variant around the same idea.
There's also GPUs have also gotten a lot more complicated and I'm not sure how many of these details we'll have time to get into, but at least we want to expose you to some of the sort of the peculiarities, I would say, of GPUs and the and give you appreciation for the types of things that one has to consider in order to squeeze as much juice out of them as possible.
And we'll write some kernels in and try.
So, what happens if you have thousands of GPUs?

So, the principle of minimize data movement is still, you know, the same.

The only thing is that moving data between different GPUs is even more expensive.

We're going to talk about how these very classic collective operations like gather and reduce and all reduce are the way to kind of think about basically distributed, you know, training.
The general game is that we have these model parameters, we have activations, gradients and optimizer states, and they need to be sharded or split across multiple GPUs, and of course, you need to bring the right data to the right nodes to make the compute and write it back.
So, there's a whole kind of orchestration that and how to do that efficiently is going to be the topic of this unit.
And there's multiple ways to shard.

You can shard by, you know, splitting up your data, splitting up your model, splitting up different layers in the model, splitting up the sequences, splitting up between experts, and we'll talk about the tradeoffs that come with each of these.

Okay, and then finally, we're going to talk about inference, which, as I mentioned, is growing in importance.
So, the goal of inference is to actually use the model.
So, you know, minor detail here.
So, inference is Of course, you need to use inference when you're chatting with the model, but it also is useful for reinforcement learning.
It's useful for doing the rollouts, test time compute, generating synthetic data, evaluation.
So, inference is a very critical part of what it means to you know, do language modeling work.
So, we're probably not going to spend as much time as I like on this because the course is already kind of filled up, but we'll see what we can do. there was some discussion around whether we would should make you write inference from scratch, but we'll we'll see.
So, the way to think about inference is that there's two phases, a prefill and a decode.
In the prefill, you take the prompt and then you feed all the tokens forward and build key value pairs.
This is very much like what happens in training. and then then the decoding part, tokens are generated one at a time.
And this is the part that is becomes quickly memory bound, and this is why inference is hard, and so, there's many things you can do to speed up inference.
You can try to use a cheaper model by pruning a larger model.
You can quantize, you can distill. you can use this technique called speculative decoding where you use a cheaper model to sort of run ahead and guess a bunch of tokens, and now you use the full model, which can operate on those tokens in parallel to see if it's good.
And if you got lucky, then you can accept all those tokens and you're you're much faster than if you're doing one token at a time.
And of course, you can do systems optimizations.
There's bunch of kernels that are designed specifically for inference.
And then one of the interesting stuff things about inference is that if you're running a service, you know, queries are coming at you at potentially different times. and then you have to figure out how to batch them up.
Whereas in training, you basically are already defining your batches and everything is much more predictable.
So, in assignment two, there's going to be implementing kernels in Triton. and doing some sort of parallel training. the details here might actually change as you know, the CAs have grand plans of revamping the systems.
So, the assignment might look a bit different from last year's, but it will be cover the roughly the same material. one thing I will mention is that there's this wonderful book out of some Google people called how to scale your model.

And I think it's it's really nice for providing a conceptual understanding of like roofline analysis and transformer math and doing LMs conceptually.
So, I highly recommend you take a look at that.
Now, the only thing is that it's from Google, so it's about TPUs, but a lot of the high level concepts are similar.
And now they have a new chapter on how to think about GPUs.

Okay.
So, So, the third assignment is about scaling laws.
So, by now, you've trained a language model, you can make it go really fast by optimizing kernels in parallel.
Now, you want to scale up.
So, how do you scale up?
So, imagine the following setting.
If you had 1e25 FLOPs, so this is tens of millions of dollars of compute, what what model would you train?

Okay, so this is, you know, I think a kind of a daunting task because if you mess up, well, that's a lot of money down the drain.
So, and you can't do your probably typical hyperparameter tuning at that scale because you only get to train one model.

And so, this is like the key problem that you have to deal with in language model large language model training that you don't have to really deal with if you're just fine-tuning a model or doing small scale stuff.

And so, the key conceptual shift here is that we shouldn't think about a single model that we're training, but really think about a scaling recipe.
And a scaling recipe is a mapping from a FLOP budget, let's say 1e25 or 1e24, to a set of hyperparameters, basically a config file.
And for a given scaling recipe, what we will do is run a bunch of experiments to compute the loss that you get at smaller scales, and then you fit a scaling law, and that enables you to predict the loss at a target scale.
So, maybe you run some small experiments, you fit the scaling the scaling law, and then you project out what you're going to get at larger scale.

Okay, so that's the primitive.
Now, using this, what you can do is now you can optimize the target res the scaling recipe targeting a larger scale using smaller scale experiments, which is wonderful.
And second of all, you can predict the loss that you're going to in theory achieve before actually running the experiment.
Which go goes allows you to go, you know, raise money and you say, "Well, look, I ran the small scale experiments and I think I can get a really like a GPT-5 level model.
Please give me a lot of money so I can train that model."
So, one thing I think is a maybe another misconception is that scaling laws are not laws of laws of nature.
They don't just happen automatically.
You kind of have to will them into existence.
And this is happens by careful construction of a scaling recipe.
Right?
And a scaling recipe, remember, it has to extrapolate.
So, what this typically means is that you have a sequence of hyperparameters, which say as the scale increases, maybe the learning rate is a constant, maybe it drops, maybe the batch size increases by how much.
And these are things that a scaling recipe has to figure out. and so, you parameterize So, then, in order to get these predictable scaling laws, one thing that you actually have to think about is how do you parameterize the model in a way to get what's called hyperparameter transfer.
Right?
Meaning that this hyperparameter you use at small scale are either the ones that you use at larger scale or are predictable functions of that.
Right?
Because if every scale, your learning rate acts like sometimes 1e-5 and sometimes 1e-4, then you're not going to be able to magically guess the right learning rate when you at larger scale.
So, one shift in thinking is that the predictability is actually at least as important as optimality.
So, you think normally think of oh, we're trying to optimize for, you know, efficiency here and we want to, you know, hyperparameter tune and make things optimal.
And yes, you do want to do that, but you also want this predictability so that you don't get at larger scale.
Okay, so that's this is kind of a stage setting. the actual scaling laws we're going to look at are fairly you know, classic.
So, these are you know, some of you have might have seen this idea of well, if you give you a FLOPs budget, should you train how should you balance training a larger model versus training on more tokens?
And this is where the classic computer optimal scaling laws from Kaplan et al. and the Chinchilla so-called Chinchilla scaling laws comes in.

The basic idea is that you for each FLOPs budget, so let's say 6e18 all the way to 3e21, you sweep across different you know model sizes and you choose the best one.
So, think about minimizing each of these.
And then you fit a curve that allows you to basically predict the number of parameters given a FLOPs budget.
And if you're lucky, it will lie on a roughly on a line.
And if you're unlucky, it's going to be all over the place, which means that you should have no confidence that you're going to be able to predict reliably.

And So, you know, the upshot of this is this is quite crude, but you know, a rule of thumb is the 20 times the number of parameters is the number of data points you should train on.
So, a 70B parameter model should be trained on roughly 1.4 trillion tokens.
Of course, depending on the data set and architecture, this number will actually vary.

Okay.
Also, this doesn't take you into account the inference cost.

a lot of models these days are small, but they're trained on way more tokens than is computer optimal because you want a smaller model for inference reasons.
Okay.
So, one fun thing that we've been doing in the Marin project is pre-registering our results.
So, we fit a bunch of scaling you know, plots at different compute budgets.
We fit a scaling law, and we basically made these predictions out to 1e22 FLOPs. this one is actually training.
If you go to the Marin site, you can follow along. it should actually be done maybe as earliest tonight.
So, maybe on Wednesday I'll I'll report back on how we did and see how we manage register the how we match the pre-registered loss.

So, the idea here is that we made a prediction on if we were to train this large model, which we've never trained before, if we can predict how well it's going to do, then that's that's really nice.
Okay.
So, in assignment three, I think I think this is a kind of a either fun or you know, s- Actually, it's a fun ass- let's just say it's a fun assignment.
So, we're going to define this training API, which basically you give us hyper parameters, and we're going to give you a loss back.
So, what we're going to try to do is simulate what happens if you could do a lot of training runs.
And of course, we don't have enough compute to actually allow everyone to train their own, you know, 8B model or anything.
So, basically, we train a bunch of models offline, and we basically provide this cache. so, it looks like you're training. and what you're going to do is submit training jobs.

you give us a config.
Basically, we give you a loss back.
And then you can do whatever you want.
I would we would recommend you fit scaling laws to these points, and then you extrapolate, and then we give you a budget, and you're we basically evaluate how well your model landed.
So, it is meant to I was going to say it's meant to kind of replicate the high-stress scenarios if you actually had a budget that like, you know, $100 million that you needed to spend. and you have to be very careful about what how you spend this. of course, this is low low stakes.
Okay.
So, so at this point, you will have you trained a model, you know how to make it fast, you know how to scale up. you know, now what's missing?
You know, what do you train the model on?
And that's going to be the subject of the data section, which is, you know, arguably one of the most important things because data quality is basically specifies how good your model is going to be.
One way to also frame it is, you know, what do you want your model to do?
Right?
Data basically is reflects what your model wants.
So, do you want to speak multiple languages, be good at, you know, having a conversation?
Do you want to do run long agentic coding tasks? and so, part of that is also going to, you know, so we're going to start by talking about evaluation, which basically defines the capabilities that you like your model to have. you know, one thing we'll talk about is evaluation is a fairly, you know, deep topic. not just it's not just about like, you know, running on some benchmarks. there are internal evaluation metrics for model development, and what's matters here is that you know, smoothness across scales so that they're, you know, remember we want things to be predictable. relative performance matters.
You don't care necessarily how well this you know, does an absolute charge because, you know, let's say perplexity number is just like what is a perplexity 1. you know, two on some, you know, held-out data really mean? and then there's external metrics.
These are things that you report to your your customers or your reviewers or whoever you're presenting your thing to.
And here ecologically validity really matters.
And I think sometimes these two things two things get completed, but I think they really serve two distinct purposes.

And so, you can think about perplexity is something that is really helpful for internal development.
And still to this day, you know, perplexity is a very good way of capturing the intrinsic quality of a model without getting kind of worrying about bench maxing.
Now, there's a separate question of what you run your evals on. and the recommended thing is, well, if you have some data that's not on the internet, that would be good because you can avoid contamination.
Okay.
So, if and then there's advanced use cases, which are more representative of external-facing use cases.

So, one thing to also note is that, you know, language models are purportedly very general purpose.
Right?
So, it's suitable only fitting that we actually need a very diverse set of evaluations.

And I would always recommend having many evaluations that look at you can average them into a single number, but often that average conflates a lot of different things.

Okay.
So, now after we set up the evals, we know what we're building. how do you get the data?

Well, first thing is that data does not just fall from the sky. it has to be actively, you know, curated. often, I think you know, in you know, especially in classes and also in research, sometimes you're just given a data set, and then it's like, okay, well, now I do stuff on the data set.
But a lot of language models, especially if you want to collect these large data sets, you have to go actively look at it.
So, web pages are called from the internet. there's books, I guess, which is controversial at this at this point, but archive papers, GitHub code, and so on.
This is an old figure from the Pile from 2021, and you can see, you know, language modeling data sets are fairly diverse. there's especially these days, I think, a lot of contention around is it fair use to train on copyrighted data? maybe sometimes you have to license data, and so on.
So, there's legal issues around data, which I think are quite in, you know, important.
For example, a lot of GitHub code doesn't have a license.
So, how do you interpret that?
Do you think assume it's permissive, or do you be conservative and assume it's not permissive?

So, there's also the fact is that data is not even text, right?
It's either HTML or PDFs, or code is directories, and this requires processing to turn it into actual text to be usable for training.
So, that's the top topic of data processing. there's a few steps that has to happen here.
Transformation, converting some non-text thing into text. filtering, keeping only the good stuff. if the random internet document in Common Crawl is extremely bad, and you the and you don't want to train on it, most likely. you want to deduplicate. there's multiple sources, so how do you combine the different sources? and then finally, more recently, there's been a lot of work on, you know, generating synthetic data, which could mean taking the real data and just rewriting it into things that are more like the downstream task, or just some more Wikipedia-like, or whatever the whatever you want.
So, this is an active area of research.

also, data can be used both for pre-training, something called mid-training, which is usually the high-quality data that you put at the end of the pre-training step.
And this includes long context data, such as, you know, maybe big code repositories or books.

And then finally there's post-training data, which are you know, maybe conversations or genetic traces with you know, tool calling.

So, in assignment four, we're going to make you start with a very raw corpus, like a raw web crawl, and do all the work to filter and to dedupe and make the data clean.
So, this is I would say I don't know if I would call it not fun, but it is certainly I think a lot of you know, what people would call you know, dirty work, but that is I think part of an important part of building a language model from scratch.
So, you have to get the full experience.
Okay.
So, finally alignment so far we've basically trained a model using full supervision.
Predict the next token or the next few tokens. now, at this point the model should already be reasonable. but we can improve it further by using weak supervision.
And why weak supervision?
It's because sometimes it's easier to critique than it is to generate.
So, you can't always have data that says this is the right response to this prompt, but maybe you can have a way of specifying what good looks like.
So, then the basic template is that you generate responses from a model, you score them either with a human or verify or LM judge, and then you update the model to prefer better responses.
This can be instantiated either through various RL algorithms such as PPO or GRPO, or in a simpler for preference data DPO.
So, the challenges around RL are that RL algorithms are unstable and you know, hard to tune.
Some of you probably know this from first-hand you know, experience.
Personally, I prefer to keep things as much in the full truly supervised case as long as possible, and then finally okay, fine, I have to do RL. but you know, some people for whatever reason like doing RL.

Also, what we'll hopefully talk about this year is that if you do RL at scale and try to maximize your throughput, there's a actually a lot of systems challenges. you have to have an inference server and a training server, and then the inference server has to generate these rollouts, especially if you do RL against environments that involve code execution.
It's a whole kind of orchestration game.
And then if your workers lag behind or then you get into off-policy issues, and then you're constantly kind of juggling this on-policyness with a desire to kind of maximize throughput.
It's it's a big wonderful mess, which hopefully we'll talk more about when we come to that.

So, assignment five, we're still deciding what exactly we want to do.
Last year it was implement DPO and GRPO and get it working for some you know, math benchmark, but we'll see how how much farther we can push it on the realistic dimension this year.
Okay.
So, again, remember it's about efficiency, and efficiency can mean either data efficiency or compute efficiency.
So, the way to think about it, you have all these resources.
You have data, you have the hardware, which has compute cores, you have memory, communication bandwidth, and you're just trying to figure out how do you build the best model according to some evaluation given a fixed set of resources.

So, you can through this lens, I think you can actually think about a lot of these design decisions as optimizing for this.
So, systems, clearly that's about compute efficiency. tokenization, as I mentioned before, it's you know, you can't just work with raw bytes, but that's going to be very compute inefficient, and at least with today's model architectures.
And so, a lot of tokenization is about improving compute efficiency.
Model architecture, many of the changes that we'll see are motivated by reducing the memory or FLOPs. in fact, a lot of them are influenced by the need to have faster inference.

data filtering, you can also view as through efficiency lens.
We don't want to waste time updating gradients on a bunch of redundant bad data, even if it might not hurt you, but it hurts you in the sense that if you have a fixed compute budget, more time on bad data means less time on good data.
And then finally, scaling laws is explicitly about how you can essentially do effective hyperparameter tuning on much smaller models.
And now, tomorrow we might become data constrained in the calculus of what design decisions you should take might change, but I think the overall mindset, which we're trying to teach you, is you know, think about the efficiency of your approach.

Okay. let me stop there.
Are there any questions about any of these assignments or topics?

Okay.
So, now let's do tokenization.
So, this is we're jumping into our first unit here.
So, Andrej Karpathy has this really good video on tokenization.
You should check it out.
So, let's starting point is raw text is what is text.
It's Unicode strings.

and on the other hand, the language model places a distribution over sequences of tokens, usually represented as indices.
So, we need a procedure that encodes these strings into tokens, and also a procedure that decodes tokens back into strings.
So, a tokenizer is basically something that can do this round trip.
So, here are some examples to give you a flavor for how tokenizers work.
Actually, I should have tried to get internet earlier, so this is not going to work.
Okay, if you go to the site, you can play around with different tokenizers.
So, some observations here.

there's and you'll you'll appreciate why tokenizers are kind of annoying, why people want to get rid of them.
So, a word and its and a word with conglomerate with its preceding space are different you know, tokens. so, many of tokens you'll actually see are space word, which is you know, fine, but kind of strange.
So, this hello and this hello is actually two completely different indices that have nothing to do with each other.
So, and sometimes, depending on the tokenizer you use, numbers are represented with this every few digits is a token.
Sometimes it's predictable, and sometimes it's it's not.
Some tokenizers try to make every digit a token, but then you're blowing up the number of tokens you have, so there's some trade-off there.

So, here's the GPT-5 tokenizer.

so, it can take the string and convert it into you know, these indices. and then you can decode it back into the string.
And a tokenizer should round trip.
If you implement a tokenizer that doesn't round trip, you have a problem.
So, the compression ratio here is the number of bytes per token.

So, in this case, we have the number of bytes is of this string is 20.
The number of tokens is eight, and you do 20 / 8, so the compression ratio is 2.5.
Okay, so 2.5 bytes per token.
The larger the compression ratio, that means the shorter the sentence, which is good because attention is quadratic, and you want to make sure the sent sequence is shorter.
Now, you could obviously increase the compression ratio by increasing the vocab size, but then you get into sparsity, where more and more because every element of vocab is treated as like a as a distinct element. so, these days tokenizers, especially multilingual tokenizers, have you know, 100k or 200k you know, tokens.
Distinct tokens. so, you can look at the GPT token vocabulary.
I think we'll skip it this in the interest of time.

Okay, so how do you build a tokenizer?
So, I'm going to go through this fast, but So, the first thing you might do is like, well, Unicode string, that's a sequence already of Unicode characters, and each character is an integer, which you can call ORD in Python, and you get some number out.
And this can be converted into characters, so let's just build a character-level tokenizer, which basically breaks up each character and encodes it in a token, and then this can decode back.
Like this is good.

So, now, there are 150k Unicode characters, so your vocab size could be 150k, which is which is you know, a lot I mean, it's not crazy, but I think the bigger problem is that many characters are actually rare. which means that it's really an inefficient use of a vocabulary.

And also the compression ratio, which kind of reflects this is not that great.
So, most of the time you're actually using, a lot of tokens to represent your your sequence and many of the indices, are actually not being very used.
So, this is like not a very good tokenizer.

so here's another attempt.

So, you can turn strings into bytes.

So, Unicode has a UTF-8 encoding, which means that you can sometimes, a string like A is just one byte and sometimes a string is, you know, multiple bytes.
So, let's build a tokenizer around that.
So, we can take this string and convert it into a sequence of bytes and notice that this is a longer sequence now and but all of the all the numbers are between 0 and 255 because that's what a byte means. and the compression ratio is, is one, which is, which is not great.
Right?
So, byte sequences are can be very long, but the vocab size is, small.
Okay, so now, okay, so the both of these are like really bad.
So, here's let's try to make some progress.
So, this is what actually people used to do in NLP, if, people remember.
So, if you take a string, I can just chunk it up into, break it up by spaces or some regular expression, and let's just call each of these chunks, a token.
Okay, so what is good about this one is that each token is meaningful because humans invented words and words tend to have a stable semantic meaning, so in there but your vocab size is the number of distinct chunks in the training data, which could be all a lot.
And also your, compression ratio, I mean, it's it's it's quite, you know, good but the vocabulary can be huge.
Actually, it's worse than that because, the, the vocabulary could be actually unbounded, right?
Because at test time you might get some sequence and you tokenize and then you have a token you've never seen before and people used to assign these like unk token but that's like really ugly and can mess up your perplexity calculations.

So, this is also not not great.
Okay, so what we're actually going to do is called byte pair encoding.

And this was introduced a long time ago for data compression way before language models, were really on the scene, really. it was first introduced to NLP for doing neural machine translation, and the first, paper that, used BPE to, for LMs was, GPT-2.

So, the basic idea is you're going to train the tokenizer on raw text to construct a vocabulary that's tailored to the data.
And you're also going to have this property that everything becomes can be tokenized if it's rare, then it just breaks up into smaller units rather than having this unk token.

so common sequences can be, are going to be represented as one token, rare sequences are going to be split into multiple tokens.
That's the idea.
Okay, so the algorithm is fairly, simple conceptually.
So, you start basically with your your corpus, let's assume it's one long sequence. you get a byte sequence, each byte starts as a token and then we're going to merge, successive pairs of adjacent tokens that are occurred the most frequently.

Okay, so let's step through how this is going to work in code.
So, here's a simple string, the cat in the hat, and, here's the implementation of the BPE algorithm. so we're going to, turn that into a sequence of bytes, and then we're going to first, we're going to basically count the number of times successive tokens appear.
So, 116 104 shows up twice.

so we get this and then we're going to find the pair that happens most number of times.
So, that's 116 104.
I guess there's a few ties but we'll just take the first one.
And then we're going to merge that pair.
And by merging that pair, what we do is we create a new token.

In this case, this is going to be called token 256.
It's going to represent this pair and we're going to add it to our vocabulary.
So, 256 is going to represent the sequence th.
So, t and h have been merged and we're going to call every time we see th, that's going to we're going to use 256 to represent that.
And then we go through indices and then we replace every occurrence of 116 104 with 256.
So, those two places have been replaced.
And then we iterate.
So, the next time, we do this, we're going to find 256 and 101, we're going to merge that and now we have 257 and then, we're going to merge that, one more time and we're going to get 258.
So, over time, the sequence is, shrinking, and the vocabulary size is growing.

Okay, so, let me and the compression ratio, here is that we get for this, I mean, this toy example is 1.5. okay, so now that you have a tokenizer, how do you tokenize new text?
Well, you take a new string and you encode it and conceptually what happens is that, you basically go through the set of merges, that you've made and then you just apply, the merges to your your string.
Okay, let me not actually not step through that code. so that will give you a sequence.

so this is the sequence, encoding of the quick brown fox and then, when you decode it, you get the same thing, you know, back.

So, I went to through this a bit fast just in the interest of time.
I will say that this, implementation works.
This is a full-blown BPE implementation.
It's extremely slow.

So, in assignment one, we're going to ask you to, basically make it faster. so currently encode, loops over all the, you know, merges, which is very slow because you might have the number of merges you have is essentially essentially, the vocab size minus, you know, 256. so you only want to loop over the merges that matter. and you have to build some indices to make that, you know, happen. there's some, you know, details around special tokens.
Conceptually not deep but important to building a modern tokenizer. another thing is that I've presented the tokenizer just for simplicity as you take an entire string and then you try to tokenize it.
Really, what happens is that you break it up into, your text into chunks and then you apply tokenizer on each, chunk.
So, that's going to be much faster. and then, you know, and then try to make it as fast as possible. at some point you might realize that Python is just not very fast and if you want to implement it in, you know, say your favorite language, Rust or C or something, then go for it.
Okay, so quick summary.
Tokenizers convert between strings and tokens or indices. the previous character-based, byte-based, word-based are highly suboptimal in this, in their own way.
BPE is, effective heuristics that is data-driven so, it seems to be pretty effective.
Now, like I said before, maybe next year I don't have to teach this but, for this year we're stuck with tokenization. you know, even if we get rid of tokenization though, I think whatever solution replaces it, I think has to satisfy the following properties, right?
If you have the model, the transformer needs to operate on some sort of abstractions of the sequence.
And this is most evident if you think about, you know, not just text but video or DNA, sequences where, you know, the individual, bytes or units are actually quite kind of low signal to noise, and you have to do some sort of abstraction to lift it into a place where you can do modeling on that. and then finally, as I mentioned, chunks should be kind of variable.
You want adaptive computation. not all, you know, bytes are treated the same and if you don't do that, I think you're going to be suboptimal.
So, if any end-to-end solution also, I think has to have these properties.
Okay, so with that, I will end.
Next time on Wednesday, we're going to start the unit on resource accounting, so which is sort of a, you know, a baby systems, I would say.

And then after that, we're going to go back into architectures and, go from there.
All right.

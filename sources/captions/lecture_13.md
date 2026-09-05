# Lecture 13 字幕

So, today we're going to talk about data.
And where we are is that you know how to train a model given data.

Last time we saw evaluation, so we know what a good model is.
And the for the next two lectures this week we're going to focus on what data should we train on.
And I want to argue that data is the most important thing to get right in language models.
And one justification for this is if you look at what companies actually disclose.
So, here is the Llama 3 paper. they have full transparency into architecture.
Well, of course, because it's open weight model. they even tell you about the training procedures. but they don't say anything about their data.

So, they just say we train from a variety of data sources. and do some stuff.

And there's good reasons for this secrecy.
One is that data is kind of your competitive secret sauce, competitive dynamics.
You don't want to let your competitors what you're doing.
And the second thing, which we'll talk a little bit more about, is copyright liability.
You don't want to get sued if you tell people you're training on certain types of data.
So, data is a long topic in machine learning.
And before foundation models, data working meant you annotate label data for supervised learning. nowadays there's less manual annotation, at least in pre-training.

But there's still a lot of curation and cleaning work that needs to be done.
And so, fundamentally this hasn't really changed.
Data has always been a bottleneck and always will be a bottleneck because it's in some sense of long-tail problem.
And that scales with human effort.
So, if you have a lot of people trying to work on a problem, well, there's only so many people that can work on architectures and systems, but data, well, that's like, you know, especially if you're training a foundation model that's trying to do all these things, you can paralyze that effort easily.
So, that's why these data teams and these model developers are actually quite quite big.
So, so data comes in at different stages of a of the pipeline.
Today, we're focusing mostly on pre-training.

So, pre-training, you take raw data, these are documents from the web, then you move into mid-training where you train on more high-quality data to enhance certain capabilities and give long context, things like that.
And then finally, post-training where you're training on things like chat transcripts or if you're doing reinforcement learning, you get some environments that you're training on.
This becomes more task-specific.
In practice, the lines are a bit blurry.

and there could be more than three stages, but this is sort of a basic template. the trend is that we go from training on large amounts of low-quality data to smaller amounts of high-quality data.
So, as a result of this process, you'll see in the literature some people talk about a base model.
This usually means after pre-training and mid-training.
Instruct models or chat models are after post-training.

but of course, the lines are becoming blurry enough that what is a base model anymore is really kind of unclear and more recently, the largest models are there's no base model.
There's just Qwen3.5-397B-A17B and that's it.
There's you don't see the intermediate checkpoints.

So, for models that are open source, like Olmo from AI2, you can see everything that's happening.
And so, we'll study some of these models. so for example, in pre-training, there's a bunch of sources, which we'll talk about what these mean, but you know, some web pages, academic papers, math pages, proofs, and so on.
There's mid-training. higher-quality web data, instruction data, a bunch of synthetic data often goes here.

And then finally, post-training, there is a bunch of you know, chat logs.
There's more math and reasoning and coding and you do safety at this at this point as well.
So, you know, what are all these data sets?
You know, how do you choose them, and how do you process them?

So, that's that's the main question.
Okay, so in the spirit of the class, we're going to talk about data.
I'm going to go start from scratch.
So, where does data come from?
Okay, so you know, one might hear in the hallway, "Oh, language models are trained on the entire internet."

So, so first of all, this doesn't really quite type check because for that to be true, it would have to be an agent that like an RL agent that goes on the internet and like does stuff, but that's not actually how pre-training works. slightly more accurately, it's trained on the public World Wide Web. this is not quite accurate for reasons I'll explain.
Okay, so first of all, the web is actually a bunch bunch of live servers that just exist in the world, and you can connect to.
So, take any website, you can go and download a page.
Or you can basically send a request and get back a response.

So, you can't actually train on these live servers unless you're training an RL agent. so typically, what you do is you have a crawler, or someone builds a crawler, not necessarily you. the crawler needs to discover what web pages are out there, starting with a seed seed set, and it downloads this discovered web pages as it's it's crawling.
So, but now still you can't just run a crawl and download all the web pages on the internet.

And there's a few reasons for this. one is that a lot of the web content is actually dynamic.
So, especially these days a lot of sites are apps.

the URL isn't even as full a specification of the content.
It's just an app that you interact with. and you need to often click buttons or submit forms to access content. for example, if you're on, you know, Discord or something, it's not like you can just like crawl Discord and get a all the content out of Discord.
Okay?
So, so that's one kind of thing, and a lot of content is actually known as kind of in the deep web, which is not just like you're, you know, doing web page crawling and following hyperlinks, which is a very kind of traditional model of web crawling.

The second thing is that you know, authentication.
So, some pages need a login and account, and generally you have to pay.

for example, there's Facebook, X, LinkedIn, New York Times. there's actually huge amounts of content that are actually locked up behind these walled gardens.
So, these are technically on the internet or on the web, but you can't just build a crawler and go crawl you know, Facebook.

I guess if you're Facebook, you can crawl Wait, you don't need to crawl Facebook.
You have the data and you can train a model.
Or if you're you're X XAI, you can train on X.
But if you're anyone else, you can't actually access this content.
So, okay, suppose you don't have the authentication issue, there's still potential restrictions here.

so, so there's something called robots.txt.
And this is a file that is usually placed at the root level of a directory such as nytimes.com.

And it basically tells you what you're allowed to crawl and what not.
So, this things are a lot of things are disallowed such as OAI search bot or perplexity bot, Cora bot.
I think basically chat GPT user, Claude bot, and all these things.

Okay?
So, this is not a tech This is not a legal restriction.
This is just like you're supposed to be a good citizen and look at robots.txt.

And if your name is on that list, then you should not crawl.
That's That's basically the contract not even a contract.

That is the thing you're supposed to do.
A lot of websites now use something like Cloudflare to detect and block bot activity.

So, sometimes you go on to a website and it gives you a capture, right?
That's because it's somehow some algorithm has detected that you might be a bot and then it's making sure you're human.
And if you're actually a bot there then well, you can't I mean, you could try to get around this, but it generally this is a obstacle in your way.
A website might block certain IP addresses or countries and they might have rate limits.
So, there's technical restrictions on what you can actually crawl.
And then there's also legal restrictions.

So, there's websites have terms of service that say, "If you go to a website, you have to obey this contract to use the website."
And the terms of service often will say, "If you are a bot, go away."
"You cannot use this the content of the site for AI training or whatever it might be."
And also, if the if you even if terms of service don't say anything, the content of the website might have, you know, a license or you might not have a license to train.
And we'll talk more about this in a bit.
So, there's legal restrictions on what you can allowed to train on.
These things are also evolving over time. so, there's a nice paper by Shane Lampray called Consent Consent in Crisis.
And what they did is they examined the restrictions, both the technical restrictions, robots.txt, and the legal restrictions, the terms of service, for various URLs in common data sets which we'll talk about later.
And the conclusion was that the restrictions have increased over time.
So, here's a graphic that shows The top one shows robots.txt.

and if you just look at this red, that's the full restrictions. and up until 2023, everything was fairly fairly constant.
And then up until 20 mid-2023, all of a sudden you see the amount of the fraction of websites that have full restrictions on it has grown to almost like, you know, 50% here.

And for terms, this there's a similar trend where I guess in 2016, no one really put any terms on their pages, and now most pages put some terms, and most of the terms say you can't use this for AI.

Okay, so even though it was possible to crawl the internet back in, you know, let's say 20 2020, the internet actually, at least what you can legally crawl is actually much smaller.
So, of course, these are you know, guidelines and sometimes either intentionally or unintentionally you could violate these things.
So, this is sometimes you get into these situations where there's this guy who what was this I forget the name of this the site was complaining about Anthropic crawling them hitting their servers a million times in 24 hours which was not good.
And then read the docs was also getting hammered.
So, there's a period where which I think probably has hopefully been corrected by now that crawlers were hammering you know, websites.
And there the here this is interesting is like even before talking about copyright issues.
There's I think problems with crawling because either violating terms of service or robots.txt or just just blow up down you know, generating a lot of server load which this costs money for the person hosting it and also degrade service for everyone else trying to use it.
So, there are you know, issues with crawling and then of course there's copyright which will talk about soon.

That's a whole another complex can of worms.
Okay, and then finally there's these things called shadow libraries there which are also part of the web.

so, examples are Libgen or Anna's Archive and these they completely disregard copyright and bypass paywalls.
So, someone has collected a ton of data usually books and articles that are all copyright and people have to pay for these and have made them available for free.

and these there have been a lot of takedown orders, lawsuits and these are blocked but these are circumvented because they just make servers in other countries. and the people doing this argue that this is making free what's freely available, what should have been freed.
But, generally from a legal perspective, this is piracy and copyright infringement.
And we'll come back to this point later.
So, there are a lot of books and papers on this on these shadow libraries.
Okay, so the summary so far, the internet is a huge and messy and you know, scary place.

and you can't actually get all of it. there are technical restrictions, there's legal restrictions on what data one can access.
So, now it's next time someone says I trained on the internet, you can tell them all of these things you learned.
Okay, so now let's go to the second part of you know, what data can we use? and this has to do with copyright.

So, suppose you were able to obey the terms of service and you are really a good citizen and you obey the rate limits.

now there's still a question of are you allowed to train on this data?
And this is a ongoing question that has not been fully resolved, but there has been a bunch of developments in the last year.
So, the legal context for thinking about this question falls into intellectual property law.
And this is a system of you know, system that is trying to incentivize the creation of intellectual goods.
So, I think it's remember to remember the spirit of the law is to not just say, you know, technically say no to everything, but to incentivize the creation of intellectual goods.

And there's many forms of intellectual property that are covered.
There's copyrights, patents, trademarks trademarks and secrets.
Trade trademarks and trade secrets.

The thing that has is most relevant for language model data is copyright law.
So copyright law has a fairly long history going back to the 1700s in England.
And so 1709 was the first time that governments and courts started actually regulating a copyright for this goal of incentivizing you know innovation and creation.
And in the US more recently the Copyright Act of 1976 is what think of defines a lot of kind of modern copyright.
So copyright protection applies to original works of authorship authorship fixed in any tangible medium of expression.
Yada yada yada.

Okay?
So a few notes.
So not everything is copyrightable.

In particular collections are not copyrightable.
So for example, if you have a telephone you know directory unless there is some creativity in how you arrange things, this is not really a copyrightable.
And furthermore copyright applies to an expression not the idea.

So you can't copyright an idea like you can't copyright the quicksort algorithm.
You can copyright the like implementation of a quicksort algorithm, but not the quicksort algorithm.

So one thing that happened in 1976 is that the barrier to getting a copyright was relaxed a lot.
So more things are copyright.
So it used to be that you had to publish something to have something copyright.
Now just have to be fixed.
And what that means is that registration of any sort is not required for copyright protection.
This is in contrast to patents.
Patents, you have to go pay a lot of money to get a patent and you would have to file a patent.
But, the threshold for copyright is very low.
For example, you put something on your website, it's copyrighted.

That's it. now, you if you want to sue someone, you have to go get it registered, but it only costs $65.
So, which is much smaller than the lawyer fees that you'll probably pay.
So, it's really still a very low low bar.
So, one thing to know about copyright is that it lasts, 75 years and then the copyright expires and whatever that content gets released into the public domain and this is now you permiss- everyone can use it freely.
Okay, so a lot of, people who have, lived in the past, all of their great works are in the public domain, which is great for everyone.
So, the rationale here is that you're incentivizing innovation. so, when an artist or a creator creates something, you want to protect their You don't want to have someone else to just take it.
So, you protect them for a while, but, you know, after 75 years, presumably, it's not worth protecting or they're passed away or something.
So, it doesn't make sense to keep things, copyrighted anymore.

Okay, so, basically, everything on the internet is, copyrighted.
Okay, so you're saying, "Hm, wait a minute.
So, that means is every everything just if I train on anything, that's a copyright violation?"
Well, not necessarily.
So, everything is copyrighted, but you can use copyrighted works.
And the way you do it is A, you either get a license for it or B, you appeal to appeal to the fair use clause.
Okay, so a license is from contract law is something that a licensor grants to a licensee.
Basically says, don't sue me.
Like you can you can use this work in the ways that the license permits.

So there's a really nice license called you know creative commons which allows something to kind of act like it was in the public domain. enables free distribution of copyrighted work.
So your examples include like Wikipedia, OpenCourseWare, Khan Academy, all these all these things.
It was created in 2001 to essentially bridge public domain and existing copyright because otherwise you have to wait 75 years.
But what if the creator says, I don't I will actually want people to use it.
They can now have a means a legal means of saying, I put a creative commons license on it.
It's going to act like it's in the public domain and you know, and everyone can use it.
So far things in the public domain and things that have creative commons licenses are you can use it. but there's many other things that times where you don't have it's not a creative commons license, it's not in the public domain.
You have to get a license.
So if you have money, you can pay someone to give you a license.
So there's a bunch of deals between model developers and content I guess platforms that license allow the model developers to license the data for training foundation models.
Okay, so that's the first thing.
You can get a license, either Creative Commons or you can pay someone for a license.

Now, if you don't want to pay, you can appeal to fair use, which is a more complicated matter. so, this is section 107 of the Copyright Act.
And there are basically, fair use says, I can use this anyway, even if I don't have a license.

And there's four factors determine whether four fair use applies.
So, the first one is the purpose or character of the use, like what are you trying to do with it?
And none of these are, you know, hard rules.
They're just tendencies which have to be weighed in court if it comes to that.
So, for example, if you're trying to do something for education, it is more likely to be fair use than if you're trying to sell and make money off of it.
If you're trying to if you transform something, then that's going to be more favored than if you just literally host the identical copy. the nature of the actual copyrighted work is also important.
So, things that are more factual are more likely to be fair use than fictional works.
Right?
Because if you write a summary of World War, you know, facts about page about facts about World War II is less There's less protection over a you know, my you know, really creative poem.

And then you know, the amount of portion of the original work.
So, using a snippet is favored over the whole work.
So, if you just take a little bit, that's not as bad as taking everything.
So, that's fairly straightforward.
And the final thing is kind of related to original motivation.
It was like, why do you why copyright at all?
It's to protect and align economic incentives.
So, the effect that what the use has on the market for the original work.
So, if you take author's work and you provide an alternative and that decreases amount that the original author could monetize their work, that is seen as potentially bad.
But, if you're transforming in a way and adding, you know, going to a new market or doing something else with it, then that's going to be treated more favorably.
Okay.
So, here's some examples of fair use.
You can watch a movie and write a summary of it.
You can reimplement an idea like an algorithm, not copy the code.

And then there was this famous you know lawsuit between Authors Guild and Google.

So, if you go to Google Books, you can see snippets of various books, which are copyright.

So, there was a huge you know deal about whether this was deemed fair use because you're sort of redistributing you know, other copyrighted work.
And eventually eventually meaning after 11 years, this lawsuit was settled in favor of Google, which is why this thing exists.

And this has set some precedent for thinking about whether you know language model training is fair use.

one thing to note is that copyright is not about verbatim memorization.
So, this might be I think for ML audience, this might be a little bit new because a lot of papers are focused on verbatim memorization, but that's that's one way you can violate copyright, but not the only one.
So, for example, plots and characters can be copyrightable.
So, you can copyright Harry Potter, the character, not any particular you know, book or anything.

But, there are some exceptions.
For example, if you're parodying something, so, you're creating something that is looks like a derivative, but as long as you're making fun of it, it's actually more likely to be fair use.
So, this Yeah.
So, a lot of copyright is about semantics.
It's definitely not about n-gram overlap. and also about the economics.
Okay.
So, what is the implication for language models?

So, first off, copying data, like even if you're not training, like even the mere fact of copying, which is in the word copyright, is potentially a violation already, even if you don't do anything with it.
Okay? training a model, you know, I guess this is This is not really a Let's see how I would have to take that.
This is not necessarily a fact, but is intuitively has a transformative flavor, right?

Because it certainly not It certainly seems different than just like re-hosting another work, right?
It's doing something transformative.

and you know, in some sense, the models are really trained on this data as a means to the end.
You know, we're trying to extract the general idea, learn about the world and how it works, rather than only just the concrete expression.

and the other thing is that regardless of cop- copyright, language models can definitely affect you know, the market.
So, so I think this is also, remember, that's the fourth item for fair use.
So, by negatively affecting the market, you're you know, more likely to be ruled not fair use.
Okay, so fair use is kind of a little bit, you know, slippery. suppose you have fair use or you have a license, remember that there's still terms of service that allow prevent you from actually just getting a piece of work.
For example, YouTube has all these videos which are licensed.
But the terms of service forbid prohibits just downloading the videos as a using a bot or like scraping.
Right?
So, which is another layer.
So, there's multiple layers of restrictions here.

Okay, so let me talk a little bit about where things are on the legal landscape.
So, back in 2023, New York Times filed suit against OpenAI saying, "You trained on our news articles and look, here's evidence we were able to actually prompt ChatGPT to generate a news article almost verbatim."

I don't think this is this is still pending.
There's a case against Anthropic where the allegation was you pirated millions of books and you trained on these books to make Claude. just last year this was a kind of a landmark ruling that said, "Actually, this particular instance of training is fair use.
However, Anthropic, you're still in trouble because you pirated all these books and that is a no-no.

Right?
So, notice that this is has nothing to do with training.
Just a mere a fact of cop- pirating, I mean, which is this is not a new thing, is is illegal.
Anthropic, actually, interestingly, had bought and scanned all of the books, which is actually fair use.
So, the court said, "You're allowed to scan.
You buy a bunch of books, rip off the binding, scan and digitize it for your own use.
That's fair use."

But, that doesn't absolve you of your sin of pirating.
So, you can't pirate and then buy it and say like, "Actually, never mind what I just did first."

so, the outcome is that Anthropic paid 1.5 billion to settle the authors.
So, this is like about $3,000 a book.

so, there's also a lawsuit against Meta.
The allegation was that you trained on our books. sorry.

And this was, as we'll see later, actually revealed in the Llama paper.
And the judgment, which came right after Anthropic, was yes, training is fair use. and then they also torrented some books, so that's still pending.
So, that's probably if precedent holds, this is also not going to be good for Meta.
Okay, so the summary so far is training, you know, so to speak, has been deemed fair use or at least has been not deemed not fair use.
The new rulings have so far been, you know, narrow.
It's not to say that any training on any copyrighted content is fair use. but in these cases, it's fine. pirating books is clearly illegal.
I guess we already knew that.

And but this is still a very active and evolving area.
Okay, any questions about the sources of data?

I think before we get into the actual data sets, I just want to kind of set the stage with this kind of a more general social context.

Yeah. the question is what do I think of voice dating at 11 labs?

I don't know too much about that, so we can talk offline.
Yeah.

>> [clears throat] >> Yeah, so the question is what happens with time if you have one license and then it gets changed, what happens?
So, first I'm not a lawyer. however, second, so this so the first license, I believe you're still allowed to train on it, but what happens is that the license usually applies to not just let's say a fixed set of documents, for example, Reddit.

There's constantly new content being created and the license being changed means that you wouldn't be able to train on the later things.

Okay, so let's look at various sources of data.
So, let's go back to crawling.

so most model developers have their own crawler because they want to have control over full control over what the data is. fortunately for the rest of us, if you don't want to build your own crawler, there's something called Common Crawl, which has been around for quite some time since 2007. and every month they run a web crawl. gets about 3 to 5 billion web pages.

each crawl has some overlap with a previous crawl, but they try to diversify and get new pages. so there's 300 billion pages so far.

which seems This was from their website.
It seems a little bit big because if you multiply this number by 20, you don't really get a 300 billion, but that's what they say. so how many URLs are out there?
That's really hard to estimate.
So, the Google search index is at least 100 petabytes, according to them.
And you know, each Common Crawl, dump has about two, billion, you know, web pages.
So, that's about 372, terabytes.
And this is not including images.
This is just mostly focused on the, the text.

So, crawling is conceptually straightforward, but all the, gory details are in the implementation.
You start with a bunch of URLs, and then you iterate.
It's basically graph traversal.
You pop a URL from a queue, you download it, and then you look at all the hyperlinks in that page, and then you add them to the queue.
And generally this is done in parallel, over many machines. and there's many decisions here.

Which paper, which pages to download. to respect You have to respect our robots.txt.

Don't overload the server. sometimes, the websites change, and so you want a policy that will download, you know, frequently change pages, but not spend time downloading pages that don't change.
So, there's some policy around that.
And like I mentioned before, URLs are dynamic, so both sometimes the same URL leads to different content depending on like some state of the browser, but also different many URLs might lead to the same content, so there's a lot of duplication that happens if you're not you know, careful. in particular, mirror sites explicitly is about duplication.

So, Common Crawl releases their dumps in two formats.
One is a WARC file.
So, this is the raw HTTP response.
Right?
So, this remember HTTP protocol is that you send a get you know, here's a a URL and then out comes a HTTP kind of response.
And it's just the WARC file is that response. they also do some processing, they convert to a WAT file, which is necessarily a lossy process.

it turns out that this is not necessarily the best way to use the web. the HTML to text there are many tools for this, including Trafilatura and resiliparse.
And the way you convert actually does matter.
So, this is from the DataComp-LM paper, which we'll talk a little about.
It is ablation, where you look at the WAT files that Common Crawl releases versus these other these other tools.
And it seems like resiliparse and jusText are you know, better.
Okay, so one thing you can do is just rely on general web crawls.

But often, the web is not a uniform place.
It's not like you there's a bunch of websites and you just like get some random fraction of them and that's your data set.
And there are specific pockets of really interesting high quality content, and I'll talk about three of those, Wikipedia, GitHub, and archive.
So, Wikipedia, all of you know about it's been around since 2001, and now there's 67 million articles of around all these different languages. and it is it doesn't Wikipedia is not all content by any means.
There's no You can't have original thought in it because everything has to be kind of referenced and cited.
So, in some sense, you could argue that Wikipedia is doesn't contain anything because that's not already on the on the web.
Well, actually, that's not true because they can also cite books, which are, you know, obviously, you can't easily get them.
So, Wikipedia is so good.
So, and also there's articles based on notability.
So, not everything Not everyone can get a Wikipedia article about them.

So, anyone on the internet can write this content.
This was the radical thing about wikis, and it's sort of still a miracle, I think, to me that this actually works.
It just happens that any vandalisms gets reverted by administrators or bots, I guess, these days.

There's you know, a small number of Wikipedians, like as any sort of peer production system, there's a small set of people who do most of the work.

And for example, this guy has like 5 million edits. and one thing about Wikipedia is that every few weeks they do a periodic dump.
So, basically, take all Wikipedia, package into a nice tar file, and then you can just download that.

So, this is important because you don't need to crawl Wikipedia.
In fact, they don't want you to crawl Wikipedia.
They just want you to download this if you're going to crawl.
And this is much better than crawling.
One kind of just a fun aside or good to know is that there's this paper that says you can actually poison Wikipedia.
So, you might think that this vandalism gets reverted.
So, if you do attacker tries to do something fishy you'll just get it rolled back.
But, what Carlini realized is that these periodic dumps happen at a regular cadence.

And so, what what you can do is you can go in and you can just edit it right before the dump happens.

And then the dump ha- the it happens and then and then the edits get rolled back, but the dump still has the malicious content.

and with if you're able to inject things into web pages, that means you can cause there's works that's show that you can cause the model to, let's say, ascribe negative sentiment to any sort of trigger phrase like the iPhone.
So, my one takeaway here is that if you can consider adversaries, even so-called high-quality contents might contain bad things in them.
I think this has been a fix since then.
Okay, so GitHub is a good place for code and code is important not just for if you want coding capabilities for your language model, but if you want general you know, reasoning. so, GitHub, all of you know again, it's a live service for hosting code repositories. been around for since 2008, about the same age as Common Crawl. it has 400 20 million repositories, 28 million are public. each repository contains an It's not a file.
It's as you know, it's it's a directory with commit history, issues, pull requests, and comments. code has generally a lot of you know, duplicates and this because either copying code or forking code.

and GitHub has deemed it you're allowed to train on any public repository with a permissive you know, license.
So, MIT or Apache licenses are fine. so you there's remember there's two types of data when you talk about GitHub.
There's like the repository data which you can just go and download through the GitHub Git protocol.
And again, you know, you shouldn't scrape GitHub. you should just download the repository.
And then the metadata associated with each repository, issues, and pull requests, and so on.
These are made actually via the GitHub archive which gives you hourly snapshots of this event stream, which I think is interest- really interesting data.
It's basically every single comment or star or action on GitHub gets that gets recorded.

There's this Software Heritage Foundation that is focused on the repository, not the metadata.
And they aggregate GitHub GitHub is not the only place where code shows up.
There's also smaller entities like GitLab, Bitbucket, and so on.
And they aggregate all that those repositories.

okay, so archive just very briefly, everyone knows archive.
This has been around as a site that allows people to share papers since 1991, started with physics, and now has a lot of other areas in it. each of the 3 million submissions has the metadata, a PDF, and optional LaTeX source.
Okay, so already you should be thinking, well, what does it mean to train on archive?
It's not clear because this is well, you have to convert the PDF into a text or you can use a LaTeX source, which is a bunch of files. there it's not peer-reviewed, but there is some approval process, and authors can choose to maintain the rights or put in the Creative Commons, which so everything is actually very clearly licensed. all the metadata is under permissive license, so you can use that regardless, and then you can go and you look at all the papers that are Creative Commons licensed, you download those, and you can use them.
And I think most archive papers are Creative you know, Commons. and then again, this is available, you don't crawl archive, you just go down to bulk download it from some website.

Okay.
So, any questions about source of data?
Yeah.
Are there restrictions on data generated by models?

Generated by models?
Yeah, by using that data.
Oh, oh, I see. we'll talk about that later on.
So, the question is, what about model generated, like synthetic data, are you allowed to use it? the short answer is probably yes.
Yeah.
Are you using crawlers like Open or crawl and stuff like that?
I guess like the cuz you mentioned earlier that there are websites that have a lot of pirated books on it, right?
How do you like How do you ensure that those types of websites don't show up in your crawl of the internet.
Yeah.

Yeah, so the question is there's a lot of websites with pirated books.
So, if you're just doing a web crawl, you can't look at all the websites.
So, how do you make sure that everything is kosher?
And that's part of the difficulty is you can't.
And so there's most likely in Common Crawl there are copyrighted books and content that you are not supposed to train on and you can appeal to you appeal Well, you appeal to fear it.
So, remember everything is copyright.
Books are not remarkably different from your your website. for example, both are you know, copyrighted.
Probably a book author has more of a you know, if they want to go into court, they'll probably be able to protect that better because it's published than your your website, but but still. and I'll talk a little little bit about later.
If you really want to be careful, you can do something I'll show later.
Okay, so what I'm going to do now is to now talk about the various data sets that are actually used in building models starting all the way back to 2019.
So, we've talked about the origin of data and source of data.
So, you have a feeling for just the general data you know, landscape. and so let's start with BERT.
Okay, so BERT was from 20 18 and they trained back in the day, they just trained on Wikipedia and books.
So, what is this books? so, there's this website called Smashwords.
It was created that allow anyone to just put a ebook up for to publish and they had about half a million books in 2024.
So, in 2015 there is this paper who just scraped Smashwords. so sorry, took the books that were free and just made a corpus.

This was back in the innocent days when no one was paying attention to any of this.
So, this books corpus was around for many years in the academic computer academic community. since then it has been taken down because it violated the terms of service.

Okay?
Just because it was free and you can get it doesn't mean it's it was legally allowed.

so one thing about to note about BERT is that the sequences were documents rather than sentences.
All the language modeling research before that was focused on sentences.
Okay, so in 2019 GPT-2 came around and so what did they do?

so they wanted to get high quality web content.
And at that time people knew about Common Crawl, but somehow the Common Crawl data was too messy.
So, what they did was came up with this clever idea where you can take you look at pages that are outgoing links from Reddit posts with greater than three karma.
So, good posts must link to good good good websites.
So, you grab those pages and you get 40 GB of text, a million pages. they never released this data, but there was an open replication of WebText which people use quite a bit.

Okay, so that's a GPT-2.
So, as you're going through this, I guess think about both you know, there's many methods of filtering and collecting data and it's interesting to look at these design choices and okay, so CCNet was developed by at that time Facebook. so the goal was to create large high-quality data sets for pre-training and they were interested especially interested in low-resource languages.

so they didn't want some very manual process that only worked for English. so they did a bunch of things.

They did deduplication and language identification.
They trained a classifier to detect a language and keep the only the language that you're looking for. and for quality filtering the idea that they had was they wanted to keep documents that look like Wikipedia.

So Wikipedia at the time was deemed good high-quality content and they use the language model that was trained on Wikipedia and you score probability of a new document under that language model to gauge how Wikipedia-like it was.
So rather than outgoing links to Reddit, they use a language model on Wikipedia.

and they you know show that because you can get much more more data, you can actually outperform training performing just training on Wikipedia and this was a tool that will later appear in some other papers.
So here is another work.
This is from Google 2019. so this is called C4.
So this paper is actually more famous for the T5 model which pushes the idea that you should just treat all the NLP tasks in a world as text-to-text.
But actually a big contribution of this paper, which is a very long report, was the C4 data set.

And the observation, which was very clear at that time, was that Common Crawl is mostly not useful for natural language.

Right?
There is sort of this idea that you had these small data sets, and they were high-quality, and then all you had Common Crawl, which was a mess.
And any attempt to train on Common Crawl just led to junk results.
So, people were thinking like, "How do I get a larger but high-quality subset of Common Crawl?"
So, we saw the Reddit idea, we saw the Wikipedia-like language model idea.

C4 used a different idea, and this is saying, "Let's just define a bunch of rules."
Which turned out to be fairly effective.

So, basically, they keep lines that end in punctuation, have more than five words, remove pages with fewer than three sentences, remove pages that contain any bad words.

remove things like terms of use, you know, boilerplate.
Interestingly, remove the curly brace, which, you know, filters out a lot of code.
So, they weren't clearly at that time they weren't thinking about code models. and they filtered out anything that was not English.

Okay, so the end result is that you have 156 billion tokens.
So, you know, 40 sorry, 800 gigabytes of text.
So, this is much larger than the GPT-2 data set, which was only 40 gigabytes of text. a bit later, there was some analysis done in C4, which showed Here is the websites that were represented.
So, you have a bunch of Wikipedia, patents was a very common thing.

they also released the WebText like document.
They also used this idea of links in Reddit posts and from Reddit posts with greater than three karma and filtered through those pages. so they were able to get using this method 17 GB of text using 12 Common Crawl dumps.
Remember WebText was 40 GB, which means that Common Crawl is, you know, maybe incomplete or doesn't have everything because if it had everything, you should be able to hit 40 GB.
And they showed that at that time you could use this data and improve on a bunch of NLP you know, benchmarks.

Okay, so we've seen using rules, things that look like Wikipedia, Reddit links.

And now let's talk about GPT-3.
So GPT-3, the data set was they did their they used Common Crawl, they did their own processing. there is this WebText which was essentially the GPT-2 data set but expanded. notice that there's some overlap.
Common Crawl does overlap with this, but this is a subset of more targeted distribution. there is this in the paper it's described as books one and books two, which is miss internet-based books corpora.
So this remains a mystery exactly what it is. and Wikipedia.
Okay, so the result was about 500, you know, GB of text, 400 billion tokens.

and to do the Common Crawl processing, they trained a quality classifier to distinguish things they determined deemed to be high quality from the rest.
And they did some fuzzy deduplication because you know, WebText and Common Crawl had dupes.
Okay, so this paper followed the idea of using a classifier to just to train to do quality classification.
So, after GPT-3 came out, this was I think a big event and there were a number of efforts to try to things in the open.
So, one of the initial efforts was the Pile, which is a grassroots effort from Eleuther AI, where they had a bunch of people in this Discord just jamming and figuring out high-quality sources and they came up with this list in the end.
And today this is still pretty interesting and diverse.
It has you know, the Common Crawl, has PubMed, this has this thing called Books 3, which you know, we'll talk about, Archive GitHub, which we talked about, you know, Wikipedia, IRC, another books corpus, philosophy papers, and so on.

And so, you know, they included this Enron emails data set.
So, Enron went bust in 20 2002 and all the emails were released and this is one of the few email data sets we have, which is a weird distribution I think for email, but that's that's what you get.

so, Project Gutenberg was so, let's talk about books a little bit.
So, Project Gutenberg was started in 1971 and it only includes books that received copyright clearance.
So, mostly books in the public domain.
And there's this packaging of Gutenberg into PG-19, which is Gutenberg's before 2019.

And which is I guess most of them because for them to be in public domain, you have to wait 75 years.

So, Books 3, which is kind of this interesting data set that appeared in The Pile.
So, this is described actually as you know, 200K books from a shadow library called Bibliotek.
It concludes all books from your favorite authors.

And at that time, again, 2020, no one was paying attention.
It just sat you know, people used it to train models.
And since then, it has been taken down.

So, you cannot use or should not use Books 3 anymore.
We'll come back to Books 3 again.

So, Stack Exchange you've probably used quite a bit.
Although, I guess these days maybe less so because of AI.
There's a bunch of user contributed questions and answers starting 2008.
And You know, one thing that's it's interesting about this data set is that it's a sort of a Q&A format.

And this is quite close to kind of a real you know, application.
So, normally you think about pre-training data as here's Wikipedia articles, just like raw text.

Right?
But some parts of the web are actually look like you know, supervised, like what you would want to ask your language model.

So, you know, which I think maybe helps the model learn certain types of question answering behaviors.

So, not everything is like super magically emergent.
It's like, well, this type of data does exist on the web.
So, there's also this data has metadata, which is like the number of votes, which are useful for filtering.
And again, this data is released in data dumps that allows you to just download without crawling it.

Okay, so now in 2021, there was another paper from DeepMind.
They trained a model called Gopher, which was never released and actually was sort of subsumed by Chinchilla. but the description of data, I think, is actually really good.
You should read this paper because I think it's very thorough in the way that they describe the data processing.
Well, except for the parts where they don't tell you what's in the data. so, there's there's this thing created this MassiveWeb, and they had you know, C4, which remember was the data set that we looked at before from Google. they trained on books, news, GitHub, and Wikipedia.
They don't say anything about how they got that.
So, for the MassiveWeb, they kept English, deduped, and their quality filtering was also based on manual rules.
And part of the reason for this is that they had more control over that.
So, I think as you can see, there is a sort of this like division between people who wanted to use rules, and people who wanted to use classifiers.
So, they got, you know, a massive as the name suggests, a massive amount of text.

but, you know, their model was only trained on like a very small fraction of this of this data set.
Okay, so in 2022, Llama 1 came out.
And the data set here, this was actually a very nice paper in that it detailed their data processing.
So, this is probably one of the last non-open, fully open models that actually talk about their data processing.

So, they have Common Crawl processed with CCNet, which you know about. the classification though was whether the page references was a reference of Wikipedia, not whether it is actually a Wikipedia article. so, the idea is that, well, maybe Wikipedia articles are like too stylized.
And we know that Wikipedia articles references a bunch of other articles, which are presumably good.
So, let's call those the good websites. they had you know, C4. they did a bunch of processing on GitHub to get permissive licenses, Wikipedia. they trained on, you know, Books 3 and Project Gutenberg.
So, Books 3 really got them in a lot of trouble because they're just announcing to world that, "Hey, I trained on this data set."
And you look back, "Oh, this came from.

Where did it come from?"
It came from The Pile.
"Oh, it came from a shadow library."
So, you know, that's why people don't to talk about their data anymore. so, they looked at arXiv and actually used the LaTeX to do the processing , Stack Exchange, and they got a 1.2 trillion tokens.
They didn't actually release this data set, but they had enough of a description in their processing that this was reproduced by Together's RedPajama V1 data set, which was then used in other, you know, sources.

So, as you can see here, RedPajama V1 initially also contained Books3, which has now been kind of you know, stripped out.

So, various decisions made by early on about copyright actually influence have a fairly big kind of watershed.
Okay, so let's talk about RefinedWeb.

So, this is a paper that tried to make a point that web data is all you need.

Right?
So, the web, in some sense, is everything.
And then we talked about all these specialized sources like there's GitHub and archive and Stack Exchange.
And they said, "Well, what happens if we just stick with the web?"

So, they did job of doing the transformation from HTML to text.
They filtered using the Gopher rules, which is basically keep things that look like English. they explicitly at that time still said, "Avoid ML-based filtering to avoid biases.

I don't want to like find an overly narrow subset of the web." they did some, you know, deduplication, and they had 5 trillion tokens.
They released about 600 billion of them. so, the FineWeb from Hugging Face was a replication of RefinedWeb, but improved it.
So, they took all the Common Crawl dumps at that time, did some filtering, again using manual rules because to not inject biases, deduped, did PI removal, and got, you know, 15 trillion, you know, tokens.
So, you can see that the size of the data these data sets is growing you know, quite a bit.

and so then there is this Dolma data set from AI2 and this is a data set that includes their own processing of Common Crawl, The Stack which we'll talk about before, C4 which you know about and a bunch of other things.
And the Reddit data set was from this project called PushShift. and at that time you could still get this data and you can you know, train on it before things got locked down.
AI2 has their own you know, crawl of academic papers so called Semantic Scholar so they derive a data set based on that.

so, let's look at their Common Crawl processing.
So, they use a language identification, they So, this was model based but quality filtering still avoid model based filtering. and then they remove toxicity using a rules and a classifier. and they got 3 trillion tokens out of that.

So, DCLM I think was maybe a point where this idea of model based quality filtering started to really kind of become the the norm here.

so, DataComp they their initial motivation was to define some sort of pipeline so that people can try different data methods in a standard way and show results.
But I think the main way that I think people use this is just using their data set that they released to train models and do other things.
So, they process Common Crawl to produce data on pool.
So, this is completely unfiltered.

And if you look at this is this is massive, 240 trillion you know, tokens. this is probably more than number of tokens that anyone really trains on.
But, a lot of this is fairly low quality.
So, then they have a pipeline where all that gets you know, filtered if you only keep and you know, English. you have some sort of rules to really narrow it down.
You do dedupe.
And then you do this model-based filtering.

And you end up with you know, something that's 1.4%.
And this is the data set that we'll see actually works you know, pretty well.

So, the model-based filtering and this was you know, s- kind of strangely good.
So, to train a classifier, they took OpenHermes, which is essentially a instruction data that was generated by GPT-4 and Eli5, which is a subreddit with various questions and answers.
So, this is let's see.
Like these are tied the type of questions that are in Eli5.

Okay.
So, anyway, you get the idea.
So, it's it's kind of weird, but somehow this works.
So, the negative examples are anything from RefinedWeb, which is remember it's just like a very loosely filtered version of the web.
It's basically the web.
And it got 3.8 trillion tokens. they train a fastText classifier.
Think about this is a linear classifier. and then they show that this magical classifier quality classifier outperforms a bunch of you know other things that they tried.

So, DCLM this became sort of a for a while at least in the open community a bit of a gold standard for quality filtering. let's move on to Nemotron.
So, Nemotron this is from Nvidia.

They said that well DCLM filters too aggressively.
It's removing like most of the data. and remember they only got I mean 3.8 trillion tokens, right?

So, we need more tokens.
So, what do we do?
We are going to do something more I guess elaborate.

We're going to prompt our existing model to score find web documents based on educational value.
So, we're going to prompt a language model and said is this education or not?
Create a bunch of labels, train a fastText model.
So, that's one classifier.
Another classifier is just using the DCLM classifier.

So, we're going to use those classifiers.
We're also going to use synthetic data.
So, this is probably the one of the I don't know if it's a it's probably not the first, but one of the main data sets that really lean into synthetic data as a for pre-training.
So, they had for low quality data as deemed by these classifiers, you use a language model to rephrase it to make it more look like Wikipedia.

for high quality data, we use a language model to generate various tasks.
For example, given a Wikipedia article I can generate question answers or I can generate like please summarize this document, extract key information from it and so on.

So they resulted in data set that was six trillion tokens.
So this is quite a you know substantially larger than DCLM.

and just for kind of reference this is still kind of small by some you know considerations.
So Llama 3 was trained on 15 trillion tokens, Qwen3 was trained on 36 trillion.
Although there is a sort of this it's not clear how big these unique tokens these are because when you look at token counts in language model papers some of it's like repeated.
Like you do multiple epochs, two epochs that's twice the number of tokens.

So you have to be careful when you look at those numbers.
And they show that this the data set is you know better.
Their high quality subset beats the previous data sets.
Okay, so up until now [clears throat] we've seen a bunch of different methods for filtering.
I think all of these sort of look very kind of similar at some level.
You take a web crawl, you either decide I'm going to use rules to filter or I'm going to use a model.
If I'm going to use a model then I have to decide what looks like good data and then try to train a classifier and classify all my documents and select.
And there's a seems to be a trade-off between you can take a lot of Common Crawl.
You can get you know 240 you know trillion tokens if you want, but that's probably going to be really low quality.
Or you can get like one trillion tokens and there's sort of some you know sweet spot in between.
Let me talk about two final things.

So code and going back to this question of licensing.
So The Stack is a very nice project that was trying to make a really good coding data set since it was cleared by 2022 that coding was going to be really kind of important.
So what they did in the initial version is that they clone you know 137 new repos.
They kept the ones that were permissively licensed, removed example near duplicates and resulted in you know one sorry 3 terabytes of code.
And then in 2024 they had an update and here they took more than the metadata like the issues, comments and PRs from GitHub, the repositories from the software heritage which we talked about.
They also scraped documentation from various websites by crawling them.
They did a bunch of processing.
So in GitHub repos you have binary files which you probably don't want to train on.
They have malware, you get rid of that.
A lot of GitHub especially these PRs are bots so you have to filter that.
Dedupe, do PI reduction.
There's a lot of pull requests and stuff and they basically subsample to kind of keep it representative and the data set manageable.
They also did this thing which was I thought was kind of nice which is that there's many programming languages out there.
There's a lot of Python.
A lot of C but low resources languages like Nim which I hadn't even heard of.
They're not that common.
So what they do is they compile this code and then into a low level intermediate language, LLVM, which everything like C compiler can compile into this, and they've juxtaposed the low-resource language and the intermediate representation.
So, then the the language model can actually learn the mapping between the shared low-level representation, which has a lot of data on, and the thing that it does has less data on. and just for fun, you add in all the other you know, good stuff that you can.
So, this is sort of The Stack v2 is like mostly code and then plus other things that were used to train their coding models.
And one thing about pull requests is that their pull requests and all the metadata is not by its nature linearized sequence.
So, there has to be some steps to take into linearize it.

So, one important thing you have to side is how much context to provide.
So, for example, event might just be I changed one line of code.
Right?
And that could just be one line, but presumably to learn that, you want to provide some context, maybe a few lines around it, maybe the entire file surrounding that diff.
And these are some design decisions that you have to make for linearization.
And so, they at the end of the day, the tokens that they train on look something like this, where it's sort of this like XML like structured data, where you have like the PR and then you have a bunch of diffs and then for comments, you have basically the events of like a comment was posted what the review state of that PR was, and so on and so forth.
So, it's not just learning how to generate code, but also the software development process around code.
Okay, so finally, I'll talk about CommonPile.
So recall that almost all the data on the internet is copyrighted. some of it is permissively licensed.

Some of it is even in the public domain.
And while you can appeal to fair use and say I'm going to train on it anyway.

this is not quite settled.
Right?
So if you're very very risk-averse, then you say well, let's Like if I don't know whether it is okay, that's a no.
If you take that attitude, then that's what CommonPile did.
So can you train a good model with only permissively licensed data?

or rather how far you can you get.
So this was a project that went and scoured the internet for all the different types of data that it could possibly find that was worth training on, that was permissively licensed.
It includes you know, The Stack v2 and code.

turns out that a lot of government proceedings are actually permissively, you know, licensed. wikis, some things on the on the web.
Some news sites are actually permissively licensed.
Academic papers, online, you know, forums, things that are in the public domain, educational resources and so on.

So in the end there were 8 terabytes of you know, data, which is actually pretty good for permissively licensed data. doing this project is actually much harder than maybe it sounds at first glance.
It's not just simple as like, oh, you look at the license and you're like, oh, yep, Apache, a good or CC, a good. there's some subtleties. first is license laundering.
So, people are kind of sloppy with licenses.
So, people might take some copyright work and just, you know, put it on a slap a CC BY on it.
Anyone can write this on the internet.
It's kind of hard to tell whether this is like a real real or not.

there's also this common practice where, you know, for example, Dolma is permissively licensed.
Remember, Dolma is the AI2 collection, but collection licenses don't extend to the individual works.
So, the collection is like you know, is you I mean, first of all, there's a question where the collections are I think collections can be copyright, but you can I guess you can put a license on it. anything. but the individual works are not necessarily if good.
So, you can't just look at data sets on Hugging Face and you Many data sets on Hugging Face you see they have a permissive license, but if you dig deeper, it's actually not permissively licensed at the individual level.
And they also made a decision to forego training on any synthetic data because training language models on synthetic unlicensed data is unclear.
So, probably it's going fine because like technically these are like open way models with MIT license.
You can you know, do whatever you can use them however you want, but you know, these language models were presumably also trained on unlicensed data, so it's a little bit of kind of data laundering if you are really kind of honest here. so, so with this data, they compared with a bunch of, other models, like the first LLaMA, MPT, and Qwen.
And on a bunch of benchmarks, they show that this is not bad.
I would say it's not nearly as It's not as good as the Qwen, models, for sure, but it's, certainly outperforming the very old kind of Pythia, This is like, you know, from 20 20 three, I guess.
So, quite old models.

so, I would say that the conclusion here is that you can do reasonably, but it's I think it's still, pretty tough to compete without, doing more tokens.
But, I don't think this is the final word on it, and I think you can probably eke more out of publicly permissive licenses if you try.

Okay, so, to summarize this, lecture here.
So, maybe hopefully I now I've, given you some appreciation that data is a very rich topic.
It's not that data just falls from the sky, or you just go on Hugging Face and you download a data dataset. data has to come from somewhere, and there's also a technical, but also social context in which, data comes. you know, because at the top, there's the internet is a bunch of live services.
Right?
There's hack to has someone has to do the work of producing raw whether it be the website that gives you dumps, or someone has to build a crawler, or something.
And then, you someone has to make a decision of how to process it into usable form, filtering, transformation, deduplication.
All of these things have an impact on the quality of your final, language model.
And notice that filtering is like probably one of the most kind of important things.
How do you go from 200 trillion tokens to you know, less than like 3 trillion tokens.
I think that obviously is a huge reduction, which I think merits a lot of attention.

Data is in some sense a key ingredient that differentiates language models.
A lot of language models are roughly the same kind of transformer architecture, but you know, data is some depending on how you process it can make a pretty big difference.

There's plenty of legal and ethical issues around data, more than I can get into this lecture.
And also, this process is very messy. unlike, you know, some of the other parts of this class, where things are maybe more based on kind of first principles, you know, data processing is right now, at least, a lot just based on kind of vibes.
You set up you define this classifier, you define this rule, you set some threshold. so, there are many opportunities to improve.
So, as you do your assignment four, maybe think about whether there's better ways to do this, and this could be maybe a research direction.
Okay, so that's it for today.
So, next time I'll continue talking about data.

I'll do talk a bit about post-training data and a bit more about you know, filtering.

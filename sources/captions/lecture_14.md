# Lecture 14 字幕

Okay, let's get started.
So, today is the second day on data. last time we talked about how data doesn't really just fall from the sky.

You actually have to think about where it comes from. so in general, the internet consists of live services. these the data on those services have to be either dumped or crawled and then there's additional step where you have to process the data and we also talked about various social societal considerations terms of service copyright you have to either get a license or appeal to fair use.
So there's a lot of complexity that goes into data. so today we're going to look more at the data pipeline from transforming the data, filtering the data, deduplication and mixing.  and then we're going to top it off by talking a bit about post-training data in particular how folks are using synthetic data these days.
So the first part is going to be mostly about pre-training.
That's what you should have in mind.
So we talked a little bit about data transformation already, but just as a kind of reminder.
So raw data doesn't come as text.
Even as if you've you've scraped something.
If you ever look at inside Common Crawl, it's it's not text.
It's either HTML.
Sometimes it could be PDFs or directories in the case of GitHub.
So most of the tension on transforming data is dealing with HTML because most of the web is in HTML and for this processing there's a lot of this is fairly heuristic there's removing of boilerplate like navigation and ads and extracting content which is like the main part of the page and there's some you know subtleties around what constitutes content usually you get rid of the footers and headers and maybe menus and those things and you try try to extract the content but you could imagine in cases where some of the navigation elements might be might be helpful to learn what web pages look like.
So what is content and what is not content is not always clear and then what do you do about images and tables are on in web pages. so inherently this is a lossy process because you need to linearize HTML which at least is either hierarchical or visual if you think about the rendered output and do a sequence of tokens.
In particular tables are a bit tricky to deal with.
Simple tables you can render using markdown but if you have nested tables then that becomes quite challenging. you have to give up at some point or approximate at some point. so typically HTML to text processing is rule-based and the reason for this is that rule-based processors are very fast and also they you're not trying to do too much here.
You don't need too much intelligence.
So rule-based generally works. now I think there could be a case for model based interventions at this point.
They have to be very fast and they have to you know do something more more intelligent.
But if you ever look at data you'll notice there are imperfections in the data just because any rule-based processing is going to be is have some failure rate.
As we showed last time, the accuracy does matter depending on which tool you choose. if you use jusText or Trafilatura, then it's you know I guess jusText parse on these extended DCLM you know eval works better than the others.
I'll talk also briefly about PDFs.
So there this hugging face work which released a data set called FinePDFs.  so PDF files if you have open them up not in PDF reader looks looks kind of like this. and this needs to get kind of rendered. so PDFs can be found on just like web pages on the internets and Common Crawl u does have you know some PDFs. generally Common Crawl focus on text, but sometimes if you're just given a URL, you might not even know before you fetch it whether it's a PDF or not if it doesn't have the extension. there's a lot if you read this blog post there's a lot of details which I'll spare you of. for example, one thing is that many of the PDFs that are in Common Crawl are truncated because PDFs are big.
So then that re demands that rec crawling is necessary and then there's a question once you have the PDF file how do you actually convert it to text and you know some PDFs are might be just scans as well right so they're essentially images so there's a bunch of tools that this paper tried out but mostly this kind of involves running OCR using a VLM and this obviously can be much more expensive than what we were doing before with text.
Fortunately PE or unfortunately the PDFs are a very small fraction of the whole internet but they are very valuable because generally if you bother to make a PDF that means you probably have something interesting to say as opposed to a web page.
So the quality of a PDF is average PDF is generally higher than for a HTML file. there's a lot of clean up and filtering and PDFs more so even then than web pages a lot of layout information is missing because in HTML you have various tags like H1 and P that gives you some semantic information PDFs are by design all about layout so you don't necessarily they preserve the kind of the semantic structure okay so there's a bunch of transformation information that happens at this point.
You have text and but you're not done.
You're far from done.
So the next step is filtering.
So I'm going to talk somewhat abstractly about what filtering is.

Here's a sort of the building block.
So suppose you have some target data that you want to get. this is usually a small amount of high quality data and lots of raw data.
This is the the fresh shipment of the tokens that you do transform from the previous step and the goal is to find a subset of this raw data that is similar to target.
So this is the general skeleton for filtering and may almost all kind of filtering falls into this schema.

So filtering, you could there's many reasons you might want to filter. if you're training an English language model or a German language model, you might want to identify the language and filter things that don't match that language. the main reason for filtering is quality filtering.
You want to find things that are high quality as opposed to low quality.
You don't want just you know spam.
You want encyclopedia information. and then another application is toxicity filtering.
Of course, the internet has plenty of nasty content and maybe you don't want to train your language model on that.
Okay, so for a filtering algorithm, you want that you want some generalization from the target data, right?
Because you already have the target data.
You don't want to just only get the target data but you also want it to be extremely fast because you have to run it on the whole internet.
So this could be 100 trillion tokens worth of worth of data.
So generally filtering is you end up with a very small fraction single digits fraction of your entire data.
Okay.
So, so remember the general framework given the target and raw you're trying to find a subset.
Okay.
So the general scheme is that you estimate some sort of model based on R and T and derive a scoring function and then you keep the examples in R based on their score.
So typically there are two types of classifiers.
One are generative models. so this is basically you have your target data you can just you know estimate a model of that data and remember this has to be cheap so probably you're not training a big language model generally KenLM basically says I'm going to train a five gram model a more common thing which I think we're mostly seeing these days is just train a classifier and The classifier says, I'm going to predict a positive label for examples that are in t my target and negative labels for things that are in raw but not in t.
So you basically it looks like t are your positive examples. some random subset of r your negative examples.
You maybe balance it and then you train a classifier.
And the tool that people generally use is fastText because it's fast and it's generally it's just a linear classifier bag of words.
And then once you have this model now for every new document you can score it and you set some appropriate threshold depending on your quality bar and you keep the examples sometimes stochastically sometimes not.

Okay.
So, so this is a very much a model based filtering approach.
Remember from the last lecture, there are many data sets from you know generally a few years ago that did not use modelbased filtering because they wanted to not bias things too much.
These days I think basically everyone does some amount of modelbased filtering because unless you are compute plentiful which in which case you probably don't need to do as much filtering.
You can just train on everything.
Most people are compute poor you have to be very smart about how you're filtering.
Otherwise you're just wasting FLOPs on kind of lowquality content.
Okay, so let me talk through some instantiations of filtering.
So remember I mentioned language identification. so the goal is just to find given a piece of text, detect whether it's of a particular language. so Meta has trained these set of fastText language identification models which often you can just use off the shelf.
So it supports 176 different languages.
It's been trained on a bunch of multilingual sites.
So, Wikipedia, remember, has a lot of languages.
There's also translation sites and sites for different languages.
So, a language ID is generally a fairly easy problem compared to many other tasks that we're dealing with.
So, a simple classifier, if you look at a few words, you can tell that it's Spanish or or Japanese.
Now this does you know there are subtleties because of code switching and text and there's some dialects.
So it's I wouldn't say it's like an absolutely solved problem but you know this is not really the bottleneck for training a good language model. okay and then you can just so once you have your classifier you just choose some threshold. this is generally fairly heuristic.
So another thing you can do with filtering is that suppose you're looking for data of a certain type.
So let's say you want to get really good at math.
So you go want to go and find a bunch of math data.
So open math text is this paper from 2023 where the goal is to create a large corpus of math text. this pipeline is consists of a few steps. so it's not training a single classifier.
So first you use some rules to filter does it contain LaTeX commands. you they also use KenLM.
So this is a generative approach and trained it on the proof pile which is a known data set of math and keep it if the perplexity is you know below some threshold. and then you also train a fastText classifier to predict whether it's mathematical writing or not.
And there's, you know, two different thresholds.
If it has LaTeX in it, then it's a higher bar.
If it's sorry, if it's it's a lower bar, but if it's doesn't have LaTeX in it, it's a higher bar.  and then as a result, they got you know 15 billion tokens which they use to train models.
And as a paper they show that this more targeted security data collection results in models that are better than math at math than models that were trained on 20 times much data which were not filtered in this way.

Okay.
So the quality filtering again think about it as a tool. you can define quality however you want.
There's no universal notion of quality.
If you want define quality to be math then you can go and get math and you get better at it. so GPT3 I think I mentioned this last time but just to put it in this framework.
So positive examples are Wikipedia WebText.
So these are pages that link out from high you know star Reddit posts and some books and negatives are generally sampled from the web. they train a linear classifier and keep documents if the linear classifier is scored highly enough. there's the first llama paper the positives were pages referenced by Wikipedia not Wikipedia articles themselves and the it's just kind of the same idea.
So 51 from Microsoft is interestingly had you know also falls into this framework.
So they started with the raw data is not already a Python subset of The Stack remember which is this all you know code and they're the prompt they define a prompt which is determine educational value and then they use GPT-4 to classify a subset of R 100k subset of the R with this prompt.

and see if it's and the ones that are positive are kept and called the target.
Right?
So the target here is actually output of an expensive you know classifier and then you train a cheaper classifier in their case a random forest but you could have probably used fastText as well.  and then you select data that is classified positive by this classifier.
And they also show that using this data set you do much better than if you're just using the raw data.
So this performance goes up in fewer steps to a higher value.
Okay.
So toxicity kind of works the same way.
There's this data set Jigsaw toxic comments where it came from this project where the goal is to help people have better discussions online.
So here the data is the Wikipedia talk pages which can I guess for controversial sites get quite heated and this has been annotated with whether there are any toxic comments on there and you can similarly define positive and negative examples and you can train a classifier on that.
Okay.
So, so now I think you have the tools or some examples and inspiration.

You can identify a particular type of data that you want.
You build a classifier for it and then you can go filter Common Crawl for that.

I want to talk about one subtlety here u which is that you know the notion of what you want for data is actually depends on what model you want to train.
So in particular, it depends on the number of tokens you're training on.
And so there's no optimal threshold.
So before when I looked at the classifier gives you a score there's you can't say 0.9 and say that's the best because it depends on what you want to do.
Intuitively if you are going to train for a longer period of time then you can tolerate lower quality data.
If you're training for shorter then you want higher quality data in general.
Okay.
And of course you know if you could wave of magic one if you're training for longer you want more high quality data but that's not an option that you're given the data pool is what it is.
So here's an a kind of a preliminary experiment that Michael Ryan did and so basically this plot shows for a 157 million parameter model we're taking a very small pool so 100 you know works which is a tiny fraction of Common Crawl and we're look training and training for more over time.
So let's take a look at the blue curve.
So the blue curve is DCLM.
And so the loss starts here and then it comes down and each of these lines is when you epoch over the data.
So this is one epoch and then the next blue is the second epoch and so on and so forth.
So there's not that much data.
So eventually you have to repeat your data and you know the loss continues going down because the second time you look at the data you're still learning things and at some point you start to overfit.
Okay. whereas if you look at resilient pars this is basically no filtering. so here it's it's much worse in the beginning. but as you continue and continue at some point you also epoch it starts to go down you know slower.
So you kind of see this trend where high quality data is better in this regime where you're not epoing but once you get to lots and lots of tokens high quality data is no longer that great.
And of course you with high quality data you wouldn't even get you wouldn't want to go into this regime because you're overfitting.
You'll probably stop here.
But even this at this point this is worse than if you had trained for longer using low quality data. >> Yeah. >> Just a question about like the computing the metrics.
This is actually not like totally about graph but like when you're like each of those dots correspond with one training one training run, right? >> Yes. do you ever need to do the confidence interval when you're doing these kind of pre-training experiments or like is it kind of enough just to do it one time? >> So the question is each of these points is a single training run and should you do it multiple times and get confidence intervals? ideally that would be good practice. often you'll see in these papers that these are kind of scarce because each training run is fairly expensive. so you but in reality when we have done these experiments it's generally tends to be stable I would say at least for pre-training. >> Yeah. >> Yeah.
So you mentioned like trading for longer and high quality data as that's not like a combination that we're considering but like if we are trading with for longer on this high quality data would it have diminishing returns compared to like training for longer with quality data? >> Yeah.
So the question is if you were able to get higher quality data and you train for longer would it still have diminishing returns? every data set is going to have the machine returns eventually. that's finite. but it would just it would probably be here.
It would just like be down here and just just keep on going down.
Okay, great.
Let's let's move on.
So, summary, filtering is pretty critical for building a good model, especially when you're computed like most of us. technically if you have infinite compute you don't need to filter and you can train on everything and it'll be a giant model but realistically everyone has to filter.

so and the recipe here is figure out what good data looks like and then you can train a classifier and that will extrapolate to the rest of your model.

and what the good data looks like.
You can either find it by saying, "Aha, there's this data set out there that I really like and I just want more of it."

Or you can craft a prompt to a language model and then use that to construct to do a sort of a preliminary filter of a large pool and then use that good quality data to train a smaller classifier and extrapolate to everyone else.
Okay, next I'm going to talk about deduplication.
So at this point we have filtered our data set so we only have what we deem to be high quality data. but you know often data still has duplicates in it and there's you know two types of duplicates.
There's exact duplicates.
So this happens when for example if you look at these mirror sites the whole point of a mirror is that it's a duplicate and you know sometimes the web crawler isn't smart enough to know that this mirror is exactly this mirror so it just crawls many sites and you'll get this exact same content. also when you fork a repo that is also a duplicate. even if you make changes probably you're making changes to a few files so 99% of that repo might be the same.
So duplicate is abundant and yeah sometimes there's actually near duplicates which are not mirrors or derivative but they just happen to be same text differing by a few tokens. usually probably this came from a kind of a it could be copying or came from a different common source.
So here's some examples of near you know duplicates.
So terms of service and licenses. so I guess the MIT license probably shows up you know on a lot of places. and so many cases it is a exact copy but only of that license unless someone made a typo when they copied it. but the rest of the page that the license might be part of is might be different.
And we talked about how a lot of websites have the same headers and footers.
Those are also you know duplicates. there's also cases where for whatever reason you know there's this is from LM1B so the one billion or benchmark there's like these articles where you just have you know typographic differences like there's one version with a comma and one version without a comma I don't really know why but that's happens and sometimes you see these templates where there this is like some you know probably low quality content where it's like a it's a essentially looks like an ad of some sort and someone just you know templatize replace Canada with USA.

so if you train like on this data but you are just training on different variations of this with different entities this is going to be wasting your GPUs.
There's a more extreme cases.
So this is why it's good to look at your data.
So this audit of the C4 data set found this product description 61,036 times in the data set.
And if you trace back, this is really bizarre.
I don't know why this is like some description of this gas mask.
And this just showed up in this data set in Common Crawl 61,036 times.
So yeah, the web is weird. so why do duplicate?
So the first clear idea is clear because you want to train more efficiently.
Duplication reduces your data set size without really losing information because you're just removing duplicates.
And it also has this benefit of avoiding memorization which this paper you know talks about. for example, if you have some copyright content that's duplicated a lot then if you train on it then you memorize it and also privacy concerns. but mostly I think it's just to make sure that you're not wasting FLOPs related to due duplication that's also decontamination which is arguably even more important to and it's the same sort of deal is that you want to make sure that your test set is not in your training set.

Okay.
So how do we dedup?
So here's the design space to think about.
So first of all, what are the items that you're deduping?
Do you do at the sentence level, the paragraph level, or the document level? how do you determine a match?
Is it an exact match?
Is it existence of a common sub item?
Is it the fraction of common sub items u for near deduplication?
And then once you found a duplicate between two pages, what do you do?
Do you remove all the instances or do you remove all but one?
Okay, so and the key algorithmic challenge is that deduplication is fundamentally about comparing items to other items. and normally if you're doing filtering this is about an individual item.
This item is it good or not?
And this can be parallelized.
It can be you know it's sort of linear time which is good. and even in linear time we're trying to make it fast by having rule-based or very small models.
But due deduplication clearly you can't do the n square thing where you compare everything to everything.
You need linear time algorithms to scale especially at this web scale.
So typically the deduplication literature uses hash functions to get around this.  so we'll develop some of these ideas.
These are quite nice ideas from thanks to you know the algorithms community.
So I think everyone knows what a hash function is.

It takes of some sort of value like a string and maps it into something like a string or integer. and a hash value is much smaller than item.
Hash collisions are when two distinct items map to the same age.
Okay.
So when you look at hash functions there are really fancy hash functions which are cryptographic nature.
These are collision resistance used in cryp cryptography and bitcoin and things like that.
And then there's these like faster ones which are used for hash tables where hash collisions aren't the end of the world.
And so we'll be using these.
Okay.
So take a string and map it to some value.
That's what hash functions do.
Okay.
So exact duplication is conceptually very simple. you take a string you see if there's an exact match and you remove all but one.
Okay.
So here you have a bunch of elements and you hash them and you dedup.
Okay.
So this is and exact duplication is is very nice.
It's very clear what's happening, but this isn't really good enough for the messy web data because often you have these near duplicates. and one note is that there's many ways to have you know written this. this is written by in a sort of like a bit of a map reduce style way which makes it more easy to easily parallelizable and scale. so C4 this paper from the T5 paper which process Common Crawl did exact duplication. the items they operate on were three sentence spans. and you do exact match and they remove all but one.
So, one, if you're kind of paying attention, you realize that there's something kind of a bit strange about this because you're looking at three sentence spans.
And if you find two documents with three sentence span, you remove all but one.
That means you're just going to rip out three sentences from that document, which is a little bit strange because it breaks the coherence, but you know, this is what they did. okay so that's exact dup deduplication.
So how do we do near deduplication?
So first of all we have to define what approximate match means here.
Okay.
So to do that we're going to define this called Jaccard similarity which is a fairly standard notion.
And Jaccard of two sets is basically the size of the intersection over the size of a union.
Okay.
So given these two sets 1 2 3 4 and 1 2 3 5.

when you compute Jaccard you take the intersection that's 1 2 3 you take the union that's 1 2 3 4 5 and you divide and the Jaccard is 0.6.
Okay.
So the Jaccard is a number between 0 and one.
Zero means that they're disjoint. one means that they're identical.
Okay, so a fairly natural notion and we're going to say that two documents are near duplicates if their Jaccard similarity is above some threshold let's say 0.99.
So now the question is how do you find near duplicates in linear time?
Okay, so fortunately this is a solved you know algorithms question and the answer is to use MinHash.
So well okay so the first step is to use MinHash that's not the final answer.
So MinHash is a random hash function so that the probability of a hash collision is exactly the Jaccard of A and B.
So this is a very kind of nice property because hash functions hashing is good for making things linear time.
Jaccard is the metric that we want and we're sort of connecting these two right now or in expectation.
So what's interesting is that normally you want hash functions to define hash functions so that everything is hashing to different distinct elements are hashing to different things.
You don't want collisions but here you actually want collisions u not arbitrary collisions but you want to control the collisions in a certain way to align with a similarity.
So similar things you want to collide more than the disperate things.

Okay.
So the MinHash is essentially you actually let's see maybe I will okay so here's the MinHash.
It's fairly simple.
You take the set and you hash every element in that set and you take the minimum element.
Okay.
So this mean might if you're seeing this for the first time it may seem a little bit strange.
Why are you taking the min you can take a max too doesn't really matter it's just a way to break ties.
So here's the picture that you should have in your mind.
So okay so if you have a remember a contains one two three four and b contains 1 2 3 5.
So this is a characteristic matrix representation of these two sets, right?
And so what the random hash function is doing is that it induces a permutation over the items.
Okay, so it might be 4 3152 or something else.
And then you look at which item is first according to this permutation in A and which item is first in B.
So each item has the same probability of being first.
Okay.
So if you have if it the random hash function puts one first then first in A will be equal to first in B.
And if it's two first then it's the same.
If it's three first it's also the same.
But if it the random permutation puts four first then the first element in A is going to be different from the first element in B.
And same with five.

Okay.
So look at this representation.
Basically the random hash function says elect one of these rows to be you know first and then when I the min is just telling me the min the the min equal of a equal the min of b is basically telling me whether that row has is the same.

Okay, so that is the I guess the proof of why MinHash why this property holds which is that you have a hash function in expectation over your random choice of hash functions that the probability of collision is the similarity metric you want okay so let's just check this in code so I'm going to generate 100 different hash you know functions S each hash function is given by a seed and I just check whether the MinHash is equal to MinHash and the estimated Jaccard.
So I'm going to just look at the number of the fraction of matches and you get 0.6.

Okay.
And the key thing with a MinHash is that I don't have to do the n squared thing.
I can compute the MinHash of A and the MinHash of B and MinHash of a different set and I just look for collisions.

Okay, any questions so far about MinHash.
So now we can hash our items, but we're not done yet because a collision doesn't tell us that Jaccard is above some threshold, which is what we want.
We want to find a and b such that Jaccard is greater than 0.99.
All we've done is said, okay, well, if we got a collision, the probability of that collision of two things colliding is the Jaccard.
But that's not really that useful by itself. it's stochastic and I can't really get anything reliable here.
So the next thing idea is this thing called locality sensitive hashing which essentially solves this problem.
So this is a very classic idea in theoretical computer science and it's quite nice. so the idea here is that we have a and b colliding with probability equal to Jaccard.
Okay, so it is true that more similar items will collide more often, but it's very stochastic, right?
The variance is, you know, quite large here.

So our goal is to have A and B collide if the Jaccard is greater than some threshold.
So we in some in some sense have to sharpen these probabilities somehow like the probability can't just be literally equal to the Jaccard.

So the solution is to use more hash functions u and these hash functions are going to be independent.

So this part is you know gets a bit technical but we'll walk through it.
So you break the n hash functions into u b bands of r hash functions.
So if you have 12 hash functions you have three bands.
Each band has our four hash functions.
Okay.
So there's a band here H1 through H4.
Here's a second band H5 through H8.
Third band H9 through H12.
Okay.
And then so what we're going to try to do so each hash function gives us either collision or not collision.
Okay. and so every hash function there's a probability of colliding.

So the key here is that we want to say compute the probability that A and B collide.
We say that A and B collide if for some band all of its hash functions return the same value.
Okay.
So if H5 and H6 and H7 and H8 all return the kind of the same value.
So sorry if h a of a equals h5 of sorry h5 of a equals h5 of b and h6 of a equals h six of b and so on that means this band is triggered and then I would say they collide or if this band if all of these say coll agree then I say it's collision or these two but you don't have to have all the hash functions you know return the same value.

Okay, so there's sort of this and or structure here that is is doing the lifting here and we'll see why this works.

So now let's say you have a particular Jaccard of A and B what is the probability that A and B collide according to this definition?

Okay.
So we can calculate this. so let's say that this Jaccard similarity is 0.8.

Okay.
And we have five you know bands and each band has 10 hash functions.
So then we can say what is the probability of a fixed band matching.
So that's 0.8 to the r because each band has r hash functions.
So the probability that they all have to match is 0.8 to r.
Okay.
So the probability of a band of a fixed band matching is generally quite low.
It's exponential in r.

Okay.
And then now the probability of collision is the probability that some band matches and that's going to be the probability of one minus the probability of so this is a probability of a band not matching and this if you raise it to the B there's B bands this is a probability that all of them don't match and then one minus that is the probability that some match okay so the probability of collision is you know four and you expect it to be higher because you're given you know beach tries to get a match.
Okay.

So if you plot this is what it looks like.
Okay.
So you have on the x- axis the similarity here and we were looking at 0.8 and you look at the probability of a collision here.
So we would expect that if it's similarly zero then it should be zero.
If it's one it's one.

But notice that this is kind of a interesting sshaped which is nice because this is what we wanted to sharpen.
We wanted to basically say if the similarity is below some threshold then we want the probability to match to be as low as possible.
And if the similar is above some threshold, we want it to be as high as possible.
So we're trying to get this to be like a phase transition.
Okay.
And if you know plot this function as a function of similarity, that's what you get.
Okay.
So Oops.
Okay.
So, so let's just look at some examples.
So, let's concretely we have similarly 0.7 to 0.98.
Okay. and we're going to look at how u b= 10 r equ= 10.

So we look at the collision probability according to our definition and we see that this gives us a range of you know 0.25 to one.

Okay.
So if you were to set a threshold you know this is this is not you know bad. we can like set it here and you know these with you know some probability will you know there's still false positives because even if these are below a threshold let's say 0.9 these might still collide but we can filter them out if we if we want but the ones above are mostly kept.
So what happens when you move so increasing r so remember r is a number of hash functions within a bucket.
So when you increase r the threshold sharpens and moves the curve to the right everything becomes harder to match right because you have more hash functions inside a bucket you have that it sharpens that exponent.
So the probabilities you see here are kind of sharper here.
And now everything moves to the to the right. and now it's fairly unlikely that if something has low Jaccard, you're going to match it.
So before it was like 0.25 and now it's, you know, 08.
Okay.
So we're sharpening the probabilities and then increasing B has the effect of moving the curve to the left.
It makes it easier to match.
B is the number of bands and some band has to match.
If you have more bands then there's more chances of matching.
So if you look at this then before and after.
So now we're making before this 0.9 was only 72 and now it's 0.92.

And of course this also increases but you know not by that much.
Okay.
So and you can drive these phase transition to be as you know as sharp as you want by increasing B and R.
But of course, if you can increase BNR too much, then it can be more expensive.
But okay, so let's look at a more real world setting.
So in this paper on deduplication, the they had you know B 20 bands and R is 450.
So to give you an idea of the magnitude of these numbers and In general this phase transition happens at the at a threshold which is one over b raised to the power of one over r.
So if you want to find a you know if you want to filter based on Jaccard greater than 0.9 then you basically have to set bnr such that this is 0.9 and then by changing BNR you can make that phase transition sharper.

Okay.
So remember probability that a fixed band matches is one over b.
If you have at this threshold then the probability that a and b collide is you know approximately it's a it's a constant.
So the probability of collision is 64.

Okay.
And this makes sense that basically if you have a phase transition the sort of center of that phase transition is like point you know 64 and everything below it should go to zero as BNR increases and everything above it should go to one as BNR increases.

Okay any questions about LSH?
So this method is called MinHash LSH because LSH really works for any hash functions.
For the duplication of language model processing, we're using the MinHash which approximates Jaccard.
And so that basically is the one you should use here.

Okay. questions about dedup.
So one note about dedup is that there's often as we'll see data sets that are coming in and sometimes the duplication will happen within a data set but you actually have to do duplication across your entire data set because often data sets can be redundant.
So you know sometimes that's not done but it should be.
Okay, let's go on to the next topic which is data mixing.
Okay, so far we've transformed our data from really you know raw HTML or PDFs into text.
We filter for high quality. we dduped.

So we have a smaller set of high quality documents and this generally happens for a given you know data source but language models are trained on multiple data sources.
So in Marin currently this website tracks the different data sources that the next model will be trained on and you can see that there's a bunch of things there's some Nemotron there's FinePDFs which we talked about there's some you know institutional books and some code and all these things.
Okay, so you have all these different you know sources and the question is how do you combine these?
So if you look at an older paper The Pile this is the set of sources that they had at that time and they essentially assign a particular weight to each component.
So basically you have a distribution of resources.

Okay.
And so where does that weight come from?
So more you know just cut and dry let's say you have three sources you want to find a data mixture and a data mixture is just a distribution over your sources.
Okay.
So if you're just thinking about this problem from first principles well not from first principles but like just from what would you what you might do vibes which is I guess opposite of first principles is you just manually set it based on some intuition which is more often than you might think what people do right so I think this is this is definitely fairly vibes based And even more recent papers you know you just look at it maybe you use some method and then you just like tweak things. you can also do uniform sampling which is you just put a uniform distribution over your sources and you sample every you know chunk you get you just sample a source. you can also do proportional mixing which is that you sample proportional to the number of tokens in a source.
So if you have a data set with more you know tokens then you put higher weight.
So this is also you know generally a rational thing to do but you might worry that the if you have a huge lowquality data set that's going to eat up a lot of your tokens.
So this doesn't seem quite optimal either.
So intuitively you should upweight your higher quality sources.
Okay.
But there's two things that are important to keep in mind.
One is that you do want to ensure some diversity, right?
So often sources are incomparable for literature, code, and papers. you can't really say that you know this paper is higher quality than this code because they're just incomparable you know objects maybe and if you want your language model to do well you don't want to just put all your your mass on just papers the second thing which we'll talk a bit more about is each source is actually finite so if you put too much weight on a small source then you need to you essentially run out of that source and you're you need to just epoch over it.

Meaning that you keep on training on the same literally the same tokens and this is going to be bad for reasons we'll see.
Okay, so let's let's try to unpack this a little bit more concretely this last point.
So imagine you had just have two sources a lowquality source you have 10 trillion tokens and you have a high quality source that has 10 billion tokens.
Generally high quality sources are smaller.
Okay.
So if you just do a naive data mixture, let's just do uniform.
So half on high, half on low.
And I'm going to train for one trillion tokens. so when I say train for one trillion tokens, that doesn't mean unique tokens.
I just means train for that many steps divided by batch size essentially or times batch size. okay.
So then you look at let's look at this which is the number of times a particular element in the lowquality data set was trained on.
So that's the number of epochs a particular data point was trained on.
So p low times train tokens is the number of times I'm want to request tokens from this lowquality source and divided by the total number of lowquality sources.
So this is less than one.
So basically that means 5% of the tokens in this lowquality data set I'm even going to touch and train on once. and then if you look at high quality, you'll see that the number of epochs is 50, right?
Because I only have 10 billion tokens here.
And if I'm half half, that means 500 billion tokens.
I'm I need to train up 500 billion tokens of high quality data.
But I don't have 500 billion tokens of data.
So I have to repeat each data point 50 times.

Okay, so this is actually really important and you know some big model trains range runs have kind of messed this up is that you can't just like naively look at the data sets and define a distribution and then you sample because if you just look at the quality if you look at the data you don't you don't you're sort of missing how many data points are there.
Okay, does this make sense?
Maybe pause for questions.
This is a kind of important point.
Okay.
So, yeah.

>> I guess like does it like why do we need in the first place? like doesn't it achieve comparable performance level way I don't know fewer you know >> so the question is why do you need 50 epochs and I guess the point is that you don't need 50 epox worst case it's wasting compute sorry best case is wasting compute and worst case you're overfitting but 50 the reason it's 50 is that if you just go in and you define this data mixture then you're going to be end up doing 50 epochs without realizing unless you're paying close attention So this is basically the lesson is like look at how many epochs you're actually doing on your data. >> Yeah. >> How does this how do the mixtures get expressed during training?
Like so when you switch every step do you train like steps of 10 saying oh I want to do 10 code 10 to see because I feel like it was like is there also any intuition between >> what decisions are made there. >> Yeah good question.
So the question is like how when you train how do the mixtures get kind of realized and so in general you sample so you have you want to fill a batch right?

So you can sample a essentially for each element which mixture component it comes from and you fill up a batch with you know sequences from that mixture component.
Usually you every sequence comes from one mixture component.
You're not sampling like per token. >> I guess so each batch should be mixed.

So I guess the unexture assumptions are at the batch level and then you just train for that.
It's not like oh there's one step is one step is >> yeah in general you want to reduce variance you want to train from multiple you know a batch should have some mixture in it.
Yeah.
Okay.
So this idea I mean this problem has been noticed you know quite a long time ago. and so in this paper they introduced a UniMax and I in there that case it was they were training in multilingual models.
It was like very clear that some languages were very low resource.
So you had to do something about this.
So in this paper they noticed that before some works would essentially take the proportional mixing and raise it to some power to kind of flatten it out the distribution. but the idea in this paper is that let's be a bit more kind of explicit about that.
Let's sample the sources uniformly but with a hard cap on the number of epochs.
Okay, so this is the key idea here is that you say I'm only going to take 20 epochs over a particular, you know, data source.
So if you've done that, then you know, too bad.
You go you don't get any more tokens.
You move on to the next thing.
So this is sort of like a safety net in some sense.
So in particular if you look at the number the probability assigned to a source times the number of training tokens that should be less or equal to the cap.
Okay.
And there's a simple procedure for actually just determine the mixture subject to this constraint. okay so let's switch gears a little bit and talk about how we define the mixture in the first place.
Right?
Because if you have 50 sources, there's 50 numbers you have to fill.
And we noticed that proportional mixing isn't really enough.
Maybe you can try to get estimate some quality metric and then you can sample proportional to that. but that is also heuristic.

So the most principled methods and most kind of easiest to understand like what's happening is these regression based mixing methods.
So DoReMi and RegMix and the idea here is kind is relatively simple.
So first you let's say you're trying to train a large scale model.
So you go to a small scale let's say one you know I guess let's say 300 million parameters and you try different data mixtures.
So let's say you have three components.
You try different mixtures and then you train these small models.
So you train a swarm of small models.
Each model gives you a loss on some target metric.
It could be some downstream evals, it could be perplexity, whatever you want.
Okay.
So you use these data points to basically fit a regression where the regression model maps these inputs which is the data mixture weights to the loss.

Okay.
So now you have basically a very cheap model that tells you if I were to train on this mixture what loss would I get?

then you can essentially optimize that you know data mixture sorry optimize that function for the optimal data mixture and then that is the data mixture you use to train the large scale model.
Okay, so that's the kind of the simple procedure and you'll notice some kind of similarities with scaling laws where you're trying to do some cheap computations to figure out some answers and then you know scale up.
Okay, so there's a few design decisions here.
One is you know what are the mixtures that you're trying out here?
You need a distribution over distributions here. so often people use some durlay distribution. you also have to define the regression method. people have tried linear models or boosted decision trees.

you have to figure out the target here. often this is based on downstream eval.
So, we have to be very careful not to overfit, right?
Because we're doing pre-training and then, you know, supposedly pre-training is supposed to be training this general purpose model and we're not trying to fit some downstream evals.
And if you're not careful, for example, if you have a bunch of code evals, then well, guess what?
You're going to upweight all the code data.
That's, you know, not rocket science.
And if you then go and you say I want to generate some poetry, you might you know realize that you've over fit.
So you have to be very careful about this.
So proportional mixing and uniform mixing don't have this problem because you're not looking there's no downstream evals.
And the final thing is that you know how much of a difference between small and large scale is there and this is a cost and accuracy trade-off.
Of course, if you train really really small models, this might not be representative.
But if you train large models to do this, you well, there's no point in doing any of this.
You just train you're basically doing hyperparameter tuning at the largest scale, which is too expensive. so this RegMix paper actually does a nice has this nice table which shows you for a number of different methods that all fall into this kind of framework. this the size of the proxy models they're generally fairly you know tens or tens of millions of parameters.
How many you know points do you sample mixtures do you sample m here being the number of parameters in your mixture sorry the number of domains rather and you can use Dirichlet you can use exponential and then the regression model you can fit a log linear model which you know tends to work pretty well and then you can use different ways is to solve this optimization you know problem.
Okay, so there's sort of two leaps of faith here that you just have to be very careful of is that the regression model was trained on a bunch of these small proxy runs and you're optimizing this.
So the hope is that your the minimizer your optimal data mixture that regression model is still accurate. and this is a little bit kind of, you know, you have to be very careful because, if you, let's say, sample a random mixture, you'll probably be fine because, because of, you know, classic generalization, you sample a mixture, you and you fit a function, it should be able to predict in distribution.
But when you're optimizing, you're essentially trying to go to potentially the extremes where you might not have as much coverage.
So that's one thing to be careful of.
And the other thing is that optimal data mixtures you just hope that they transfer for small scale to large scale.

and this in general it seems like at least at the scales that open community works with tends to be true or not blatantly false. but clearly there are scale dependent effects.
If you just think about the the previous slide when we were talking about filtering, if you're training for many more tokens, then probably low quality data is okay.
So clearly the optimum is not the same, but you just kind of hope for the best here.
Okay, so hold on.
There's one scale dependent effect that we already talked about but and we need to address here which is that you imagine so we have a lot of lowquality data 10 trillion tokens 10 billion tokens of high quality data and if we train a small model on low token counts you know remember what what happens is that you might this is a madeup you know example But you might put a lot of mass on your high quality data because if a low token counts, you're not epoching.
So it's like, oh wow, Wikipedia is so great.
Let's just train on Wikipedia. but if you then go and train your large mixture on this large model on this mixture, then you're going to end up epoing a lot on this high quality data and you'll overfit and that's definitely bad.
So one way to fix this which is done in the RegMix paper is you can just cap the number of epox solution.
There's another solution here which goes by probably multiple names but one name for it is simulate epoching.
And the the principle here is that make your small scale look like your large scale.
This is kind of a general theme for the course.
We saw it in kind of when Tatsu talked about muP.  you parameterize your model so that your your hyperparameters transfer.
And so the kind of same principle applies here.
And the problem with you know doing this kind of naive scaling is that at low you can't be not epoing at small scale and epoch then epoing at large scales because then those are qualitatively different operations on your data set.
So the way you would handle this is you downstream your sources proportionally.
So if you have your small runs were 10 billion tokens and your big run were one trillion tokens then you have a you know one in 100 kind of downsampling and then you basically use the down sample mixture.
And so the idea is that if you have the down sampled you know data then this solution is not going to look good because you're not training on all you're not you don't get to just train on all Wikipedia you're going to train on some minuscule fraction of Wikipedia and you'll realize oh actually I'm going to be epoing a lot and that's going to have really bad loss which means that in the optimization you're going to look for more balanced approaches. is so you're kind of simulating the data scarcity that you would get at in your high regime at a low scale.
Okay.

So to summarize this section the problem is how do you weigh different sources?
You have Wikipedia, you have your Common Crawl, you have your your code, you have some math data that you scrape from somewhere. how do you balance them? regression based mixing is a nice framework I think to think about things.
You estimate your a function form that maps your mixture weights to a loss at small scale optimize and then you generalize to large scale.
And you just have to be very careful with this because of epoing and overfitting kind of issues.

And you have to either use capped epoing or simulate epoing.
So one lesson here is that if you're trying to optimize anything you have to be very careful because you are in danger of optimizing the wrong thing.

Okay.
Any questions about data mixing before I move on? when you're a dev sample assume something is like too small that like you can't really generalize as well.

>> So the question is if you're downsampling is there's a risk maybe you get too small of a data set. yeah absolutely.

So I think right you might get into a case where you just have so few tokens.

I think then what will happen is that your optimum will just put very small mass on that and when you scale up maybe you get the right set of tokens.
So in principle like it should work out but there might be like rounding errors where like you end up instead of training once on this data set you train like zero times by accident.
But you can you can sort of work around these by just like think you always train once on this data.
Yeah.
Okay.
So, yeah.

>> Within data mixing, do you ever because like I guess these sources just happen to align with different topics, but do you ever take like a diverse data set and then try to do like data mixing within that? >> Yeah. >> So, the question is do you do what do you apply data mixing to the like different you know domains or even within a data set?
That's actually I forgot to mention that.
So I'm glad you brought it up.
So in Nemotron paper they actually in some of the OMO stuff as well you basically think about let's say I give you a just Common Crawl right you can actually break this up so you can group by domain.
So they the AI2 folks have this like web organizer so you can group into topics but you can also quality filter. so then you have this like two-dimensional grid where you have domains and you have quality and each of these like cells is something that you would data mix.
So that's one automatic way to determine the domains and then on top of that you add ex extra sources that people hand you.
Okay.
So I'm going to quickly go through post training data here. so up until now this is basically pre-training or maybe mid-training data.
It's generally fairly task agnostic except for the reg mix stuff where you're you're trying to minimize the loss but still the data itself is still task relatively task agnostic and you're trying to develop basic skills.
So when you look at post- trainining a lot of the data becomes very kind of task you know dependent and I'm not going to do a comprehensive review I just want to point out some of interesting post-training you know data sets that have been recently released in particular for coding since that's of I think great interest these days. so the general recipe is you define a set of you know environments in the case think about code which might be GitHub you know repos you define a set of tasks or prompts and then you collect responses from a strong model or teacher.
So implicitly you know at least in the open community almost all the post- training data that's most of it is you know synthetically you know generated you can also replace strong model with a human but you know that is slow and costs a lot of money but you know if you're at the frontier and of if you I guess a few years ago you kind of had to go and pay a lot of people a lot of money to give you responses. and nowadays, even at the frontier, you can do sort of hybrid human AI things.

But anyway, the point is that there's some teacher out there that's giving you responses.
Okay.
So, a few kind of I'm just going to talk to a few works here.
So, one is this works, called open thoughts. and this idea here is this came around it was motivated by you know when 01 came out and there was a lot of intention on reasoning mostly for like math and science how do we get really good post-training data sets for this and you know they eventually came out with 1.2 two million examples using this teacher model. so the steps are first of all there are a bunch of so what are the environments and the tasks and questions. there are a lot of different sources that they drew from there human sources like sack exchange or num math and then there's like some synthetic sources as well.
These are just a subset of the coding ones, but there's also more for math and chemistry.
This is a big project that involved a lot of different contributions from many people. so you can see that there's you know some of these are like coding exercises, some of these are you know more realistic things that show up on I guess code golf is not realistic but there are some more realistic examples here. so code review this is probably more realistic. and they noticed that they did a fairly comprehensive analysis of you know given this data set what how do you generate responses.
They show that having a few sources was actually good but rather than trying to do all sources sampling multiple generations is helpful like 16. one thing that was interesting is that having better models aren't necessarily you know better teachers.
So for example QwQ-32B which is you know now a very old and small model is was a better teacher than DeepSeek-R1 which was at the time probably one of the strongest open models.  they also found that basic answer filtering wasn't helpful. and so the entire pipeline looks something you know like this where you have a bunch of these sources you know going in and then you duplicate you randomly you know sample I guess this down sample the questions and you generate multiple answers and that gets into the final data set.
So the 1.2 2 million isn't is examples but divided by 16 gives you the number of actual questions.

Okay.
So there's an another so that was for you know math and science and some code.
I think more recently there's been a lot of interest in developing in particular a gentic coding model.
So not just a model that can just generate some code but actually a model that can do software development.

so this SWE-smith paper had the idea of you're given a repository, you're using an language model to automatically generate tasks.
So they have an agent that takes a repository and actually makes it usable for like installing dependencies and so on. and then you generate tasks which are generally you know you modify the code in some way maybe introduce some bugs and those get verified and you get some task instances.
So these are synthetic tasks but you get you know 50k of them which is actually at the time which was last year quite large so since then there's been a bunch of other work which I'll you know mention so there's this SWE-Zero paper from you know Nvidia which had this kind of actually Okay, I'll also talk about that.
So this is a kind of an interesting idea where the observation was that unlike math su task have just a lot of heavy dependencies.
Most GitHub repos like don't even run and you have to install all these dependencies and out of date especially if you roll back to like when a PR was.
It's just kind of a mess. and this is just a nightmare. and so they were thinking about how can we actually get a data set that just on all the the repos and rather than having a repo specific docker image they noticed that the models are actually good enough that they can solve a lot of these tasks without execution feedback. so you know if you look at some of these models if you were allowing educ execution you get like 80 if you don't allow educa not education execution you get almost 70.
So which is you know not bad for not being able to execute code.
So somehow these models have some internal semantics of of code. so they were able to generate 300,000 agent trajectories. all of these are real GitHub you know PR so they're realistic unlike the Sweenith Smith examples. they use the open hands scaffold and did some there's a lot of details like how to prevent agent hacking.
So normally you hand an agent this instruction you explore you test you implement and their 30 version was basically saying you cannot run Python code you can only do like set and grap and all these like basic operations.
Okay.
So then they distilled from a big coding model and filtered because sometimes a coding model will ignore these instructions and still try to execute anyway. and then there they showed that with this oh they also had 13k agent trajectories that do require execution feedback.
So they train models where they first fine-tune on these SWE-Zero examples and then fine-tune again on these SWE-Hero examples and they were able to you know I guess the frontier is still pretty high up but they were able to make some you know progress here.
I'll quickly go through SWE-rebench was essentially another attempt to grab tons and tons of you know PRs. so and all of these kind of look like get a bunch of GitHub repos, try to install the repo, like most of them probably, you know, fail and try harder and then you use a language model to give you the responses.

So in this actually just came out today. so you can take the SWE-Zero idea and you can now scale up to 12 million agent trajectories. the nice benefit of SWE-Zero is that it's just you know almost it's very lightweight. here they use the SWE- rebench tasks.
So in this one remember they were trying to get things to execute.
They only got 32,000 of them to execute and 120 of them didn't execute.
So you just but zero doesn't care. you can use all of them. this is a very small model because now this data set is quite large. and so you know I think as you can see the data sets are you know getting more and more sophisticated.
You go from you know environment free things like math to now coding and now the coding data sets are growing quite a bit. but the general idea is that you know you have these prompts which you know there's kind of trade-offs.
You can have fully synthetic you can have semi-ynthetic where you have a real environment and synthetic tasks or you can have real and the responses generally are coming from just capable models but they need also need to be good teachers.
Code environments are a pain and you there's a lot of filtering and other details which we don't have time for.
Okay, to summarize this lecture, we talked about filtering.
So define what good looks like and then train a lightweight classifier.
Go over your web crawl and you can get a small subset that matches what you're looking for.
Deduplication is important to avoid overfitting and saving FLOPs. mixing try mixture at small scales extrapolate to large scale and then we looked at some you know post-training data I will say that though that a lot of the data work is can be very grungy it's very domain specific and requires looking at concrete examples to make these high quality data sets so this lecture is not really representative what data work is like but hopefully I've given you a idea of the data landscape out there.
Okay, that will be it for today.

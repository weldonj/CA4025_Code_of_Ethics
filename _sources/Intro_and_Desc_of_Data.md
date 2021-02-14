## DATASET AND TYPES OF RECOMMENDER SYSTEMS

### The Dataset

The dataset I am using here is a list of 200,000 Steam Video Game user behaviors, with user-id, game-title, behavior-name, value. The behaviors included are 'own' and 'play'. The behaviour I am interested in is 'play' as this is a numeric figure, representing how many hours the particular user has spent playing the games that they have purchased. I will be using this to create a derived rating score that the recommender system will then be able to use.

### System Types

**Collaborative Filtering -**

This approach to recommendation requires a lot of data about users’ preferences. Collaborative filtering doesn’t know anything about the content itself, it doesn’t attempt to analyse them at all. Instead it makes the assumption that if User A tends to give similar weights to content that User B does, they both enjoy similar content. Therefore, if User B likes content that User A has not been exposed to yet, it’s likely that User A will be happy to have that content recommended to them. This user preference data can either be explicit (the user actually rates the content) or implicit (analyse which content the user engages with most often and for long periods of time). Due to Collaborative Filtering’s reliance on user data, it suffers from what’s known as the “Cold Start” problem. Brand new users have no preference data, and so the system will always struggle to know what to recommend.

**Content-Based Filtering -**

Unlike the previous approach, here the content being recommended is also closely analysed. It can be a lot more difficult to collect the data as it’s not at all easy to identify how the individual features relate to user preferences. Two main data categories are used in content based music recommendation systems. Metadata about the content, and also the content itself. For music as an example, this can be features such as beats per minute, what chords are played in the song, and even just how loud it is. One big advantage that content based filtering has over collaborative filtering is that it can be difficult to get users to actually rate content, whereas the content itself can often be freely analysed.

**Hybrid Model -**

This is exactly as it sounds, with a blend of both collaborative filtering and content based filtering. These models can outperform either of the other two on their own, but it is difficult to find data that combines both user ratings and actual data about the content itself.

**My Approach -**

For this recommender system, I will be using Collaborative Filtering with Spark ML. The dataset doesn't contain any information about the video games themselves, just user preference, and so this is the only real choice that I have here.

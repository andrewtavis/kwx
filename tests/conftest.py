"""
Fixtures
--------
"""

import pytest
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
)  # required or the import within kwx.visuals will fail

from kwx import utils
from kwx.utils import load_data
from kwx.utils import _combine_tokens_to_str
from kwx.utils import _clean_text_strings
from kwx.utils import clean_and_tokenize_texts
from kwx.utils import prepare_data
from kwx.utils import _prepare_corpus_path
from kwx.utils import translate_output
from kwx.utils import organize_by_pos
from kwx.utils import prompt_for_word_removal

from kwx.autoencoder import Autoencoder
from kwx.topic_model import TopicModel

from kwx.languages import lem_abbr_dict
from kwx.languages import stem_abbr_dict
from kwx.languages import sw_abbr_dict

from kwx.visuals import save_vis
from kwx.visuals import graph_topic_num_evals
from kwx.visuals import gen_word_cloud
from kwx.visuals import pyLDAvis_topics
from kwx.visuals import t_sne

from kwx.model import get_topic_words
from kwx.model import get_coherence
from kwx.model import _order_and_subset_by_coherence
from kwx.model import extract_kws
from kwx.model import gen_files


texts = [
    "@VirginAmerica SFO-PDX schedule is still MIA.",
    "@VirginAmerica So excited for my first cross country flight LAX to MCO I've heard nothing but great things about Virgin America. #29DaysToGo",
    "@VirginAmerica  I flew from NYC to SFO last week and couldn't fully sit in my seat due to two large gentleman on either side of me. HELP!",
    "I ❤️ flying @VirginAmerica. ☺️👍",
    "@VirginAmerica you know what would be amazingly awesome? BOS-FLL PLEASE!!!!!!! I want to fly with only you.",
    "@VirginAmerica why are your first fares in May over three times more than other carriers when all seats are available to select???",
    "@VirginAmerica I love this graphic. http://t.co/UT5GrRwAaA",
    "@VirginAmerica I love the hipster innovation. You are a feel good brand.",
    "@VirginAmerica will you be making BOS&gt;LAS non stop permanently anytime soon?",
    "@VirginAmerica you guys messed up my seating.. I reserved seating with my friends and you guys gave my seat away ... 😡 I want free internet",
]

texts_long = [
    "@VirginAmerica What @dhepburn said.",
    "@VirginAmerica plus you've added commercials to the experience... tacky.",
    "@VirginAmerica I didn't today... Must mean I need to take another trip!",
    "@VirginAmerica it's really aggressive to blast obnoxious 'entertainment' in your guests' faces &amp; they have little recourse",
    "@VirginAmerica and it's a really big bad thing about it",
    "@VirginAmerica seriously would pay $30 a flight for seats that didn't have this playing. It's really the only bad thing about flying VA",
    "@VirginAmerica yes, nearly every time I fly VX this “ear worm” won’t go away :)",
    "@VirginAmerica Really missed a prime opportunity for Men Without Hats parody, there. https://t.co/mWpG7grEZP",
    "@virginamerica Well, I didn't…but NOW I DO! :-D",
    "@VirginAmerica it was amazing, and arrived an hour early. You're too good to me.",
    "@VirginAmerica did you know that suicide is the second leading cause of death among teens 10-24",
    "@VirginAmerica I &lt;3 pretty graphics. so much better than minimal iconography. :D",
    "@VirginAmerica This is such a great deal! Already thinking about my 2nd trip to @Australia &amp; I haven't even gone on my 1st trip yet! ;p",
    "@VirginAmerica @virginmedia I'm flying your #fabulous #Seductive skies again! U take all the #stress away from travel http://t.co/ahlXHhKiyn",
    "@VirginAmerica Thanks!",
    "@VirginAmerica SFO-PDX schedule is still MIA.",
    "@VirginAmerica So excited for my first cross country flight LAX to MCO I've heard nothing but great things about Virgin America. #29DaysToGo",
    "@VirginAmerica  I flew from NYC to SFO last week and couldn't fully sit in my seat due to two large gentleman on either side of me. HELP!",
    "I ❤️ flying @VirginAmerica. ☺️👍",
    "@VirginAmerica you know what would be amazingly awesome? BOS-FLL PLEASE!!!!!!! I want to fly with only you.",
    "@VirginAmerica why are your first fares in May over three times more than other carriers when all seats are available to select???",
    "@VirginAmerica I love this graphic. http://t.co/UT5GrRwAaA",
    "@VirginAmerica I love the hipster innovation. You are a feel good brand.",
    "@VirginAmerica will you be making BOS&gt;LAS non stop permanently anytime soon?",
    "@VirginAmerica you guys messed up my seating.. I reserved seating with my friends and you guys gave my seat away ... 😡 I want free internet",
    "@VirginAmerica status match program.  I applied and it's been three weeks.  Called and emailed with no response.",
    "@VirginAmerica What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
    "@VirginAmerica do you miss me? Don't worry we'll be together very soon.",
    "@VirginAmerica amazing to me that we can't get any cold air from the vents. #VX358 #noair #worstflightever #roasted #SFOtoBOS",
    "@VirginAmerica LAX to EWR - Middle seat on a red eye. Such a noob maneuver. #sendambien #andchexmix",
    "@VirginAmerica hi! I just bked a cool birthday trip with you, but i can't add my elevate no. cause i entered my middle name during Flight Booking Problems 😢",
    "@VirginAmerica Are the hours of operation for the Club at SFO that are posted online current?",
    "@VirginAmerica help, left expensive headphones on flight 89 IAD to LAX today. Seat 2A. No one answering L&amp;F number at LAX!",
    "@VirginAmerica awaiting my return phone call, just would prefer to use your online self-service option :(",
    "@VirginAmerica this is great news!  America could start flights to Hawaii by end of year http://t.co/r8p2Zy3fe4 via @Pacificbiznews",
    "Nice RT @VirginAmerica: Vibe with the moodlight from takeoff to touchdown. #MoodlitMonday #ScienceBehindTheExperience http://t.co/Y7O0uNxTQP",
    "@VirginAmerica Moodlighting is the only way to fly! Best experience EVER! Cool and calming. 💜✈ #MoodlitMonday",
    "@VirginAmerica @freddieawards Done and done! Best airline around, hands down!",
    "@VirginAmerica when can I book my flight to Hawaii??",
    "@VirginAmerica Your chat support is not working on your site: http://t.co/vhp2GtDWPk",
    "@VirginAmerica View of downtown Los Angeles, the Hollywood Sign, and beyond that rain in the mountains! http://t.co/Dw5nf0ibtr",
    "@VirginAmerica Hey, first time flyer next week - excited! But I'm having a hard time getting my flights added to my Elevate account. Help?",
    "@VirginAmerica plz help me win my bid upgrade for my flight 2/27 LAX---&gt;SEA!!!  🍷👍💺✈️",
    "@VirginAmerica I have an unused ticket but moved to a new city where you don't fly. How can I fly with you before it expires? #travelhelp",
    "@VirginAmerica are flights leaving Dallas for Seattle on time Feb 24?",
    "@VirginAmerica I'm #elevategold for a good reason: you rock!!",
    "@VirginAmerica  DREAM http://t.co/oA2dRfAoQ2 http://t.co/lWWdAc2kHx",
    "@VirginAmerica wow this just blew my mind",
    "@VirginAmerica @ladygaga @carrieunderwood After last night #tribute #SoundOfMusic #Oscars2015 @ladygaga! I think @carrieunderwood agree",
    "@VirginAmerica @ladygaga @carrieunderwood All were entertaining",
    "@VirginAmerica Is flight 769 on it's way? Was supposed to take off 30 minutes ago. Website still shows 'On Time' not 'In Flight'. Thanks.",
    "@VirginAmerica @ladygaga @carrieunderwood Julie Andrews all the way though @ladygaga was very impressive! NO to @Carrieunderwood",
    "@VirginAmerica wish you flew out of Atlanta... Soon?",
    "@VirginAmerica @ladygaga @carrieunderwood Julie Andrews. Hands down.",
    "@VirginAmerica Will flights be leaving Dallas for LA on February 24th?",
    "@VirginAmerica hi! i'm so excited about your $99 LGA-&gt;DAL deal- but i've been trying 2 book since last week &amp; the page never loads. thx!",
    "@VirginAmerica you know it. Need it on my spotify stat #guiltypleasures",
    "@VirginAmerica @ladygaga @carrieunderwood  I'm Lady Gaga!!! She is amazing! 😊",
    "@VirginAmerica @ladygaga @carrieunderwood - Carrie!",
    "@VirginAmerica New marketing song? https://t.co/F2LFULCbQ7 let us know what you think?",
    "@VirginAmerica @ladygaga @carrieunderwood Julie Andrews first but Lady Gaga wow'd me last night. Carrie? Meh.",
    "@VirginAmerica I called a 3-4 weeks ago about adding 3 flights from 2014 to my Elevate...they still haven't shown up...help!",
    "@VirginAmerica @ladygaga @carrieunderwood all are great , but I have to go with #CarrieUnderwood 😍👌",
    "@VirginAmerica @LadyGaga @CarrieUnderwood Sorry, Mary Martin had it first!",
    "@VirginAmerica @ladygaga @carrieunderwood  love all three but you really can't beat the classics!",
    "@VirginAmerica Flight 0736 DAL to DCA 2/24 2:10pm. Tried to check in could not. Status please.",
    "@VirginAmerica heyyyy guyyyys.. been trying to get through for an hour. can someone call me please? :/",
    "@VirginAmerica Hi, Virgin! I'm on hold for 40-50 minutes -- are there any earlier flights from LA to NYC tonight; earlier than 11:50pm?",
    "@VirginAmerica Congrats on winning the @Travelzoo award for Best Deals from an Airline (US) http://t.co/kj1iljaebV",
    "@VirginAmerica everything was fine until you lost my bag",
    "@virginamerica Need to change reservation. Have Virgin credit card. Do I need to modify on phone to waive change fee? Or can I do online?",
    "@VirginAmerica I emailed your customer service team. Let me know if you need the tracking number.",
    "@VirginAmerica hi I just booked a flight but need to add baggage, how can I do this?",
    "@VirginAmerica your airline is awesome but your lax loft needs to step up its game. $40 for dirty tables and floors? http://t.co/hy0VrfhjHt",
    "@VirginAmerica not worried, it's been a great ride in a new plane with great crew. All airlines should be like this.",
    "@VirginAmerica awesome. I flew yall Sat morning. Any way we can correct my bill ?",
    "@VirginAmerica Or watch some of the best student films in the country at 35,000 feet! #CMFat35000feet http://t.co/KEK5pDMGiF",
    "@VirginAmerica first time flying you all. do you have a different rate/policy for media Bags? Thanks",
    "@VirginAmerica what is going on with customer service? Is there anyway to speak to a human asap? Thank you.",
    "@VirginAmerica what happened to Doom?!",
    "@VirginAmerica why can't you supp the biz traveler like @SouthwestAir  and have customer service like @JetBlue #neverflyvirginforbusiness",
    "@VirginAmerica I've applied more then once to be a member of the #inflight crew team...Im 100% interested. #flightattendant #dreampath -G",
    "@VirginAmerica you're the best!! Whenever I (begrudgingly) use any other airline I'm delayed and Late Flight :(",
    "@VirginAmerica I have no interesting flying with you after this. I will Cancelled Flight my next four flights I planned.#neverflyvirginforbusiness",
    "@VirginAmerica it was a disappointing experience which will be shared with every business traveler I meet. #neverflyvirgin",
    "@VirginAmerica I’m having trouble adding this flight my wife booked to my Elevate account. Help? http://t.co/pX8hQOKS3R",
    "@VirginAmerica Can't bring up my reservation online using Flight Booking Problems code",
    "@VirginAmerica Random Q: what's the distribution of elevate avatars? I bet that kitty has a disproportionate share http://t.co/APtZpuROp4",
    "@VirginAmerica I &lt;3 Flying VA But Life happens and I am trying to #change my trip JPERHI  Can you help.VA home page will not let me ?",
    "@VirginAmerica Why is the site down?  When will it be back up?",
    "@VirginAmerica 'You down with RNP?' 'Yeah you know me!'",
    "@VirginAmerica hi, i did not get points on my elevate account for my most recent flight, how do i add the flight and points to my account?",
    "@VirginAmerica I like the TV and interesting video . Just disappointed in Cancelled Flightled flight when other flights went out to jfk on Saturday .",
    "@VirginAmerica just landed in LAX, an hour after I should of been here. Your no Late Flight bag check is not business travel friendly #nomorevirgin",
    "@VirginAmerica why is flight 345 redirected?",
    "@VirginAmerica Is it me, or is your website down?  BTW, your new website isn't a great user experience.  Time for another redesign.",
    "@VirginAmerica I can't check in or add a bag. Your website isn't working. I've tried both desktop and mobile http://t.co/AvyqdMpi1Y",
    "@VirginAmerica - Let 2 scanned in passengers leave the plane than told someone to remove their bag from 1st class bin? #uncomfortable",
    "@virginamerica What is your phone number. I can't find who to call about a flight reservation.",
]


@pytest.fixture(params=[texts])
def list_texts(request):
    return request.param


@pytest.fixture(params=[pd.DataFrame(texts, columns=["text"])])
def df_texts(request):
    return request.param


@pytest.fixture(
    params=[
        utils.clean_and_tokenize_texts(
            texts=texts,
            input_language="english",
            min_freq=2,
            min_word_len=3,
            sample_size=1,
        )[0]
    ]
)
def short_text_corpus(request):
    return request.param


@pytest.fixture(
    params=[
        utils.clean_and_tokenize_texts(
            texts=texts_long,
            input_language="english",
            min_freq=2,
            min_word_len=3,
            sample_size=1,
        )[0]
    ]
)
def long_text_corpus(request):
    return request.param

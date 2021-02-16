"""
Fixtures
--------
"""

import pytest
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
)  # required or the import within kwx.visuals will fail

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


@pytest.fixture(params=[texts])
def list_texts(request):
    return request.param


@pytest.fixture(params=[pd.DataFrame(texts, columns=["text"])])
def df_texts(request):
    return request.param

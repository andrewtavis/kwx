"""
Fixtures
--------
"""

import pytest
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


@pytest.fixture(
    params=[
        [
            "This",
            "is",
            "a",
            "miniature",
            "corpus",
            "that",
            "has",
            "exactly",
            "ten",
            "words",
        ]
    ]
)
def ten_word_corpus(request):
    return request.param
